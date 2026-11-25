# Application entrypoint.
#
# Why wiring happens in lifespan rather than at import time: nothing should
# connect to an external service just because a module was imported, and the
# test suite needs to substitute a fake Redis and a stub provider without
# patching module globals.
#
# The module level `app` below does not contradict that. create_app only builds
# routes and handlers; the lifespan, which does the connecting, runs on startup
# rather than on import. That distinction is what lets `uvicorn app.main:app`
# work while the smoke test still asserts that importing opens nothing.

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import chat, health
from app.cache.embedder import OnnxEmbedder
from app.cache.eviction import start_eviction, stop_eviction
from app.cache.exact_cache import ExactCache
from app.cache.semantic_cache import SemanticCache
from app.config import Settings, get_settings
from app.errors import GatewayError, error_response, request_id
from app.models import ErrorCode
from app.providers.factory import install_providers
from app.request_limits import BodySizeLimit
from app.telemetry.metrics import Metrics
from app.telemetry.tracing import install_tracing


def chroma_client_factory(settings: Settings) -> Callable[[], Any]:
    """Return a function that builds a Chroma client from settings.

    A factory, not a client: the Chroma client raises at construction when the
    server is down, so building it here would stop the gateway starting during
    a vector store outage. The semantic cache calls this when it first needs a
    connection, and again after each reconnect cooldown.
    """
    host, port, ssl = settings.chroma_endpoint

    def factory() -> Any:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        return chromadb.HttpClient(
            host=host,
            port=port,
            ssl=ssl,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    return factory


async def build_semantic_cache(settings: Settings) -> SemanticCache:
    """Load the embedder and build the Tier 2 cache.

    The embedder is loaded here, off the event loop, so no request pays for it.
    A failure to load it stops startup, deliberately, and that is the opposite
    of how a Chroma outage is handled. The weights are baked into the image, so
    failing to load them is a broken deployment that should be loud. Chroma is
    a remote service whose outages are transient, so losing it degrades instead.
    """
    embedder = OnnxEmbedder(settings.embedding_model)
    await asyncio.to_thread(embedder.load)
    return SemanticCache(
        chroma_client_factory(settings),
        embedder,
        collection_name=settings.chroma_collection,
        ttl_seconds=settings.cache_ttl_seconds,
        threshold=settings.similarity_threshold,
        max_entries=settings.tier2_max_entries,
        reconnect_seconds=settings.chroma_reconnect_seconds,
        timeout_seconds=settings.chroma_timeout_seconds,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Build dependencies and attach them to app.state, then tear them down.

    Every dependency is created only if it is not already there. That one
    getattr per dependency is the whole test seam: a test sets app.state.redis
    to a fake before entering the client context and the lifespan leaves it
    alone. No dependency injection framework, and no patching of module globals.

    Only connections this function opened are closed on the way out, so a fake
    handed in by a test is not closed underneath it.
    """
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings

    owns_redis = False
    if getattr(app.state, "redis", None) is None:
        from redis.asyncio import Redis

        app.state.redis = Redis.from_url(settings.redis_url, decode_responses=True)
        owns_redis = True

    # The provider, and the router that puts it and any secondary behind their own
    # breakers. The client they share is closed below, only if this opened it.
    http_client = await install_providers(app.state, settings)

    if getattr(app.state, "cache", None) is None:
        app.state.cache = ExactCache(app.state.redis, settings.cache_ttl_seconds)

    # The admission controller is not built yet. None rather than a null object,
    # because a null object would force reserve and reconcile to be given
    # signatures now, before Reservation exists and before tiktoken has shown
    # what an estimate looks like. The route guards on it explicitly, so the
    # position of the check is fixed even though the implementation is not.
    if getattr(app.state, "limiter", None) is None:
        app.state.limiter = None

    if getattr(app.state, "metrics", None) is None:
        app.state.metrics = Metrics(settings.metrics_latency_window)

    # Tier 2. None when switched off, and the route guards on it, the same
    # pattern as the limiter, so the position of the lookup in the pipeline is
    # fixed whether or not a vector store is configured.
    #
    # The housekeeping task is started only for a cache built here, the same rule
    # as every connection: a cache a test supplied is not ours to run jobs on.
    eviction_task: asyncio.Task[None] | None = None
    if getattr(app.state, "semantic_cache", None) is None:
        app.state.semantic_cache = (
            await build_semantic_cache(settings) if settings.semantic_cache_enabled else None
        )
        eviction_task = start_eviction(
            app.state.semantic_cache,
            interval_seconds=settings.eviction_interval_seconds,
            on_evicted=app.state.metrics.record_evicted,
        )

    # Tracing. install_tracing returns None when tracing is disabled, which
    # is the default, so the gateway starts with no collector and no errors.
    owns_tracing = False
    if getattr(app.state, "tracer_provider", None) is None:
        provider = install_tracing(
            endpoint=settings.otel_exporter_endpoint,
            enabled=settings.tracing_enabled,
        )
        app.state.tracer_provider = provider
        owns_tracing = provider is not None

    try:
        yield
    finally:
        await stop_eviction(eviction_task)
        if owns_tracing and app.state.tracer_provider is not None:
            app.state.tracer_provider.shutdown()
        if owns_redis:
            await app.state.redis.aclose()
        if http_client is not None:
            await http_client.aclose()


def create_app() -> FastAPI:
    """Construct the app, register routers and install the error handlers.

    A factory rather than a module level app so a test can build an isolated
    instance per case and seed its state before startup runs.
    """
    app = FastAPI(
        title="SemCache-Router",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(chat.router)
    app.add_middleware(BodySizeLimit)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Map a schema failure to 400, not FastAPI's default 422.

        The published error contract says 400 for a malformed body. FastAPI
        returns 422 out of the box, so without this handler the API would
        quietly disagree with its own documentation.
        """
        return error_response(
            400,
            ErrorCode.INVALID_REQUEST,
            "Request body failed schema validation.",
            request_id(request),
        )

    @app.exception_handler(GatewayError)
    async def on_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        """Render any error the routes raise deliberately."""
        return error_response(
            exc.status_code,
            exc.code,
            exc.message,
            request_id(request),
            exc.retry_after_seconds,
        )

    return app


app = create_app()
