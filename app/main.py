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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api import chat, health
from app.cache.exact_cache import ExactCache
from app.config import Settings, get_settings
from app.errors import GatewayError, request_id
from app.models import ErrorCode, ErrorDetail, ErrorResponse
from app.providers.base import LLMProvider
from app.providers.stub import StubProvider


def build_provider(name: str, settings: Settings) -> LLMProvider:
    """Construct the configured provider.

    Only the stub exists today. The real adapters slot in here without the route
    changing, which is the point of the common interface.
    """
    if name == "stub":
        return StubProvider()
    raise ValueError(
        f"provider {name!r} is configured but not implemented yet; "
        "set SEMCACHE_PRIMARY_PROVIDER=stub"
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

    if getattr(app.state, "provider", None) is None:
        app.state.provider = build_provider(settings.primary_provider, settings)

    if getattr(app.state, "cache", None) is None:
        app.state.cache = ExactCache(app.state.redis, settings.cache_ttl_seconds)

    # The admission controller is not built yet. None rather than a null object,
    # because a null object would force reserve and reconcile to be given
    # signatures now, before Reservation exists and before tiktoken has shown
    # what an estimate looks like. The route guards on it explicitly, so the
    # position of the check is fixed even though the implementation is not.
    if getattr(app.state, "limiter", None) is None:
        app.state.limiter = None

    try:
        yield
    finally:
        if owns_redis:
            await app.state.redis.aclose()


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

    def envelope(
        status_code: int,
        code: ErrorCode,
        message: str,
        request_id: str,
        retry_after_seconds: int | None = None,
    ) -> JSONResponse:
        """Render the one error shape the API is documented to return."""
        body = ErrorResponse(
            error=ErrorDetail(
                code=code,
                message=message,
                retry_after_seconds=retry_after_seconds,
                request_id=request_id,
            )
        )
        return JSONResponse(
            status_code=status_code,
            content=body.model_dump(mode="json"),
            headers={"x-request-id": request_id},
        )

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Map a schema failure to 400, not FastAPI's default 422.

        The published error contract says 400 for a malformed body. FastAPI
        returns 422 out of the box, so without this handler the API would
        quietly disagree with its own documentation.
        """
        return envelope(
            400,
            ErrorCode.INVALID_REQUEST,
            "Request body failed schema validation.",
            request_id(request),
        )

    @app.exception_handler(GatewayError)
    async def on_gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        """Render any error the routes raise deliberately."""
        return envelope(
            exc.status_code,
            exc.code,
            exc.message,
            request_id(request),
            exc.retry_after_seconds,
        )

    return app


app = create_app()
