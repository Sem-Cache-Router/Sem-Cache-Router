# Building the providers, their breakers and the failover router from settings.
#
# Kept out of main.py so the lifespan stays a list of what is wired rather than how,
# and so the construction can be tested without starting an app. The rule is the
# lifespan's own: build only what is missing from app.state, so a test that seeds a
# provider or a whole router is never overridden, and close only what was opened
# here, so a client a test supplied is not closed underneath it.

from __future__ import annotations

from typing import Any

import httpx

from app.config import ProviderName, Settings
from app.providers.anthropic_client import AnthropicClient
from app.providers.base import LLMProvider
from app.providers.breaker import CircuitBreaker
from app.providers.openai_client import OpenAIClient
from app.providers.router import FailoverRouter, ProviderSlot
from app.providers.stub import StubProvider

# The model a fallback is called with when settings do not name one. One vendor's
# model name means nothing to another, so a fallback cannot be called with the
# model the caller asked the primary for.
DEFAULT_FALLBACK_MODEL: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
}


def uses_a_real_provider(settings: Settings) -> bool:
    """Whether the configured primary or secondary needs an HTTP client."""
    configured = {settings.primary_provider, settings.secondary_provider}
    return bool(configured & {"openai", "anthropic"})


def build_provider(
    name: ProviderName, settings: Settings, http_client: httpx.AsyncClient | None
) -> LLMProvider:
    """Construct the named provider.

    A real provider refuses to construct without its credentials, which is what
    turns a missing key into a gateway that will not start rather than one that
    fails its first request.
    """
    if name == "stub":
        return StubProvider(latency_ms=settings.stub_latency_ms)
    if http_client is None:
        raise ValueError(f"provider {name!r} needs an HTTP client")
    if name == "openai":
        return OpenAIClient(http_client, settings)
    return AnthropicClient(http_client, settings)


def build_breaker(settings: Settings) -> CircuitBreaker:
    """One breaker from the settings. Every provider gets its own, never a shared one."""
    return CircuitBreaker(
        failure_threshold=settings.breaker_failure_threshold,
        window_seconds=settings.breaker_window_seconds,
        base_backoff=settings.breaker_base_backoff,
        backoff_ceiling=settings.breaker_backoff_ceiling,
    )


def build_router(
    settings: Settings, http_client: httpx.AsyncClient | None, primary: LLMProvider
) -> FailoverRouter:
    """The primary behind its breaker, and the secondary after it if one is configured.

    The primary is passed in rather than built here, because it is also what the
    rest of the app reads as app.state.provider, and the two must be one object.
    """
    slots = [ProviderSlot(primary, build_breaker(settings))]
    if settings.secondary_provider is not None:
        try:
            secondary = build_provider(settings.secondary_provider, settings, http_client)
        except ValueError as exc:
            raise ValueError(
                f"SEMCACHE_SECONDARY_PROVIDER is {settings.secondary_provider} but it cannot "
                f"be built: {exc}. Provide its credentials, or unset the setting to run "
                "without failover."
            ) from exc
        model = settings.secondary_model or DEFAULT_FALLBACK_MODEL.get(settings.secondary_provider)
        slots.append(ProviderSlot(secondary, build_breaker(settings), model))
    return FailoverRouter(slots)


async def install_providers(state: Any, settings: Settings) -> httpx.AsyncClient | None:
    """Build whatever of state.provider and state.router is missing.

    Returns the HTTP client it opened, if it opened one, so the caller can close
    it on shutdown; None means there is nothing to close. The client is opened only
    when a real provider is actually going to be built, so the default stub setup
    holds no connection pool.
    """
    building_provider = getattr(state, "provider", None) is None
    building_router = getattr(state, "router", None) is None
    if not (building_provider or building_router) or not uses_a_real_provider(settings):
        http_client = None
    else:
        http_client = httpx.AsyncClient()

    try:
        if building_provider:
            state.provider = build_provider(settings.primary_provider, settings, http_client)
        if building_router:
            state.router = build_router(settings, http_client, primary=state.provider)
    except Exception:
        # A missing key is a startup failure, and must not leave a pool behind it.
        if http_client is not None:
            await http_client.aclose()
        raise
    return http_client
