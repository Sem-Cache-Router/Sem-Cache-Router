# Liveness and dependency reachability.
#
# Why dependency state is reported rather than folded into a single up or down:
# losing ChromaDB degrades the gateway to Tier 1 while losing Redis stops it
# entirely, and an operator has to be able to tell those apart at a glance.
#
# This endpoint never raises. A health check that returns 500 tells you less
# than one that returns a body saying which dependency is unreachable.

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


async def _redis_state(request: Request) -> str:
    """ok, unreachable, or not_configured."""
    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        return "not_configured"
    try:
        await redis.ping()
    except Exception:
        # Deliberately broad. Any failure to reach Redis is the same fact to an
        # operator, and narrowing it would mean importing and tracking every
        # exception the client might raise.
        return "unreachable"
    return "ok"


async def _chroma_state(request: Request) -> str:
    """ok, unreachable, or disabled when Tier 2 is switched off.

    A real round trip through the semantic cache, not a cached belief, since a
    vector store can vanish after the gateway connected to it. is_available
    never raises, so neither does this.
    """
    semantic_cache = getattr(request.app.state, "semantic_cache", None)
    if semantic_cache is None:
        return "disabled"
    return "ok" if await semantic_cache.is_available() else "unreachable"


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Report liveness plus whether each dependency can be reached.

    Status is "ok" only when every configured dependency answers. Losing Redis
    and losing Chroma both report "degraded", and the dependencies map says
    which: an unreachable Redis means requests fail, while an unreachable Chroma
    means they are served from Tier 1 alone. The distinction lives in the map
    because a single word cannot carry it.
    """
    dependencies = {
        "redis": await _redis_state(request),
        "chroma": await _chroma_state(request),
    }
    healthy = dependencies["redis"] == "ok" and dependencies["chroma"] in ("ok", "disabled")

    return {
        "status": "ok" if healthy else "degraded",
        "dependencies": dependencies,
    }


@router.get("/metrics")
async def get_metrics(request: Request) -> dict[str, Any]:
    """Expose aggregate counters and latency percentiles for benchmarking.

    Reads from app.state.metrics rather than a process-global singleton, so
    each test gets its own instance and counters never leak between cases.
    """
    return request.app.state.metrics.snapshot()
