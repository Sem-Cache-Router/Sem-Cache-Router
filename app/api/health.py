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


@router.get("/health")
async def health(request: Request) -> dict[str, Any]:
    """Report liveness plus whether each dependency can be reached.

    Status is "ok" only when nothing required is missing. Redis is required,
    because admission cannot be decided without it, so an unreachable Redis is
    reported as degraded rather than glossed over.
    """
    dependencies: dict[str, str] = {}

    redis = getattr(request.app.state, "redis", None)
    if redis is None:
        dependencies["redis"] = "not_configured"
    else:
        try:
            await redis.ping()
            dependencies["redis"] = "ok"
        except Exception:
            # Deliberately broad. Any failure to reach Redis is the same fact to
            # an operator, and narrowing it would mean importing and tracking
            # every exception the client might raise.
            dependencies["redis"] = "unreachable"

    # Tier 2 is not built. Reported honestly as absent rather than stubbed as
    # healthy, because a health endpoint that lies about a dependency it never
    # checked is worse than one that admits the gap.
    dependencies["chroma"] = "not_configured"

    healthy = dependencies["redis"] == "ok"

    return {
        "status": "ok" if healthy else "degraded",
        "dependencies": dependencies,
    }
