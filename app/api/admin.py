# Cache administration.
#
# Holds two routes for operators:
#   GET    /admin/cache   how many entries each tier holds
#   DELETE /admin/cache   empty both tiers
#
# Why they exist: a benchmark run has to start cold, and a run started on a warm
# cache reports a hit ratio that measures the previous run. Until now that meant
# reaching into Redis and Chroma by hand (scripts/reset-cache.sh); in containers
# neither store is published, so an endpoint is the only way in.
#
# Both need SEMCACHE_ADMIN_API_KEY as the bearer token, compared in constant
# time. Without the setting the routes answer 404, as if they were not there,
# so a default deployment exposes no way to empty its cache.
#
# A Chroma outage does not fail either route. Tier 2 is reported as null with
# its state, and Tier 1 is still counted or cleared.

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from app.errors import GatewayError
from app.models import ErrorCode

router = APIRouter(tags=["admin"])


def _require_admin(request: Request) -> None:
    """Refuse unless admin is configured and the bearer token is the admin key."""
    expected = request.app.state.settings.admin_api_key
    if not expected:
        raise HTTPException(status_code=404, detail="Not Found")
    header = request.headers.get("authorization", "")
    token = header.removeprefix("Bearer ").strip() if header.startswith("Bearer ") else ""
    if not token:
        raise GatewayError(ErrorCode.MISSING_API_KEY, "The admin key is required.",
                           status_code=401)
    if not hmac.compare_digest(token.encode(), expected.encode()):
        raise GatewayError(ErrorCode.INVALID_API_KEY, "That is not the admin key.",
                           status_code=403)


def _tier2_state(semantic_cache: Any, value: int | None) -> str:
    if semantic_cache is None:
        return "disabled"
    return "unavailable" if value is None else "ok"


@router.get("/admin/cache")
async def cache_stats(request: Request) -> dict[str, Any]:
    """Count the entries in each tier."""
    _require_admin(request)
    state = request.app.state
    semantic_cache = getattr(state, "semantic_cache", None)
    tier2 = await semantic_cache.count() if semantic_cache is not None else None
    return {
        "tier1_entries": await state.cache.count(),
        "tier2_entries": tier2,
        "tier2": _tier2_state(semantic_cache, tier2),
    }


@router.delete("/admin/cache")
async def flush_cache(request: Request) -> dict[str, Any]:
    """Empty both tiers and report how many entries each lost."""
    _require_admin(request)
    state = request.app.state
    semantic_cache = getattr(state, "semantic_cache", None)
    tier2 = await semantic_cache.clear() if semantic_cache is not None else None
    return {
        "tier1_removed": await state.cache.clear(),
        "tier2_removed": tier2,
        "tier2": _tier2_state(semantic_cache, tier2),
    }
