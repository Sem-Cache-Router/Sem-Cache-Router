# What the gateway is running with, for the showcase page and for operators.
#
# Holds two read-only routes:
#   GET /config     the settings that shape behaviour: cache tiers, threshold,
#                   TTL, limiter budget, providers and breaker, as plain values
#   GET /benchmark  the committed results of the benchmark sweep
#
# /config never returns a secret. It is built from an explicit list of fields
# rather than by dumping the settings and removing the dangerous ones, so a
# credential added to the settings later cannot appear here by accident; for the
# key settings it says only whether one is set.
#
# /benchmark serves app/static/benchmark.json, written by
# `python -m benchmarks.report ... --json app/static/benchmark.json`. It says
# whether the runs were simulated, and the page shows that beside the numbers.

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

router = APIRouter(tags=["info"])

BENCHMARK = Path(__file__).resolve().parent.parent / "static" / "benchmark.json"

# (group, setting, label) for every value /config shows. Secrets are never listed.
SHOWN = (
    ("cache", "semantic_cache_enabled", "Semantic cache (Tier 2)"),
    ("cache", "similarity_threshold", "Similarity threshold"),
    ("cache", "cache_ttl_seconds", "Entry lifetime, seconds"),
    ("cache", "semantic_negation_guard", "Negation guard"),
    ("cache", "tier2_max_entries", "Tier 2 capacity"),
    ("cache", "embedding_model", "Embedding model"),
    ("cache", "chroma_embedded", "Chroma inside the gateway"),
    ("limiter", "limiter_enabled", "Rate limiter"),
    ("limiter", "bucket_capacity", "Budget per key, tokens"),
    ("limiter", "bucket_refill_rate", "Refill, tokens per second"),
    ("limiter", "limiter_default_completion_tokens", "Reply allowance, tokens"),
    ("providers", "primary_provider", "Primary provider"),
    ("providers", "secondary_provider", "Secondary provider"),
    ("providers", "provider_timeout_seconds", "Provider timeout, seconds"),
    ("providers", "breaker_failure_threshold", "Breaker opens after, failures"),
    ("providers", "breaker_base_backoff", "Breaker backoff, seconds"),
    ("access", "require_api_key", "API key required"),
    ("access", "max_request_bytes", "Largest request, bytes"),
)


@router.get("/config")
async def config(request: Request) -> dict[str, Any]:
    """The behaviour shaping settings, grouped, with no secret in them."""
    settings = request.app.state.settings
    groups: dict[str, list[dict[str, Any]]] = {}
    for group, name, label in SHOWN:
        groups.setdefault(group, []).append(
            {"setting": name, "label": label, "value": getattr(settings, name)}
        )
    groups["access"].append({
        "setting": "allowed_api_keys",
        "label": "API key allowlist",
        "value": bool(settings.allowed_api_keys),
    })
    groups["access"].append({
        "setting": "admin_api_key",
        "label": "Admin routes",
        "value": bool(settings.admin_api_key),
    })
    return groups


@router.get("/benchmark")
async def benchmark() -> dict[str, Any]:
    """The committed benchmark summary, or 404 when none has been committed."""
    if not BENCHMARK.exists():
        raise HTTPException(status_code=404, detail="No benchmark results committed")
    result: dict[str, Any] = json.loads(BENCHMARK.read_text())
    return result
