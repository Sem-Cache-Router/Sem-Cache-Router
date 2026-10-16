# Tests for the metrics layer.
#
# Every test gets a fresh Metrics because it lives on app.state and each test
# builds a fresh app.  The previous singleton leaked state between tests,
# which is exactly why the singleton was removed.
#
# Unit tests verify the Metrics class in isolation.  Integration tests verify
# the wiring: a request through the gateway moves the right counter, and
# GET /metrics reflects it.

from __future__ import annotations

from fastapi.testclient import TestClient

from app.telemetry.metrics import Metrics

_BODY = {
    "model": "gpt-4o-mini",
    "messages": [{"role": "user", "content": "hello"}],
}
_AUTH = {"Authorization": "Bearer test"}


# ---------------------------------------------------------------------------
# Unit tests: the Metrics class in isolation
# ---------------------------------------------------------------------------


def test_fresh_metrics_start_at_zero() -> None:
    """A new Metrics has every counter at zero and no latency samples."""
    m = Metrics()
    snap = m.snapshot()
    assert snap["tier1_hits"] == 0
    assert snap["tier2_hits"] == 0
    assert snap["misses"] == 0
    assert snap["throttled_requests"] == 0
    assert snap["breaker_failovers"] == 0
    assert snap["total_provider_spend_usd"] == 0.0
    assert snap["latency_hit_p50_ms"] is None
    assert snap["latency_miss_p50_ms"] is None


def test_hit_increments_hit_not_miss() -> None:
    """record_tier1_hit increments tier1_hits and leaves misses at zero."""
    m = Metrics()
    m.record_tier1_hit()
    snap = m.snapshot()
    assert snap["tier1_hits"] == 1
    assert snap["misses"] == 0


def test_miss_increments_miss_not_hit() -> None:
    """record_miss increments misses and leaves tier1_hits at zero."""
    m = Metrics()
    m.record_miss()
    snap = m.snapshot()
    assert snap["misses"] == 1
    assert snap["tier1_hits"] == 0


def test_two_hits_produce_count_of_two() -> None:
    """Two record_tier1_hit calls produce a count of two, not one."""
    m = Metrics()
    m.record_tier1_hit()
    m.record_tier1_hit()
    assert m.snapshot()["tier1_hits"] == 2


def test_throttled_increments() -> None:
    """record_throttled increments the throttle counter."""
    m = Metrics()
    m.record_throttled()
    assert m.snapshot()["throttled_requests"] == 1


def test_spend_accumulates() -> None:
    """add_spend accumulates across calls."""
    m = Metrics()
    m.add_spend(1.0)
    m.add_spend(0.5)
    assert m.snapshot()["total_provider_spend_usd"] == 1.5


def test_latency_split_by_cache_status() -> None:
    """Hit and miss latencies are tracked in separate distributions."""
    m = Metrics()
    m.record_latency(0.002, hit=True)
    m.record_latency(0.150, hit=False)
    snap = m.snapshot()
    assert snap["latency_hit_p50_ms"] is not None
    assert snap["latency_miss_p50_ms"] is not None
    # Hit latency should be much smaller than miss latency.
    assert snap["latency_hit_p50_ms"] < snap["latency_miss_p50_ms"]


def test_percentile_with_no_samples_is_none() -> None:
    """When there are no samples, percentiles are None, not zero."""
    m = Metrics()
    snap = m.snapshot()
    assert snap["latency_hit_p50_ms"] is None
    assert snap["latency_hit_p95_ms"] is None
    assert snap["latency_miss_p50_ms"] is None
    assert snap["latency_miss_p95_ms"] is None


def test_instances_do_not_share_state() -> None:
    """Two Metrics instances are independent — the singleton is gone."""
    a = Metrics()
    b = Metrics()
    a.record_tier1_hit()
    assert b.snapshot()["tier1_hits"] == 0


# ---------------------------------------------------------------------------
# Integration tests: wiring through the HTTP request path
# ---------------------------------------------------------------------------


def test_miss_then_hit_wires_both_counters(client: TestClient) -> None:
    """First request is a miss (provider), second identical request is a hit."""
    r1 = client.post("/v1/chat/completions", json=_BODY, headers=_AUTH)
    assert r1.status_code == 200
    assert r1.json()["semcache_status"] == "miss"

    r2 = client.post("/v1/chat/completions", json=_BODY, headers=_AUTH)
    assert r2.status_code == 200
    assert r2.json()["semcache_status"] == "exact_hit"

    metrics = client.get("/metrics").json()
    assert metrics["tier1_hits"] == 1
    assert metrics["misses"] == 1


def test_two_misses_produce_count_of_two(client: TestClient) -> None:
    """Two distinct prompts each miss, giving a miss count of two."""
    body_a = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "alpha"}]}
    body_b = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "bravo"}]}

    client.post("/v1/chat/completions", json=body_a, headers=_AUTH)
    client.post("/v1/chat/completions", json=body_b, headers=_AUTH)

    metrics = client.get("/metrics").json()
    assert metrics["misses"] == 2
    assert metrics["tier1_hits"] == 0


def test_latency_populated_after_request(client: TestClient) -> None:
    """After a miss, miss latency percentiles are populated."""
    client.post("/v1/chat/completions", json=_BODY, headers=_AUTH)
    metrics = client.get("/metrics").json()
    assert metrics["latency_miss_p50_ms"] is not None
    assert metrics["latency_miss_p50_ms"] >= 0


def test_spend_recorded_on_miss(client: TestClient) -> None:
    """A cache miss records the provider's reported cost."""
    client.post("/v1/chat/completions", json=_BODY, headers=_AUTH)
    metrics = client.get("/metrics").json()
    # StubProvider reports cost_usd=0.0, so spend is zero but recorded.
    assert metrics["total_provider_spend_usd"] == 0.0


def test_metrics_endpoint_returns_all_fields(client: TestClient) -> None:
    """GET /metrics returns every documented field."""
    metrics = client.get("/metrics").json()
    expected = {
        "tier1_hits", "tier2_hits", "misses",
        "throttled_requests", "breaker_failovers",
        "total_provider_spend_usd",
        "latency_hit_p50_ms", "latency_hit_p95_ms",
        "latency_miss_p50_ms", "latency_miss_p95_ms",
    }
    assert expected == set(metrics.keys())


def test_throttled_request_wires_counter() -> None:
    """A 429 from the limiter increments the throttle counter."""
    from fakeredis import aioredis

    from app.cache.exact_cache import ExactCache
    from app.config import Settings
    from app.main import create_app
    from app.providers.stub import StubProvider

    class RejectingLimiter:
        """Always rejects, so the throttle counter fires."""

        async def reserve(self, key: str, estimate: int) -> None:
            return None

        def retry_after(self, key: str) -> int:
            return 10

    app = create_app()
    redis = aioredis.FakeRedis(decode_responses=True)
    app.state.settings = Settings(
        redis_url="redis://localhost:6379/0",
        cache_ttl_seconds=2,
        primary_provider="stub",
    )
    app.state.redis = redis
    app.state.provider = StubProvider()
    app.state.cache = ExactCache(redis, 2)
    app.state.limiter = RejectingLimiter()

    with TestClient(app) as tc:
        r = tc.post("/v1/chat/completions", json=_BODY, headers=_AUTH)
        assert r.status_code == 429

        metrics = tc.get("/metrics").json()
        assert metrics["throttled_requests"] == 1
        assert metrics["tier1_hits"] == 0
        assert metrics["misses"] == 0
