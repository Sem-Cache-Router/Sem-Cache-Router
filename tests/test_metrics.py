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

import pytest
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
    assert snap["tier2_evicted_total"] == 0
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


def test_evictions_accumulate_across_passes() -> None:
    """record_evicted adds the size of each pass, so the total is entries removed to date."""
    m = Metrics()
    m.record_evicted(3)
    m.record_evicted(2)
    assert m.snapshot()["tier2_evicted_total"] == 5


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
    """Two Metrics instances are independent -- the singleton is gone."""
    a = Metrics()
    b = Metrics()
    a.record_tier1_hit()
    assert b.snapshot()["tier1_hits"] == 0


# ---------------------------------------------------------------------------
# Bounded deque behaviour
# ---------------------------------------------------------------------------


def test_window_evicts_oldest_samples() -> None:
    """After 3N records with window=N only the N most recent samples remain."""
    n = 5
    m = Metrics(latency_window=n)
    # First batch: values 0.001, 0.002, ..., 0.005 (old)
    for i in range(1, n + 1):
        m.record_latency(i * 0.001, hit=True)
    # Second batch: values 0.100, 0.200, ..., 0.500 (also old, but newer than first)
    for i in range(1, n + 1):
        m.record_latency(i * 0.100, hit=True)
    # Third batch: values 1.0, 2.0, ..., 5.0 (the most recent N, should be kept)
    for i in range(1, n + 1):
        m.record_latency(i * 1.0, hit=True)

    snap = m.snapshot()
    # Only the N most recent samples are kept.
    assert snap["hit_sample_count"] == n
    # The oldest batches are gone; the minimum kept sample is 1.0 s = 1000 ms.
    assert snap["latency_hit_p50_ms"] is not None
    assert snap["latency_hit_p50_ms"] >= 1000.0


def test_window_size_in_snapshot() -> None:
    """latency_window in the snapshot equals the value passed to the constructor."""
    m = Metrics(latency_window=42)
    assert m.snapshot()["latency_window"] == 42


def test_sample_counts_reflect_deque_length() -> None:
    """hit_sample_count and miss_sample_count track the deque size, not total records."""
    n = 3
    m = Metrics(latency_window=n)
    for _ in range(10):
        m.record_latency(0.001, hit=True)
        m.record_latency(0.001, hit=False)
    snap = m.snapshot()
    assert snap["hit_sample_count"] == n
    assert snap["miss_sample_count"] == n


def test_percentiles_correct_on_known_window() -> None:
    """Percentile calculation is still exact over the retained samples."""
    m = Metrics(latency_window=100)
    # Record 0.001 through 0.100 seconds (100 samples); p50 should be near 0.050 s.
    for i in range(1, 101):
        m.record_latency(i * 0.001, hit=False)
    snap = m.snapshot()
    # p50 of [1, 2, ..., 100] ms by linear interpolation: idx=49.5, so
    # between s[49]=50 ms and s[50]=51 ms, frac=0.5 -> 50.5 ms.
    assert snap["latency_miss_p50_ms"] == pytest.approx(50.5, abs=0.1)
    # p95: idx = 99*0.95 = 94.05, s[94]=95 ms, s[95]=96 ms, frac=0.05 -> 95.05 ms.
    assert snap["latency_miss_p95_ms"] == pytest.approx(95.05, abs=0.1)


# ---------------------------------------------------------------------------
# Ratio fields
# ---------------------------------------------------------------------------


def test_hit_ratio_is_none_with_no_traffic() -> None:
    """hit_ratio is null before any requests, never 0, to avoid a false zero rate."""
    m = Metrics()
    snap = m.snapshot()
    assert snap["hit_ratio"] is None
    assert snap["tier1_hit_ratio"] is None
    assert snap["tier2_hit_ratio"] is None


def test_hit_ratio_is_zero_with_all_misses() -> None:
    """When every request misses, hit_ratio is 0.0 (a real zero, not null)."""
    m = Metrics()
    m.record_miss()
    m.record_miss()
    snap = m.snapshot()
    assert snap["hit_ratio"] == 0.0
    assert snap["tier1_hit_ratio"] == 0.0
    assert snap["tier2_hit_ratio"] == 0.0


def test_hit_ratio_correct_with_mixed_traffic() -> None:
    """hit_ratio and per-tier ratios are consistent with the raw counters."""
    m = Metrics()
    m.record_tier1_hit()
    m.record_tier1_hit()
    m.record_tier2_hit()
    m.record_miss()
    # 3 hits, 1 miss, 4 total.
    snap = m.snapshot()
    assert snap["requests_total"] == 4
    assert snap["hit_ratio"] == pytest.approx(3 / 4)
    assert snap["tier1_hit_ratio"] == pytest.approx(2 / 4)
    assert snap["tier2_hit_ratio"] == pytest.approx(1 / 4)
    # Ratios and total hits agree with raw counters.
    assert snap["tier1_hits"] + snap["tier2_hits"] + snap["misses"] == snap["requests_total"]


def test_requests_total_is_sum_of_all_outcomes() -> None:
    """requests_total equals tier1_hits + tier2_hits + misses."""
    m = Metrics()
    m.record_tier1_hit()
    m.record_tier2_hit()
    m.record_tier2_hit()
    m.record_miss()
    snap = m.snapshot()
    assert snap["requests_total"] == 4


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
    """GET /metrics returns every documented field.

    Checked as a subset so that new fields added in future PRs do not silently
    break this assertion while still guaranteeing every field the benchmark
    report reads is present.
    """
    metrics = client.get("/metrics").json()
    required = {
        "tier1_hits", "tier2_hits", "misses",
        "tier2_unavailable",
        "requests_total", "hit_ratio", "tier1_hit_ratio", "tier2_hit_ratio",
        "throttled_requests", "breaker_failovers", "tier2_evicted_total",
        "total_provider_spend_usd",
        "routed_cheap", "routed_unchanged",
        "latency_window", "hit_sample_count", "miss_sample_count",
        "latency_hit_p50_ms", "latency_hit_p95_ms",
        "latency_miss_p50_ms", "latency_miss_p95_ms",
    }
    assert required <= set(metrics.keys())
