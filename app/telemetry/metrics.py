# Counters and latency distributions exposed on GET /metrics.
#
# A plain class, instantiated once in the lifespan and attached to app.state.
# Not a singleton: the previous __new__ pattern leaked counter state between
# tests, and contradicted the dependency wiring used everywhere else in the
# app, where every dependency is built in the lifespan so a test can substitute
# a fresh instance per case.
#
# Latency is recorded as elapsed seconds and stored in a plain list, split by
# cache status.  The PRD requires p50 and p95 reported separately for hits and
# misses, because an average across both describes neither.  Storing every
# sample is simplest, gives exact percentiles, and is fine at benchmark scale
# where the workload is hundreds to low thousands of queries.  At production
# scale this would need a streaming sketch; that is not this project.

from __future__ import annotations

import threading
from typing import Any


class Metrics:
    """Aggregate counters and latency distributions for the /metrics endpoint.

    Thread-safe because the embedding call runs in a thread pool, so a
    record call from the pool and a snapshot from the event loop can overlap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.tier1_hits = 0
        self.tier2_hits = 0
        self.misses = 0
        self.throttled_requests = 0
        self.breaker_failovers = 0
        self.total_provider_spend_usd = 0.0
        self._hit_latencies: list[float] = []
        self._miss_latencies: list[float] = []

    def record_tier1_hit(self) -> None:
        """Count a Tier 1 exact cache hit."""
        with self._lock:
            self.tier1_hits += 1

    def record_tier2_hit(self) -> None:
        """Count a Tier 2 semantic cache hit."""
        with self._lock:
            self.tier2_hits += 1

    def record_miss(self) -> None:
        """Count a cache miss that reached a provider."""
        with self._lock:
            self.misses += 1

    def record_throttled(self) -> None:
        """Count a request rejected by the rate limiter."""
        with self._lock:
            self.throttled_requests += 1

    def record_failover(self) -> None:
        """Count a request rerouted by the circuit breaker."""
        with self._lock:
            self.breaker_failovers += 1

    def add_spend(self, amount: float) -> None:
        """Accumulate provider cost in USD."""
        with self._lock:
            self.total_provider_spend_usd += amount

    def record_latency(self, elapsed_seconds: float, *, hit: bool) -> None:
        """Store one latency sample, split by whether the response was a cache hit.

        The split matters because hit latency and miss latency describe
        different systems — a hash lookup versus a provider round trip — and
        mixing them into one distribution would let each hide the other.
        """
        with self._lock:
            if hit:
                self._hit_latencies.append(elapsed_seconds)
            else:
                self._miss_latencies.append(elapsed_seconds)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy of all counters and percentiles.

        Percentiles are computed on read by sorting the sample list.  At
        benchmark scale this is negligible; the sort is O(n log n) over the
        number of requests in the run, which is bounded by the committed
        workload size.
        """
        with self._lock:
            return {
                "tier1_hits": self.tier1_hits,
                "tier2_hits": self.tier2_hits,
                "misses": self.misses,
                "throttled_requests": self.throttled_requests,
                "breaker_failovers": self.breaker_failovers,
                "total_provider_spend_usd": self.total_provider_spend_usd,
                "latency_hit_p50_ms": self._percentile(self._hit_latencies, 0.50),
                "latency_hit_p95_ms": self._percentile(self._hit_latencies, 0.95),
                "latency_miss_p50_ms": self._percentile(self._miss_latencies, 0.50),
                "latency_miss_p95_ms": self._percentile(self._miss_latencies, 0.95),
            }

    @staticmethod
    def _percentile(samples: list[float], pct: float) -> float | None:
        """Compute an exact percentile by linear interpolation over sorted samples.

        Returns None when there are no samples, so the /metrics body carries an
        explicit null rather than a misleading zero.  Converts to milliseconds,
        because that is the natural unit for gateway latency and what the
        benchmark report expects.
        """
        if not samples:
            return None
        s = sorted(samples)
        idx = (len(s) - 1) * pct
        lo = int(idx)
        hi = min(lo + 1, len(s) - 1)
        frac = idx - lo
        return round((s[lo] + (s[hi] - s[lo]) * frac) * 1000, 2)
