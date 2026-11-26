# Counters and latency distributions exposed on GET /metrics.
#
# A plain class, instantiated once in the lifespan and attached to app.state.
# Not a singleton: the previous __new__ pattern leaked counter state between
# tests, and contradicted the dependency wiring used everywhere else in the
# app, where every dependency is built in the lifespan so a test can substitute
# a fresh instance per case.
#
# Latency is recorded as elapsed seconds and stored in a bounded deque, split by
# cache status.  The PRD requires p50 and p95 reported separately for hits and
# misses, because an average across both describes neither.  Using a deque with
# maxlen caps memory at O(latency_window) samples per bucket regardless of how
# long the gateway runs.  Percentiles are exact over the most recent
# latency_window samples; the /metrics snapshot includes latency_window so
# a reader knows what window the p95 was computed from.

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Sequence
from typing import Any


class Metrics:
    """Aggregate counters and latency distributions for the /metrics endpoint.

    Thread-safe because the embedding call runs in a thread pool, so a
    record call from the pool and a snapshot from the event loop can overlap.
    """

    def __init__(self, latency_window: int = 10_000) -> None:
        self._lock = threading.Lock()
        self._latency_window = latency_window
        self.tier1_hits = 0
        self.tier2_hits = 0
        self.misses = 0
        self.cache_bypassed = 0
        self.tier2_unavailable = 0
        self.throttled_requests = 0
        self.breaker_failovers = 0
        self.tier2_evicted_total = 0
        self.total_provider_spend_usd = 0.0
        self.routed_cheap = 0
        self.routed_unchanged = 0
        self._hit_latencies: deque[float] = deque(maxlen=latency_window)
        self._miss_latencies: deque[float] = deque(maxlen=latency_window)

    def record_tier1_hit(self) -> None:
        """Count a Tier 1 exact cache hit."""
        with self._lock:
            self.tier1_hits += 1

    def record_tier2_hit(self) -> None:
        """Count a Tier 2 semantic cache hit."""
        with self._lock:
            self.tier2_hits += 1

    def record_bypass(self) -> None:
        """Count a request that skipped the cache because the caller asked it to.

        Kept apart from misses, so the hit ratio describes only requests the
        cache was allowed to answer.
        """
        with self._lock:
            self.cache_bypassed += 1

    def record_miss(self) -> None:
        """Count a cache miss that reached a provider."""
        with self._lock:
            self.misses += 1

    def record_tier2_unavailable(self) -> None:
        """Count a request where Tier 2 was skipped because Chroma was unreachable.

        Incremented once per request that reached the Tier 2 step and found
        Chroma down, including requests served during the reconnect cooldown.
        Each such request is a lost hit opportunity, so the counter rises even
        when Tier 1 ultimately serves the answer. Tier 2 disabled is not an
        outage, so it does not increment this counter.
        """
        with self._lock:
            self.tier2_unavailable += 1

    def record_throttled(self) -> None:
        """Count a request rejected by the rate limiter."""
        with self._lock:
            self.throttled_requests += 1

    def record_failover(self) -> None:
        """Count a request rerouted by the circuit breaker."""
        with self._lock:
            self.breaker_failovers += 1

    def record_evicted(self, count: int) -> None:
        """Count expired Tier 2 entries removed by a housekeeping pass."""
        with self._lock:
            self.tier2_evicted_total += count

    def record_routed_cheap(self) -> None:
        """Count a request routed to the cheaper model."""
        with self._lock:
            self.routed_cheap += 1

    def record_routed_unchanged(self) -> None:
        """Count a request whose model routing left unchanged."""
        with self._lock:
            self.routed_unchanged += 1

    def add_spend(self, amount: float) -> None:
        """Accumulate provider cost in USD."""
        with self._lock:
            self.total_provider_spend_usd += amount

    def record_latency(self, elapsed_seconds: float, *, hit: bool) -> None:
        """Store one latency sample, split by whether the response was a cache hit.

        The split matters because hit latency and miss latency describe
        different systems -- a hash lookup versus a provider round trip -- and
        mixing them into one distribution would let each hide the other.
        """
        with self._lock:
            if hit:
                self._hit_latencies.append(elapsed_seconds)
            else:
                self._miss_latencies.append(elapsed_seconds)

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time copy of all counters and percentiles.

        Percentiles are computed on read by sorting the sample deque.  The
        deque holds at most latency_window samples, so the sort is bounded
        regardless of run time.  latency_window is included in the response so
        callers know the percentiles are over the most recent N samples, not
        since process start.  hit_ratio and the per-tier ratios are null when
        there are no requests, never zero, so a zero in production is always
        a real zero hit rate and never an artefact of an empty counter.
        """
        with self._lock:
            requests_total = self.tier1_hits + self.tier2_hits + self.misses
            total_hits = self.tier1_hits + self.tier2_hits

            if requests_total == 0:
                hit_ratio = None
                tier1_hit_ratio = None
                tier2_hit_ratio = None
            else:
                hit_ratio = round(total_hits / requests_total, 6)
                tier1_hit_ratio = round(self.tier1_hits / requests_total, 6)
                tier2_hit_ratio = round(self.tier2_hits / requests_total, 6)

            return {
                "tier1_hits": self.tier1_hits,
                "tier2_hits": self.tier2_hits,
                "misses": self.misses,
                "cache_bypassed": self.cache_bypassed,
                "tier2_unavailable": self.tier2_unavailable,
                "requests_total": requests_total,
                "hit_ratio": hit_ratio,
                "tier1_hit_ratio": tier1_hit_ratio,
                "tier2_hit_ratio": tier2_hit_ratio,
                "throttled_requests": self.throttled_requests,
                "breaker_failovers": self.breaker_failovers,
                "tier2_evicted_total": self.tier2_evicted_total,
                "total_provider_spend_usd": self.total_provider_spend_usd,
                "routed_cheap": self.routed_cheap,
                "routed_unchanged": self.routed_unchanged,
                "latency_window": self._latency_window,
                "hit_sample_count": len(self._hit_latencies),
                "miss_sample_count": len(self._miss_latencies),
                "latency_hit_p50_ms": self._percentile(self._hit_latencies, 0.50),
                "latency_hit_p95_ms": self._percentile(self._hit_latencies, 0.95),
                "latency_miss_p50_ms": self._percentile(self._miss_latencies, 0.50),
                "latency_miss_p95_ms": self._percentile(self._miss_latencies, 0.95),
            }

    @staticmethod
    def _percentile(samples: Sequence[float], pct: float) -> float | None:
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
