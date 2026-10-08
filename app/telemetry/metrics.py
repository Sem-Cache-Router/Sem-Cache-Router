from __future__ import annotations

import threading
from typing import Any


class Metrics:
    """Thread-safe singleton for tracking aggregate telemetry."""

    _instance: Metrics | None = None
    _lock = threading.Lock()

    def __new__(cls) -> Metrics:
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init()
            return cls._instance

    def _init(self) -> None:
        self.tier1_hits = 0
        self.tier2_hits = 0
        self.misses = 0
        self.throttled_requests = 0
        self.breaker_failovers = 0
        self.total_provider_spend_usd = 0.0
        self._metrics_lock = threading.Lock()

    def record_tier1_hit(self) -> None:
        with self._metrics_lock:
            self.tier1_hits += 1

    def record_tier2_hit(self) -> None:
        with self._metrics_lock:
            self.tier2_hits += 1

    def record_miss(self) -> None:
        with self._metrics_lock:
            self.misses += 1

    def record_throttled(self) -> None:
        with self._metrics_lock:
            self.throttled_requests += 1

    def record_failover(self) -> None:
        with self._metrics_lock:
            self.breaker_failovers += 1

    def add_spend(self, amount: float) -> None:
        with self._metrics_lock:
            self.total_provider_spend_usd += amount

    def snapshot(self) -> dict[str, Any]:
        """Return a snapshot of current metrics."""
        with self._metrics_lock:
            return {
                "tier1_hits": self.tier1_hits,
                "tier2_hits": self.tier2_hits,
                "misses": self.misses,
                "throttled_requests": self.throttled_requests,
                "breaker_failovers": self.breaker_failovers,
                "total_provider_spend_usd": self.total_provider_spend_usd,
            }
