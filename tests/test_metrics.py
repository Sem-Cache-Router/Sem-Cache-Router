from app.telemetry.metrics import Metrics


def test_metrics_singleton():
    m1 = Metrics()
    m2 = Metrics()
    assert m1 is m2


def test_metrics_recording():
    metrics = Metrics()
    # Reset for test isolation since it's a singleton
    metrics._init()
    
    metrics.record_tier1_hit()
    metrics.record_tier2_hit()
    metrics.record_tier2_hit()
    metrics.record_miss()
    metrics.record_throttled()
    metrics.record_failover()
    metrics.add_spend(1.50)
    
    snapshot = metrics.snapshot()
    assert snapshot["tier1_hits"] == 1
    assert snapshot["tier2_hits"] == 2
    assert snapshot["misses"] == 1
    assert snapshot["throttled_requests"] == 1
    assert snapshot["breaker_failovers"] == 1
    assert snapshot["total_provider_spend_usd"] == 1.50
