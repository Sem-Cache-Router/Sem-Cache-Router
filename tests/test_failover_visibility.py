# The one failover case the telemetry does not otherwise show.
#
# app/providers/router.py owns the breaker now, and test_provider_wiring.py
# already proves the route fails over, counts it once, and names the provider
# that answered. In all of those the log reports a closed breaker, because it
# reports the breaker of the provider that answered.
#
# An open breaker therefore only ever reaches a log line when nothing answered:
# the request that gets the 502 is the one carrying breaker_state=open. That is
# the line an on-call engineer searches for to tell a provider outage apart from
# an ordinary 502, so it is pinned here.

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.providers.breaker import CircuitBreaker
from app.providers.router import FailoverRouter, ProviderSlot
from tests.test_failover_router import ScriptedProvider

AUTH = {"Authorization": "Bearer test-key"}
MODEL = "gpt-4o-mini"


@pytest.fixture
def tripping(client: TestClient) -> tuple[TestClient, ScriptedProvider, ScriptedProvider]:
    """A two provider router whose breakers open on the first failure.

    The backoff is a minute, so an open breaker stays open for the whole test.
    With the default second on the real clock, a slow machine could let one
    pass between two requests, and the breaker would rightly go half open.
    """
    primary, secondary = ScriptedProvider("primary"), ScriptedProvider("secondary")
    client.app.state.router = FailoverRouter(  # type: ignore[attr-defined]
        [
            ProviderSlot(primary, CircuitBreaker(failure_threshold=1, base_backoff=60.0)),
            ProviderSlot(secondary, CircuitBreaker(failure_threshold=1, base_backoff=60.0)),
        ]
    )
    return client, primary, secondary


def ask(client: TestClient, content: str = "a question never asked before") -> Any:
    return client.post(
        "/v1/chat/completions",
        json={"model": MODEL, "messages": [{"role": "user", "content": content}]},
        headers=AUTH,
    )


def test_an_open_breaker_is_the_difference_between_an_outage_and_a_bad_request(
    tripping: tuple[TestClient, ScriptedProvider, ScriptedProvider],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Both 502s look alike to the caller; only breaker_state tells them apart.

    The first request fails through both providers, which opens both breakers
    because the threshold is one. The second is refused without a call being
    made, and that is the one whose log line has to say open.
    """
    client, primary, secondary = tripping
    primary.fail_with(503)
    secondary.fail_with(500)

    with caplog.at_level("WARNING", logger="semcache.request"):
        assert ask(client, "the request that opens both breakers").status_code == 502
        assert ask(client, "asked once every breaker is open").status_code == 502

    lines = [json.loads(r.getMessage()) for r in caplog.records if r.name == "semcache.request"]
    assert [line["breaker_state"] for line in lines] == ["closed", "open"]
    assert {line["error_code"] for line in lines} == {"ALL_PROVIDERS_UNAVAILABLE"}
    assert primary.calls == 1, "the second request must not reach a provider at all"
