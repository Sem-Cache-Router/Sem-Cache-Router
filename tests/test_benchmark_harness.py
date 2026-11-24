"""The benchmark harness: scoring answers against groups, and summarising a run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from app.config import Settings
from app.models import Usage
from app.providers.openai_client import OpenAIClient
from benchmarks import harness
from benchmarks.harness import RequestResult, percentile, price, score, served_prompt, summarise
from benchmarks.workload import WorkloadItem

PARIS = WorkloadItem("paris-1", "paris", "paraphrase", "What is the capital of France?")
PARIS_2 = WorkloadItem("paris-2", "paris", "paraphrase", "Which city is the capital of France?")
OPEN = WorkloadItem("open-yes", "open-yes", "negation", "Is the store open?")
CLOSED = WorkloadItem("open-no", "open-no", "negation", "Is the store not open?")
GROUPS = {item.prompt: item.group for item in (PARIS, PARIS_2, OPEN, CLOSED)}
USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}


def _reply(
    answer_for: str, status: str | None = None, similarity: float | None = None
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "choices": [{"message": {"role": "assistant", "content": f"Mock answer to: {answer_for}"}}],
        "usage": USAGE,
    }
    if status is not None:
        body["semcache_status"] = status
        body["semcache_similarity"] = similarity
    return body


def test_served_prompt_reads_back_the_prompt_a_mock_answer_was_made_for() -> None:
    assert served_prompt("Mock answer to: Is the store open?") == "Is the store open?"
    assert served_prompt("Paris.") is None


def test_the_price_matches_the_gateways_own_table() -> None:
    client = OpenAIClient(httpx.AsyncClient(), Settings(openai_api_key="x", _env_file=None))  # type: ignore[call-arg]
    assert price(USAGE) == pytest.approx(client.price_of(Usage(**USAGE)))


def test_a_miss_is_correct_and_paid_for() -> None:
    result = score(PARIS, "gateway", 200, _reply(PARIS.prompt, "miss"), GROUPS, 800.0)
    assert result.correct is True
    assert result.cost_usd == pytest.approx(price(USAGE))


def test_a_semantic_hit_from_the_same_group_is_correct_and_free() -> None:
    reply = _reply(PARIS.prompt, "semantic_hit", 0.94)
    result = score(PARIS_2, "gateway", 200, reply, GROUPS, 13.0)
    assert (result.correct, result.cost_usd, result.similarity) == (True, 0.0, 0.94)
    assert result.served_group == "paris"


def test_a_hit_serving_the_other_side_of_a_negation_is_a_false_hit() -> None:
    result = score(CLOSED, "gateway", 200, _reply(OPEN.prompt, "semantic_hit", 0.97), GROUPS, 13.0)
    assert result.correct is False
    assert result.served_group == "open-yes"


def test_an_answer_the_workload_never_asked_for_is_not_correct() -> None:
    result = score(PARIS, "gateway", 200, _reply("something else", "exact_hit"), GROUPS, 1.0)
    assert (result.served_group, result.correct) == (None, False)


def test_an_error_response_is_recorded_without_a_verdict() -> None:
    result = score(PARIS, "gateway", 503, {}, GROUPS, 5.0)
    assert (result.http_status, result.cache_status, result.correct, result.cost_usd) == (
        503, None, None, 0.0,
    )


def test_a_direct_call_is_marked_direct_and_paid_for() -> None:
    result = score(PARIS, "direct", 200, _reply(PARIS.prompt), GROUPS, 800.0)
    assert result.cache_status == "direct"
    assert result.cost_usd > 0


def test_percentile_uses_nearest_rank() -> None:
    assert percentile([], 0.5) is None
    assert percentile([5.0, 1.0, 3.0, 2.0, 4.0], 0.5) == 3.0
    assert percentile([float(n) for n in range(1, 101)], 0.95) == 95.0


def _result(item: WorkloadItem, status: str | None, correct: bool | None, latency: float,
            http: int = 200, cost: float = 0.0) -> RequestResult:
    return RequestResult(item.id, item.group, item.kind, http, status, None, latency,
                         None, correct, cost)


def test_summarise_puts_false_hits_next_to_the_hit_ratio() -> None:
    results = [
        _result(PARIS, "miss", True, 800.0, cost=0.001),
        _result(PARIS_2, "semantic_hit", True, 13.0),
        _result(OPEN, "miss", True, 800.0, cost=0.001),
        _result(CLOSED, "semantic_hit", False, 14.0),
        _result(PARIS, "exact_hit", True, 1.0),
        _result(PARIS, None, None, 5.0, http=503),
    ]
    summary = summarise(results)
    assert summary["requests"] == 6
    assert summary["errors"] == 1
    assert summary["hit_ratio"] == pytest.approx(3 / 5)
    assert summary["false_hits"] == 1
    assert summary["false_hit_rate"] == pytest.approx(1 / 3)
    assert summary["false_hit_rate_by_kind"] == {"negation": 1.0, "paraphrase": 0.0}
    assert summary["provider_calls"] == 2
    assert summary["cost_usd"] == pytest.approx(0.002)
    assert summary["latency_ms"]["miss_p50"] == 800.0
    assert summary["latency_ms"]["hit_p50"] == 13.0


def test_summarise_an_empty_run_has_no_rates() -> None:
    summary = summarise([])
    assert (summary["hit_ratio"], summary["false_hit_rate"]) == (None, None)


async def test_run_sends_each_prompt_in_order_and_scores_it() -> None:
    sent: list[str] = []
    cache: dict[str, str] = {}

    def gateway(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        sent.append(prompt)
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer bench-key"
        # A deliberately careless cache: anything mentioning the store gets the first answer.
        key = "store" if "store" in prompt else prompt
        if key in cache:
            return httpx.Response(200, json=_reply(cache[key], "semantic_hit", 0.97))
        cache[key] = prompt
        return httpx.Response(200, json=_reply(prompt, "miss"))

    order = [OPEN, CLOSED, PARIS]
    async with httpx.AsyncClient(transport=httpx.MockTransport(gateway)) as client:
        results = await harness.run(order, client, "http://gw/v1/", "gateway", GROUPS)

    assert sent == [item.prompt for item in order]
    assert [r.correct for r in results] == [True, False, True]
    assert all(r.latency_ms >= 0 for r in results)


async def test_run_records_a_failed_request_instead_of_raising() -> None:
    def failing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(502, json={"error": {"code": "UPSTREAM"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(failing)) as client:
        results = await harness.run([PARIS], client, "http://gw/v1", "gateway", GROUPS)
    assert results[0].http_status == 502
    assert results[0].correct is None


def test_git_commit_falls_back_when_git_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_git(*args: object, **kwargs: object) -> None:
        raise OSError("no git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert harness.git_commit() == "unknown"


def test_git_commit_reads_the_short_hash() -> None:
    assert harness.git_commit() not in ("", None)


async def test_main_writes_a_labelled_simulated_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def vendor(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][-1]["content"]
        return httpx.Response(200, json=_reply(prompt))

    real_client = httpx.AsyncClient

    def mocked_client(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(vendor))

    monkeypatch.setattr(harness.httpx, "AsyncClient", mocked_client)
    out = tmp_path / "runs" / "direct.json"
    args = harness.parse_args(["--label", "direct", "--out", str(out), "--direct", "http://mock/v1"])
    await harness._main(args)

    record = json.loads(out.read_text())
    assert (record["mode"], record["simulated"], record["label"]) == ("direct", True, "direct")
    assert record["summary"]["errors"] == 0
    assert record["summary"]["provider_calls"] == record["summary"]["requests"]
    assert record["summary"]["hit_ratio"] == 0.0
