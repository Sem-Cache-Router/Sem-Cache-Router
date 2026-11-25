"""Quality parity: which answers are judged, against what, and how the scores are summarised.

The judge itself (RAGAS with a model) is replaced by a fake, so none of this
needs RAGAS installed or a model running.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evaluation import ragas_pipeline as rp

WORKLOAD = {"items": [
    {"id": "paris-1", "group": "paris", "kind": "paraphrase", "prompt": "Capital of France?"},
    {"id": "paris-2", "group": "paris", "kind": "paraphrase", "prompt": "France's capital?"},
    {"id": "open-yes", "group": "open-yes", "kind": "negation", "prompt": "Is it open?"},
    {"id": "open-no", "group": "open-no", "kind": "negation", "prompt": "Is it not open?"},
]}


def _result(item_id: str, status: str, answer: str | None, correct: bool | None = True,
            http: int = 200) -> dict[str, Any]:
    return {"item_id": item_id, "cache_status": status, "answer": answer, "correct": correct,
            "similarity": 0.95 if status == "semantic_hit" else None, "http_status": http}


BASELINE = {"results": [
    _result("paris-1", "direct", "Paris."),
    _result("paris-2", "direct", "It is Paris."),
    _result("open-yes", "direct", "Yes, it is open."),
    _result("open-no", "direct", "No, it is open."),
    _result("paris-1", "direct", "Paris, again."),  # a repeat: the first answer is the reference
]}
RUN = {"label": "t0.90", "results": [
    _result("paris-1", "miss", "Paris is the capital."),
    _result("paris-2", "semantic_hit", "Paris is the capital."),
    _result("open-yes", "miss", "Yes, it is open."),         # same text as baseline: no control
    _result("open-no", "semantic_hit", "Yes, it is open.", correct=False),
    _result("open-no", "semantic_hit", "Yes, it is open.", correct=False),  # repeat: judged once
    _result("paris-1", "exact_hit", "Paris is the capital."),  # exact hits are not judged
    _result("paris-2", "semantic_hit", None),
]}


def test_fresh_answers_are_the_first_answer_per_item() -> None:
    failed = {"results": [_result("paris-1", "direct", None, http=502), *BASELINE["results"]]}
    assert rp.fresh_answers(failed)["paris-1"] == "Paris."


def test_every_distinct_semantic_hit_becomes_one_case_against_a_fresh_answer() -> None:
    cases = rp.build_cases(RUN, BASELINE, WORKLOAD)
    assert [(c.item_id, c.response, c.reference, c.kind) for c in cases] == [
        ("paris-2", "Paris is the capital.", "It is Paris.", "paraphrase"),
        ("open-no", "Yes, it is open.", "No, it is open.", "negation"),
    ]
    assert cases[0].question == "France's capital?"
    assert (cases[1].correct, cases[1].role, cases[1].similarity) == (False, "hit", 0.95)


def test_a_hit_with_no_fresh_answer_to_compare_is_left_out() -> None:
    baseline = {"results": [_result("open-no", "direct", "No.")]}
    assert [c.item_id for c in rp.build_cases(RUN, baseline, WORKLOAD)] == ["open-no"]


def test_controls_pair_a_fresh_miss_with_the_baseline_and_skip_identical_text() -> None:
    controls = rp.build_controls(RUN, BASELINE, WORKLOAD, limit=10)
    assert [(c.item_id, c.response, c.reference, c.role) for c in controls] == [
        ("paris-1", "Paris is the capital.", "Paris.", "control"),
    ]


def test_controls_stop_at_the_limit() -> None:
    run = {"results": [_result(i["id"], "miss", f"fresh {i['id']}") for i in WORKLOAD["items"]]}
    assert len(rp.build_controls(run, BASELINE, WORKLOAD, limit=2)) == 2


async def test_cases_are_scored_by_the_judge_with_question_response_and_reference() -> None:
    calls: list[tuple[str, str, str]] = []

    async def judge(question: str, response: str, reference: str) -> float:
        calls.append((question, response, reference))
        return 1.0 if response.startswith("Paris") else 0.0

    cases = rp.build_cases(RUN, BASELINE, WORKLOAD)
    scored = await rp.score_cases(cases, judge)
    assert [s.score for s in scored] == [1.0, 0.0]
    assert calls[0] == ("France's capital?", "Paris is the capital.", "It is Paris.")


def _scored(role: str, kind: str, score: float, correct: bool | None = True) -> rp.Scored:
    case = rp.Case("x", kind, "q", "r", "ref", role, correct)
    return rp.Scored(case, score)


def test_summary_reads_hits_against_the_controls() -> None:
    summary = rp.summarise([
        _scored("hit", "paraphrase", 1.0),
        _scored("hit", "paraphrase", 0.5),
        _scored("hit", "negation", 0.0, correct=False),
        _scored("control", "paraphrase", 0.75),
        _scored("control", "negation", 1.0),
    ])
    assert summary == {
        "hits": 3, "controls": 2, "hit_mean": 0.5, "control_mean": 0.875,
        "hit_mean_by_kind": {"negation": 0.0, "paraphrase": 0.75},
        "hits_from_another_group_mean": 0.0,
    }


def test_an_empty_summary_has_no_means() -> None:
    summary = rp.summarise([])
    assert (summary["hit_mean"], summary["control_mean"]) == (None, None)
    assert summary["hits_from_another_group_mean"] is None


def test_arguments_default_to_the_local_model(tmp_path: Path) -> None:
    args = rp.parse_args(["run.json", "direct.json", "--out", str(tmp_path / "q.json")])
    assert (args.base_url, args.model, args.controls) == (
        "http://localhost:11434/v1", "qwen2.5:7b", 20,
    )
    assert args.workload.name == "workload.json" and args.workload.exists()


@pytest.mark.parametrize("limit", [0])
def test_no_controls_when_none_are_asked_for(limit: int) -> None:
    assert rp.build_controls(RUN, BASELINE, WORKLOAD, limit=limit) == []
