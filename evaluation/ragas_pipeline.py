# Quality parity scoring with RAGAS.
#
# Holds:
#   build_cases(), the answers to judge: every distinct semantic hit in a gateway
#   run, set against a fresh answer to the same question from the baseline run
#   build_controls(), two independent fresh answers to the same question, which
#   show what score "equally good" gets from this judge
#   score_cases(), which runs a judge over the cases
#   summarise(), mean scores for hits overall, by kind, and for the controls
#   a command line entrypoint that judges with RAGAS AnswerAccuracy
#
# The question this answers: when the cache serves an answer stored for a
# different wording, is it as good as asking the model fresh? Exact hits are left
# out, since they serve the same question's own answer. Each case gives the
# judge the question asked, the cached answer, and the fresh answer as the
# reference, and AnswerAccuracy rates the match from 0 to 1 with two judge
# prompts, the second swapping the roles to offset the judge's bias.
#
# Why the controls: a model asked the same question twice does not word its
# answer the same way, so even a perfect cache would not score 1. The controls
# are pairs of fresh answers to one question (the gateway run's own misses
# against the baseline), and the hits are read against them, not against 1.
#
# Why RAGAS rather than a judging prompt written here: a prompt written for this
# project invites the fair objection that it was tuned until it agreed with the
# desired conclusion. AnswerAccuracy is a published metric whose prompts this
# project does not control. The judge is still a model, so its scores carry
# that model's mistakes; the false hit rate from the benchmark harness, which
# needs no judge, is reported beside it.
#
# RAGAS and the OpenAI client are imported only when the judge runs, so neither
# is a dependency of the gateway or of CI. scripts/evaluate-quality.sh runs this
# in an environment of its own, judging with a local model through Ollama.

from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

Judge = Callable[[str, str, str], Awaitable[float]]


@dataclass(frozen=True)
class Case:
    """One answer to judge against a fresh answer to the same question."""

    item_id: str
    kind: str
    question: str
    response: str
    reference: str
    role: str
    correct: bool | None = None
    similarity: float | None = None


@dataclass(frozen=True)
class Scored:
    """A case and the judge's score for it, 0 to 1."""

    case: Case
    score: float


def _prompts(workload: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """item id to (prompt, kind), from the workload file."""
    return {item["id"]: (item["prompt"], item["kind"]) for item in workload["items"]}


def fresh_answers(baseline: dict[str, Any]) -> dict[str, str]:
    """The first answer the baseline run got for each item, straight from the provider."""
    answers: dict[str, str] = {}
    for result in baseline["results"]:
        if result["http_status"] == 200 and result.get("answer"):
            answers.setdefault(result["item_id"], result["answer"])
    return answers


def build_cases(
    run: dict[str, Any], baseline: dict[str, Any], workload: dict[str, Any]
) -> list[Case]:
    """Every distinct semantic hit in the run, each with a fresh reference answer."""
    references = fresh_answers(baseline)
    prompts = _prompts(workload)
    seen: set[tuple[str, str]] = set()
    cases = []
    for result in run["results"]:
        if result["cache_status"] != "semantic_hit" or not result.get("answer"):
            continue
        key = (result["item_id"], result["answer"])
        if key in seen or result["item_id"] not in references:
            continue
        seen.add(key)
        question, kind = prompts[result["item_id"]]
        cases.append(Case(result["item_id"], kind, question, result["answer"],
                          references[result["item_id"]], "hit", result.get("correct"),
                          result.get("similarity")))
    return cases


def build_controls(
    run: dict[str, Any], baseline: dict[str, Any], workload: dict[str, Any], limit: int
) -> list[Case]:
    """Up to limit pairs of fresh answers to one question: the run's misses against the baseline."""
    references = fresh_answers(baseline)
    prompts = _prompts(workload)
    controls: list[Case] = []
    seen: set[str] = set()
    for result in run["results"]:
        if len(controls) >= limit:
            break
        item_id = result["item_id"]
        if result["cache_status"] != "miss" or item_id in seen or item_id not in references:
            continue
        if not result.get("answer") or result["answer"] == references[item_id]:
            continue
        seen.add(item_id)
        question, kind = prompts[item_id]
        controls.append(Case(item_id, kind, question, result["answer"], references[item_id],
                             "control"))
    return controls


async def score_cases(cases: Iterable[Case], judge: Judge) -> list[Scored]:
    """Judge every case in turn. One at a time: a local model gains nothing from more."""
    scored = []
    for case in cases:
        scored.append(Scored(case, await judge(case.question, case.response, case.reference)))
    return scored


def _mean(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None


def summarise(scored: list[Scored]) -> dict[str, Any]:
    """Mean scores: hits overall and by kind, hits the harness found wrong, and the controls."""
    by_role: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, list[float]] = defaultdict(list)
    wrong_group: list[float] = []
    for item in scored:
        by_role[item.case.role].append(item.score)
        if item.case.role == "hit":
            by_kind[item.case.kind].append(item.score)
            if item.case.correct is False:
                wrong_group.append(item.score)
    return {
        "hits": len(by_role["hit"]),
        "controls": len(by_role["control"]),
        "hit_mean": _mean(by_role["hit"]),
        "control_mean": _mean(by_role["control"]),
        "hit_mean_by_kind": {kind: _mean(values) for kind, values in sorted(by_kind.items())},
        "hits_from_another_group_mean": _mean(wrong_group),
    }


def ragas_judge(base_url: str, model: str) -> Judge:  # pragma: no cover - needs RAGAS and a model
    """AnswerAccuracy from RAGAS, judging with model at an OpenAI compatible base_url."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerAccuracy

    client = AsyncOpenAI(base_url=base_url, api_key="unused")
    metric = AnswerAccuracy(llm=llm_factory(model, provider="openai", client=client))

    async def judge(question: str, response: str, reference: str) -> float:
        result = await metric.ascore(user_input=question, response=response, reference=reference)
        return float(result.value)

    return judge


async def _main(args: argparse.Namespace) -> None:  # pragma: no cover - needs RAGAS and a model
    run = json.loads(args.run.read_text())
    baseline = json.loads(args.baseline.read_text())
    workload = json.loads(args.workload.read_text())
    cases = build_cases(run, baseline, workload)
    controls = build_controls(run, baseline, workload, limit=args.controls)
    print(f"judging {len(cases)} semantic hits and {len(controls)} controls with {args.model}")
    scored = await score_cases([*cases, *controls], ragas_judge(args.base_url, args.model))
    record = {
        "run": run["label"],
        "threshold": run.get("threshold"),
        "provider": run.get("provider"),
        "judge": f"RAGAS AnswerAccuracy, {args.model}",
        "summary": summarise(scored),
        "scored": [dict(asdict(item.case), score=item.score) for item in scored],
    }
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record["summary"], indent=2))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge cached answers against fresh ones.")
    parser.add_argument("run", type=Path, help="a gateway run written by the benchmark harness")
    parser.add_argument("baseline", type=Path, help="the direct run from the same sweep")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://localhost:11434/v1")
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument("--controls", type=int, default=20)
    parser.add_argument("--workload", type=Path,
                        default=Path(__file__).parent.parent / "benchmarks" / "workload.json")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(_main(parse_args()))
