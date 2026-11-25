# Workload replay harness.
#
# Holds:
#   RequestResult, what one request did: cache status, latency, and whether the
#   answer it got was correct for the question asked
#   run(), which replays the workload against a gateway, or straight against the
#   vendor API for the no cache baseline
#   summarise(), hit ratio per tier, false hits, provider calls, cost, latency
#   a command line entrypoint that writes one run to a JSON file
#
# How correctness is scored without a judge model: the harness works out which
# prompt a served answer was originally produced for, and looks up that prompt's
# group in the workload. A hit serving an answer produced for a prompt of another
# group is a false hit. Against the mock vendor the prompt is in the answer
# itself ("Mock answer to: <prompt>"). Against a real model, the harness
# remembers every answer the provider gave during the run and which prompt it
# was for; a cached answer is the same text, so it maps back. A hit whose origin
# cannot be found is counted as unscored, never as right or wrong.
#
# The provider is recorded with every run. Only a run against the mock is
# marked simulated; against a real model the latency is real, and the cost is
# what the tokens would cost at the gateway's gpt-4o-mini rates, which the
# output says.
#
# Threshold is not set here. It is a gateway setting, so a sweep restarts the
# gateway for each threshold (scripts/run-benchmark.sh) and labels each run.
# Unfavourable thresholds are run and reported too. A sweep that keeps only the
# flattering number is not a measurement.

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import httpx

from benchmarks.workload import DEFAULT_PATH, DEFAULT_SEED, WorkloadItem, load, replay_order

Mode = Literal["gateway", "direct"]

ANSWER_PREFIX = "Mock answer to: "
MODEL = "gpt-4o-mini"
# The gateway's own price for gpt-4o-mini (app/providers/openai_client.py), so the
# baseline and the gateway runs are priced by the same table.
INPUT_USD_PER_TOKEN = 0.150 / 1_000_000
OUTPUT_USD_PER_TOKEN = 0.600 / 1_000_000
HIT_STATUSES = ("exact_hit", "semantic_hit")


@dataclass(frozen=True)
class RequestResult:
    """What one request did and whether its answer was right for its question."""

    item_id: str
    group: str
    kind: str
    http_status: int
    cache_status: str | None
    similarity: float | None
    latency_ms: float
    served_group: str | None
    correct: bool | None
    cost_usd: float
    answer: str | None = None


def served_prompt(answer: str) -> str | None:
    """The prompt a mock answer was produced for, or None if it is not a mock answer."""
    return answer[len(ANSWER_PREFIX):] if answer.startswith(ANSWER_PREFIX) else None


def price(usage: dict[str, Any]) -> float:
    """Dollar cost of one provider reply, by the gateway's gpt-4o-mini rates."""
    return (
        int(usage.get("prompt_tokens", 0)) * INPUT_USD_PER_TOKEN
        + int(usage.get("completion_tokens", 0)) * OUTPUT_USD_PER_TOKEN
    )


def score(
    item: WorkloadItem, mode: Mode, http_status: int, body: dict[str, Any], groups: dict[str, str],
    latency_ms: float, origins: dict[str, str] | None = None,
) -> RequestResult:
    """Turn one response into a result, scoring the answer against the item's group.

    origins maps answers a real provider gave during this run to the prompt each
    was for; the mock needs none, since its answers name their prompt.
    """
    if http_status != 200:
        return RequestResult(item.id, item.group, item.kind, http_status, None, None,
                             latency_ms, None, None, 0.0)
    answer = body["choices"][0]["message"]["content"]
    cache_status = "direct" if mode == "direct" else body.get("semcache_status")
    # A reply the gateway fetched from the provider is paid for; a hit is free.
    paid = mode == "direct" or cache_status == "miss"
    # A paid reply is the provider answering this very prompt. The mock's answer
    # still names its prompt, which keeps the check honest against it.
    origin = served_prompt(answer) or (origins or {}).get(answer) or (item.prompt if paid else None)
    served_group = groups.get(origin) if origin is not None else None
    return RequestResult(
        item_id=item.id,
        group=item.group,
        kind=item.kind,
        http_status=http_status,
        cache_status=cache_status,
        similarity=body.get("semcache_similarity"),
        latency_ms=latency_ms,
        served_group=served_group,
        correct=None if served_group is None else served_group == item.group,
        cost_usd=price(body.get("usage") or {}) if paid else 0.0,
        answer=answer,
    )


async def run(
    order: list[WorkloadItem], client: httpx.AsyncClient, url: str, mode: Mode,
    groups: dict[str, str], api_key: str = "bench-key", model: str = MODEL,
    max_tokens: int | None = None,
) -> list[RequestResult]:
    """Send every prompt in order, one at a time, and score each answer.

    Sequential on purpose: the order decides which member of a group is stored
    first, and concurrency would make that, and so every hit, depend on timing.
    Every answer the provider itself gave is remembered against its prompt, so a
    later hit serving that text can be traced back to it.
    """
    endpoint = f"{url.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    origins: dict[str, str] = {}
    results = []
    for item in order:
        message = {"role": "user", "content": item.prompt}
        body: dict[str, Any] = {"model": model, "messages": [message]}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        started = time.perf_counter()
        response = await client.post(endpoint, json=body, headers=headers)
        latency_ms = (time.perf_counter() - started) * 1000
        payload = response.json() if response.status_code == 200 else {}
        result = score(item, mode, response.status_code, payload, groups, latency_ms, origins)
        if result.cache_status in ("miss", "direct") and result.answer:
            origins.setdefault(result.answer, item.prompt)
        results.append(result)
    return results


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest rank percentile, or None for no samples."""
    if not values:
        return None
    ranked = sorted(values)
    return ranked[max(0, math.ceil(fraction * len(ranked)) - 1)]


def _rate(part: int, whole: int) -> float | None:
    return part / whole if whole else None


def summarise(results: list[RequestResult]) -> dict[str, Any]:
    """The headline numbers of one run. Every hit ratio sits next to its false hit rate."""
    ok = [r for r in results if r.http_status == 200]
    statuses = Counter(r.cache_status for r in ok)
    hits = [r for r in ok if r.cache_status in HIT_STATUSES]
    paid = [r for r in ok if r.cache_status not in HIT_STATUSES]
    scored = [r for r in hits if r.correct is not None]
    false_hits = [r for r in scored if not r.correct]
    false_by_kind = Counter(r.kind for r in false_hits)
    hits_by_kind = Counter(r.kind for r in scored)
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "statuses": dict(statuses),
        "hit_ratio": _rate(len(hits), len(ok)),
        "false_hits": len(false_hits),
        "false_hit_rate": _rate(len(false_hits), len(scored)),
        "unscored_hits": len(hits) - len(scored),
        "false_hit_rate_by_kind": {
            kind: _rate(false_by_kind[kind], count) for kind, count in sorted(hits_by_kind.items())
        },
        "provider_calls": len(paid),
        "cost_usd": round(sum(r.cost_usd for r in ok), 8),
        "latency_ms": {
            "all_p50": percentile([r.latency_ms for r in ok], 0.50),
            "all_p95": percentile([r.latency_ms for r in ok], 0.95),
            "hit_p50": percentile([r.latency_ms for r in hits], 0.50),
            "hit_p95": percentile([r.latency_ms for r in hits], 0.95),
            "miss_p50": percentile([r.latency_ms for r in paid], 0.50),
            "miss_p95": percentile([r.latency_ms for r in paid], 0.95),
        },
    }


def git_commit() -> str:
    """The commit the run was made at, so a result can be traced to its code."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


async def _main(args: argparse.Namespace) -> None:
    items = load(args.workload)
    groups = {item.prompt: item.group for item in items}
    order = replay_order(items, repeats=args.repeats, seed=args.seed)
    mode: Mode = "direct" if args.direct else "gateway"
    url = args.direct or f"{args.gateway.rstrip('/')}/v1"
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        results = await run(order, client, url, mode, groups, model=args.model,
                            max_tokens=args.max_tokens)
    simulated = args.provider == "mock"
    record = {
        "label": args.label,
        "mode": mode,
        "threshold": args.threshold,
        "provider": args.provider,
        "model": args.model,
        "max_tokens": args.max_tokens,
        "simulated": simulated,
        "note": (
            "Run against the mock vendor API: latency and cost are simulated."
            if simulated else
            f"Run against {args.provider}: latency is real; cost is the tokens priced at "
            "the gateway's gpt-4o-mini rates."
        ),
        "commit": git_commit(),
        "seed": args.seed,
        "repeats": args.repeats,
        "workload": str(args.workload),
        "summary": summarise(results),
        "results": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2) + "\n")
    summary = record["summary"]
    print(f"{args.label}: {summary['requests']} requests, hit ratio {summary['hit_ratio']}, "
          f"false hit rate {summary['false_hit_rate']}, cost ${summary['cost_usd']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the benchmark workload once.")
    parser.add_argument("--label", required=True, help="name of this run, e.g. t0.90")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    parser.add_argument("--gateway", default="http://localhost:8000")
    parser.add_argument("--direct", help="vendor base URL, e.g. http://localhost:8002/v1; "
                        "skips the gateway for the no cache baseline")
    parser.add_argument("--threshold", type=float, help="the gateway's threshold, recorded only")
    parser.add_argument("--provider", default="mock",
                        help="what answered, recorded: 'mock', or e.g. 'ollama qwen2.5:7b'")
    parser.add_argument("--model", default=MODEL, help="model name sent with every request")
    parser.add_argument("--max-tokens", type=int, help="cap on each answer's length")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per request")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workload", type=Path, default=DEFAULT_PATH)
    return parser.parse_args(argv)


if __name__ == "__main__":
    asyncio.run(_main(parse_args()))
