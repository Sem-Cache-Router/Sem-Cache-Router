# Load profile for the concurrency target.
#
# A Locust user that mixes three kinds of request:
#   repeats, a small fixed set of prompts asked again and again (Tier 1)
#   paraphrases, rewordings of a few questions (Tier 2 once one of them is stored)
#   fresh prompts, unique every time, so they always reach the provider
#
# Before the users start, each repeat prompt is asked once, so the paraphrases
# meet a stored question and Tier 2 is measured. Started cold, fifty users ask
# everything at once, most paraphrases miss and are then stored in Tier 1, and
# the run measured three semantic hits in one attempt and 173 in another.
#
# What it checks, at the end of the run:
#   end to end p95 for exact hits and for semantic hits, against budgets, on at
#   least MIN_SAMPLES of each; fewer is reported as a failure, not a pass
#   queue growth: p95 of the hits in the last third of the run against the
#   first third. A server that keeps up holds steady; one that falls behind
#   gets slower as the queue builds, whatever its average looks like
#   that no request failed
# A failed check sets the exit code, so the script running this can fail.
#
# Latency is recorded per cache status, read from each response, rather than per
# task: a paraphrase that misses is a miss, and counting it as a Tier 2 request
# would hide exactly what this is meant to show.
#
# The PRD's 5 ms and 60 ms targets are for the cache lookups alone. This
# measures whole requests through HTTP, so its budgets are end to end and are
# set with LOAD_EXACT_P95_MS and LOAD_SEMANTIC_P95_MS.
#
# Run it with scripts/load-test.sh, which starts what it needs.

from __future__ import annotations

import json
import math
import os
import random
import time
import urllib.request
from collections import defaultdict
from typing import Any

from locust import HttpUser, between, events, task

REPEATS = [
    "What is the capital of France?",
    "How do I reset my password?",
    "Why is the sky blue?",
    "How do I cancel my order?",
    "Who wrote Hamlet?",
]
PARAPHRASES = [
    "Which city is the capital of France?",
    "Tell me France's capital city.",
    "How can I reset my password?",
    "What are the steps to reset my password?",
    "What makes the sky look blue?",
    "Explain why the sky is blue.",
    "How can I cancel an order I placed?",
    "Who is the author of Hamlet?",
    "What is the boiling point of water?",
    "At what temperature does water boil?",
]
BUDGETS_MS = {
    "exact_hit": float(os.environ.get("LOAD_EXACT_P95_MS", "50")),
    "semantic_hit": float(os.environ.get("LOAD_SEMANTIC_P95_MS", "150")),
}
MAX_GROWTH = float(os.environ.get("LOAD_MAX_GROWTH", "1.5"))
MIN_SAMPLES = int(os.environ.get("LOAD_MIN_SAMPLES", "30"))
HEADERS = {"Authorization": "Bearer load-test-key"}

# Fresh prompts are six random words from this list. A template with a counter in
# it ("question number 7") embeds so close to its neighbours that the semantic
# cache serves them to each other, and they stop being fresh. Random word sets
# were measured with the real model: the closest of 1,770 pairs scored 0.82.
WORDS = (
    "apple river engine violin tax glacier poem kernel harbor tomato satellite ledger cactus "
    "orbit fabric lantern piano canyon vaccine mortgage tulip quartz bicycle opera turbine "
    "falcon meadow compiler volcano yogurt sculpture radar pepper galaxy cello marathon fossil "
    "bakery prism tunnel"
).split()

_samples: dict[str, list[tuple[float, float]]] = defaultdict(list)


def _body(prompt: str) -> dict[str, Any]:
    return {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]}


class GatewayUser(HttpUser):
    """One caller: mostly repeats and paraphrases, with a steady trickle of new questions."""

    wait_time = between(0.05, 0.2)

    def _ask(self, prompt: str, name: str) -> None:
        self.client.post("/v1/chat/completions", json=_body(prompt), headers=HEADERS, name=name)

    @task(5)
    def repeat(self) -> None:
        self._ask(random.choice(REPEATS), "repeat")

    @task(4)
    def paraphrase(self) -> None:
        self._ask(random.choice(PARAPHRASES), "paraphrase")

    @task(1)
    def fresh(self) -> None:
        self._ask(" ".join(random.sample(WORDS, 6)) + "?", "fresh")


@events.test_start.add_listener
def _warm(environment: Any, **kwargs: Any) -> None:
    """Ask each repeat prompt once, in turn, so the run starts with them stored."""
    for prompt in REPEATS:
        request = urllib.request.Request(
            f"{environment.host}/v1/chat/completions",
            data=json.dumps(_body(prompt)).encode(),
            headers={**HEADERS, "Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response.read()


@events.request.add_listener
def _record(response_time: float, response: Any, exception: Any, **kwargs: Any) -> None:
    """Keep each successful request's latency under the cache status the gateway reported."""
    if exception is not None or response is None or response.status_code != 200:
        return
    try:
        status = response.json().get("semcache_status", "unknown")
    except ValueError:
        status = "unknown"
    _samples[status].append((time.monotonic(), response_time))


def p95(values: list[float]) -> float | None:
    """Nearest rank 95th percentile, or None for no samples."""
    if not values:
        return None
    ranked = sorted(values)
    return ranked[max(0, math.ceil(0.95 * len(ranked)) - 1)]


def growth(samples: list[tuple[float, float]]) -> float | None:
    """p95 of the last third of the run over p95 of the first third, or None if too few."""
    if len(samples) < 30:
        return None
    ordered = sorted(samples)
    third = len(ordered) // 3
    first = p95([latency for _, latency in ordered[:third]])
    last = p95([latency for _, latency in ordered[-third:]])
    return None if not first else (last or 0.0) / first


def verdict(samples: dict[str, list[tuple[float, float]]], failures: int) -> list[str]:
    """The failed checks, as sentences. Empty means the run met every target."""
    problems = []
    if failures:
        problems.append(f"{failures} requests failed")
    for status, budget in BUDGETS_MS.items():
        latencies = [latency for _, latency in samples.get(status, [])]
        measured = p95(latencies)
        if measured is None or len(latencies) < MIN_SAMPLES:
            problems.append(f"only {len(latencies)} {status} measured, fewer than {MIN_SAMPLES}")
        elif measured > budget:
            problems.append(f"{status} p95 {measured:.0f} ms is over its {budget:.0f} ms budget")
    hits = samples.get("exact_hit", []) + samples.get("semantic_hit", [])
    ratio = growth(hits)
    if ratio is not None and ratio > MAX_GROWTH:
        problems.append(f"hit latency grew {ratio:.2f}x from the first third to the last")
    return problems


@events.quitting.add_listener
def _check(environment: Any, **kwargs: Any) -> None:
    """Print the per status summary and fail the run if a target was missed."""
    print("\nend to end latency by cache status (ms)")
    for status, samples in sorted(_samples.items()):
        latencies = [latency for _, latency in samples]
        print(f"  {status:<13} n={len(latencies):<6} p95={p95(latencies):.0f}")
    hits = _samples.get("exact_hit", []) + _samples.get("semantic_hit", [])
    ratio = growth(hits)
    print(f"  hit p95 growth, last third over first: {'n/a' if ratio is None else f'{ratio:.2f}x'}")
    problems = verdict(_samples, environment.stats.total.num_failures)
    for problem in problems:
        print(f"FAIL {problem}")
    if problems:
        environment.process_exit_code = 1
    else:
        print("PASS every target met")
