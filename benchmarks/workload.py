# Workload loading.
#
# Holds:
#   WorkloadItem, one prompt and the paraphrase group it belongs to
#   load(), which reads the committed workload from disk
#   replay_order(), the fixed, seeded order a run sends the prompts in
#
# The workload is written by hand and committed rather than sampled from a public
# dataset. It holds three kinds of prompt: paraphrase groups, which are the
# traffic a semantic cache is meant to exploit; negation pairs, where the two
# sides look alike but need opposite answers; and unrelated prompts, which
# should always miss. Every item names its group, and that is the ground truth a
# run is scored against: a hit is correct only when the answer it serves was
# produced for a prompt of the same group.
#
# Loading from disk rather than fetching means a run needs no network and cannot
# silently change between two runs that are meant to be compared.

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Kind = Literal["paraphrase", "negation", "unrelated"]
KINDS: tuple[Kind, ...] = ("paraphrase", "negation", "unrelated")

DEFAULT_PATH = Path(__file__).with_name("workload.json")
DEFAULT_SEED = 20261124


@dataclass(frozen=True)
class WorkloadItem:
    """One prompt of the workload and the group whose answers are correct for it."""

    id: str
    group: str
    kind: Kind
    prompt: str


def load(path: Path = DEFAULT_PATH) -> list[WorkloadItem]:
    """Read and validate the workload file.

    Fails on a duplicate id or prompt, an unknown kind, or an empty file, because
    each of those would make the scoring quietly wrong rather than loudly broken.
    """
    raw = json.loads(path.read_text())
    items = [
        WorkloadItem(
            id=entry["id"], group=entry["group"], kind=entry["kind"], prompt=entry["prompt"]
        )
        for entry in raw["items"]
    ]
    if not items:
        raise ValueError(f"{path} holds no items")
    for field in ("id", "prompt"):
        values = [getattr(item, field) for item in items]
        if len(set(values)) != len(values):
            raise ValueError(f"{path} has a duplicate {field}")
    unknown = {item.kind for item in items} - set(KINDS)
    if unknown:
        raise ValueError(f"{path} has unknown kinds: {sorted(unknown)}")
    return items


def replay_order(
    items: list[WorkloadItem], repeats: int = 1, seed: int = DEFAULT_SEED
) -> list[WorkloadItem]:
    """The order a run sends prompts in: every item, then `repeats` exact repeats of each.

    The first pass is shuffled with a fixed seed, so which member of a group
    arrives first, and is therefore the one that misses and gets stored, is the
    same on every run. Exact repeats follow in a second shuffled pass, standing
    in for the same question asked again later.
    """
    if repeats < 0:
        raise ValueError("repeats cannot be negative")
    rng = random.Random(seed)
    first = list(items)
    rng.shuffle(first)
    again: list[WorkloadItem] = []
    for _ in range(repeats):
        batch = list(items)
        rng.shuffle(batch)
        again.extend(batch)
    return first + again
