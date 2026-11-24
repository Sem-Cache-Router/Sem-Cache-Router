"""The benchmark workload: the committed file loads, validates, and replays in a fixed order."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.workload import DEFAULT_SEED, KINDS, WorkloadItem, load, replay_order


def _write(path: Path, items: list[dict[str, str]]) -> Path:
    path.write_text(json.dumps({"version": 1, "items": items}))
    return path


def _item(n: int, **overrides: str) -> dict[str, str]:
    entry = {"id": f"i{n}", "group": f"g{n}", "kind": "unrelated", "prompt": f"prompt {n}"}
    entry.update(overrides)
    return entry


def test_the_committed_workload_loads_with_every_kind() -> None:
    items = load()
    kinds = Counter(item.kind for item in items)
    assert set(kinds) == set(KINDS)
    assert len({item.prompt for item in items}) == len(items)


def test_negation_pairs_sit_in_different_groups() -> None:
    # The scoring depends on it: a hit across a negation pair must count as wrong.
    negations = [item for item in load() if item.kind == "negation"]
    assert len(negations) % 2 == 0
    assert len({item.group for item in negations}) == len(negations)


def test_paraphrase_groups_have_more_than_one_member() -> None:
    sizes = Counter(item.group for item in load() if item.kind == "paraphrase")
    assert min(sizes.values()) >= 2


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ([], "no items"),
        ([_item(1), _item(2, id="i1")], "duplicate id"),
        ([_item(1), _item(2, prompt="prompt 1")], "duplicate prompt"),
        ([_item(1, kind="mystery")], "unknown kinds"),
    ],
)
def test_a_malformed_workload_is_refused(
    tmp_path: Path, items: list[dict[str, str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load(_write(tmp_path / "w.json", items))


def test_replay_order_is_the_same_for_the_same_seed() -> None:
    items = load()
    assert replay_order(items) == replay_order(items, seed=DEFAULT_SEED)
    assert replay_order(items, seed=1) != replay_order(items, seed=2)


def test_the_first_pass_is_shuffled_so_no_group_member_is_always_first() -> None:
    # The file lists each group's members in order; replaying it unshuffled would
    # always store the first wording and test only the others against it.
    items = load()
    first_pass = replay_order(items, repeats=0)
    assert first_pass != items
    assert replay_order(items, repeats=0, seed=1) != replay_order(items, repeats=0, seed=2)


def test_replay_order_sends_every_item_once_then_each_repeat() -> None:
    items = [WorkloadItem(f"i{n}", f"g{n}", "unrelated", f"p{n}") for n in range(5)]
    order = replay_order(items, repeats=2)
    assert len(order) == 15
    assert sorted(item.id for item in order[:5]) == sorted(item.id for item in items)
    assert Counter(item.id for item in order) == {item.id: 3 for item in items}


def test_negative_repeats_are_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        replay_order(load(), repeats=-1)
