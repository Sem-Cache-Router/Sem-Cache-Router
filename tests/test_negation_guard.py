"""The negation guard: polarity of a prompt, and a semantic hit refused across it."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.cache.negation_guard import negation_count, polarity, same_polarity
from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.models import Message
from tests.conftest import FakeClock, MappingEmbedder, base_vector, unit_vector_at

MODEL = "gpt-4o-mini"
BODY = {"choices": [{"message": {"role": "assistant", "content": "Yes, 9 to 5."}}]}
OPEN = "Is the store open on Sunday?"
NOT_OPEN = "Is the store not open on Sunday?"
REWORDED = "Does the store open on Sundays?"


@pytest.mark.parametrize(
    ("text", "count"),
    [
        ("Is the store open?", 0),
        ("Is the store not open?", 1),
        ("Isn't the store open?", 1),
        ("Isn’t the store open?", 1),
        ("I can't say it is never open", 2),
        ("Nobody knows", 1),
        ("Is it cannot or can not?", 2),
        ("Note the notice on the knot", 0),
    ],
)
def test_negation_count(text: str, count: int) -> None:
    assert negation_count(text) == count


def test_an_odd_count_is_negated_and_a_double_negative_is_not() -> None:
    assert polarity(NOT_OPEN) is True
    assert polarity(OPEN) is False
    assert polarity("It is not never open") is False


def test_same_polarity() -> None:
    assert same_polarity(OPEN, REWORDED)
    assert not same_polarity(OPEN, NOT_OPEN)
    assert same_polarity(NOT_OPEN, "Isn't the store open on Sunday?")


def _cache(factory: Callable[[], Any], embedder: MappingEmbedder, clock: FakeClock,
           name: str, guard: bool) -> SemanticCache:
    return SemanticCache(factory, embedder, collection_name=name, ttl_seconds=60,
                         threshold=0.90, negation_guard=guard, clock=clock)


async def _ask_after_storing(
    cache: SemanticCache, embedder: MappingEmbedder, asked: str, cosine: float
) -> Any:
    embedder.register(OPEN, base_vector())
    await cache.add(MODEL, [Message(role="user", content=OPEN)], "e1", BODY)
    embedder.register(asked, unit_vector_at(cosine))
    return await cache.search(MODEL, [Message(role="user", content=asked)])


async def test_with_the_guard_a_negated_question_is_not_served_the_plain_answer(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, guard=True)
    assert await _ask_after_storing(cache, mapping_embedder, NOT_OPEN, 0.97) is None


async def test_with_the_guard_a_paraphrase_of_the_same_polarity_still_hits(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, guard=True)
    hit = await _ask_after_storing(cache, mapping_embedder, REWORDED, 0.95)
    assert hit is not None
    assert hit.response_body == BODY


async def test_without_the_guard_the_negated_question_is_a_hit_as_before(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    # The behaviour the guard exists to change, pinned so the default stays visible.
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, guard=False)
    hit = await _ask_after_storing(cache, mapping_embedder, NOT_OPEN, 0.97)
    assert hit is not None
    assert hit.similarity == pytest.approx(0.97, abs=1e-3)


async def test_the_guard_never_turns_a_below_threshold_miss_into_a_hit(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, guard=True)
    assert await _ask_after_storing(cache, mapping_embedder, REWORDED, 0.85) is None


def test_the_guard_is_off_unless_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings(_env_file=None).semantic_negation_guard is False  # type: ignore[call-arg]
    monkeypatch.setenv("SEMCACHE_SEMANTIC_NEGATION_GUARD", "true")
    assert Settings(_env_file=None).semantic_negation_guard is True  # type: ignore[call-arg]


async def test_the_gateway_builds_tier2_with_the_configured_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    class NoLoad:
        def __init__(self, model: str, **options: object) -> None:
            pass

        def load(self) -> None:
            pass

    monkeypatch.setattr(main, "OnnxEmbedder", NoLoad)
    settings = Settings(_env_file=None, semantic_negation_guard=True)  # type: ignore[call-arg]
    cache = await main.build_semantic_cache(settings)
    assert cache._negation_guard is True


def test_the_guard_lets_a_hit_through_when_the_stored_prompt_is_unknown(
    chroma_factory: Callable[[], Any], mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock, collection_name: str,
) -> None:
    # Entries written before the guard existed still carry their prompt, but a
    # result without documents must not crash the lookup or invent a refusal.
    cache = _cache(chroma_factory, mapping_embedder, fake_clock, collection_name, guard=True)
    result = {"ids": [["e1"]], "distances": [[0.03]],
              "metadatas": [[{"response_body": '{"ok": true}'}]]}
    assert cache._best_hit(result, NOT_OPEN) is not None
    assert cache._best_hit(dict(result, documents=[[OPEN]]), NOT_OPEN) is None
    assert cache._best_hit(dict(result, documents=[[OPEN]])) is not None
