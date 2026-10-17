# Tier 2 semantic cache tests.
#
# Vectors are built by hand with an exact, known cosine against a base vector,
# so these tests can assert behaviour either side of the threshold. Real
# embeddings cannot be steered that way; their behaviour is measured separately
# by the adversarial probe.

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from app.cache.semantic_cache import (
    SemanticCache,
    prompt_text,
    similarity_from_distance,
)
from app.models import Message
from tests.conftest import FakeClock, MappingEmbedder, base_vector, unit_vector_at

MODEL = "gpt-4o-mini"
BODY = {"choices": [{"message": {"role": "assistant", "content": "Paris."}}]}
STORED = "What is the capital of France?"
ASKED = "Tell me France's capital city."


def user(content: str) -> list[Message]:
    """One user message."""
    return [Message(role="user", content=content)]


@pytest.fixture
def cache(
    chroma_factory: Callable[[], Any],
    mapping_embedder: MappingEmbedder,
    fake_clock: FakeClock,
    collection_name: str,
) -> SemanticCache:
    """A semantic cache over in-memory Chroma, threshold 0.90, TTL 60 seconds."""
    return SemanticCache(
        chroma_factory,
        mapping_embedder,
        collection_name=collection_name,
        ttl_seconds=60,
        threshold=0.90,
        clock=fake_clock,
    )


async def store(cache: SemanticCache, embedder: MappingEmbedder, entry_id: str = "e1") -> None:
    """Seed one entry whose prompt embeds to the base vector."""
    embedder.register(STORED, base_vector())
    await cache.add(MODEL, user(STORED), entry_id, BODY)


# --- threshold ------------------------------------------------------------


async def test_a_close_paraphrase_hits(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """Cosine 0.95 clears a threshold of 0.90."""
    await store(cache, mapping_embedder)
    mapping_embedder.register(ASKED, unit_vector_at(0.95))
    hit = await cache.search(MODEL, user(ASKED))
    assert hit is not None
    assert hit.response_body == BODY


async def test_the_reported_similarity_is_the_true_cosine(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """Guards the most dangerous trap in this module.

    Chroma defaults to squared L2 distance. For these unit vectors L2 gives a
    distance of 0.10, so a cache left on the default would report 0.90 for a
    pair whose real cosine is 0.95, and still pass a 0.90 threshold. Only the
    value catches it; a hit or miss check does not.
    """
    await store(cache, mapping_embedder)
    mapping_embedder.register(ASKED, unit_vector_at(0.95))
    hit = await cache.search(MODEL, user(ASKED))
    assert hit is not None
    assert hit.similarity == pytest.approx(0.95, abs=1e-3)


async def test_a_distant_prompt_misses(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """Cosine 0.85 does not clear 0.90, so the provider gets the request."""
    await store(cache, mapping_embedder)
    mapping_embedder.register(ASKED, unit_vector_at(0.85))
    assert await cache.search(MODEL, user(ASKED)) is None


def test_a_score_exactly_at_the_threshold_hits(cache: SemanticCache) -> None:
    """The comparison is at or above. Tested on the decision itself, because a
    vector built to sit exactly on the boundary comes back from Chroma in
    float32 and lands a hair either side."""
    result = {
        "ids": [["e1"]],
        "distances": [[0.10]],
        "metadatas": [[{"response_body": '{"ok": true}'}]],
    }
    hit = cache._best_hit(result)
    assert hit is not None
    assert hit.similarity == pytest.approx(0.90)


# --- what else must match -------------------------------------------------


async def test_a_different_model_never_matches(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """The same question to a different model is a different answer."""
    await store(cache, mapping_embedder)
    mapping_embedder.register(STORED, base_vector())
    assert await cache.search("gpt-4o", user(STORED)) is None


async def test_an_expired_entry_is_not_served(
    cache: SemanticCache, mapping_embedder: MappingEmbedder, fake_clock: FakeClock
) -> None:
    """Expiry is enforced in the query, so a stale entry is never served,
    even though eviction has not removed it yet."""
    await store(cache, mapping_embedder)
    fake_clock.advance(61)
    assert await cache.search(MODEL, user(STORED)) is None
    assert cache._get_collection().count() == 1


async def test_an_entry_just_inside_its_ttl_is_served(
    cache: SemanticCache, mapping_embedder: MappingEmbedder, fake_clock: FakeClock
) -> None:
    """The expiry filter must not cut entries off early."""
    await store(cache, mapping_embedder)
    fake_clock.advance(59)
    assert await cache.search(MODEL, user(STORED)) is not None


async def test_an_empty_collection_returns_nothing_rather_than_raising(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """Chroma answers an empty search with [[]]; indexing it blindly raises."""
    mapping_embedder.register(STORED, base_vector())
    assert await cache.search(MODEL, user(STORED)) is None


# --- what is stored -------------------------------------------------------


async def test_the_hit_carries_tier_one_entry_id(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """The shared id is what traces a hit in either tier to one origin request."""
    await store(cache, mapping_embedder, entry_id="abc123")
    hit = await cache.search(MODEL, user(STORED))
    assert hit is not None
    assert hit.entry_id == "abc123"


async def test_re_adding_the_same_entry_does_not_duplicate_it(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """Writes are upserts keyed on the entry id."""
    await store(cache, mapping_embedder)
    await store(cache, mapping_embedder)
    assert cache._get_collection().count() == 1


async def test_the_response_body_round_trips_exactly(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """A hit replays exactly what the miss returned."""
    body = {"id": "x", "choices": [{"message": {"content": "naïve café, 日本"}}], "usage": {"n": 1}}
    mapping_embedder.register(STORED, base_vector())
    await cache.add(MODEL, user(STORED), "e1", body)
    hit = await cache.search(MODEL, user(STORED))
    assert hit is not None
    assert hit.response_body == body


# --- guards ---------------------------------------------------------------


async def test_chroma_never_embeds_text_itself(
    cache: SemanticCache, mapping_embedder: MappingEmbedder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If Chroma is handed text without a vector it embeds the text with its own
    bundled model, which on a clean machine is a 79 MB download mid request, and
    it does so even for a collection created with embedding_function=None. This
    makes Chroma's embedder fail, then proves add and search never reach it."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Chroma tried to embed text itself")

    monkeypatch.setattr(ONNXMiniLM_L6_V2, "__call__", refuse)

    await store(cache, mapping_embedder)
    assert await cache.search(MODEL, user(STORED)) is not None


async def test_the_embedder_sees_normalised_text(
    cache: SemanticCache, mapping_embedder: MappingEmbedder
) -> None:
    """Whitespace noise is folded before embedding, as it is before hashing."""
    await store(cache, mapping_embedder)
    await cache.search(MODEL, user("  What is   the capital\nof France?  "))
    assert mapping_embedder.calls[-1] == "What is the capital of France?"


def test_prompt_text_joins_messages_in_order() -> None:
    """Every message contributes, in conversation order."""
    messages = [Message(role="system", content=" Be brief. "), Message(role="user", content="Hi")]
    assert prompt_text(messages) == "Be brief.\nHi"


def test_similarity_is_one_minus_cosine_distance() -> None:
    """Named and tested on its own, because it is only true in cosine space."""
    assert similarity_from_distance(0.0) == 1.0
    assert similarity_from_distance(0.4) == pytest.approx(0.6)
