"""Chroma inside the gateway process, for hosts with nowhere else to run it."""

from __future__ import annotations

import uuid

import pytest

from app.cache.semantic_cache import SemanticCache
from app.config import Settings
from app.main import chroma_client_factory
from app.models import Message
from tests.conftest import FakeClock, MappingEmbedder, base_vector, unit_vector_at


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


def test_embedded_chroma_is_off_by_default() -> None:
    assert _settings().chroma_embedded is False


def test_the_embedded_factory_builds_an_in_process_client() -> None:
    client = chroma_client_factory(_settings(chroma_embedded=True))()
    name = f"embedded-{uuid.uuid4().hex[:8]}"
    collection = client.get_or_create_collection(name, embedding_function=None)
    assert collection.count() == 0


def test_without_embedded_the_factory_still_points_at_the_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import chromadb

    seen: dict[str, object] = {}

    def fake_http_client(**kwargs: object) -> str:
        seen.update(kwargs)
        return "http client"

    monkeypatch.setattr(chromadb, "HttpClient", fake_http_client)
    factory = chroma_client_factory(_settings(chroma_host="http://chroma.internal:9000"))
    assert factory() == "http client"
    assert (seen["host"], seen["port"]) == ("chroma.internal", 9000)


async def test_the_semantic_cache_works_over_embedded_chroma(
    mapping_embedder: MappingEmbedder, fake_clock: FakeClock,
) -> None:
    cache = SemanticCache(
        chroma_client_factory(_settings(chroma_embedded=True)), mapping_embedder,
        collection_name=f"embedded-{uuid.uuid4().hex[:8]}", ttl_seconds=60, threshold=0.90,
        clock=fake_clock,
    )
    mapping_embedder.register("What is the capital of France?", base_vector())
    mapping_embedder.register("Which city is the capital of France?", unit_vector_at(0.95))
    await cache.add("gpt-4o-mini", [Message(role="user", content="What is the capital of France?")],
                    "e1", {"answer": "Paris"})
    hit = await cache.search(
        "gpt-4o-mini", [Message(role="user", content="Which city is the capital of France?")]
    )
    assert hit is not None and hit.response_body == {"answer": "Paris"}
    assert await cache.is_available()
