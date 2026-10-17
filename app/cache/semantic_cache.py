# Tier 2: semantic cache backed by ChromaDB.
#
# Holds a SemanticCache class with:
#   search          nearest live entry for the model and its score, if the score
#                   meets the threshold. The threshold is configuration, set once,
#                   so a benchmark sweeps it by running each value separately.
#   add             index a response against its prompt embedding
#   evict_expired   remove entries past their TTL, since Chroma has no native one
#   is_available    whether Chroma is reachable
#
# This tier is the actual subject of the project. Unlike a normal cache, a hit
# here can be wrong, so every hit carries the similarity score that produced it
# and that score is recorded on the trace span. That is what makes a wrong
# answer traceable back to the decision that produced it.
#
# is_available returning false makes the gateway serve Tier 1 only and record
# the degradation, rather than failing the request. Losing the vector store
# should cost hit rate, not availability.
#
# Every call into Chroma passes vectors from our own embedder. Chroma will
# otherwise embed text itself with a bundled model, and that is true even when a
# collection is created with embedding_function=None: verified on 1.5.9, an add
# with documents only and a query with query_texts both succeed by embedding
# silently. On a machine without the model cached that is a 79 MB download in
# the middle of a request. The guard is that no method here ever hands Chroma
# text without a vector, and a test patches Chroma's embedder to raise and
# proves it is never reached.

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.cache.embedder import Embedder
from app.cache.keys import normalise_messages
from app.models import Message

# Chroma defaults to squared L2 distance. Cosine has to be requested when the
# collection is created; under the default, 1 minus distance is not a
# similarity at all and every threshold comparison would be silently wrong.
COSINE_SPACE = {"hnsw:space": "cosine"}

ClientFactory = Callable[[], Any]


@dataclass(frozen=True)
class SemanticHit:
    """A cached answer, the score that justified serving it, and where it came from.

    The score travels with the answer because a semantic hit can be wrong, and
    a wrong answer has to be traceable to the decision that produced it.
    """

    response_body: dict[str, Any]
    similarity: float
    entry_id: str


def similarity_from_distance(distance: float) -> float:
    """Convert a Chroma cosine distance to a cosine similarity.

    Only valid in cosine space, which is why the space is fixed where the
    collection is created rather than left to Chroma's default.
    """
    return 1.0 - distance


def prompt_text(messages: list[Message]) -> str:
    """The text that gets embedded for a conversation.

    The normalised content of every message, in order, so the same whitespace
    rules apply as at Tier 1. The model reads roughly the first 256 word pieces,
    so a long conversation is effectively judged on how it opens.
    """
    return "\n".join(message.content for message in normalise_messages(messages))


class SemanticCache:
    """Nearest neighbour cache over prompt embeddings, with a similarity threshold."""

    def __init__(
        self,
        client_factory: ClientFactory,
        embedder: Embedder,
        *,
        collection_name: str,
        ttl_seconds: int,
        threshold: float,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Hold what the cache needs; connect to nothing yet.

        Takes a factory rather than a client because the Chroma client raises
        at construction when the server is down. Building it here would turn a
        vector store outage into a gateway that will not start.
        """
        self._client_factory = client_factory
        self._embedder = embedder
        self._collection_name = collection_name
        self._ttl_seconds = ttl_seconds
        self._threshold = threshold
        self._clock = clock
        self._collection: Any = None

    def _get_collection(self) -> Any:
        """Connect on first use and fetch the collection in cosine space."""
        if self._collection is None:
            client = self._client_factory()
            self._collection = client.get_or_create_collection(
                self._collection_name, metadata=COSINE_SPACE
            )
        return self._collection

    async def add(
        self,
        model: str,
        messages: list[Message],
        entry_id: str,
        response_body: dict[str, Any],
    ) -> None:
        """Index a provider response against its prompt embedding.

        Uses Tier 1's entry id as the document id, so a hit in either tier
        traces to the same origin request. Expiry is a timestamp in the metadata,
        because Chroma has no TTL of its own.
        """
        text = prompt_text(messages)
        vector = await self._embedder.encode(text)
        now = self._clock()
        metadata = {
            "model": model,
            "created_at": now,
            "expires_at": now + self._ttl_seconds,
            "response_body": json.dumps(response_body),
        }
        collection = await asyncio.to_thread(self._get_collection)
        await asyncio.to_thread(
            collection.upsert,
            ids=[entry_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata],
        )

    async def search(self, model: str, messages: list[Message]) -> SemanticHit | None:
        """Return the nearest live entry for this model if it clears the threshold.

        Expired entries are filtered out in the query itself, so one is never
        served even before eviction gets to it. The model is filtered too: the
        same prompt against a different model is a different answer.
        """
        vector = await self._embedder.encode(prompt_text(messages))
        collection = await asyncio.to_thread(self._get_collection)
        result = await asyncio.to_thread(
            collection.query,
            query_embeddings=[vector],
            n_results=1,
            where={"$and": [{"model": model}, {"expires_at": {"$gt": self._clock()}}]},
            include=["metadatas", "distances"],
        )
        return self._best_hit(result)

    def _best_hit(self, result: Any) -> SemanticHit | None:
        """Turn a raw query result into a hit, or None if empty or below threshold.

        An empty or fully filtered collection comes back as [[]] rather than an
        error, so the emptiness check comes before any indexing.
        """
        ids = result["ids"][0]
        if not ids:
            return None
        similarity = similarity_from_distance(result["distances"][0][0])
        if similarity < self._threshold:
            return None
        metadata = result["metadatas"][0][0]
        return SemanticHit(
            response_body=json.loads(metadata["response_body"]),
            similarity=similarity,
            entry_id=ids[0],
        )
