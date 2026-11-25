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
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.cache.embedder import Embedder
from app.cache.keys import normalise_messages
from app.cache.negation_guard import same_polarity
from app.models import Message

# Chroma defaults to squared L2 distance. Cosine has to be requested when the
# collection is created; under the default, 1 minus distance is not a
# similarity at all and every threshold comparison would be silently wrong.
COSINE_SPACE = {"hnsw:space": "cosine"}

log = logging.getLogger(__name__)

# Returned by _call when Chroma could not be used. A sentinel rather than None,
# because several Chroma methods legitimately return None on success, delete
# and upsert among them, so None cannot also mean failure.
UNAVAILABLE: Any = object()

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


@dataclass(frozen=True)
class Tier2Result:
    """Result of a Tier 2 search, distinguishing a cache miss from a Chroma outage.

    When unavailable is True, Chroma could not be reached and hit is always
    None. When False, hit is None on a genuine miss and a SemanticHit on a
    cache hit. The distinction matters because an outage looks identical to a
    miss at the SemanticHit | None level, so the pipeline cannot tell them
    apart and cannot record the degradation.
    """

    hit: SemanticHit | None
    unavailable: bool


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
        max_entries: int | None = None,
        reconnect_seconds: float = 30.0,
        timeout_seconds: float = 2.0,
        negation_guard: bool = False,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Hold what the cache needs; connect to nothing yet.

        Takes a factory rather than a client because the Chroma client raises
        at construction when the server is down. Building it here would turn a
        vector store outage into a gateway that will not start.

        After a failure the cache stays unavailable for reconnect_seconds before
        trying again. Without that, every request during an outage would pay for
        a connection attempt that is going to fail.

        Every call into Chroma is also given timeout_seconds. A vector store that
        has hung, rather than refused the connection, raises nothing, so without
        a bound the failure handling above would never run and each request would
        wait on it indefinitely.

        With negation_guard on, a candidate that clears the threshold is still
        refused when it and the new prompt disagree on negation; see
        app/cache/negation_guard.py.
        """
        self._client_factory = client_factory
        self._embedder = embedder
        self._collection_name = collection_name
        self._ttl_seconds = ttl_seconds
        self._threshold = threshold
        self._max_entries = max_entries
        self._reconnect_seconds = reconnect_seconds
        self._timeout_seconds = timeout_seconds
        self._negation_guard = negation_guard
        self._clock = clock
        self._collection: Any = None
        self._unavailable_until = 0.0

    def _get_collection(self) -> Any:
        """Connect on first use and fetch the collection in cosine space.

        Returns None while a previous failure's cooldown is running, and starts
        a new cooldown if connecting fails now. Never raises: losing the vector
        store should cost hit rate, not availability.
        """
        if self._collection is not None:
            return self._collection
        if self._clock() < self._unavailable_until:
            return None
        try:
            client = self._client_factory()
            self._collection = client.get_or_create_collection(
                self._collection_name, metadata=COSINE_SPACE
            )
        except Exception:
            self._mark_unavailable("connect")
            return None
        return self._collection

    def _mark_unavailable(self, operation: str) -> None:
        """Drop the connection and wait out the cooldown before trying again.

        Logged rather than raised, so the request carries on with Tier 1 only.
        The connection is dropped so the next attempt after the cooldown starts
        clean instead of reusing a client whose server has gone away.
        """
        log.warning("semantic cache unavailable during %s; serving Tier 1 only", operation,
                    exc_info=True)
        self._collection = None
        self._unavailable_until = self._clock() + self._reconnect_seconds

    def _cooling_down(self) -> bool:
        """Whether a recent failure is still being waited out.

        Answered on the event loop, from two attributes, so a request that arrives
        during an outage learns it without taking a worker thread. The threads
        are what a hung Chroma holds on to, and a cooldown check that needed one
        would queue behind them.
        """
        return self._collection is None and self._clock() < self._unavailable_until

    async def _within_timeout(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Run a blocking call in a worker thread, giving up after timeout_seconds.

        Raises TimeoutError if the call has not finished in time. Giving up
        releases the request, not the thread: a thread cannot be interrupted, so
        an abandoned call runs on until Chroma answers or the connection fails.
        That is bounded by the cooldown, because no new call is started while it
        runs.
        """
        return await asyncio.wait_for(
            asyncio.to_thread(func, *args, **kwargs), timeout=self._timeout_seconds
        )

    async def _call(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        """Run one collection method off the event loop, degrading on any failure.

        Returns UNAVAILABLE when Chroma cannot be reached, does not answer in
        time, or the call fails, after marking the cache unavailable. Every Chroma
        call goes through here, so there is one place that decides what an outage
        means.
        """
        if self._cooling_down():
            return UNAVAILABLE
        try:
            collection = await self._within_timeout(self._get_collection)
        except TimeoutError:
            self._mark_unavailable("connect")
            return UNAVAILABLE
        if collection is None:
            return UNAVAILABLE
        try:
            return await self._within_timeout(getattr(collection, operation), *args, **kwargs)
        except Exception:
            self._mark_unavailable(operation)
            return UNAVAILABLE

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
        await self._call(
            "upsert",
            ids=[entry_id],
            embeddings=[vector],
            documents=[text],
            metadatas=[metadata],
        )

    async def search_with_status(self, model: str, messages: list[Message]) -> Tier2Result:
        """Return the nearest live entry and whether Chroma was reachable.

        Expired entries are filtered out in the query itself, so one is never
        served even before eviction gets to it. The model is filtered too: the
        same prompt against a different model is a different answer.

        Returns Tier2Result.unavailable=True when Chroma could not be reached,
        so the caller can record the degradation rather than treating it as a
        routine miss. search() delegates here and discards the status, so
        existing callers need no change.
        """
        text = prompt_text(messages)
        vector = await self._embedder.encode(text)
        result = await self._call(
            "query",
            query_embeddings=[vector],
            n_results=1,
            where={"$and": [{"model": model}, {"expires_at": {"$gt": self._clock()}}]},
            include=["metadatas", "distances", "documents"],
        )
        if result is UNAVAILABLE:
            return Tier2Result(hit=None, unavailable=True)
        return Tier2Result(hit=self._best_hit(result, text), unavailable=False)

    async def search(self, model: str, messages: list[Message]) -> SemanticHit | None:
        """Return the nearest live entry for this model if it clears the threshold.

        Convenience wrapper over search_with_status that discards the outage
        flag. Use search_with_status when the caller needs to record whether
        Chroma was reachable.
        """
        return (await self.search_with_status(model, messages)).hit

    async def evict_over_capacity(self) -> int:
        """Delete the oldest entries beyond max_entries and return how many went.

        Oldest by when they were stored, since a hit does not write back to the
        entry. Reads every entry's metadata to sort them, which is why it runs on
        the eviction interval and never on a request, and only once the
        collection is over its bound.
        """
        if self._max_entries is None:
            return 0
        count = await self._call("count")
        if count is UNAVAILABLE or count <= self._max_entries:
            return 0
        everything = await self._call("get", include=["metadatas"])
        if everything is UNAVAILABLE:
            return 0
        by_age = sorted(
            zip(everything["ids"], everything["metadatas"], strict=True),
            key=lambda pair: pair[1].get("created_at", 0.0),
        )
        excess = [entry_id for entry_id, _ in by_age[: len(by_age) - self._max_entries]]
        if not excess or await self._call("delete", ids=excess) is UNAVAILABLE:
            return 0
        return len(excess)

    async def is_available(self) -> bool:
        """Report whether Chroma can be reached right now.

        Makes a real round trip rather than trusting a cached connection, since
        a server can disappear after the client was built. The health endpoint
        uses this to tell a degraded gateway from a healthy one.
        """
        return await self._call("count") is not UNAVAILABLE

    async def evict_expired(self) -> int:
        """Delete entries past their expiry and return how many went.

        Housekeeping only. Expiry is already enforced in every query, so an
        expired entry is never served whether or not this has run; this just
        stops the collection growing without bound.
        """
        expired = await self._call(
            "get", where={"expires_at": {"$lte": self._clock()}}, include=[]
        )
        if expired is UNAVAILABLE or not expired["ids"]:
            return 0
        if await self._call("delete", ids=expired["ids"]) is UNAVAILABLE:
            return 0
        return len(expired["ids"])

    def _best_hit(self, result: Any, text: str | None = None) -> SemanticHit | None:
        """Turn a raw query result into a hit, or None if empty, below threshold, or guarded.

        An empty or fully filtered collection comes back as [[]] rather than an
        error, so the emptiness check comes before any indexing. text is the new
        prompt, which the negation guard compares with the stored one; search
        always passes it.
        """
        ids = result["ids"][0]
        if not ids:
            return None
        similarity = similarity_from_distance(result["distances"][0][0])
        if similarity < self._threshold:
            return None
        if self._refused_by_guard(result, text):
            log.info("semantic hit at %.4f refused: the prompts differ in negation", similarity)
            return None
        metadata = result["metadatas"][0][0]
        return SemanticHit(
            response_body=json.loads(metadata["response_body"]),
            similarity=similarity,
            entry_id=ids[0],
        )

    def _refused_by_guard(self, result: Any, text: str | None) -> bool:
        """Whether the negation guard refuses this hit: the stored and new prompts disagree."""
        if not self._negation_guard or text is None:
            return False
        stored = (result.get("documents") or [[None]])[0][0]
        return stored is not None and not same_polarity(stored, text)
