# Tier 1: exact match cache in Redis.
#
# Checked before the semantic tier because embedding is the expensive step in
# the lookup path and literal repeats, which are common, do not need it.
#
# A hit updates hit count and last hit timestamp but leaves the remaining TTL
# untouched. TTL here means freshness, not popularity, and a sliding expiry
# would let a frequently requested answer live indefinitely, which is exactly
# where a stale answer does the most damage.
#
# Writes happen only on a provider miss, so the cost is paid once per genuinely
# new answer.
#
# STORAGE SHAPE. The entry is a Redis hash, not a string. The PRD specifies a
# string, and this is a deliberate, recorded deviation. A string holds the whole
# entry as one blob, so incrementing the hit count means reading it, editing it
# and writing it back, and a plain SET clears the TTL. That resurrects the entry
# for a full fresh lifetime on every hit: the sliding expiry the design rejects,
# arriving by accident, with no error and no failing test. SET with KEEPTTL
# avoids the TTL bug but leaves the read, modify, write racing with itself, so
# two simultaneous hits lose an increment. A hash fixes both: HINCRBY is atomic,
# and HSET on an existing key does not touch its TTL at all, so preservation
# becomes a property of the data model rather than a flag somebody has to
# remember on every write.
#
# Forbidden on the read path, all of which rewrite or clear the TTL:
#   SET, SETEX, GETEX, EXPIRE, PERSIST

from __future__ import annotations

import uuid
from collections.abc import Awaitable
from datetime import UTC, datetime, timedelta
from typing import Any

from redis.asyncio import Redis

from app.cache.keys import make_key
from app.models import CacheEntry, Message


async def _resolve[T](result: Awaitable[T] | T) -> T:
    """Await a redis-py command result.

    redis-py declares its commands as returning either an awaitable or a plain
    value, because one class serves both the sync and the async client. On the
    async client it is always the awaitable branch, but a type checker cannot
    narrow that from the signature alone. Narrowing once here keeps the call
    sites readable and avoids a cast on every command.
    """
    if isinstance(result, Awaitable):
        return await result
    return result


FIELD_ENTRY = "entry"
FIELD_HIT_COUNT = "hit_count"
FIELD_LAST_HIT_AT = "last_hit_at"


class ExactCache:
    """Hash keyed response cache with a TTL applied once, at write time."""

    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        """Hold the redis handle and the write time TTL.

        The client is injected rather than constructed here, so the route never
        sees Redis directly and a test can substitute a fake without patching
        module globals.
        """
        self._redis = redis
        self._ttl_seconds = ttl_seconds

    def _key(self, model: str, messages: list[Message]) -> str:
        """Derive the Redis key for this prompt.

        Delegates and adds nothing. It exists as the single seam where a per
        instance namespace could later be introduced, for test isolation or for
        the per tenant cache keys any real deployment would need, without
        touching the pure hashing module. The namespace policy lives in
        keys.make_key and is not duplicated here, because a prefix applied twice
        produces a cache that never hits.
        """
        return make_key(model, messages)

    async def get(self, model: str, messages: list[Message]) -> CacheEntry | None:
        """Return the cached entry for this prompt, or None on a miss.

        Records the hit, and deliberately does not touch the TTL. Expiry is
        checked against the stored timestamp as well as relying on Redis, so an
        entry is never served past its stated life even if the key outlived it.
        """
        key = self._key(model, messages)

        # redis-py types this as possibly bytes, because decode_responses is a
        # runtime flag the type checker cannot see. The client is configured to
        # decode, so normalise once here and let nothing downstream care.
        raw: Any = await _resolve(self._redis.hgetall(key))
        stored = {str(field): str(value) for field, value in raw.items()}

        if not stored or FIELD_ENTRY not in stored:
            return None

        entry = CacheEntry.model_validate_json(stored[FIELD_ENTRY])

        if entry.expires_at <= datetime.now(UTC):
            await self._redis.delete(key)
            return None

        # HINCRBY is atomic, so two concurrent hits cannot lose an increment,
        # and neither it nor HSET disturbs the key's remaining TTL.
        hit_count = await _resolve(self._redis.hincrby(key, FIELD_HIT_COUNT, 1))
        last_hit_at = datetime.now(UTC)
        await _resolve(self._redis.hset(key, FIELD_LAST_HIT_AT, last_hit_at.isoformat()))

        entry.hit_count = int(hit_count)
        entry.last_hit_at = last_hit_at
        return entry

    async def set(
        self,
        model: str,
        messages: list[Message],
        response_body: dict[str, Any],
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float = 0.0,
    ) -> CacheEntry:
        """Write a provider response into Tier 1 with a fresh TTL.

        Called only on a provider miss. EXPIRE is issued here and nowhere else,
        which is what makes the no-sliding-expiry rule hold by construction
        rather than by discipline.

        prompt_text stores the normalised prompt, which is what Tier 2 will
        embed and what makes a cached answer reviewable during evaluation.
        """
        key = self._key(model, messages)
        now = datetime.now(UTC)

        entry = CacheEntry(
            entry_id=uuid.uuid4().hex,
            prompt_hash=key,
            prompt_text=messages[-1].content,
            response_body=response_body,
            model_used=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            created_at=now,
            expires_at=now + timedelta(seconds=self._ttl_seconds),
        )

        async with self._redis.pipeline(transaction=True) as pipe:
            pipe.hset(
                key,
                mapping={
                    FIELD_ENTRY: entry.model_dump_json(exclude={"hit_count", "last_hit_at"}),
                    FIELD_HIT_COUNT: 0,
                    FIELD_LAST_HIT_AT: "",
                },
            )
            pipe.expire(key, self._ttl_seconds)
            await pipe.execute()

        return entry

    async def delete(self, model: str, messages: list[Message]) -> bool:
        """Remove one entry. Returns whether anything was there.

        Needed by the cache flush endpoint and by any test that has to force a
        miss without waiting out a TTL.
        """
        removed: int = await self._redis.delete(self._key(model, messages))
        return removed > 0
