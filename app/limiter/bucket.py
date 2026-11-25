# The token bucket, over the two Lua scripts that do the work.
#
# Everything that must be atomic is in the scripts, because Redis runs a script
# without interleaving anything else: reserving tokens (scripts.lua) and crediting
# or debiting a balance (adjust.lua). This class does no arithmetic of its own. It
# names the bucket, calls the script, and reads the answer into types, so there is
# one implementation of the rules and Python cannot drift from it.
#
# A bucket belongs to a caller, identified by their API key. The key is hashed
# before it goes into Redis, so the credential itself never appears in the keyspace,
# where anyone able to list keys, a monitoring agent or an operator, could read it.

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import redis.asyncio as redis_async

# Read relative to this file, not the working directory. The scripts used to be
# opened by a path that only resolved when the process started in the repository
# root, which a container or a service manager does not guarantee.
_SCRIPTS = Path(__file__).resolve().parent
RESERVE_SCRIPT = (_SCRIPTS / "scripts.lua").read_text()
ADJUST_SCRIPT = (_SCRIPTS / "adjust.lua").read_text()

# Where a caller's key lands: long enough that two callers colliding is not a
# concern, short enough to keep keys small.
_KEY_DIGEST_CHARACTERS = 32


class ReserveStatus(IntEnum):
    """What the reserve script decided. The values are the ones it returns."""

    DENIED = 0
    ADMITTED = 1
    TOO_LARGE = 2


@dataclass(frozen=True)
class Reserve:
    """The outcome of trying to reserve tokens."""

    status: ReserveStatus
    remaining: int
    retry_after_seconds: int

    @property
    def admitted(self) -> bool:
        return self.status is ReserveStatus.ADMITTED


class TokenBucket:
    """Per caller token buckets held in Redis."""

    def __init__(
        self,
        redis_client: redis_async.Redis,
        capacity: int,
        refill_rate: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """Register the scripts and hold the bucket's shape.

        clock is a test seam, like the one on the semantic cache. In production it
        is None and the scripts read Redis's own clock, which is the point: every
        gateway in front of one Redis then agrees on what time it is. A test passes
        a clock it can move, so refill needs no sleeping.
        """
        if capacity <= 0 or refill_rate <= 0:
            raise ValueError("a bucket needs a positive capacity and refill rate")
        self.capacity = capacity
        self.refill_rate = refill_rate
        self._clock = clock
        self._reserve = redis_client.register_script(RESERVE_SCRIPT)
        self._adjust = redis_client.register_script(ADJUST_SCRIPT)

    @staticmethod
    def key(api_key: str, window: str) -> str:
        """The Redis key for one caller's bucket, with the caller's key hashed."""
        digest = hashlib.sha256(api_key.encode()).hexdigest()[:_KEY_DIGEST_CHARACTERS]
        return f"rl:{digest}:{window}"

    async def reserve(self, api_key: str, window: str, estimate: int) -> Reserve:
        """Try to take an estimated number of tokens from the caller's bucket."""
        args: list[float] = [self.capacity, self.refill_rate, estimate]
        if self._clock is not None:
            args.append(self._clock())
        status, remaining, retry = await self._reserve(
            keys=[self.key(api_key, window)], args=args
        )
        return Reserve(ReserveStatus(int(status)), int(remaining), int(retry))

    async def adjust(self, api_key: str, window: str, delta: int) -> int:
        """Credit (positive) or debit (negative) the bucket, clamped to what it can hold.

        Returns the tokens now in the bucket, or -1 if there is no bucket, which
        means it expired and is full again, so there was nothing to settle.
        """
        balance = await self._adjust(
            keys=[self.key(api_key, window)], args=[self.capacity, delta]
        )
        return int(balance)

    async def release(self, api_key: str, window: str, tokens: int) -> None:
        """Return a reservation in full, as when a cache answered and no provider was paid."""
        await self.adjust(api_key, window, tokens)

    async def reconcile(self, api_key: str, window: str, estimate: int, actual: int) -> None:
        """Settle a reservation against what the provider reported.

        In either direction. A reply that cost less than was reserved returns the
        difference; one that cost more takes it. Settling only ever in the caller's
        favour would let a caller who always spends more than they estimate pay
        for none of the excess.
        """
        await self.adjust(api_key, window, estimate - actual)
