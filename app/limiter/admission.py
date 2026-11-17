# Admission control: whether a request may run, and the settling of what it cost.
#
# The request path asks for tokens before it does anything else (PRD section 16 and
# the ordering in app/api/chat.py), runs, and then either gives the tokens back, if
# a cache answered, or settles them against what the provider reported. This is
# that conversation, over the token bucket.
#
# Two failures are treated opposite ways, deliberately. If the limiter cannot be
# reached when a request asks to be admitted, the request is refused: admission
# cannot be decided, and letting traffic through unmetered whenever Redis is down
# would make the budget something a caller could escape by waiting for an outage.
# If the limiter cannot be reached afterwards, to give tokens back or settle, the
# request is left alone. The caller already has their answer, failing it now would
# punish them for the gateway's problem, and the reservation simply stays spent,
# which errs on the side of the budget.

from __future__ import annotations

import logging
from dataclasses import dataclass

from redis.exceptions import RedisError

from app.limiter.bucket import ReserveStatus, TokenBucket

log = logging.getLogger(__name__)

# One bucket per caller. The bucket supports several windows per caller, such as a
# minute and a day, and a fixed name keeps this to the single budget the gateway
# has today without closing off the others.
DEFAULT_WINDOW = "tokens"


class LimiterUnavailable(Exception):
    """The limiter could not be reached, so admission cannot be decided."""


@dataclass(frozen=True)
class Reservation:
    """Tokens held against a caller's bucket for one request."""

    api_key: str
    tokens: int


@dataclass(frozen=True)
class Admission:
    """The decision on one request.

    A reservation exactly when it was admitted. When it was not, status says why:
    denied for now, with the seconds until it could be admitted, or too large for
    the bucket to ever hold, in which case no wait helps and the retry hint is zero.
    """

    status: ReserveStatus
    reservation: Reservation | None
    remaining: int
    retry_after_seconds: int

    @property
    def admitted(self) -> bool:
        return self.reservation is not None


class AdmissionController:
    """Reserves tokens for requests and settles them afterwards."""

    def __init__(self, bucket: TokenBucket, window: str = DEFAULT_WINDOW) -> None:
        self._bucket = bucket
        self._window = window

    async def reserve(self, api_key: str, estimate: int) -> Admission:
        """Decide whether this caller may spend an estimated number of tokens.

        Raises LimiterUnavailable if the limiter cannot be reached, which the
        caller must turn into a refusal rather than an admission.
        """
        try:
            result = await self._bucket.reserve(api_key, self._window, estimate)
        except (RedisError, OSError) as exc:
            raise LimiterUnavailable("the rate limiter could not be reached") from exc

        reservation = Reservation(api_key, estimate) if result.admitted else None
        return Admission(result.status, reservation, result.remaining, result.retry_after_seconds)

    async def release(self, reservation: Reservation) -> None:
        """Give a reservation back in full, as when a cache answered and nothing was spent.

        Never raises: see the note at the top for why a failure here must not fail
        the request.
        """
        try:
            await self._bucket.release(reservation.api_key, self._window, reservation.tokens)
        except (RedisError, OSError):
            log.warning("could not release %d tokens; the reservation stays spent",
                        reservation.tokens, exc_info=True)

    async def reconcile(self, reservation: Reservation, actual_tokens: int) -> None:
        """Settle a reservation against the tokens the provider reported using.

        Returns the difference when the reply cost less than was reserved and takes
        it when it cost more. Never raises, for the same reason as release.
        """
        try:
            await self._bucket.reconcile(
                reservation.api_key, self._window, reservation.tokens, actual_tokens
            )
        except (RedisError, OSError):
            log.warning("could not settle a reservation of %d tokens against %d used",
                        reservation.tokens, actual_tokens, exc_info=True)
