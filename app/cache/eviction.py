# Scheduled removal of expired Tier 2 entries.
#
# Chroma has no TTL of its own. Every Tier 2 query already filters out entries
# past their expiry, so an expired answer is never served whether or not anything
# removes it. What the filter cannot do is reclaim the space: with nothing
# deleting them, expired entries accumulate for the life of the collection and
# every query searches an index that grows without bound.
#
# SemanticCache.evict_expired does the deleting. This module only decides when,
# which is the part that was missing: nothing called it.
#
# A pass never raises, and the loop never stops on its own. Housekeeping that
# died on the first Chroma hiccup would fail silently, and the first sign would
# be the collection growing. During an outage a pass costs nothing, because
# evict_expired answers the reconnect cooldown without touching Chroma, so there
# is no separate "is Chroma up" check to keep in step with it.

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable

from app.cache.semantic_cache import SemanticCache

log = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


async def evict_once(cache: SemanticCache, on_evicted: Callable[[int], None]) -> None:
    """Run one pass: remove what has expired, then any excess, and report how many went.

    Expired entries go first, so the capacity bound only ever removes live ones
    when expiry alone was not enough.

    Swallows every ordinary failure, including one from the callback, and says so
    in the log. The next interval simply tries again.
    """
    try:
        expired = await cache.evict_expired()
        excess = await cache.evict_over_capacity()
        if expired or excess:
            log.info("evicted %d expired and %d excess entries from Tier 2", expired, excess)
            on_evicted(expired + excess)
    except Exception:
        log.warning("Tier 2 eviction pass failed; trying again next interval", exc_info=True)


async def run_eviction_loop(
    cache: SemanticCache,
    *,
    interval_seconds: float,
    on_evicted: Callable[[int], None],
    sleep: Sleep = asyncio.sleep,
) -> None:
    """Evict expired entries every interval_seconds, until cancelled.

    Waits before the first pass rather than running one at startup: a gateway
    that has just started is busiest, and nothing is lost by waiting, since reads
    filter expired entries regardless. Each pass is a delete of rows that are
    already invisible to callers, so several gateways running this against one
    Chroma do redundant work but never conflicting work.

    sleep is a parameter so a test can step the loop without waiting in real time.
    """
    while True:
        await sleep(interval_seconds)
        await evict_once(cache, on_evicted)


def start_eviction(
    cache: SemanticCache | None,
    *,
    interval_seconds: float,
    on_evicted: Callable[[int], None],
) -> asyncio.Task[None] | None:
    """Start the loop as a background task, or return None if there is nothing to do.

    Nothing to do means no semantic cache, or an interval of zero, which is how
    the setting switches eviction off.
    """
    if cache is None or interval_seconds <= 0:
        return None
    return asyncio.create_task(
        run_eviction_loop(cache, interval_seconds=interval_seconds, on_evicted=on_evicted),
        name="tier2-eviction",
    )


async def stop_eviction(task: asyncio.Task[None] | None) -> None:
    """Cancel the loop and wait for it to finish. Safe to call with None, or twice."""
    if task is None:
        return
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
