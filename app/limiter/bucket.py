from __future__ import annotations

import time

import redis.asyncio as redis_async


class TokenBucket:
    def __init__(self, redis_client: redis_async.Redis, capacity: int, refill_rate: float) -> None:
        self.redis = redis_client
        self.capacity = capacity
        self.refill_rate = refill_rate

        with open("app/limiter/scripts.lua") as f:
            script_content = f.read()
        self.script = self.redis.register_script(script_content)

    async def reserve(self, api_key: str, window: str, estimate: int) -> tuple[bool, int, int]:
        """Attempt to reserve an estimated token count.

        Returns (admitted, tokens_remaining, retry_after_seconds).
        """
        key = f"rl:{api_key}:{window}"
        now = int(time.time())
        result = await self.script(
            keys=[key],
            args=[self.capacity, self.refill_rate, estimate, now]
        )
        return bool(result[0]), int(result[1]), int(result[2])

    async def release(self, api_key: str, window: str, tokens: int) -> None:
        """Return a reservation in full (e.g. on cache hit)."""
        key = f"rl:{api_key}:{window}"
        release_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local return_tokens = tonumber(ARGV[2])
        local current = redis.call("HGET", key, "tokens_remaining")
        if current then
            local new_val = math.min(capacity, tonumber(current) + return_tokens)
            redis.call("HSET", key, "tokens_remaining", new_val)
        end
        """
        await self.redis.eval(release_script, 1, key, str(self.capacity), str(tokens))

    async def reconcile(self, api_key: str, window: str, estimate: int, actual: int) -> None:
        """Settle a reservation against reported usage."""
        diff = estimate - actual
        if diff > 0:
            await self.release(api_key, window, diff)
