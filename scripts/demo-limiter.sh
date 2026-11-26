#!/usr/bin/env bash
# Demonstrate the rate limiter: a burst of new questions until the budget runs out,
# the refusals with how long to wait, and admission again once it refills.
#
# Starts its own Redis, the mock vendor API answering after 800 ms like a real
# provider, and a gateway with a small budget (3000 tokens, refilling at 50 a
# second) so the story fits in a few seconds, and stops what it started. No keys,
# no cost.
#
# Every question is new, because a cache hit spends nothing and gives its
# reservation back, so repeats would never reach the limit. And they are sent at
# once: each request reserves its estimate (about 266 tokens here, most of it the
# allowance for the reply) and settles to what it really cost when it finishes,
# so one at a time the budget would barely move. While the provider is answering,
# the reservations of every request in flight are held together, and that is
# what a burst runs into.
source "$(dirname "$0")/common.sh"
require_venv

gateway_up && die "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"

cleanup() { "$ROOT/scripts/gateway-stop.sh" >/dev/null; "$ROOT/scripts/mock-upstream-stop.sh" >/dev/null; }
trap cleanup EXIT

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/mock-upstream-start.sh"
curl -s -X POST "$MOCK_URL/reset" >/dev/null
curl -s -X PUT "$MOCK_URL/control/openai" -H 'Content-Type: application/json' \
  -d '{"delay_ms": 800}' >/dev/null

export SEMCACHE_PRIMARY_PROVIDER=openai OPENAI_API_KEY=demo-key SEMCACHE_OPENAI_BASE_URL="$MOCK_URL/v1"
export SEMCACHE_LIMITER_ENABLED=true SEMCACHE_BUCKET_CAPACITY=3000 SEMCACHE_BUCKET_REFILL_RATE=50
"$ROOT/scripts/gateway-start.sh"
reset_cache >/dev/null
# A fresh budget too: the limiter's keys live beside the cache's in Redis.
"$PY" -c "import redis, sys; r = redis.Redis(port=int(sys.argv[1])); keys = list(r.scan_iter('rl:*')); keys and r.delete(*keys)" "$REDIS_PORT"

attempt() {
  # attempt <n>: one new question; prints its status, the tokens left or the wait.
  local headers code
  headers=$(mktemp)
  code=$(curl -s -o /dev/null -D "$headers" -w '%{http_code}' "$GATEWAY_URL/v1/chat/completions" \
    -H 'Authorization: Bearer demo-key' -H 'Content-Type: application/json' \
    -d "$(question "Brand new question number $1, asked at $(date +%s%N)")")
  if [ "$code" = 200 ]; then
    printf '  %2d  HTTP 200  tokens left %s\n' "$1" "$(grep -i '^x-ratelimit-remaining-tokens:' "$headers" | tr -dc 0-9)"
  else
    printf '  %2d  HTTP %s  retry after %s s\n' "$1" "$code" "$(grep -i '^retry-after:' "$headers" | tr -dc 0-9)"
  fi
  rm -f "$headers"
}

step "1. 15 new questions at once against a 3000 token budget"
burst=$(mktemp)
for n in $(seq 1 15); do attempt "$n" >>"$burst" & done
wait
sort -n "$burst"
rm -f "$burst"

step "2. waiting 6 seconds while the budget refills at 50 tokens a second"
sleep 6

step "3. admitted again"
attempt 16
curl -s "$GATEWAY_URL/metrics" | jq -r '"  throttled_requests: \(.throttled_requests)"'
