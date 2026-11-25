#!/usr/bin/env bash
# Load test the gateway at 50 concurrent callers and check it keeps up.
#
# Starts Redis, Chroma, the mock vendor API (answering after a delay, like a real
# provider) and a gateway with both cache tiers on, then runs locustfile.py
# headless. Locust runs through uvx, in an environment of its own, so neither it
# nor gevent becomes a dependency of the gateway or of its image.
#
#   scripts/load-test.sh                 50 users for 60 seconds
#   USERS=100 DURATION=120s scripts/load-test.sh
#
# Exits non zero when a request failed, an end to end p95 budget was missed, or
# hit latency grew during the run; see locustfile.py for the budgets.
source "$(dirname "$0")/common.sh"
require_venv
command -v uvx >/dev/null || die "uvx is needed to run Locust; install uv first"

USERS="${USERS:-50}"
DURATION="${DURATION:-60s}"
DELAY_MS="${DELAY_MS:-300}"

gateway_up && die "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"

cleanup() { "$ROOT/scripts/gateway-stop.sh" >/dev/null; "$ROOT/scripts/mock-upstream-stop.sh" >/dev/null; }
trap cleanup EXIT

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/chroma-start.sh"
"$ROOT/scripts/mock-upstream-start.sh"
curl -s -X POST "$MOCK_URL/reset" >/dev/null
curl -s -X PUT "$MOCK_URL/control/openai" -H 'Content-Type: application/json' \
  -d "{\"delay_ms\": $DELAY_MS}" >/dev/null

export SEMCACHE_PRIMARY_PROVIDER=openai OPENAI_API_KEY=load-test-key
export SEMCACHE_OPENAI_BASE_URL="$MOCK_URL/v1"
export SEMCACHE_SEMANTIC_CACHE_ENABLED=true SEMCACHE_REQUIRE_API_KEY=false
"$ROOT/scripts/gateway-start.sh"
reset_cache >/dev/null

step "$USERS users for $DURATION; the mock vendor answers after ${DELAY_MS} ms"
uvx --from locust==2.47.0 locust -f "$ROOT/locustfile.py" --headless \
  --users "$USERS" --spawn-rate "$USERS" --run-time "$DURATION" \
  --host "$GATEWAY_URL" --only-summary
