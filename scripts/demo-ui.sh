#!/usr/bin/env bash
# Open the showcase page with everything behind it running, ready to click through.
#
# Starts Redis, Chroma, the mock vendor API (answering after 800 ms, like a real
# provider), and a gateway with both cache tiers on, the mock as primary and
# secondary provider so the Failover buttons work, and a fresh admin key for the
# "Empty the cache" button, printed here. Opens http://localhost:8000/ in the
# browser and keeps running until Enter is pressed, then stops what it started.
#
# Needs no keys and costs nothing: every answer comes from the mock.
source "$(dirname "$0")/common.sh"
require_venv

gateway_up && die "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"

cleanup() { "$ROOT/scripts/gateway-stop.sh" >/dev/null; "$ROOT/scripts/mock-upstream-stop.sh" >/dev/null; }
trap cleanup EXIT

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/chroma-start.sh"
"$ROOT/scripts/mock-upstream-start.sh"
curl -s -X POST "$MOCK_URL/reset" >/dev/null
curl -s -X PUT "$MOCK_URL/control/openai" -H 'Content-Type: application/json' \
  -d '{"delay_ms": 800}' >/dev/null

ADMIN_KEY="$("$PY" -c 'import secrets; print(secrets.token_hex(12))')"

# The same failover settings as demo-failover.sh: a low threshold and a short
# backoff, so a broken primary is noticed in a few requests and retried soon.
export SEMCACHE_PRIMARY_PROVIDER=openai SEMCACHE_SECONDARY_PROVIDER=anthropic
export OPENAI_API_KEY=demo-key ANTHROPIC_API_KEY=demo-key
export SEMCACHE_OPENAI_BASE_URL="$MOCK_URL/v1" SEMCACHE_ANTHROPIC_BASE_URL="$MOCK_URL/v1"
export SEMCACHE_BREAKER_FAILURE_THRESHOLD=3 SEMCACHE_BREAKER_BASE_BACKOFF=1 SEMCACHE_BREAKER_BACKOFF_CEILING=3
export SEMCACHE_SEMANTIC_CACHE_ENABLED=true SEMCACHE_SIMILARITY_THRESHOLD=0.85
export SEMCACHE_ADMIN_API_KEY="$ADMIN_KEY"
"$ROOT/scripts/gateway-start.sh"
reset_cache >/dev/null

echo
ok "page:      $GATEWAY_URL/"
ok "admin key: $ADMIN_KEY   (paste it into the Admin panel to empty the cache)"
if command -v xdg-open >/dev/null; then
  xdg-open "$GATEWAY_URL/" >/dev/null 2>&1 || warn "could not open a browser; open the page above by hand"
fi
echo
read -r -p "Press Enter to stop everything. " _
