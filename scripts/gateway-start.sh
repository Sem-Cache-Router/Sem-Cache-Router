#!/usr/bin/env bash
# Start the gateway on localhost:8000 in the background.
#
# Configured through the environment, like the gateway itself. Examples:
#   scripts/gateway-start.sh                                   Tier 1 only
#   SEMCACHE_SEMANTIC_CACHE_ENABLED=true scripts/gateway-start.sh  Tier 2 on
# Most of the time, use run-tier1.sh or run-tier2.sh instead, which also start
# what the gateway depends on.
source "$(dirname "$0")/common.sh"
require_venv

if gateway_up; then
  warn "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"
  exit 0
fi
redis_ping || die "Redis is not running; start it with scripts/redis-start.sh"

# Point the gateway at the Redis and Chroma these scripts manage, unless the
# caller set them explicitly. Without this, a gateway started on another port
# would still read the default addresses from .env.
export SEMCACHE_REDIS_URL="${SEMCACHE_REDIS_URL:-redis://localhost:$REDIS_PORT/0}"
export SEMCACHE_CHROMA_HOST="${SEMCACHE_CHROMA_HOST:-http://localhost:$CHROMA_PORT}"

step "starting the gateway on port $GATEWAY_PORT"
[ "${SEMCACHE_SEMANTIC_CACHE_ENABLED:-false}" = "true" ] && \
  warn "Tier 2 on: loading the embedding model takes a few seconds (a 79 MB download the first time)"

nohup "$VENV/bin/uvicorn" app.main:app --host 127.0.0.1 --port "$GATEWAY_PORT" \
  >"$RUN_DIR/gateway.log" 2>&1 &
echo $! >"$RUN_DIR/gateway.pid"

wait_for 120 "the gateway" gateway_up
ok "gateway up at $GATEWAY_URL   (log: $RUN_DIR/gateway.log)"
curl -s "$GATEWAY_URL/health" | jq -c .
