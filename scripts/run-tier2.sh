#!/usr/bin/env bash
# Start the full stack: Redis, ChromaDB, then the gateway with both cache tiers.
#
# The similarity threshold defaults to 0.85 here so the France paraphrase in
# demo-tier2.sh (measured at 0.903) hits clearly rather than by a hair at the
# gateway's own default of 0.90. Override it to see the trade off move:
#   SEMCACHE_SIMILARITY_THRESHOLD=0.95 scripts/run-tier2.sh
source "$(dirname "$0")/common.sh"

# A gateway already running with Tier 2 off would be left alone by
# gateway-start.sh, and every Tier 2 demo would then quietly miss.
if gateway_up && [ "$(health_field .dependencies.chroma)" = "disabled" ]; then
  die "a Tier 1 gateway is running on port $GATEWAY_PORT; stop it with scripts/gateway-stop.sh first"
fi

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/chroma-start.sh"
SEMCACHE_SEMANTIC_CACHE_ENABLED=true \
SEMCACHE_SIMILARITY_THRESHOLD="${SEMCACHE_SIMILARITY_THRESHOLD:-0.85}" \
  "$ROOT/scripts/gateway-start.sh"

echo
ok "full stack running. Try: scripts/demo-tier2.sh"
