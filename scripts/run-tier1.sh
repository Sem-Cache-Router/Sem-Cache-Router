#!/usr/bin/env bash
# Start the Tier 1 stack: Redis, then the gateway with the semantic cache off.
# Needs nothing beyond scripts/setup.sh. Stop it with scripts/stop-all.sh.
source "$(dirname "$0")/common.sh"

"$ROOT/scripts/redis-start.sh"
SEMCACHE_SEMANTIC_CACHE_ENABLED=false "$ROOT/scripts/gateway-start.sh"

echo
ok "Tier 1 stack running. Try: scripts/demo-tier1.sh"
