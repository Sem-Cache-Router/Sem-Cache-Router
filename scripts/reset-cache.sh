#!/usr/bin/env bash
# Empty both cache tiers without restarting anything, so the next request is a
# miss. Removes only the gateway's own Redis keys and its Chroma entries.
source "$(dirname "$0")/common.sh"
require_venv

redis_ping || die "Redis is not running on port $REDIS_PORT"
step "clearing the cache"
reset_cache
