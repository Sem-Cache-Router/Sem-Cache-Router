#!/usr/bin/env bash
# Stop whichever Redis redis-start.sh started. A Redis that was already running
# before is left alone.
source "$(dirname "$0")/common.sh"

mode="$(cat "$RUN_DIR/redis.mode" 2>/dev/null || true)"
case "$mode" in
  docker) docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 && ok "removed container $REDIS_CONTAINER" ;;
  native|fake) stop_pid redis ;;
  *) warn "no Redis started by these scripts" ;;
esac
rm -f "$RUN_DIR/redis.mode"
