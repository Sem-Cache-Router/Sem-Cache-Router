#!/usr/bin/env bash
# Follow a service's log: scripts/logs.sh [gateway|chroma|redis], gateway by
# default. The gateway log carries one structured line per request.
source "$(dirname "$0")/common.sh"

name="${1:-gateway}"
case "$name" in
  gateway|chroma|redis) ;;
  *) die "usage: scripts/logs.sh [gateway|chroma|redis]" ;;
esac

if [ "$name" = redis ] && [ "$(cat "$RUN_DIR/redis.mode" 2>/dev/null)" = docker ]; then
  exec docker logs -f "$REDIS_CONTAINER"
fi
file="$RUN_DIR/$name.log"
[ -f "$file" ] || die "no $name log at $file; was $name started by these scripts?"
exec tail -n 50 -f "$file"
