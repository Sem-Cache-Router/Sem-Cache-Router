#!/usr/bin/env bash
# Start Redis on localhost:6379, using the first option that is available:
#   1. a Redis that is already running        (left alone)
#   2. Docker, if the daemon is running       (container "semcache-redis")
#   3. a native redis-server                  (sudo pacman -S redis)
#   4. an in-memory fake Redis from the venv  (no install needed; a dev fallback)
# Force one with REDIS_MODE=docker, native or fake.
source "$(dirname "$0")/common.sh"
require_venv

if redis_ping; then ok "Redis already answering on port $REDIS_PORT"; exit 0; fi
port_open "$REDIS_PORT" && die "port $REDIS_PORT is taken by something that is not Redis"

mode="${REDIS_MODE:-}"
if [ -z "$mode" ]; then
  if docker info >/dev/null 2>&1; then mode=docker
  elif command -v redis-server >/dev/null; then mode=native
  else mode=fake
  fi
fi

case "$mode" in
  docker)
    step "starting Redis in Docker"
    docker rm -f "$REDIS_CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$REDIS_CONTAINER" -p "$REDIS_PORT:6379" redis:7-alpine \
      redis-server --appendonly yes >/dev/null
    ;;
  native)
    step "starting native redis-server"
    redis-server --port "$REDIS_PORT" --daemonize yes \
      --pidfile "$RUN_DIR/redis.pid" --logfile "$RUN_DIR/redis.log" --dir "$RUN_DIR"
    ;;
  fake)
    step "starting in-memory fake Redis (dev fallback)"
    warn "not real Redis: nothing persists, and Lua scripts are unsupported."
    warn "fine for the cache demos; for real Redis start Docker or install redis."
    nohup "$PY" -c "
from fakeredis import TcpFakeServer
TcpFakeServer(('127.0.0.1', $REDIS_PORT), server_type='redis').serve_forever()
" >"$RUN_DIR/redis.log" 2>&1 &
    echo $! >"$RUN_DIR/redis.pid"
    ;;
  *) die "REDIS_MODE must be docker, native or fake" ;;
esac

wait_for 15 "Redis" redis_ping
echo "$mode" >"$RUN_DIR/redis.mode"
ok "Redis ($mode) answering on port $REDIS_PORT"
