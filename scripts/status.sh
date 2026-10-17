#!/usr/bin/env bash
# Show what is running: each service, whether these scripts started it, and
# the gateway's own view of its dependencies.
source "$(dirname "$0")/common.sh"

owner() {
  # Whether a pid file from these scripts is behind the service, or something else is.
  if pid_alive "$1"; then echo "started by scripts, pid $(cat "$RUN_DIR/$1.pid")"
  else echo "started elsewhere"
  fi
}

line() { printf '  %-8s %-6s %-6s %s\n' "$1" "$2" "$3" "$4"; }

echo "${BOLD}service  port   state  owner${RESET}"
if redis_ping; then
  mode="$(cat "$RUN_DIR/redis.mode" 2>/dev/null || true)"
  if [ "$mode" = "docker" ]; then line redis "$REDIS_PORT" up "docker container $REDIS_CONTAINER"
  else line redis "$REDIS_PORT" up "$(owner redis)${mode:+ ($mode)}"
  fi
else line redis "$REDIS_PORT" down ""
fi
if chroma_up; then line chroma "$CHROMA_PORT" up "$(owner chroma)"; else line chroma "$CHROMA_PORT" down ""; fi
if gateway_up; then line gateway "$GATEWAY_PORT" up "$(owner gateway)"; else line gateway "$GATEWAY_PORT" down ""; fi

if gateway_up; then
  echo
  echo "${BOLD}gateway /health${RESET}"
  curl -s "$GATEWAY_URL/health" | jq .
fi
echo
echo "logs and pid files: $RUN_DIR"
