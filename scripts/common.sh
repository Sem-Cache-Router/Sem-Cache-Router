#!/usr/bin/env bash
# Shared settings and helpers for every script in this folder.
#
# Sourced, not run. Every other script starts with:
#   source "$(dirname "$0")/common.sh"
#
# Ports and paths can be overridden from the environment, so two copies of the
# stack, or a port already in use, never need a script edited.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"

GATEWAY_PORT="${GATEWAY_PORT:-8000}"
CHROMA_PORT="${CHROMA_PORT:-8001}"
REDIS_PORT="${REDIS_PORT:-6379}"
GATEWAY_URL="http://localhost:$GATEWAY_PORT"

# Process ids, logs and Chroma's data live outside the repository, so running
# the stack never leaves files for git to pick up.
RUN_DIR="${SEMCACHE_RUN_DIR:-/tmp/semcache-router}"
mkdir -p "$RUN_DIR"

REDIS_CONTAINER="semcache-redis"

if [ -t 1 ]; then
  BOLD=$'\e[1m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; RESET=$'\e[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; RESET=""
fi

step() { echo "${BOLD}==> $*${RESET}"; }
ok()   { echo "${GREEN}ok${RESET}   $*"; }
warn() { echo "${YELLOW}warn${RESET} $*"; }
die()  { echo "${RED}error${RESET} $*" >&2; exit 1; }

require_venv() {
  [ -x "$PY" ] || die "no virtualenv at .venv; run scripts/setup.sh first"
}

port_open() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

redis_ping() {
  # A real PING over the socket, so it works whether or not redis-cli exists.
  # Sent in the protocol's array form: real Redis also accepts a bare "PING",
  # but fakeredis rejects it. The connection is opened inside the command
  # substitution because a failed redirect on exec would end the whole script.
  local reply
  reply=$( { exec 3<>"/dev/tcp/127.0.0.1/$REDIS_PORT" \
             && printf '*1\r\n$4\r\nPING\r\n' >&3 \
             && timeout 2 head -c 7 <&3; } 2>/dev/null ) || return 1
  [[ "$reply" == *PONG* ]]
}

wait_for() {
  # wait_for <seconds> <description> <command...>
  local seconds="$1" what="$2"; shift 2
  for _ in $(seq 1 "$((seconds * 2))"); do
    if "$@" >/dev/null 2>&1; then return 0; fi
    sleep 0.5
  done
  die "$what did not come up within ${seconds}s"
}

pid_alive() {
  local file="$RUN_DIR/$1.pid"
  [ -f "$file" ] && kill -0 "$(cat "$file")" 2>/dev/null
}

stop_pid() {
  # stop_pid <name>: stop a process this folder started in the background.
  local file="$RUN_DIR/$1.pid"
  if pid_alive "$1"; then
    kill "$(cat "$file")" 2>/dev/null || true
    for _ in $(seq 1 20); do pid_alive "$1" || break; sleep 0.25; done
    pid_alive "$1" && kill -9 "$(cat "$file")" 2>/dev/null || true
    ok "stopped $1"
  fi
  rm -f "$file"
}

gateway_up() { curl -sf "$GATEWAY_URL/health" >/dev/null; }

ask() {
  # ask <description> <json body> [curl args]: one completion request, printed
  # with its HTTP status. Extra arguments replace the default Authorization
  # header; pass -H 'Authorization:' to send none at all.
  local label="$1" body="$2"; shift 2
  local auth=(-H 'Authorization: Bearer demo-key')
  [ $# -gt 0 ] && auth=("$@")
  local out code
  echo
  echo "${BOLD}> $label${RESET}"
  if ! out=$(curl -s -w '\n%{http_code}' "$GATEWAY_URL/v1/chat/completions" "${auth[@]}" \
               -H 'Content-Type: application/json' -d "$body"); then
    echo "(no response; is the gateway running?)"
    return 0
  fi
  code="${out##*$'\n'}"
  out="${out%$'\n'*}"
  echo "HTTP $code"
  jq 'if .error then {error: .error.code, message: .error.message}
      else {semcache_status, semcache_similarity, answer: .choices[0].message.content} end' \
    <<<"$out" 2>/dev/null || echo "$out"
}

chroma_up() { curl -sf "http://localhost:$CHROMA_PORT/api/v2/heartbeat" >/dev/null; }

health_field() {
  # health_field <jq path>: one field of the gateway's /health, e.g. .dependencies.chroma
  curl -sf "$GATEWAY_URL/health" | jq -r "$1"
}

require_gateway() {
  gateway_up || die "no gateway on port $GATEWAY_PORT; start one with scripts/run-tier1.sh or scripts/run-tier2.sh"
}

require_tier2() {
  require_gateway
  [ "$(health_field .dependencies.chroma)" = "ok" ] \
    || die "Tier 2 is not on, or Chroma is down; start the stack with scripts/run-tier2.sh"
}

reset_cache() {
  # Empty both cache tiers so a demo starts from a miss. Deletes only the
  # gateway's own Redis keys and the entries of its Chroma collection, never
  # the collection itself, which a running gateway holds open.
  "$PY" - "$REDIS_PORT" "$CHROMA_PORT" "${SEMCACHE_CHROMA_COLLECTION:-semcache-v1}" <<'EOF'
import sys

import redis

redis_port, chroma_port, collection_name = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]

client = redis.Redis(port=redis_port)
keys = list(client.scan_iter("semcache:*"))
if keys:
    client.delete(*keys)
print(f"Tier 1: removed {len(keys)} Redis keys")

try:
    import chromadb
    from chromadb.config import Settings

    chroma = chromadb.HttpClient(
        host="localhost", port=chroma_port, settings=Settings(anonymized_telemetry=False)
    )
    collection = chroma.get_collection(collection_name)
    ids = collection.get(include=[])["ids"]
    if ids:
        collection.delete(ids=ids)
    print(f"Tier 2: removed {len(ids)} Chroma entries")
except Exception:
    print("Tier 2: nothing to clear (Chroma not running, or no collection yet)")
EOF
}

question() {
  # question <text>: the JSON body for a single user message.
  jq -cn --arg q "$1" '{model: "gpt-4o-mini", messages: [{role: "user", content: $q}]}'
}
