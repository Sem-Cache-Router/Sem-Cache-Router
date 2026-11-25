#!/usr/bin/env bash
# Demonstrate a hung Chroma: frozen rather than stopped, so it accepts the
# connection and never answers.
#
# A stopped Chroma refuses the connection and the client fails at once. A frozen
# one raises nothing, and before the timeout every request, and the health check
# too, waited on it indefinitely. This freezes Chroma with SIGSTOP and times each
# request: the first waits out the timeout, the ones after it skip Tier 2
# entirely for the cooldown, and once Chroma is thawed the gateway reconnects on
# its own and the reworded question is a semantic hit again.
#
# Starts its own Redis, Chroma and gateway, with a 1 second timeout and a 5 second
# cooldown so the story fits in a few seconds, and stops what it started. Chroma
# is always thawed on exit, even if the demo fails halfway.
source "$(dirname "$0")/common.sh"
require_venv

gateway_up && die "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"

thaw() { pid_alive chroma && kill -CONT "$(cat "$RUN_DIR/chroma.pid")" 2>/dev/null || true; }
cleanup() { thaw; "$ROOT/scripts/gateway-stop.sh" >/dev/null; }
trap cleanup EXIT

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/chroma-start.sh"
pid_alive chroma || die "Chroma on port $CHROMA_PORT was not started by these scripts, so this demo will not freeze it"

export SEMCACHE_SEMANTIC_CACHE_ENABLED=true
export SEMCACHE_CHROMA_TIMEOUT_SECONDS=1 SEMCACHE_CHROMA_RECONNECT_SECONDS=5
"$ROOT/scripts/gateway-start.sh"
reset_cache >/dev/null

timed() {
  # timed <label> <question>: one request, with its status and how long it took.
  local out
  out=$(curl -s -w '\n%{time_total}' "$GATEWAY_URL/v1/chat/completions" \
    -H 'Authorization: Bearer demo-key' -H 'Content-Type: application/json' -d "$(question "$2")")
  printf '  %-44s %6.0f ms  %s\n' "$1" "$(awk -v t="${out##*$'\n'}" 'BEGIN{print t*1000}')" \
    "$(jq -r '.semcache_status // .error.code' <<<"${out%$'\n'*}")"
}
timed_health() {
  local out
  out=$(curl -s -w '\n%{time_total}' "$GATEWAY_URL/health")
  printf '  %-44s %6.0f ms  chroma %s\n' "health" "$(awk -v t="${out##*$'\n'}" 'BEGIN{print t*1000}')" \
    "$(jq -r '.dependencies.chroma' <<<"${out%$'\n'*}")"
}

step "1. Chroma answering"
timed "store the question" "What is the capital of France?"
timed "a rewording, from Tier 2" "Which city is the capital of France?"

step "2. Chroma frozen (SIGSTOP): it accepts connections and never answers"
kill -STOP "$(cat "$RUN_DIR/chroma.pid")"
timed "a new question: waits out the 1 s timeout" "How do I reset my password?"
timed "another: skips Tier 2 during the cooldown" "Why is the sky blue?"
timed "the stored question, from Tier 1" "What is the capital of France?"
timed_health
curl -s "$GATEWAY_URL/metrics" | jq -r '"  tier2_unavailable: \(.tier2_unavailable)"'

step "3. Chroma thawed (SIGCONT); waiting out the 5 s cooldown"
thaw
sleep 6
timed "the rewording again, from Tier 2" "Which city is the capital of France?"
timed_health
