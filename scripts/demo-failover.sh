#!/usr/bin/env bash
# Demonstrate failover: the primary provider is failed during a live run, the
# breaker opens, traffic moves to the secondary with no error reaching the caller,
# and the primary takes traffic back once it recovers. Acceptance demonstration 3.
#
# Runs the real OpenAI and Anthropic adapters against a mock vendor API, so it
# needs no keys and costs nothing. Starts its own gateway, configured for it, and
# stops that gateway and the mock when it finishes; it refuses to run if a gateway
# is already up, since that one would not be configured for this.
source "$(dirname "$0")/common.sh"
require_venv

gateway_up && die "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"

cleanup() { "$ROOT/scripts/gateway-stop.sh" >/dev/null; "$ROOT/scripts/mock-upstream-stop.sh" >/dev/null; }
trap cleanup EXIT

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/mock-upstream-start.sh"
curl -s -X POST "$MOCK_URL/reset" >/dev/null

# A low threshold and a short backoff, so the whole story fits in a few seconds.
# The keys are placeholders: the mock accepts anything.
export SEMCACHE_PRIMARY_PROVIDER=openai
export SEMCACHE_SECONDARY_PROVIDER=anthropic
export OPENAI_API_KEY=demo-key ANTHROPIC_API_KEY=demo-key
export SEMCACHE_OPENAI_BASE_URL="$MOCK_URL/v1" SEMCACHE_ANTHROPIC_BASE_URL="$MOCK_URL/v1"
export SEMCACHE_BREAKER_FAILURE_THRESHOLD=3 SEMCACHE_BREAKER_BASE_BACKOFF=1 SEMCACHE_BREAKER_BACKOFF_CEILING=3
"$ROOT/scripts/gateway-start.sh"
reset_cache >/dev/null

failures=0
send() {
  # send <question>: one request; counts any status that is not 200 as a caller visible error.
  local code
  code=$(curl -s -o /dev/null -w '%{http_code}' "$GATEWAY_URL/v1/chat/completions" \
    -H 'Authorization: Bearer demo-key' -H 'Content-Type: application/json' -d "$(question "$1")")
  [ "$code" = 200 ] || failures=$((failures + 1))
  echo "  HTTP $code  $1"
}
calls() {
  # Who has been called so far, from the mock's own count.
  curl -s "$MOCK_URL/stats" | jq -r '"  upstream calls: openai \(.openai.calls), anthropic \(.anthropic.calls)"'
}
vendors() {
  curl -s "$GATEWAY_URL/metrics" | jq -r '"  breaker_failovers: \(.breaker_failovers)"'
}

step "1. both providers healthy: the primary answers"
send "What is the capital of France?"
send "How do I bake sourdough bread?"
calls

step "2. the primary is failed (every call returns 503)"
curl -s -X PUT "$MOCK_URL/control/openai" -H 'Content-Type: application/json' -d '{"status": 503}' >/dev/null
echo "  new questions, so nothing is served from the cache:"
send "Why is the sky blue?"
send "What is the speed of light?"
send "Who wrote Hamlet?"
calls
vendors
echo "  Every request above returned 200. After 3 failures the breaker opened."

step "3. the breaker is open: the failing primary is no longer called at all"
before=$(curl -s "$MOCK_URL/stats" | jq .openai.calls)
send "What is the boiling point of water?"
send "How many continents are there?"
after=$(curl -s "$MOCK_URL/stats" | jq .openai.calls)
echo "  openai calls before: $before, after: $after (unchanged: the breaker is rejecting without calling)"
calls

step "4. the primary recovers; after the backoff a probe succeeds and traffic returns"
curl -s -X PUT "$MOCK_URL/control/openai" -H 'Content-Type: application/json' -d '{"status": 200}' >/dev/null
echo "  waiting for the breaker backoff..."
sleep 4
send "What is the largest planet?"
send "What is the capital of Japan?"
calls

echo
if [ "$failures" -eq 0 ]; then
  ok "zero caller visible errors across the whole run"
else
  die "$failures request(s) returned an error to the caller"
fi
