#!/usr/bin/env bash
# Demonstrate graceful degradation: Chroma is stopped mid run, the gateway
# reports degraded but keeps answering from Tier 1 and the provider, then
# reconnects on its own once Chroma is back, without a restart.
#
# Needs the full stack from scripts/run-tier2.sh, with Chroma started by these
# scripts, since this demo stops and restarts it. Clears the cache first; set
# KEEP_CACHE=1 to skip that.
source "$(dirname "$0")/common.sh"
require_venv
require_tier2
pid_alive chroma || die "Chroma on port $CHROMA_PORT was not started by these scripts, so this demo will not stop it"

if [ "${KEEP_CACHE:-0}" != "1" ]; then step "clearing the cache"; reset_cache; fi

step "before the outage"
ask "1. stored in both tiers (expect miss)" "$(question 'What is the capital of France?')"

step "stopping Chroma"
"$ROOT/scripts/chroma-stop.sh"
echo "health: $(curl -s "$GATEWAY_URL/health" | jq -c .)"
echo "        (expect status degraded, chroma unreachable, redis still ok)"

ask "2. a new question with Chroma down: Tier 2 is skipped, not fatal (expect HTTP 200, miss)" \
  "$(question 'How do I bake sourdough bread?')"
ask "3. the first question again: Tier 1 still serves it (expect HTTP 200, exact_hit)" \
  "$(question 'What is the capital of France?')"

step "restarting Chroma"
"$ROOT/scripts/chroma-start.sh"
cooldown="${SEMCACHE_CHROMA_RECONNECT_SECONDS:-30}"
echo "the gateway retries only after its reconnect cooldown (${cooldown}s), waiting for it..."
chroma_back() { [ "$(health_field .dependencies.chroma)" = "ok" ]; }
wait_for "$(( ${cooldown%.*} + 30 ))" "the gateway's reconnection to Chroma" chroma_back
echo "health: $(curl -s "$GATEWAY_URL/health" | jq -c .)"

ask "4. a paraphrase of the first question: Tier 2 is back, data intact (expect semantic_hit)" \
  "$(question "Tell me France's capital city.")"
