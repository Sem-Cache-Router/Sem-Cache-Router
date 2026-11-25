#!/usr/bin/env bash
# Run the benchmark sweep: a no cache baseline, the gateway at each threshold, then the report.
#
#   scripts/run-benchmark.sh                  against the mock vendor: fast, latency and cost simulated
#   PROVIDER=ollama scripts/run-benchmark.sh  against a model running locally in Ollama: real answers
#                                             and real latency, at no cost and with no key
#
# Settings, all optional:
#   THRESHOLDS    "0.75 0.90 0.95"
#   REPEATS       1, exact repeats of the workload after the first pass
#   OUT           where the runs and the report go; default $SEMCACHE_RUN_DIR/benchmark/<time>
#   DELAY_MS      800, how long the mock takes to answer (mock only)
#   OLLAMA_URL    http://localhost:11434, OLLAMA_MODEL qwen2.5:7b, MAX_TOKENS 128 (ollama only)
# Any other gateway setting can be passed through the environment, for example
# SEMCACHE_SEMANTIC_NEGATION_GUARD=true to measure the guard.
#
# The baseline sends every prompt straight to the provider, which is what an app
# without the gateway would pay. Each threshold gets a fresh gateway and an empty
# cache, so no run measures the one before it.
source "$(dirname "$0")/common.sh"
require_venv

PROVIDER="${PROVIDER:-mock}"
THRESHOLDS="${THRESHOLDS:-0.75 0.90 0.95}"
REPEATS="${REPEATS:-1}"
OUT="${OUT:-$RUN_DIR/benchmark/$(date +%Y%m%d-%H%M%S)}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:7b}"

gateway_up && die "a gateway is already running on port $GATEWAY_PORT; stop it first with scripts/gateway-stop.sh"

cleanup() { "$ROOT/scripts/gateway-stop.sh" >/dev/null; [ "$PROVIDER" = mock ] && "$ROOT/scripts/mock-upstream-stop.sh" >/dev/null; true; }
trap cleanup EXIT

"$ROOT/scripts/redis-start.sh"
"$ROOT/scripts/chroma-start.sh"

case "$PROVIDER" in
  mock)
    "$ROOT/scripts/mock-upstream-start.sh"
    curl -s -X POST "$MOCK_URL/reset" >/dev/null
    curl -s -X PUT "$MOCK_URL/control/openai" -H 'Content-Type: application/json' \
      -d "{\"delay_ms\": ${DELAY_MS:-800}}" >/dev/null
    VENDOR="$MOCK_URL/v1"
    HARNESS=(--provider mock)
    ;;
  ollama)
    curl -sf "$OLLAMA_URL/api/tags" >/dev/null \
      || die "Ollama is not answering at $OLLAMA_URL; start it with: sudo systemctl start ollama"
    curl -sf "$OLLAMA_URL/api/tags" | jq -e --arg m "$OLLAMA_MODEL" 'any(.models[]; .name == $m)' >/dev/null \
      || die "model $OLLAMA_MODEL is not downloaded; run: ollama pull $OLLAMA_MODEL"
    VENDOR="$OLLAMA_URL/v1"
    # A local model on a CPU can take tens of seconds for a long answer.
    export SEMCACHE_PROVIDER_TIMEOUT_SECONDS="${SEMCACHE_PROVIDER_TIMEOUT_SECONDS:-300}"
    HARNESS=(--provider "ollama $OLLAMA_MODEL" --model "$OLLAMA_MODEL"
             --max-tokens "${MAX_TOKENS:-128}" --timeout 330)
    ;;
  *)
    die "PROVIDER must be mock or ollama, not '$PROVIDER'"
    ;;
esac

# The gateway's OpenAI adapter speaks to whichever vendor is chosen; Ollama
# ignores the key, and the mock accepts any.
export SEMCACHE_PRIMARY_PROVIDER=openai OPENAI_API_KEY=benchmark
export SEMCACHE_OPENAI_BASE_URL="$VENDOR"
export SEMCACHE_SEMANTIC_CACHE_ENABLED=true SEMCACHE_REQUIRE_API_KEY=false
mkdir -p "$OUT"

step "baseline: every prompt straight to the provider ($PROVIDER)"
"$PY" -m benchmarks.harness --label direct --direct "$VENDOR" --repeats "$REPEATS" \
  --out "$OUT/direct.json" "${HARNESS[@]}"

for threshold in $THRESHOLDS; do
  step "gateway at threshold $threshold"
  "$ROOT/scripts/gateway-stop.sh" >/dev/null
  SEMCACHE_SIMILARITY_THRESHOLD="$threshold" "$ROOT/scripts/gateway-start.sh" >/dev/null
  reset_cache >/dev/null
  "$PY" -m benchmarks.harness --label "t$threshold" --threshold "$threshold" --repeats "$REPEATS" \
    --out "$OUT/t$threshold.json" "${HARNESS[@]}"
done

step "report"
"$PY" -m benchmarks.report "$OUT"/direct.json "$OUT"/t*.json --out "$OUT/report.md"
echo
sed -n '/^| run/,/^$/p' "$OUT/report.md"
echo "Full report: $OUT/report.md"
