#!/usr/bin/env bash
# Judge cached answers against fresh ones with RAGAS, using a local model through Ollama.
#
#   scripts/evaluate-quality.sh <folder written by: PROVIDER=ollama scripts/run-benchmark.sh>
#
# For every threshold run in the folder, each distinct semantic hit is judged
# against the baseline's fresh answer to the same question, beside controls made
# of two fresh answers to one question. See evaluation/ragas_pipeline.py.
#
# RAGAS runs in an environment of its own through uv, so it never becomes a
# dependency of the gateway. Its package cache goes to $SEMCACHE_RUN_DIR unless
# UV_CACHE_DIR is set, since it is several hundred megabytes. Settings:
# OLLAMA_URL (http://localhost:11434), JUDGE_MODEL (qwen2.5:7b), CONTROLS (20).
source "$(dirname "$0")/common.sh"

FOLDER="${1:-}"
[ -n "$FOLDER" ] && [ -f "$FOLDER/direct.json" ] \
  || die "give the folder of a benchmark sweep, the one holding direct.json"
command -v uv >/dev/null || die "uv is needed to run RAGAS in its own environment"

OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
JUDGE_MODEL="${JUDGE_MODEL:-qwen2.5:7b}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$RUN_DIR/uv-cache}"

[ "$(jq -r .simulated "$FOLDER/direct.json")" = false ] \
  || die "that sweep ran against the mock vendor, whose answers only repeat the question; run one with PROVIDER=ollama"
curl -sf "$OLLAMA_URL/api/tags" | jq -e --arg m "$JUDGE_MODEL" 'any(.models[]; .name == $m)' >/dev/null \
  || die "Ollama is not answering at $OLLAMA_URL, or $JUDGE_MODEL is not downloaded"

for run in "$FOLDER"/t*.json; do
  [ -e "$run" ] || die "no threshold runs in $FOLDER"
  out="${run%.json}.quality.json"
  [ "${run%.quality.json}" = "$run" ] || continue
  step "judging $(basename "$run")"
  uv run --quiet --no-project --python 3.14 --with ragas==0.4.4 --with openai \
    python -m evaluation.ragas_pipeline "$run" "$FOLDER/direct.json" --out "$out" \
      --base-url "$OLLAMA_URL/v1" --model "$JUDGE_MODEL" --controls "${CONTROLS:-20}"
done
