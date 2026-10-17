#!/usr/bin/env bash
# Demonstrate Tier 1, the exact cache: a miss, an exact hit, whitespace
# normalisation, model isolation, the error contract, and the metrics.
#
# Works against either stack. Clears the cache first so the first request is a
# miss; set KEEP_CACHE=1 to skip that.
source "$(dirname "$0")/common.sh"
require_venv
require_gateway

step "gateway health"
curl -s "$GATEWAY_URL/health" | jq -c .

if [ "${KEEP_CACHE:-0}" != "1" ]; then step "clearing the cache"; reset_cache; fi

step "the exact cache"
ask "1. first time: goes to the provider (expect miss)" \
  "$(question 'What is the capital of France?')"
ask "2. same request again: served from Redis (expect exact_hit)" \
  "$(question 'What is the capital of France?')"
ask "3. extra spaces only: normalised to the same key (expect exact_hit)" \
  "$(question '   What   is the capital of    France?  ')"
ask "4. same prompt, different model: the model is part of the key (expect miss)" \
  "$(jq -cn '{model: "gpt-4o", messages: [{role: "user", content: "What is the capital of France?"}]}')"

step "the error contract"
ask "5. no API key (expect 401 MISSING_API_KEY)" \
  "$(question 'What is the capital of France?')" -H 'Authorization:'
ask "6. empty messages list (expect 400 INVALID_REQUEST, not FastAPI's 422)" \
  '{"model": "gpt-4o-mini", "messages": []}'

step "metrics after the run"
curl -s "$GATEWAY_URL/metrics" | jq .
