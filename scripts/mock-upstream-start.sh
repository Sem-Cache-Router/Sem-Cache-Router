#!/usr/bin/env bash
# Start the mock vendor API on localhost:8002 in the background.
#
# A stand in for OpenAI and Anthropic that the real adapters talk to unmodified,
# so a provider can be told to fail on command and a benchmark costs nothing. See
# benchmarks/mock_upstream.py. Control it with:
#   curl -X PUT localhost:8002/control/openai -H 'Content-Type: application/json' -d '{"status": 503}'
#   curl localhost:8002/stats
source "$(dirname "$0")/common.sh"
require_venv

if mock_up; then ok "mock upstream already answering on port $MOCK_PORT"; exit 0; fi
port_open "$MOCK_PORT" && die "port $MOCK_PORT is taken by something that is not the mock upstream"

step "starting the mock upstream on port $MOCK_PORT"
nohup "$VENV/bin/uvicorn" benchmarks.mock_upstream:app --host 127.0.0.1 --port "$MOCK_PORT" \
  >"$RUN_DIR/mock.log" 2>&1 &
echo $! >"$RUN_DIR/mock.pid"

wait_for 15 "the mock upstream" mock_up
ok "mock upstream answering on port $MOCK_PORT"
