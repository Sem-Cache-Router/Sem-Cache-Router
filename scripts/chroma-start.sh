#!/usr/bin/env bash
# Start a ChromaDB server on localhost:8001 in the background, storing its data
# under the run directory. Port 8001 because the gateway owns 8000.
source "$(dirname "$0")/common.sh"
require_venv

if chroma_up; then ok "Chroma already answering on port $CHROMA_PORT"; exit 0; fi
port_open "$CHROMA_PORT" && die "port $CHROMA_PORT is taken by something that is not Chroma"

step "starting ChromaDB on port $CHROMA_PORT"
mkdir -p "$RUN_DIR/chroma"
nohup "$VENV/bin/chroma" run --path "$RUN_DIR/chroma" --port "$CHROMA_PORT" \
  >"$RUN_DIR/chroma.log" 2>&1 &
echo $! >"$RUN_DIR/chroma.pid"

wait_for 30 "Chroma" chroma_up
ok "Chroma answering on port $CHROMA_PORT (data in $RUN_DIR/chroma)"
