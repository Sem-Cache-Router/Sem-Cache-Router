#!/usr/bin/env bash
# Stop the Chroma server chroma-start.sh started. Add --wipe to delete its data.
source "$(dirname "$0")/common.sh"
stop_pid chroma
if [ "${1:-}" = "--wipe" ]; then
  rm -rf "$RUN_DIR/chroma"
  ok "deleted Chroma data"
fi
