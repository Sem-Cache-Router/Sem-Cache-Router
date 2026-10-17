#!/usr/bin/env bash
# Run the tests that load the real all-MiniLM-L6-v2 weights. Skipped by default
# everywhere else, because the first load downloads 79 MB.
source "$(dirname "$0")/common.sh"
require_venv
step "real embedding model tests"
SEMCACHE_RUN_MODEL_TESTS=1 exec "$VENV/bin/pytest" -m model "$@"
