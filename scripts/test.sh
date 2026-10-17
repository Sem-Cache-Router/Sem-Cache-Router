#!/usr/bin/env bash
# Run the test suite. Any arguments go straight to pytest, for example:
#   scripts/test.sh tests/test_semantic_cache.py
#   scripts/test.sh -k negation
#   scripts/test.sh -m "not slow"
source "$(dirname "$0")/common.sh"
require_venv
exec "$VENV/bin/pytest" "$@"
