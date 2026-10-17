#!/usr/bin/env bash
# Every gate CI runs: lint and the complexity ceiling, type checking, and the
# test suite with the coverage floor. Needs no Redis, Chroma or network.
source "$(dirname "$0")/common.sh"
require_venv

step "lint and cyclomatic complexity (ruff, ceiling 16)"
"$VENV/bin/ruff" check .
ok "clean"

step "type check (mypy)"
"$VENV/bin/mypy"

step "tests with coverage (floor 85 percent)"
"$VENV/bin/pytest" --cov --cov-fail-under=85 -q

echo
ok "all gates pass"
