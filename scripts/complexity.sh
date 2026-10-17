#!/usr/bin/env bash
# Cyclomatic complexity of every function in app/, highest first. The ceiling
# enforced in CI is 16; this shows how much headroom each function has.
source "$(dirname "$0")/common.sh"
require_venv
step "cyclomatic complexity by function (ceiling 16)"
"$VENV/bin/ruff" check app --select C901 --config 'lint.mccabe.max-complexity=1' --output-format concise 2>/dev/null \
  | grep -oE '[^ ]+\.py:[0-9]+:[0-9]+: C901 `[^`]+` is too complex \([0-9]+' \
  | sed -E 's/^([^:]+):([0-9]+):[0-9]+: C901 `([^`]+)` is too complex \(([0-9]+)/\4\t\3\t\1:\2/' \
  | sort -rn \
  | awk -F'\t' '{ printf "  %3s  %-28s %s\n", $1, $2, $3 }'
