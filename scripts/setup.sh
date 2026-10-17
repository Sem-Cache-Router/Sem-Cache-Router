#!/usr/bin/env bash
# One time setup: virtualenv on Python 3.14, pinned dependencies, a local .env,
# and the pre-commit hooks. Safe to run again; it only fills in what is missing.
source "$(dirname "$0")/common.sh"

step "checking for uv"
command -v uv >/dev/null || die "uv is not installed; run: curl -LsSf https://astral.sh/uv/install.sh | sh"
ok "uv $(uv --version | awk '{print $2}')"

step "virtualenv on Python 3.14"
if [ -x "$PY" ] && "$PY" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 14))'; then
  ok "existing .venv is 3.14"
else
  uv python install 3.14
  uv venv -p 3.14 "$VENV"
  ok "created .venv"
fi

step "installing pinned dependencies from requirements.lock.txt"
uv pip sync --python "$PY" requirements.lock.txt
ok "dependencies match the lockfile"

step "local environment file"
if [ -f .env ]; then
  ok ".env already exists, left untouched"
else
  cp .env.example .env
  ok "created .env from .env.example (stub provider, Tier 2 off, no key needed)"
fi

step "pre-commit hooks"
"$VENV/bin/pre-commit" install >/dev/null
ok "hooks installed in .git/hooks"

echo
ok "setup complete. Next: scripts/check.sh, then scripts/run-tier1.sh"
