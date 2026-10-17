#!/usr/bin/env bash
# List every script in this folder with what it does, read from each script's
# own opening comment so the list cannot drift from the scripts.
source "$(dirname "$0")/common.sh"

summary() {
  # The first sentence of the comment block after the shebang, which may run
  # over several lines.
  # It ends at a full stop, at a line ending in a colon (a list follows), or at
  # an empty comment line.
  awk 'NR == 1 { next }
       !/^#/ || /^#$/ { exit }
       { sub(/^# ?/, ""); text = text (text ? " " : "") $0 }
       match(text, /\.( |$)/) { text = substr(text, 1, RSTART); exit }
       /:$/ { sub(/:$/, "", text); exit }
       END { print text }' "$1"
}

group() {
  echo; echo "${BOLD}$1${RESET}"; shift
  for name in "$@"; do
    printf '  %-32s %s\n' "scripts/$name.sh" "$(summary "$ROOT/scripts/$name.sh")"
  done
}

group "set up and check"   setup check test test-model complexity
group "run the stack"      run-tier1 run-tier2 status logs reset-cache stop-all
group "single services"    redis-start redis-stop chroma-start chroma-stop gateway-start gateway-stop
group "demonstrations"     demo-tier1 demo-tier2 demo-negation demo-degradation adversarial demo-all
group "this list"          help

echo
echo "Ports default to gateway 8000, Chroma 8001, Redis 6379. Override with"
echo "GATEWAY_PORT, CHROMA_PORT and REDIS_PORT; logs and pid files go to"
echo "SEMCACHE_RUN_DIR (default /tmp/semcache-router)."
