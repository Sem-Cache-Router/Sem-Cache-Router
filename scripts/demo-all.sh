#!/usr/bin/env bash
# The whole walkthrough, start to finish: bring up the full stack, run every
# demo in order, and measure the false hit rate. Leaves the stack running;
# stop it afterwards with scripts/stop-all.sh.
#
# Set PAUSE=1 to wait for Enter between demos, for presenting.
source "$(dirname "$0")/common.sh"

pause() {
  if [ "${PAUSE:-0}" = "1" ]; then
    echo; read -r -p "press Enter for $1 " _
  fi
  echo; echo "${BOLD}############ $1${RESET}"
}

pause "the stack";            "$ROOT/scripts/run-tier2.sh"
pause "Tier 1";               "$ROOT/scripts/demo-tier1.sh"
pause "Tier 2";               "$ROOT/scripts/demo-tier2.sh"
pause "the negation finding"; "$ROOT/scripts/demo-negation.sh"
pause "degradation";          "$ROOT/scripts/demo-degradation.sh"
pause "the adversarial probe"; "$ROOT/scripts/adversarial.sh"

# Failover needs a gateway configured for it, so the Tier 2 one is stopped first.
# The failover demo starts its own and stops it again when it finishes.
"$ROOT/scripts/gateway-stop.sh"
pause "failover";             "$ROOT/scripts/demo-failover.sh"

echo
ok "walkthrough complete. Redis and Chroma are still running: scripts/status.sh, scripts/stop-all.sh"
