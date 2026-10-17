#!/usr/bin/env bash
# Demonstrate the central finding: a semantic cache serves the answer to a
# question whose meaning is the opposite of the one asked.
#
# Needs the full stack (scripts/run-tier2.sh). Clears the cache first; set
# KEEP_CACHE=1 to skip that.
source "$(dirname "$0")/common.sh"
require_venv
require_tier2

if [ "${KEEP_CACHE:-0}" != "1" ]; then step "clearing the cache"; reset_cache; fi

step "a negated question"
ask "1. the original question (expect miss)" \
  "$(question 'Is the store open on Sunday?')"
ask "2. its negation: the opposite question, measured at 0.972 (expect a wrong semantic_hit)" \
  "$(question 'Is the store not open on Sunday?')"

echo
echo "Request 2 is answered with request 1's answer. all-MiniLM-L6-v2 places the"
echo "two within 0.03 of identical, above every threshold the benchmark sweeps"
echo "(0.75, 0.90 and 0.95). A threshold high enough to stop it would also stop"
echo "the genuine paraphrase in scripts/demo-tier2.sh, which scores 0.90. This is"
echo "the false hit rate scripts/adversarial.sh measures across a whole suite."
