#!/usr/bin/env bash
# Show that both cache tiers survive a full restart of the container stack.
#
# Asks a question twice, takes every container down and brings the stack back,
# then asks the same question and a reworded one. Without the named volumes the
# second half would be two misses: a restart would empty the cache, and a
# benchmark started after it would measure the restart instead of the cache.
source "$(dirname "$0")/common.sh"

compose ps --status running --services 2>/dev/null | grep -qx gateway \
  || die "the container stack is not running; start it with scripts/compose-up.sh"
require_tier2

SAME="What is the capital of France?"
REWORDED="Which city is the capital of France?"

step "before the restart"
ask "first time (a miss, unless an earlier run already cached it)" "$(question "$SAME")"
ask "the same question again: exact_hit" "$(question "$SAME")"

step "taking every container down, volumes kept"
compose down
step "bringing the stack back"
compose up -d
wait_for 180 "the gateway" gateway_up
wait_for 60 "Chroma behind the gateway" test "$(health_field .dependencies.chroma)" = ok

step "after the restart"
ask "the same question: still an exact_hit, from Redis's volume" "$(question "$SAME")"
ask "a rewording: a semantic_hit, from Chroma's volume" "$(question "$REWORDED")"
