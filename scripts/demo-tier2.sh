#!/usr/bin/env bash
# Demonstrate Tier 2, the semantic cache: a reworded question answered from
# Chroma with its similarity score, and paraphrases the model cannot match.
#
# Needs the full stack (scripts/run-tier2.sh). Clears the cache first; set
# KEEP_CACHE=1 to skip that.
source "$(dirname "$0")/common.sh"
require_venv
require_tier2

if [ "${KEEP_CACHE:-0}" != "1" ]; then step "clearing the cache"; reset_cache; fi

step "a paraphrase the semantic tier catches"
ask "1. first time: goes to the provider, stored in both tiers (expect miss)" \
  "$(question 'What is the capital of France?')"
ask "2. reworded, so Tier 1 misses; Chroma matches it (expect semantic_hit, about 0.90)" \
  "$(question "Tell me France's capital city.")"

step "where the semantic tier stops"
ask "3. an unrelated question (expect miss)" \
  "$(question 'How do I bake sourdough bread?')"
ask "4. stored first, for the next request to be compared against (expect miss)" \
  "$(question 'What is your refund policy?')"
ask "5. same meaning, different words: measured at 0.551, below any usable threshold (expect miss)" \
  "$(question 'How do I get my money back?')"

echo
echo "Request 5 is the PRD's own paraphrase example, and the model scores it far"
echo "below request 2. A threshold low enough to catch it would also catch the"
echo "opposite meanings in scripts/demo-negation.sh."
