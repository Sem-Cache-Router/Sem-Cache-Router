# Negation guard for semantic hits.
#
# Holds polarity(), whether a prompt is negated, and same_polarity(), whether a
# cached prompt and a new one agree on it.
#
# Why it exists: the embedding model places "Is the store open on Sunday?" and
# "Is the store not open on Sunday?" at a similarity of 0.97, above any useful
# threshold, so a semantic cache serves the opposite answer. Section 33 of the
# PRD measured it: 100 percent of negated questions were false hits at 0.90.
# Raising the threshold does not fix it, because the negated pair scores higher
# than most genuine paraphrases.
#
# What it does: counts negation words (not, never, no, ...) and contractions
# ending in n't. An odd count means the sentence is negated. A semantic hit is
# served only when the cached prompt and the new one have the same polarity.
#
# What it does not do: understand meaning. "Is it closed?" and "Is it not open?"
# agree in meaning but differ in polarity, so the guard turns that hit into a
# miss. That costs one provider call, never a wrong answer, which is the trade
# this guard is for. It is off by default and measured both ways by the
# benchmark, so the cost is a number rather than a guess.

from __future__ import annotations

import re

NEGATION_WORDS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "nowhere",
    "cannot", "without",
})

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def negation_count(text: str) -> int:
    """How many negations a text holds: negation words plus n't contractions."""
    words = _WORD.findall(text.lower().replace("’", "'"))
    return sum(1 for word in words if word in NEGATION_WORDS or word.endswith("n't"))


def polarity(text: str) -> bool:
    """True when the text is negated: an odd number of negations."""
    return negation_count(text) % 2 == 1


def same_polarity(cached: str, incoming: str) -> bool:
    """Whether a cached prompt may answer an incoming one, as far as negation goes."""
    return polarity(cached) == polarity(incoming)
