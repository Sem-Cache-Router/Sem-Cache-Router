# Adversarial false hit generators.
#
# Holds:
#   a case record: the original prompt, the mutation, and the expectation
#   a negation generator
#   an entity swap generator
#   a numeric change generator
#   a suite builder across all three mutation kinds
#   a false hit rate calculator
#
# These target the cases where two prompts sit close together in embedding space
# while meaning different things, which is the failure mode a semantic cache is
# built to commit. They are expected to be the hardest tests to pass.
#
# The false hit rate is reported as a headline metric rather than buried, since
# it is the honest cost of the cost saving.
#
# The generators are deliberately simple and deterministic: a fixed list of
# auxiliary verbs for negation, a fixed swap table for entities, and one numeric
# rewrite. They produce near duplicates whose meaning has changed, which is all
# the probe needs, and their output is reproducible run to run, which a model
# based paraphraser would not be.
#
# This is a measurement, not a gate. Against the real model these cases DO hit:
# measured on all-MiniLM-L6-v2, "Is the store open on Sunday?" against its
# negation scores 0.972, above every threshold the benchmark sweeps. Asserting
# that they must miss would fail CI for a property of the model, not a bug in
# the code. The rate is reported instead, which is the honest form of the claim.
#
# Run against the real model with:  python -m evaluation.adversarial

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

THRESHOLDS = (0.75, 0.90, 0.95)

# Negation forms for the auxiliary verbs. A statement takes "not" after the
# verb ("Paris is not lovely"). A question has the verb before its subject, so
# "not" there produces text no person writes ("Is not the store open"); the
# contracted form goes in the verb's place instead ("Isn't the store open").
# Probing with ungrammatical text would measure the model on input it never
# sees in real traffic.
CONTRACTIONS = {
    "is": "isn't", "are": "aren't", "was": "wasn't", "were": "weren't",
    "does": "doesn't", "do": "don't", "did": "didn't", "can": "can't",
    "will": "won't", "should": "shouldn't", "could": "couldn't",
    "would": "wouldn't", "has": "hasn't", "have": "haven't",
}

QUESTION_WORDS = frozenset({"what", "how", "why", "when", "where", "who", "which"})

# Pairs that keep a sentence's shape while changing its subject. A fixed table
# rather than named entity recognition, so the suite needs no model of its own.
ENTITY_SWAPS = {
    "France": "Germany", "Paris": "Berlin", "Python": "Java", "order": "account",
    "Monday": "Friday", "Sunday": "Saturday", "refund": "credit", "cat": "dog",
    "London": "Madrid", "iPhone": "Android", "Tesla": "Ford", "red": "blue",
}

_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")


@dataclass(frozen=True)
class AdversarialCase:
    """An original prompt and a near duplicate that must not share its answer."""

    original: str
    mutated: str
    mutation: str


@dataclass(frozen=True)
class Measured:
    """A case and the cosine similarity the embedder gave the pair."""

    case: AdversarialCase
    similarity: float


class Embedder(Protocol):
    """Anything that embeds text. Matches app.cache.embedder.Embedder."""

    async def encode(self, text: str) -> list[float]:
        """Embed one prompt."""
        ...


def _swap_word(prompt: str, find: str, replace: str) -> str:
    """Replace the first whole-word match of `find`, keeping everything else."""
    return re.sub(rf"\b{re.escape(find)}\b", replace, prompt, count=1)


def negate(prompt: str) -> AdversarialCase | None:
    """Negate the first auxiliary verb, or None if there is none.

    The mutation that worries a semantic cache most: one short word flips the
    meaning while leaving almost every token of the sentence unchanged.
    """
    words = prompt.split(" ")
    for index, word in enumerate(words):
        bare = word.strip("?.,!").lower()
        if bare not in CONTRACTIONS:
            continue
        previous = words[index - 1].lower() if index else ""
        inverted = index == 0 or previous in QUESTION_WORDS
        if inverted:
            negative = CONTRACTIONS[bare]
            if word[0].isupper():
                negative = negative[0].upper() + negative[1:]
            words[index] = word.replace(word.strip("?.,!"), negative, 1)
        else:
            words[index] = f"{word} not"
        return AdversarialCase(prompt, " ".join(words), "negation")
    return None


def swap_entity(prompt: str) -> AdversarialCase | None:
    """Replace the first known entity with its counterpart, or None if none appear.

    Same structure, different subject, different correct answer.
    """
    for find, replace in ENTITY_SWAPS.items():
        if re.search(rf"\b{re.escape(find)}\b", prompt):
            return AdversarialCase(prompt, _swap_word(prompt, find, replace), "entity_swap")
    return None


def change_number(prompt: str) -> AdversarialCase | None:
    """Change the first number in the prompt by ten, or None if there is none.

    A changed number changes the answer while barely moving the embedding,
    because numbers carry little weight in a sentence model.
    """
    match = _NUMBER.search(prompt)
    if match is None:
        return None
    text = match.group(0)
    changed = str(float(text) + 10) if "." in text else str(int(text) + 10)
    mutated = prompt[: match.start()] + changed + prompt[match.end():]
    return AdversarialCase(prompt, mutated, "numeric_change")


def generate_suite(prompts: Iterable[str]) -> list[AdversarialCase]:
    """Every mutation that applies to every prompt."""
    suite: list[AdversarialCase] = []
    for prompt in prompts:
        for generator in (negate, swap_entity, change_number):
            case = generator(prompt)
            if case is not None:
                suite.append(case)
    return suite


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity, computed here rather than trusted from a store."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


async def measure(cases: Sequence[AdversarialCase], embedder: Embedder) -> list[Measured]:
    """Embed both sides of every case and record their similarity."""
    measured: list[Measured] = []
    for case in cases:
        original = await embedder.encode(case.original)
        mutated = await embedder.encode(case.mutated)
        measured.append(Measured(case, cosine(original, mutated)))
    return measured


def false_hit_rate(measured: Sequence[Measured], threshold: float) -> float:
    """Proportion of cases that would hit when they should have missed.

    Every case here must miss, because each mutation changes the correct answer.
    Any pair at or above the threshold is therefore a false hit. Zero cases is
    reported as zero rather than as an error, so an empty filter is visible in
    the output rather than crashing the run.
    """
    if not measured:
        return 0.0
    hits = sum(1 for item in measured if item.similarity >= threshold)
    return hits / len(measured)


def report(measured: Sequence[Measured]) -> list[dict[str, object]]:
    """False hit rate per mutation kind and overall, at each swept threshold."""
    kinds = sorted({item.case.mutation for item in measured})
    rows: list[dict[str, object]] = []
    for kind in [*kinds, "all"]:
        subset = [m for m in measured if kind == "all" or m.case.mutation == kind]
        row: dict[str, object] = {"mutation": kind, "cases": len(subset)}
        for threshold in THRESHOLDS:
            row[f"false_hit_rate_at_{threshold:.2f}"] = round(false_hit_rate(subset, threshold), 3)
        rows.append(row)
    return rows


SAMPLE_PROMPTS = (
    "Is the store open on Sunday?",
    "How do I cancel my order?",
    "What is 15 percent of 200?",
    "What is the capital of France?",
    "Can I get a refund within 30 days?",
    "Does the iPhone support wireless charging?",
    "Should I water a cat's plant 3 times a week?",
    "Was Python released before 1995?",
)


async def _main() -> None:
    """Run the sample suite against the real model and print the table."""
    from app.cache.embedder import SUPPORTED_MODEL, OnnxEmbedder

    embedder = OnnxEmbedder(SUPPORTED_MODEL)
    embedder.load()
    measured = await measure(generate_suite(SAMPLE_PROMPTS), embedder)
    for item in sorted(measured, key=lambda m: -m.similarity):
        print(f"{item.similarity:.3f}  {item.case.mutation:<15} {item.case.mutated}")
    print()
    for row in report(measured):
        print(row)


if __name__ == "__main__":
    asyncio.run(_main())
