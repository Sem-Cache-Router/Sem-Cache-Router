# Adversarial false hit suite.
#
# Covers TEST-006 (negation), TEST-007 (entity swap) and TEST-008 (changed
# number). The PRD states these must miss. Against the real model they do not:
# measured on all-MiniLM-L6-v2, negated questions score 0.96 to 0.98 against
# their originals, above every threshold the benchmark sweeps. So these tests
# check the generators and the arithmetic, and the one test that loads the real
# model pins the finding itself rather than asserting a property the model does
# not have. A failure there means the finding changed, not that the code broke.

from __future__ import annotations

import pytest

from evaluation.adversarial import (
    SAMPLE_PROMPTS,
    THRESHOLDS,
    AdversarialCase,
    Measured,
    change_number,
    cosine,
    false_hit_rate,
    generate_suite,
    measure,
    negate,
    report,
    swap_entity,
)
from tests.conftest import MappingEmbedder, base_vector, unit_vector_at

# --- negation (TEST-006) --------------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Is the store open on Sunday?", "Isn't the store open on Sunday?"),
        ("What is 15 percent of 200?", "What isn't 15 percent of 200?"),
        ("Can I get a refund?", "Can't I get a refund?"),
        ("Paris is lovely", "Paris is not lovely"),
        ("The meeting was cancelled.", "The meeting was not cancelled."),
    ],
)
def test_negation_produces_grammatical_text(prompt: str, expected: str) -> None:
    """Questions take the contracted form in the verb's place, statements take
    "not" after it. Ungrammatical probes would measure the model on input it
    never sees in real traffic."""
    case = negate(prompt)
    assert case is not None
    assert case.mutated == expected
    assert case.mutation == "negation"


def test_negation_skips_a_prompt_with_no_auxiliary_verb() -> None:
    """Nothing to negate means no case, rather than a nonsense mutation."""
    assert negate("Hello there") is None


# --- entity swap (TEST-007) -----------------------------------------------


def test_entity_swap_changes_the_subject_and_nothing_else() -> None:
    """Same shape, different subject, different correct answer."""
    case = swap_entity("What is the capital of France?")
    assert case is not None
    assert case.mutated == "What is the capital of Germany?"


def test_entity_swap_matches_whole_words_only() -> None:
    """"category" contains "cat" but is not about one."""
    assert swap_entity("Which category is this?") is None


# --- numeric change (TEST-008) --------------------------------------------


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("What is 15 percent of 200?", "What is 25 percent of 200?"),
        ("Is 2.5 more than 2?", "Is 12.5 more than 2?"),
    ],
)
def test_numeric_change_alters_only_the_first_number(prompt: str, expected: str) -> None:
    """One changed number changes the answer while barely moving the embedding."""
    case = change_number(prompt)
    assert case is not None
    assert case.mutated == expected


def test_numeric_change_skips_a_prompt_without_numbers() -> None:
    """No number, no case."""
    assert change_number("What is the capital of France?") is None


# --- suite and arithmetic -------------------------------------------------


def test_the_suite_applies_every_mutation_that_fits() -> None:
    """A prompt with a verb, an entity and a number yields all three kinds."""
    suite = generate_suite(["Is Paris 5 hours away?"])
    assert sorted(case.mutation for case in suite) == ["entity_swap", "negation", "numeric_change"]


def test_the_sample_suite_covers_all_three_mutation_kinds() -> None:
    """The bundled sample is broad enough to report every kind."""
    kinds = {case.mutation for case in generate_suite(SAMPLE_PROMPTS)}
    assert kinds == {"negation", "entity_swap", "numeric_change"}


def test_cosine_of_a_vector_with_itself_is_one() -> None:
    """Sanity check on the similarity used for the report."""
    assert cosine([3.0, 4.0], [3.0, 4.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def _measured(*similarities: float, mutation: str = "negation") -> list[Measured]:
    return [Measured(AdversarialCase("a", "b", mutation), s) for s in similarities]


def test_a_score_exactly_at_the_threshold_counts_as_a_false_hit() -> None:
    """The cache serves at or above the threshold, so the rate must count it."""
    assert false_hit_rate(_measured(0.90, 0.80), 0.90) == 0.5


def test_an_empty_measurement_reports_zero_rather_than_failing() -> None:
    """An empty filter shows up in the output instead of crashing the run."""
    assert false_hit_rate([], 0.90) == 0.0


def test_the_report_has_a_row_per_kind_plus_overall_at_every_threshold() -> None:
    """Every swept threshold is reported, including the unflattering ones."""
    rows = report(_measured(0.97, mutation="negation") + _measured(0.70, mutation="entity_swap"))
    assert [row["mutation"] for row in rows] == ["entity_swap", "negation", "all"]
    for threshold in THRESHOLDS:
        assert f"false_hit_rate_at_{threshold:.2f}" in rows[-1]


async def test_measure_records_the_similarity_of_each_pair(
    mapping_embedder: MappingEmbedder,
) -> None:
    """Both sides are embedded and compared."""
    mapping_embedder.register("original", base_vector())
    mapping_embedder.register("mutated", unit_vector_at(0.93))
    measured = await measure([AdversarialCase("original", "mutated", "negation")], mapping_embedder)
    assert measured[0].similarity == pytest.approx(0.93)


# --- the finding, against the real model ----------------------------------


@pytest.mark.model
async def test_negation_defeats_the_similarity_threshold() -> None:
    """Characterises the model rather than the code.

    Negated questions score above 0.90 against their originals, so a semantic
    cache at the PRD's default threshold serves the opposite answer. If this
    starts failing, the finding the evaluation reports has changed and the PRD
    and the report need revisiting. Opt in with SEMCACHE_RUN_MODEL_TESTS=1.
    """
    from app.cache.embedder import SUPPORTED_MODEL, OnnxEmbedder

    embedder = OnnxEmbedder(SUPPORTED_MODEL)
    embedder.load()
    negations = [case for case in generate_suite(SAMPLE_PROMPTS) if case.mutation == "negation"]
    measured = await measure(negations, embedder)

    assert false_hit_rate(measured, 0.90) >= 0.75
