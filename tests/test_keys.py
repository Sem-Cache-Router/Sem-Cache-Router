# Key derivation tests.
#
# These functions are pure, so the tests are the specification. Each case below
# encodes a decision from the design rather than an observation about the code,
# and the names say which decision.

from __future__ import annotations

from app.cache.keys import EXACT_PREFIX, SCHEMA_VERSION, make_key, normalise_messages
from app.models import Message


def user(content: str) -> list[Message]:
    """One user message, the shape almost every case needs."""
    return [Message(role="user", content=content)]


def test_key_is_namespaced_and_versioned() -> None:
    """The prefix and schema version are part of the key, owned by make_key alone."""
    key = make_key("gpt-4o-mini", user("hello"))
    assert key.startswith(f"{EXACT_PREFIX}{SCHEMA_VERSION}:")
    assert len(key.rsplit(":", 1)[1]) == 64


def test_identical_input_gives_identical_key() -> None:
    """The whole cache rests on this being stable across calls."""
    assert make_key("m", user("hello")) == make_key("m", user("hello"))


# --- normalisation: what it does ------------------------------------------


def test_doubled_spaces_collapse() -> None:
    """TEST-003. Formatting noise must not cost a provider call."""
    assert make_key("m", user("a  b")) == make_key("m", user("a b"))


def test_tabs_and_newlines_collapse() -> None:
    """Any run of whitespace is one space, not just literal spaces."""
    assert make_key("m", user("a\t\nb")) == make_key("m", user("a b"))


def test_leading_and_trailing_whitespace_is_trimmed() -> None:
    """A trailing newline from a shell heredoc is not a different question."""
    assert make_key("m", user("  hello  ")) == make_key("m", user("hello"))


def test_unicode_whitespace_collapses() -> None:
    """A non-breaking space is whitespace too, and str.split treats it as such."""
    assert make_key("m", user("a b")) == make_key("m", user("a b"))


def test_normalisation_returns_new_messages() -> None:
    """The caller still needs the original text to store on the entry."""
    original = user("  spaced   out  ")
    normalised = normalise_messages(original)
    assert original[0].content == "  spaced   out  "
    assert normalised[0].content == "spaced out"


def test_role_is_preserved() -> None:
    """Normalisation touches content only."""
    messages = [Message(role="system", content=" be brief ")]
    assert normalise_messages(messages)[0].role == "system"


# --- normalisation: what it deliberately does not do ----------------------


def test_case_difference_is_a_miss_by_design() -> None:
    """Case is preserved at Tier 1 on purpose.

    Two prompts differing only in case are not reliably the same question, and
    folding them together here would bury a correctness decision inside what
    looks like a performance optimisation. That judgement belongs in Tier 2,
    where the similarity threshold makes the risk explicit and measurable.
    """
    assert make_key("m", user("Hello")) != make_key("m", user("hello"))


def test_punctuation_difference_is_a_miss_by_design() -> None:
    """Same reasoning as case. "Stop." and "Stop?" are not the same request."""
    assert make_key("m", user("Stop.")) != make_key("m", user("Stop?"))


# --- what else is part of the key -----------------------------------------


def test_different_model_gives_a_different_key() -> None:
    """TEST-002. The same prompt against a different model is a different answer."""
    assert make_key("gpt-4o-mini", user("hello")) != make_key("gpt-4o", user("hello"))


def test_message_order_matters() -> None:
    """A conversation is ordered, so two orderings are two different prompts."""
    forward = [Message(role="user", content="a"), Message(role="assistant", content="b")]
    backward = [Message(role="assistant", content="b"), Message(role="user", content="a")]
    assert make_key("m", forward) != make_key("m", backward)


def test_role_is_part_of_the_key() -> None:
    """The same text from a system prompt and a user turn are different inputs."""
    as_user = [Message(role="user", content="be brief")]
    as_system = [Message(role="system", content="be brief")]
    assert make_key("m", as_user) != make_key("m", as_system)


def test_content_boundaries_cannot_be_confused() -> None:
    """Two messages must not hash the same as one message holding both texts.

    Guards against a naive implementation that concatenates content before
    hashing, where ["ab"] and ["a", "b"] would collide.
    """
    one = [Message(role="user", content="ab")]
    two = [Message(role="user", content="a"), Message(role="user", content="b")]
    assert make_key("m", one) != make_key("m", two)
