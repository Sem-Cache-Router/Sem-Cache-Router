# Prompt normalisation and Tier 1 key derivation.
#
# Normalisation folds whitespace and does nothing else. Case and punctuation are
# preserved on purpose: two prompts differing only in case are not reliably the
# same question, and folding them here would hide a correctness decision inside
# what looks like a performance optimisation. That kind of near match belongs in
# Tier 2, where the threshold makes the risk explicit and measurable.
#
# Why the model is part of the key: the same prompt against a different model is
# a different answer.
#
# Why the key carries a schema version: if the stored entry shape changes, old
# entries are orphaned and expire on their existing TTL, instead of needing a
# migration or a flush that would invalidate a benchmark baseline.

from __future__ import annotations

import hashlib
import json

from app.models import Message

SCHEMA_VERSION = "v1"
EXACT_PREFIX = "semcache:exact:"


def normalise_messages(messages: list[Message]) -> list[Message]:
    """Collapse whitespace runs and trim each message, preserving everything else.

    Deterministic normalisation is what makes the Tier 1 hash stable across
    callers who format the same question slightly differently. str.split() with
    no argument does the whole job in one call: it splits on runs of any
    whitespace, which collapses doubled spaces, tabs and newlines and trims the
    ends, and it treats Unicode whitespace the same way.

    Returns new messages rather than mutating the input, because the caller
    still needs the original text to store on the cache entry.
    """
    return [
        Message(role=message.role, content=" ".join(message.content.split()))
        for message in messages
    ]


def canonical_payload(model: str, messages: list[Message]) -> str:
    """Serialise the model and normalised messages to a canonical JSON string.

    Exposed rather than inlined so a test can assert on the exact bytes that get
    hashed. A key function whose input cannot be inspected is one you have to
    debug by guessing.

    Canonical means fixed key order and no incidental whitespace, so that two
    equal inputs always produce byte identical output.
    """
    payload = {
        "model": model,
        "messages": [
            {"role": message.role, "content": message.content}
            for message in normalise_messages(messages)
        ],
    }
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, sort_keys=True)


def make_key(model: str, messages: list[Message]) -> str:
    """Return the Tier 1 cache key: a schema versioned SHA-256 over model and messages.

    Hashes canonical JSON rather than a Python repr. A pydantic model's repr is
    library version dependent, so hashing it would let a dependency upgrade
    silently change the whole key space and orphan every cached entry with no
    error and no failing test.

    This function owns the namespace. Callers get the full key including prefix
    and schema version and must not add either themselves, or a double prefix
    produces a cache that never hits.
    """
    digest = hashlib.sha256(canonical_payload(model, messages).encode("utf-8")).hexdigest()
    return f"{EXACT_PREFIX}{SCHEMA_VERSION}:{digest}"
