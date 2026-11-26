"""The look-alike questions on the console page: well formed, and still true of the real model.

The page shows a similarity score beside each pair, measured offline with
all-MiniLM-L6-v2. The model test re-measures every pair, so an upgrade that moves
a score fails here instead of leaving the page claiming something no longer true.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from app.api.demo import PAGE
from app.cache.embedder import SUPPORTED_MODEL, OnnxEmbedder
from evaluation.adversarial import cosine


def load() -> dict[str, Any]:
    html = PAGE.read_text()
    match = re.search(r'<script type="application/json" id="weak-spots-data">(.*?)</script>',
                      html, re.S)
    assert match, "the weak spot data block is missing from the page"
    data: dict[str, Any] = json.loads(match.group(1))
    return data


def prompt(data: dict[str, Any], case: dict[str, Any], which: str) -> str:
    return f"{data['context']} {case[which]}" if case.get("long") else str(case[which])


def test_every_case_is_well_formed() -> None:
    data = load()
    ids = [case["id"] for case in data["cases"]]
    assert len(ids) == len(set(ids)) >= 6
    for case in data["cases"]:
        assert {"id", "title", "a", "b", "score", "why"} <= set(case)
        assert case["a"] != case["b"]
        assert 0.0 <= case["score"] <= 1.0


def test_there_is_a_control_the_cache_gets_right_and_traps_it_gets_wrong() -> None:
    cases = load()["cases"]
    controls = [c for c in cases if c["id"] == "control"]
    traps = [c for c in cases if c["id"] != "control"]
    assert len(controls) == 1 and controls[0]["score"] < 0.75
    assert len(traps) >= 6
    assert all(c["score"] >= 0.90 for c in traps)


def test_the_long_prompt_case_carries_a_document() -> None:
    data = load()
    long_cases = [c for c in data["cases"] if c.get("long")]
    assert long_cases and len(data["context"].split()) > 100


def test_the_page_wires_the_cases_to_buttons() -> None:
    html = PAGE.read_text()
    assert 'id="weak"' in html and 'id="weak-all"' in html and 'id="cases"' in html
    script = html.split("<script>")[1].split("</script>")[0]
    assert "runAll" in script and "runCase" in script
    assert "innerHTML" not in script


@pytest.mark.model
async def test_the_scores_on_the_page_are_what_the_real_model_measures() -> None:
    data = load()
    embedder = OnnxEmbedder(SUPPORTED_MODEL)
    embedder.load()
    for case in data["cases"]:
        first = await embedder.encode(prompt(data, case, "a"))
        second = await embedder.encode(prompt(data, case, "b"))
        measured = cosine(first, second)
        assert measured == pytest.approx(case["score"], abs=0.02), case["id"]
