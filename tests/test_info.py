"""The read-only info routes: live settings without secrets, and the benchmark summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api import info
from app.config import Settings

SECRET = "sk-test-do-not-leak-1234"


def flat(body: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {row["setting"]: row["value"] for rows in body.values() for row in rows}  # type: ignore[misc]


def test_config_shows_the_behaviour_shaping_settings(client: TestClient) -> None:
    body = client.get("/config").json()
    assert set(body) == {"cache", "limiter", "providers", "access"}
    values = flat(body)
    assert values["similarity_threshold"] == 0.90
    assert values["primary_provider"] == "stub"
    assert values["limiter_enabled"] is False
    assert all("label" in row for rows in body.values() for row in rows)


def test_config_never_returns_a_key_only_whether_one_is_set(
    client: TestClient, settings: Settings,
) -> None:
    client.app.state.settings = settings.model_copy(update={  # type: ignore[attr-defined]
        "openai_api_key": SECRET, "anthropic_api_key": SECRET,
        "admin_api_key": SECRET, "allowed_api_keys": SECRET,
    })
    response = client.get("/config")
    assert SECRET not in response.text
    values = flat(response.json())
    assert values["admin_api_key"] is True and values["allowed_api_keys"] is True
    assert "openai_api_key" not in values and "anthropic_api_key" not in values
    assert "redis_url" not in values


def test_without_keys_the_config_says_none_are_set(client: TestClient) -> None:
    values = flat(client.get("/config").json())
    assert values["admin_api_key"] is False and values["allowed_api_keys"] is False


def test_the_committed_benchmark_is_served(client: TestClient) -> None:
    body = client.get("/benchmark").json()
    assert body == json.loads(info.BENCHMARK.read_text())
    labels = [run["label"] for run in body["runs"]]
    assert labels[0] == "direct" and len(labels) >= 2
    assert isinstance(body["simulated"], bool)


def test_no_committed_benchmark_is_a_404(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(info, "BENCHMARK", tmp_path / "missing.json")
    assert client.get("/benchmark").status_code == 404
