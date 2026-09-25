# Route level tests: wire compatibility, the cache path, and error mapping.
#
# These run against the real app factory with fake dependencies seeded into
# app.state, so the lifespan wiring is exercised rather than bypassed. A feature
# that works only when called directly from a test is not finished.

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict

from app.providers.stub import StubProvider

AUTH = {"Authorization": "Bearer test-key"}
PROMPT = "What is the capital of France?"


def body(content: str = PROMPT, model: str = "gpt-4o-mini") -> dict[str, Any]:
    """A minimal valid chat completion request."""
    return {"model": model, "messages": [{"role": "user", "content": content}]}


def post(client: TestClient, payload: dict[str, Any], auth: bool = True) -> Any:
    """Send a completion request, with the caller key unless told otherwise."""
    return client.post("/v1/chat/completions", json=payload, headers=AUTH if auth else {})


# --- the claim the whole slice exists to make ----------------------------


def test_first_request_is_a_miss(client: TestClient) -> None:
    """An unseen prompt reaches the provider and says so."""
    response = post(client, body())
    assert response.status_code == 200
    payload = response.json()
    assert payload["semcache_status"] == "miss"
    assert payload["semcache_similarity"] is None


def test_second_identical_request_is_an_exact_hit(
    client: TestClient, stub_provider: StubProvider
) -> None:
    """TEST-001. The provider is called once, not twice."""
    first = post(client, body())
    second = post(client, body())

    assert second.json()["semcache_status"] == "exact_hit"
    assert stub_provider.call_count == 1
    assert first.json()["choices"][0]["message"] == second.json()["choices"][0]["message"]


def test_whitespace_difference_still_hits(
    client: TestClient, stub_provider: StubProvider
) -> None:
    """The demonstration for this slice, in one request pair.

    Proves normalisation, hashing and Tier 1 together: a prompt that differs
    only in spacing is served from cache without a second provider call.
    """
    post(client, body(PROMPT))
    spaced = post(client, body("  What is   the\tcapital\nof France?  "))

    assert spaced.json()["semcache_status"] == "exact_hit"
    assert stub_provider.call_count == 1


def test_case_difference_reaches_the_provider(
    client: TestClient, stub_provider: StubProvider
) -> None:
    """Case is preserved at Tier 1 by design, so this is a second call."""
    post(client, body("Hello"))
    second = post(client, body("hello"))

    assert second.json()["semcache_status"] == "miss"
    assert stub_provider.call_count == 2


def test_different_model_is_a_separate_entry(
    client: TestClient, stub_provider: StubProvider
) -> None:
    """TEST-002. The model is part of the key."""
    post(client, body(model="gpt-4o-mini"))
    second = post(client, body(model="gpt-4o"))

    assert second.json()["semcache_status"] == "miss"
    assert stub_provider.call_count == 2


def test_a_hit_returns_the_model_that_answered(client: TestClient) -> None:
    """The cached body is replayed, so the response is identical apart from status."""
    first = post(client, body()).json()
    second = post(client, body()).json()
    assert first["model"] == second["model"]
    assert first["usage"] == second["usage"]


# --- wire compatibility ---------------------------------------------------


class NaiveOpenAIResponse(BaseModel):
    """What an unmodified OpenAI client models, and nothing more.

    Parsing a real response with this is the actual test of the additive fields
    claim. Asserting that semcache_status is present would only prove we added
    it; this proves a client that has never heard of it still works.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    object: str
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


def test_a_naive_client_parses_the_response(client: TestClient) -> None:
    """BR-001. Adoption costs a base URL change and nothing else."""
    parsed = NaiveOpenAIResponse.model_validate(post(client, body()).json())
    assert parsed.choices[0]["message"]["content"]
    assert parsed.usage["total_tokens"] > 0


def test_unmodelled_openai_fields_are_accepted(client: TestClient) -> None:
    """Real clients send n, user, stream_options and more.

    The request model ignores extras rather than forbidding them, because
    rejecting a field we simply do not implement would break the one promise
    the gateway makes.
    """
    payload = {**body(), "user": "u1", "n": 1, "response_format": {"type": "text"}}
    assert post(client, payload).status_code == 200


def test_response_carries_a_request_id(client: TestClient) -> None:
    """Every response is traceable back to one request in the logs."""
    assert post(client, body()).headers.get("x-request-id")


# --- error mapping --------------------------------------------------------


def test_missing_api_key_is_401(client: TestClient) -> None:
    """The key is the budget identity, so it cannot be optional."""
    response = post(client, body(), auth=False)
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "MISSING_API_KEY"


def test_malformed_body_is_400_not_422(client: TestClient) -> None:
    """The published contract says 400.

    FastAPI returns 422 for a validation failure by default. Without the
    handler in create_app the API would quietly disagree with its own
    documentation, and this is the test that holds that line.
    """
    response = client.post("/v1/chat/completions", json={"model": "m"}, headers=AUTH)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"


def test_empty_messages_is_rejected(client: TestClient) -> None:
    """A completion request with nothing to complete is malformed."""
    response = client.post(
        "/v1/chat/completions", json={"model": "m", "messages": []}, headers=AUTH
    )
    assert response.status_code == 400


def test_error_body_carries_a_request_id(client: TestClient) -> None:
    """A caller reporting a failure can be matched to the exact request."""
    error = post(client, body(), auth=False).json()["error"]
    assert error["request_id"].startswith("req_")


def test_provider_failure_is_502(client: TestClient) -> None:
    """With nothing left to fail over to, the honest answer is 502."""
    client.app.state.provider = StubProvider(fail_times=1)  # type: ignore[attr-defined]
    response = post(client, body("something never cached"))
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "ALL_PROVIDERS_UNAVAILABLE"


def test_cache_outage_fails_closed_with_503(client: TestClient) -> None:
    """Redis is the one hard dependency.

    Admission cannot be decided without it. Admission is a no-op today, so a
    pass through would be possible, but the ordering contract is the invariant
    being protected: a gateway whose failure behaviour changes once the limiter
    ships is one nobody can reason about.
    """

    class BrokenCache:
        async def get(self, *args: Any, **kwargs: Any) -> None:
            raise ConnectionError("redis is gone")

    client.app.state.cache = BrokenCache()  # type: ignore[attr-defined]
    response = post(client, body())
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "DEPENDENCY_UNAVAILABLE"


# --- health ---------------------------------------------------------------


def test_health_reports_a_reachable_redis(client: TestClient) -> None:
    """The happy path, which is also what Compose will poll."""
    payload = client.get("/health").json()
    assert payload["status"] == "ok"
    assert payload["dependencies"]["redis"] == "ok"


def test_health_reports_an_unreachable_redis(client: TestClient) -> None:
    """Degraded rather than a 500. A health check that raises tells you less."""

    class BrokenRedis:
        async def ping(self) -> None:
            raise ConnectionError("redis is gone")

    client.app.state.redis = BrokenRedis()  # type: ignore[attr-defined]
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["dependencies"]["redis"] == "unreachable"


def test_health_does_not_claim_chroma_is_healthy(client: TestClient) -> None:
    """Tier 2 is not built, and the endpoint says so rather than stubbing ok.

    A health endpoint that reports a dependency it never checked as healthy is
    worse than one that admits the gap.
    """
    assert client.get("/health").json()["dependencies"]["chroma"] == "not_configured"


@pytest.mark.parametrize("path", ["/health", "/v1/chat/completions"])
def test_routes_are_published_in_the_schema(client: TestClient, path: str) -> None:
    """Guards against a router that was written but never included.

    Read from the generated OpenAPI document rather than by walking app.routes,
    because newer Starlette wraps an included router in an object that does not
    expose a path. The schema is the contract callers read anyway, so asserting
    on it is closer to what actually matters.
    """
    assert path in client.get("/openapi.json").json()["paths"]
