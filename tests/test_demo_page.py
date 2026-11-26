"""The showcase page: served at /, self contained, and safe with what it displays."""

from __future__ import annotations

import re

from fastapi.testclient import TestClient

from app.api.demo import PAGE


def test_the_page_is_served_at_the_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == PAGE.read_text()


def test_every_panel_is_on_the_page(client: TestClient) -> None:
    html = client.get("/").text
    for panel in ("ask", "limit", "numbers", "health", "failover", "admin"):
        assert f'id="{panel}"' in html


def test_the_page_calls_only_endpoints_the_gateway_has(client: TestClient) -> None:
    html = client.get("/").text
    called = set(re.findall(r'fetch\("(/[a-z0-9/._-]*)"', html))
    assert called == {"/v1/chat/completions", "/metrics", "/health", "/admin/cache"}
    for path in ("/metrics", "/health"):
        assert client.get(path).status_code == 200


def test_the_page_never_writes_a_response_as_markup() -> None:
    # Answers come from a model; written as HTML they could inject script into the page.
    script = PAGE.read_text().split("<script>")[1].split("</script>")[0]
    assert "innerHTML" not in script and "outerHTML" not in script
    assert "insertAdjacentHTML" not in script and "document.write" not in script


def test_the_page_loads_nothing_from_outside() -> None:
    html = PAGE.read_text()
    assert not re.search(r'(src|href)="https?://', html)


def test_the_page_is_not_in_the_api_schema(client: TestClient) -> None:
    assert "/" not in client.get("/openapi.json").json()["paths"]


def test_the_api_still_answers_beside_the_page(client: TestClient) -> None:
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    response = client.post("/v1/chat/completions", json=body,
                           headers={"Authorization": "Bearer demo"})
    assert response.status_code == 200
