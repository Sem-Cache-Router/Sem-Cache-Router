# A stand in for the OpenAI and Anthropic APIs.
#
# Serves the two endpoints the gateway's adapters call, in the vendors' wire
# formats, so the real adapters run unmodified against it: only their base URL
# points here. That is what makes two things possible that a real vendor would not
# allow. One is acceptance demonstration 3, failing the primary during a live run,
# which needs a provider that can be told to fail on command. The other is a
# benchmark that is repeatable and free: the same prompt always gets the same
# answer, with usage the adapters price exactly as they would a real reply, and no
# key is needed and no money is spent.
#
# Behaviour is set per vendor through the control endpoints: an HTTP status to
# answer with, and a delay in milliseconds. Everything else about a reply is a
# deterministic function of the prompt.
#
# Run it with:  uvicorn benchmarks.mock_upstream:app --port 8002
# and point the gateway at it with SEMCACHE_OPENAI_BASE_URL=http://localhost:8002/v1
# and SEMCACHE_ANTHROPIC_BASE_URL=http://localhost:8002/v1.

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

Vendor = Literal["openai", "anthropic"]


class Behaviour(BaseModel):
    """How one vendor answers: with this status, after this long."""

    status: int = Field(default=200, ge=200, le=599)
    delay_ms: float = Field(default=0.0, ge=0)


@dataclass
class VendorState:
    behaviour: Behaviour
    calls: int = 0


def reply_text(prompt: str) -> str:
    """The answer to a prompt. Deterministic, so a repeat is distinguishable from a fresh call."""
    return f"Mock answer to: {prompt}"


def token_count(text: str) -> int:
    """Four characters to a token, the same rule of thumb the gateway estimates with."""
    return max(1, len(text) // 4)


def _prompt_of(body: Any) -> str | None:
    """The text of the last message, or None if the body is not a chat request."""
    messages = body.get("messages") if isinstance(body, dict) else None
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return None
    content = messages[-1].get("content")
    return content if isinstance(content, str) else None


def create_app() -> FastAPI:
    """Build an isolated instance, so a test never shares counters with another."""
    app = FastAPI(title="mock upstream")
    vendors = {
        "openai": VendorState(Behaviour()),
        "anthropic": VendorState(Behaviour()),
    }

    async def serve(vendor: Vendor, request: Request) -> JSONResponse:
        state = vendors[vendor]
        state.calls += 1
        if state.behaviour.delay_ms:
            await asyncio.sleep(state.behaviour.delay_ms / 1000)
        if state.behaviour.status != 200:
            return JSONResponse(
                {"error": {"message": f"mock {vendor} is failing on request"}},
                status_code=state.behaviour.status,
            )
        try:
            prompt = _prompt_of(await request.json())
        except ValueError:
            prompt = None
        if prompt is None:
            return JSONResponse({"error": {"message": "not a chat request"}}, status_code=400)

        answer = reply_text(prompt)
        prompt_tokens, completion_tokens = token_count(prompt), token_count(answer)
        if vendor == "openai":
            return JSONResponse(
                {
                    "id": "chatcmpl-mock",
                    "object": "chat.completion",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": answer},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                }
            )
        return JSONResponse(
            {
                "id": "msg-mock",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": prompt_tokens, "output_tokens": completion_tokens},
            }
        )

    @app.post("/v1/chat/completions")
    async def openai_chat(request: Request) -> JSONResponse:
        return await serve("openai", request)

    @app.post("/v1/messages")
    async def anthropic_messages(request: Request) -> JSONResponse:
        return await serve("anthropic", request)

    @app.put("/control/{vendor}")
    async def set_behaviour(vendor: Vendor, behaviour: Behaviour) -> dict[str, Any]:
        """Make a vendor answer with this status and delay from now on."""
        vendors[vendor].behaviour = behaviour
        return {vendor: behaviour.model_dump()}

    @app.get("/stats")
    async def stats() -> dict[str, Any]:
        """Calls received and current behaviour, per vendor."""
        return {
            name: {"calls": state.calls, **state.behaviour.model_dump()}
            for name, state in vendors.items()
        }

    @app.post("/reset")
    async def reset() -> dict[str, str]:
        """Return both vendors to healthy and zero the counters."""
        for state in vendors.values():
            state.behaviour = Behaviour()
            state.calls = 0
        return {"status": "reset"}

    return app


app = create_app()
