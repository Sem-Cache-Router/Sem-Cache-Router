# Model routing.
#
# A pure function: no I/O, no imports from other app modules that touch the
# event loop. That makes it trivially testable and keeps it out of the lifespan.
#
# Rule is deliberately simple and explainable in a viva: a prompt is "simple"
# when it is a single user message, its estimated token count is under the
# configured ceiling, and it contains no code fence. Simple prompts go to the
# cheap model; everything else stays on the requested model.
#
# Cache key design: the cache key uses the model the CALLER REQUESTED, not the
# model that routing selected. See section 33.7 in the PRD for the rationale.
# The short version: a cache key built from the routed model would be
# invalidated every time the routing rule changes, orphaning every entry. Using
# the requested model means the cache is stable across routing updates, at the
# cost of sometimes serving a cheap model's answer to a caller who asked for
# the frontier model -- which is exactly what routing is, so it should be said
# plainly.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.models import Message

if TYPE_CHECKING:
    from app.config import Settings


@dataclass(frozen=True)
class RoutingResult:
    """The model to call and why the routing decision landed where it did."""

    model: str
    reason: str  # "routed_cheap" or "unchanged"


def _estimate_tokens(text: str) -> int:
    """Approximate four characters per token.

    The same rule the stub provider uses. Accurate enough for a routing
    threshold and needs no tiktoken dependency in this pure module.
    """
    return max(1, len(text) // 4)


def _is_simple(messages: list[Message], max_tokens: int) -> bool:
    """True when the prompt is judged cheap enough to route to a smaller model.

    Three conditions must all hold: the conversation is a single user message
    (multi-turn context typically needs the frontier model), the estimated
    token count is below the configured ceiling, and the message contains no
    code fence (code requests usually need careful reasoning).
    """
    if len(messages) != 1:
        return False
    msg = messages[0]
    if msg.role != "user":
        return False
    if "```" in msg.content:
        return False
    return _estimate_tokens(msg.content) < max_tokens


def choose_model(
    requested_model: str,
    messages: list[Message],
    settings: Settings,
) -> RoutingResult:
    """Return the model to call and the reason for the choice.

    When routing is disabled the requested model is always returned with
    reason "unchanged". When enabled, a prompt judged simple goes to
    settings.routing_cheap_model with reason "routed_cheap"; everything else
    keeps the requested model with reason "unchanged".

    The caller is responsible for using the REQUESTED model as the cache key,
    not the model this function returns. See section 33.7 in the PRD.
    """
    if not settings.routing_enabled:
        return RoutingResult(model=requested_model, reason="unchanged")

    if _is_simple(messages, settings.routing_simple_max_tokens):
        # routing_enabled with no cheap model is rejected at startup, so
        # routing_cheap_model is guaranteed non-None here.
        return RoutingResult(model=settings.routing_cheap_model, reason="routed_cheap")  # type: ignore[arg-type]

    return RoutingResult(model=requested_model, reason="unchanged")
