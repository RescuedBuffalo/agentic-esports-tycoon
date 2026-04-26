"""``claude_call`` — the only sanctioned way to hit the Claude API.

Every call site in the Nexus pipeline routes through this function so
every dollar is accounted for and every call respects the weekly cap. New
call sites should *not* construct ``anthropic.Anthropic`` directly; instead
they pass ``messages`` / ``system`` / etc. to ``claude_call`` and let the
governor handle the rest.

The wrapper is deliberately thin:

* Cost estimate via ``client.messages.count_tokens`` (worst-case output =
  ``max_tokens``, the upper bound).
* Governor pre-flight: refuse if over cap, else write a ``pre`` row.
* Forward to ``client.messages.create`` (caller-supplied kwargs flow through).
* Reconcile post-flight from ``response.usage``.

Prompt caching: callers attach ``cache_control={"type": "ephemeral"}`` to
the ``messages.create`` call (auto-cache the last cacheable block) or as
a per-block annotation on ``system`` / ``messages``. Both shapes flow
through unchanged — the wrapper does not touch caching parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from esports_sim.budget.governor import Governor
from esports_sim.budget.pricing import (
    DEFAULT_MODEL,
    cost_from_usage_obj,
    estimate_cost,
)

if TYPE_CHECKING:
    import anthropic
    from anthropic.types import Message


# Default ceiling for a single response. Conservative — long outputs need an
# explicit override at the call site so the budget check is realistic.
_DEFAULT_MAX_TOKENS = 4096


def claude_call(
    *,
    governor: Governor,
    purpose: str,
    messages: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    system: str | list[dict[str, Any]] | None = None,
    cache_write_ttl: str = "5m",
    client: anthropic.Anthropic | None = None,
    notes: str | None = None,
    **extra: Any,
) -> Message:
    """Run one Claude call through the budget governor.

    Parameters
    ----------
    governor:
        The :class:`Governor` instance — usually a process-wide singleton.
    purpose:
        Free-form attribution string. Used to look up the per-purpose soft
        cap and to bucket spend in reports. Conventional values include
        ``personality``, ``patch_intent``, ``agent_decision``.
    messages, model, max_tokens, system:
        Forwarded to ``client.messages.create``. Defaults match the issue:
        Opus 4.7, 4096 max_tokens.
    cache_write_ttl:
        ``"5m"`` (default) or ``"1h"``. Only used to price cache-write
        tokens for the post-flight reconciliation; the actual TTL is set on
        the ``cache_control`` blocks the caller passes through.
    client:
        Anthropic SDK client. If omitted, one is constructed from the
        process environment (``ANTHROPIC_API_KEY``).
    notes:
        Free-form annotation that ends up on the ledger row.
    **extra:
        Anything else ``messages.create`` accepts — ``tools``, ``thinking``,
        ``cache_control``, etc. Passed through unchanged.
    """
    # Import lazily so tests can mock the SDK client without forcing the
    # whole anthropic package to load at import time.
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        create_kwargs["system"] = system
    create_kwargs.update(extra)

    # ---- pre-flight cost estimate -----------------------------------------
    # ``count_tokens`` mirrors the actual prompt accounting (including system,
    # tools, cache markers) so the estimate is the closest pre-flight number
    # we can get to reality.
    count_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if system is not None:
        count_kwargs["system"] = system
    # ``count_tokens`` also accepts ``tools`` — forward when present so the
    # estimate for tool-heavy prompts isn't wildly off.
    if "tools" in extra:
        count_kwargs["tools"] = extra["tools"]
    token_count = client.messages.count_tokens(**count_kwargs)
    projected_input = int(token_count.input_tokens)
    projected_cost = estimate_cost(
        model=model,
        input_tokens=projected_input,
        max_output_tokens=max_tokens,
    )

    # ---- governor gate -----------------------------------------------------
    ticket = governor.preflight(
        purpose=purpose,
        model=model,
        endpoint="messages.create",
        projected_cost_usd=projected_cost,
        notes=notes,
    )

    # ---- the actual API call ----------------------------------------------
    try:
        response = client.messages.create(**create_kwargs)
    except Exception as exc:  # pragma: no cover - network failures
        # Mark the pre row as errored so reports can surface partial calls.
        # The full exception type name is in the notes; the original is
        # re-raised so callers see real anthropic exceptions.
        governor.record_error(
            ticket,
            notes=f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
        )
        raise

    # ---- post-flight reconciliation ---------------------------------------
    actual_cost = cost_from_usage_obj(model, response.usage, cache_write_ttl=cache_write_ttl)
    governor.record_post(
        ticket,
        input_tokens=int(response.usage.input_tokens or 0),
        output_tokens=int(response.usage.output_tokens or 0),
        cache_creation_input_tokens=int(
            getattr(response.usage, "cache_creation_input_tokens", 0) or 0
        ),
        cache_read_input_tokens=int(getattr(response.usage, "cache_read_input_tokens", 0) or 0),
        usd_cost=actual_cost,
        request_id=getattr(response, "_request_id", None),
    )
    # ``messages.create`` returns ``Message`` at runtime; the cast is just
    # to satisfy mypy when the call site uses an injected mock client.
    return cast("Message", response)
