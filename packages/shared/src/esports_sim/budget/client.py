"""``claude_call`` and ``claude_stream`` — sanctioned wrappers around the
Anthropic SDK.

Every call site in the Nexus pipeline routes through one of these two
functions so every dollar is accounted for and every call respects the
weekly cap. New call sites should *not* construct ``anthropic.Anthropic``
directly; instead they pass ``messages`` / ``system`` / etc. to one of the
wrappers and let the governor handle the rest.

Choose the wrapper that matches your need:

* :func:`claude_call` for non-streaming requests. Returns the full
  :class:`anthropic.types.Message`. Rejects ``stream=True`` with a clear
  error pointing here so the wrapper can't silently corrupt the ledger
  (the streaming return type has no ``.usage`` attribute).
* :func:`claude_stream` for streaming requests. Used as a context manager;
  yields the SDK's stream object so callers can consume tokens
  incrementally, and reconciles the ledger from
  ``stream.get_final_message()`` after the context exits.

Both wrappers share the same pre-flight estimate (cache-aware), governor
gate, and post-flight reconciliation logic — see the private helpers
``_pricing_inputs``, ``_count_tokens``, ``_preflight_governor``, and
``_record_post`` below.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

from esports_sim.budget.governor import Governor, PreflightTicket
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


# --- internal helpers ------------------------------------------------------


def _has_cache_control(
    *,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None,
    extra: dict[str, Any],
) -> bool:
    """Detect whether the request has any prompt-caching markers.

    Matters for the pre-flight estimate: when caching is in use,
    ``cache_creation_input_tokens`` are billed at 1.25× (5m) or 2× (1h)
    the base input rate. If we estimated with the base rate we could let
    a near-cap call through and have post-flight reconciliation tip us
    over.

    We check four locations:

    * Top-level ``cache_control`` (auto-cache the last cacheable block).
    * Per-block ``cache_control`` on any ``messages[*].content[*]`` block.
    * Per-block ``cache_control`` on any ``system[*]`` block.
    * Per-tool ``cache_control`` on any ``tools[*]`` definition.

    Returns True on any hit; the caller treats the whole input as
    potentially cache-write.
    """
    if extra.get("cache_control"):
        return True
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("cache_control"):
                    return True
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("cache_control"):
                return True
    tools = extra.get("tools")
    if tools:
        for t in tools:
            if isinstance(t, dict) and t.get("cache_control"):
                return True
    return False


def _build_create_kwargs(
    *,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system is not None:
        kwargs["system"] = system
    kwargs.update(extra)
    return kwargs


def _projected_cost(
    *,
    client: anthropic.Anthropic,
    model: str,
    max_tokens: int,
    messages: list[dict[str, Any]],
    system: str | list[dict[str, Any]] | None,
    extra: dict[str, Any],
    cache_write_ttl: str,
) -> float:
    """Pre-flight cost: ask ``count_tokens`` for input size, price pessimistically."""
    count_kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if system is not None:
        count_kwargs["system"] = system
    # ``count_tokens`` also accepts ``tools`` — forward when present so the
    # estimate for tool-heavy prompts isn't wildly off.
    if "tools" in extra:
        count_kwargs["tools"] = extra["tools"]
    token_count = client.messages.count_tokens(**count_kwargs)
    return estimate_cost(
        model=model,
        input_tokens=int(token_count.input_tokens),
        max_output_tokens=max_tokens,
        has_cache_control=_has_cache_control(messages=messages, system=system, extra=extra),
        cache_write_ttl=cache_write_ttl,
    )


def _record_post(
    governor: Governor,
    ticket: PreflightTicket,
    response: Any,
    *,
    model: str,
    cache_write_ttl: str,
) -> None:
    """Reconcile the pre row with actual usage from a Message-shaped object."""
    usage = response.usage
    actual_cost = cost_from_usage_obj(model, usage, cache_write_ttl=cache_write_ttl)
    governor.record_post(
        ticket,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        usd_cost=actual_cost,
        request_id=getattr(response, "_request_id", None),
    )


# --- public API ------------------------------------------------------------


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
    """Run one non-streaming Claude call through the budget governor.

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
        ``"5m"`` (default) or ``"1h"``. Used both for the pre-flight cache-
        write multiplier and for pricing the post-flight cache-creation
        tokens. The actual TTL is set on ``cache_control`` blocks the
        caller passes through; this kwarg just tells the governor which
        rate to charge.
    client:
        Anthropic SDK client. If omitted, one is constructed from the
        process environment (``ANTHROPIC_API_KEY``).
    notes:
        Free-form annotation that ends up on the ledger row.
    **extra:
        Anything else ``messages.create`` accepts — ``tools``, ``thinking``,
        ``cache_control``, etc. Passed through unchanged. ``stream=True``
        is rejected (use :func:`claude_stream` instead).
    """
    if extra.get("stream"):
        raise TypeError(
            "claude_call does not support stream=True — the streaming "
            "return type has no `.usage` attribute and the post-flight "
            "reconciliation would crash, leaving the ledger row stuck in "
            "the `pre` phase. Use esports_sim.budget.claude_stream() for "
            "streaming requests; it consumes the stream and reconciles "
            "from stream.get_final_message()."
        )

    # Import lazily so tests can mock the SDK client without forcing the
    # whole anthropic package to load at import time.
    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    projected_cost = _projected_cost(
        client=client,
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        extra=extra,
        cache_write_ttl=cache_write_ttl,
    )

    ticket = governor.preflight(
        purpose=purpose,
        model=model,
        endpoint="messages.create",
        projected_cost_usd=projected_cost,
        notes=notes,
    )

    create_kwargs = _build_create_kwargs(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        extra=extra,
    )

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

    _record_post(governor, ticket, response, model=model, cache_write_ttl=cache_write_ttl)
    # ``messages.create`` returns ``Message`` at runtime; the cast is just
    # to satisfy mypy when the call site uses an injected mock client.
    return cast("Message", response)


@contextmanager
def claude_stream(
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
) -> Iterator[Any]:
    """Streaming counterpart to :func:`claude_call`.

    Used as a context manager that yields the SDK's
    ``MessageStreamManager``-yielded stream object — the caller iterates
    ``stream.text_stream`` (or `stream.events`) for incremental tokens,
    and on context exit the wrapper pulls ``stream.get_final_message()``
    to reconcile the ledger.

    Example::

        with claude_stream(governor=gov, purpose="summarise", messages=[...]) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
        # Ledger is reconciled here via stream.get_final_message().

    On caller exception inside the ``with`` block, the ledger row is
    marked ``error`` (matching :func:`claude_call`'s behaviour). The
    governor's ``preflight`` runs *before* the SDK stream opens, so an
    over-cap request never even contacts the network.
    """
    # Anthropic's streaming API doesn't take ``stream=True`` — it has its
    # own ``messages.stream()`` entry point. Reject the kwarg so callers
    # don't accidentally double-stream.
    extra.pop("stream", None)

    import anthropic

    if client is None:
        client = anthropic.Anthropic()

    projected_cost = _projected_cost(
        client=client,
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        extra=extra,
        cache_write_ttl=cache_write_ttl,
    )

    ticket = governor.preflight(
        purpose=purpose,
        model=model,
        endpoint="messages.stream",
        projected_cost_usd=projected_cost,
        notes=notes,
    )

    create_kwargs = _build_create_kwargs(
        model=model,
        max_tokens=max_tokens,
        messages=messages,
        system=system,
        extra=extra,
    )

    final_message: Any = None
    try:
        with client.messages.stream(**create_kwargs) as stream:
            yield stream
            # Caller's ``with`` block has exited cleanly. Pull the final
            # message *before* the SDK stream context exits, so the
            # accumulated state is still available.
            final_message = stream.get_final_message()
    except BaseException as exc:
        # Caller bailed out, or the SDK raised. Mark the pre row as errored
        # and re-raise so the caller sees the original exception.
        governor.record_error(
            ticket,
            notes=f"{type(exc).__module__}.{type(exc).__name__}: {exc}",
        )
        raise

    if final_message is not None:
        _record_post(governor, ticket, final_message, model=model, cache_write_ttl=cache_write_ttl)
