"""Cache-aware preflight estimates + streaming wrapper.

Covers two review fixes:

1. **Cache-aware preflight**: when ``cache_control`` is anywhere in the
   request, ``estimate_cost`` must price input tokens at the cache-write
   rate (1.25× base for 5m TTL, 2× for 1h). Otherwise a request near the
   cap can pass the gate and post-flight reconciliation tips us over.
2. **Stream support**: ``claude_call`` rejects ``stream=True`` cleanly
   (no silent crash on ``response.usage``). ``claude_stream`` is a
   context manager that pre-flights, opens the SDK stream, and reconciles
   the ledger from ``stream.get_final_message()`` after the caller is
   done iterating.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from esports_sim.budget import (
    BudgetCaps,
    BudgetExhausted,
    Governor,
    Ledger,
    claude_call,
    claude_stream,
)
from esports_sim.budget.client import _has_cache_control
from esports_sim.budget.pricing import estimate_cost

# ---- cache-aware estimate -------------------------------------------------


def test_estimate_cost_without_cache_uses_base_input_rate() -> None:
    """Baseline: no caching → base $5/MTok input on Opus 4.7."""
    cost = estimate_cost(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        max_output_tokens=0,
    )
    assert cost == pytest.approx(5.00, abs=1e-9)


def test_estimate_cost_with_cache_5m_uses_125x_input_rate() -> None:
    """5-minute TTL: input rate is base × 1.25."""
    cost = estimate_cost(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        max_output_tokens=0,
        has_cache_control=True,
        cache_write_ttl="5m",
    )
    # 1M tokens * $5 base * 1.25 = $6.25
    assert cost == pytest.approx(6.25, abs=1e-9)


def test_estimate_cost_with_cache_1h_uses_2x_input_rate() -> None:
    """1-hour TTL: input rate is base × 2.0 — the riskiest underestimate."""
    cost = estimate_cost(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        max_output_tokens=0,
        has_cache_control=True,
        cache_write_ttl="1h",
    )
    # 1M tokens * $5 base * 2.00 = $10.00
    assert cost == pytest.approx(10.00, abs=1e-9)


def test_has_cache_control_detects_top_level_kwarg() -> None:
    assert _has_cache_control(
        messages=[],
        system=None,
        extra={"cache_control": {"type": "ephemeral"}},
    )


def test_has_cache_control_detects_per_block_in_messages() -> None:
    assert _has_cache_control(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
            }
        ],
        system=None,
        extra={},
    )


def test_has_cache_control_detects_per_block_in_system() -> None:
    assert _has_cache_control(
        messages=[],
        system=[{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}],
        extra={},
    )


def test_has_cache_control_detects_per_tool() -> None:
    assert _has_cache_control(
        messages=[],
        system=None,
        extra={
            "tools": [{"name": "x", "input_schema": {}, "cache_control": {"type": "ephemeral"}}]
        },
    )


def test_has_cache_control_returns_false_when_no_markers() -> None:
    assert not _has_cache_control(
        messages=[{"role": "user", "content": "hi"}],
        system="You are helpful.",
        extra={"max_tokens": 100},
    )


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


def _make_mock_client(
    *,
    input_tokens_estimate: int = 1000,
    actual_input_tokens: int = 1000,
    actual_output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    request_id: str = "req_mock",
) -> MagicMock:
    client = MagicMock(name="anthropic_client")
    client.messages.count_tokens.return_value = SimpleNamespace(
        input_tokens=input_tokens_estimate,
    )
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hi")],
        usage=SimpleNamespace(
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
        _request_id=request_id,
    )
    client.messages.create.return_value = response
    return client


def test_cache_aware_preflight_blocks_call_that_base_estimate_would_admit(
    ledger: Ledger,
) -> None:
    """The exact scenario the reviewer flagged.

    Hard cap: $30. Already spent: $29.50 — leaves $0.50 of headroom.

    Request with caching, 1h TTL, 50k input tokens, 1k max_tokens, on
    Opus 4.7:

    * Base estimate: 50k * $5/MTok + 1k * $25/MTok = $0.275  → would admit
    * Cache-aware:   50k * $5 * 2.0 + 1k * $25       = $0.525 → blocks

    Without the cache-aware fix, the call slips through and post-flight
    reconciliation pushes us to $30.025 — over the hard cap.
    """
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=29.50,
    )
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_mock_client(
        input_tokens_estimate=50_000,
        actual_input_tokens=0,
        actual_output_tokens=0,
        cache_creation_input_tokens=50_000,
    )

    with pytest.raises(BudgetExhausted) as exc_info:
        claude_call(
            governor=gov,
            purpose="patch_intent",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "patch notes here",
                            "cache_control": {"type": "ephemeral", "ttl": "1h"},
                        }
                    ],
                }
            ],
            model="claude-opus-4-7",
            max_tokens=1000,
            cache_write_ttl="1h",
            client=client,
        )

    # The exception's projected cost is the cache-aware figure ($0.525),
    # not the naive base-rate one ($0.275).
    assert exc_info.value.projected_cost_usd == pytest.approx(0.525, abs=1e-9)
    # SDK was never called — gate refused before the network.
    assert not client.messages.create.called


# ---- stream=True rejection ------------------------------------------------


def test_claude_call_rejects_stream_true(ledger: Ledger) -> None:
    """The reviewer's second concern: ``stream=True`` previously crashed
    on ``response.usage`` after preflight, leaving the row stuck in
    ``pre``. Now it raises a clear TypeError before any side effects.
    """
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_mock_client()

    with pytest.raises(TypeError, match="claude_stream"):
        claude_call(
            governor=gov,
            purpose="patch_intent",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
            stream=True,
        )

    # Critically: nothing was written to the ledger. No orphan ``pre``
    # row, no SDK call.
    assert ledger.all_entries() == []
    assert not client.messages.count_tokens.called
    assert not client.messages.create.called


# ---- claude_stream end-to-end --------------------------------------------


class _FakeStream:
    """Stand-in for the SDK's ``MessageStream`` context-managed object.

    Yields a few text chunks, then surfaces a ``get_final_message`` that
    looks like a regular ``Message`` so the wrapper's reconciliation path
    can exercise the same code as ``claude_call``.
    """

    def __init__(self, *, chunks: list[str], usage: SimpleNamespace, request_id: str):
        self._chunks = chunks
        self._final = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="".join(chunks))],
            usage=usage,
            _request_id=request_id,
        )
        self.text_stream = iter(chunks)

    def __enter__(self) -> _FakeStream:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self._final


def _make_streaming_mock_client(
    *,
    input_tokens_estimate: int = 1000,
    actual_input_tokens: int = 1000,
    actual_output_tokens: int = 200,
    chunks: list[str] | None = None,
    request_id: str = "req_streamed",
) -> MagicMock:
    client = MagicMock(name="anthropic_streaming_client")
    client.messages.count_tokens.return_value = SimpleNamespace(input_tokens=input_tokens_estimate)
    fake_stream = _FakeStream(
        chunks=chunks or ["hello ", "world"],
        usage=SimpleNamespace(
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        request_id=request_id,
    )
    # ``client.messages.stream(**kwargs)`` returns the context-manager.
    client.messages.stream.return_value = fake_stream
    return client


def test_claude_stream_yields_stream_and_reconciles_on_exit(ledger: Ledger) -> None:
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_streaming_mock_client(
        input_tokens_estimate=1000,
        actual_input_tokens=950,
        actual_output_tokens=300,
    )

    collected: list[str] = []
    with claude_stream(
        governor=gov,
        purpose="agent_decision",
        messages=[{"role": "user", "content": "tell me a story"}],
        model="claude-haiku-4-5",
        max_tokens=512,
        client=client,
    ) as stream:
        for text in stream.text_stream:
            collected.append(text)

    # Caller saw the streamed chunks.
    assert collected == ["hello ", "world"]

    # Ledger has exactly one row, in `post` phase, with actual cost.
    rows = ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].phase == "post"
    expected_cost = (950 * 1.0 + 300 * 5.0) / 1_000_000  # haiku 4.5 pricing
    assert rows[0].usd_cost == pytest.approx(expected_cost, abs=1e-12)
    assert rows[0].request_id == "req_streamed"
    assert rows[0].endpoint == "messages.stream"


def test_claude_stream_blocks_over_cap_before_opening_stream(ledger: Ledger) -> None:
    """Streaming respects the same governor as non-streaming."""
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=29.99,
    )
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_streaming_mock_client(input_tokens_estimate=1000)

    with (
        # Combined `with` so ruff stays happy. The streaming wrapper raises
        # from preflight *before* yielding, so the body never runs.
        pytest.raises(BudgetExhausted),
        claude_stream(
            governor=gov,
            purpose="patch_intent",
            messages=[{"role": "user", "content": "..."}],
            client=client,
        ),
    ):
        pass  # pragma: no cover - preflight raises first

    # SDK stream was never opened; only count_tokens ran for the estimate.
    assert client.messages.count_tokens.called
    assert not client.messages.stream.called


def test_claude_stream_marks_error_on_caller_exception(ledger: Ledger) -> None:
    """If the caller raises mid-stream, the row is recorded as ``error``."""
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_streaming_mock_client(input_tokens_estimate=1000)

    class _Boom(RuntimeError):
        pass

    with (
        pytest.raises(_Boom),
        claude_stream(
            governor=gov,
            purpose="agent_decision",
            messages=[{"role": "user", "content": "hi"}],
            client=client,
        ) as stream,
    ):
        next(iter(stream.text_stream))  # consume one chunk
        raise _Boom("caller bailed")

    rows = ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].phase == "error"
    assert "_Boom" in (rows[0].notes or "")


def test_claude_stream_silently_ignores_stream_kwarg(ledger: Ledger) -> None:
    """``stream=True`` passed to the streaming wrapper is dropped (it's the
    SDK's non-streaming kwarg; the streaming entry point doesn't accept it).
    """
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_streaming_mock_client()

    with claude_stream(
        governor=gov,
        purpose="agent_decision",
        messages=[{"role": "user", "content": "hi"}],
        client=client,
        stream=True,  # caller copy-pasted from a non-streaming example
    ) as stream:
        list(stream.text_stream)

    # The SDK streaming entry point was called without stream=True.
    sdk_kwargs = client.messages.stream.call_args.kwargs
    assert "stream" not in sdk_kwargs
