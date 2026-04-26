"""``claude_call`` wrapper — verifies pre-flight gating, post-flight reconciliation,
and that prompt-caching parameters flow through unchanged.
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
)


def _make_mock_client(
    *,
    input_tokens_estimate: int = 1000,
    actual_input_tokens: int = 1000,
    actual_output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    request_id: str = "req_mock",
) -> MagicMock:
    """Build an anthropic.Anthropic stand-in with the two methods we use."""
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


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


def test_under_cap_reaches_sdk_and_logs_post_with_actual_cost(ledger: Ledger) -> None:
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_mock_client(
        input_tokens_estimate=1000,
        actual_input_tokens=950,
        actual_output_tokens=100,
    )

    response = claude_call(
        governor=gov,
        purpose="patch_intent",
        messages=[{"role": "user", "content": "hi"}],
        model="claude-haiku-4-5",
        max_tokens=512,
        client=client,
    )

    # SDK was called with the model + max_tokens we passed.
    create_call = client.messages.create.call_args.kwargs
    assert create_call["model"] == "claude-haiku-4-5"
    assert create_call["max_tokens"] == 512

    # Ledger has exactly one row, in ``post`` phase, with actual cost
    # (cheaper than the pre-flight estimate that assumed full max_tokens).
    rows = ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].phase == "post"
    expected_cost = (950 * 1.0 + 100 * 5.0) / 1_000_000  # haiku 4.5 pricing
    assert rows[0].usd_cost == pytest.approx(expected_cost, abs=1e-12)
    assert rows[0].request_id == "req_mock"

    # Caller gets the response back.
    assert response.usage.output_tokens == 100


def test_over_cap_raises_before_sdk_is_hit(ledger: Ledger) -> None:
    """BUF-22 acceptance: attempting a call over cap raises BudgetExhausted,
    writes to ledger, never touches the SDK."""
    # Seed the ledger above the hard cap.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=29.99,
    )
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_mock_client(input_tokens_estimate=1000)

    with pytest.raises(BudgetExhausted):
        claude_call(
            governor=gov,
            purpose="patch_intent",
            messages=[{"role": "user", "content": "hi"}],
            model="claude-opus-4-7",
            max_tokens=4096,
            client=client,
        )

    # Pre-flight count_tokens still ran (that's how we computed the projected
    # cost), but messages.create did NOT.
    assert client.messages.count_tokens.called
    assert not client.messages.create.called

    # Blocked row is in the ledger.
    blocked = [r for r in ledger.all_entries() if r.phase == "blocked"]
    assert len(blocked) == 1
    assert blocked[0].purpose == "patch_intent"


def test_cache_control_kwargs_pass_through_unchanged(ledger: Ledger) -> None:
    """Prompt-caching parameters are forwarded verbatim — the wrapper must
    not strip ``cache_control`` blocks or rewrite the system list.
    """
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_mock_client(
        input_tokens_estimate=2000,
        cache_creation_input_tokens=1500,
        cache_read_input_tokens=0,
    )

    system_blocks = [
        {
            "type": "text",
            "text": "frozen patch notes context",
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]
    claude_call(
        governor=gov,
        purpose="patch_intent",
        messages=[{"role": "user", "content": "what changed?"}],
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system_blocks,
        cache_write_ttl="1h",
        client=client,
    )

    sent = client.messages.create.call_args.kwargs
    # System blocks reach the SDK identical to what we passed in.
    assert sent["system"] is system_blocks
    # The cache-write TTL is reflected in the ledger cost (1h = 2× write rate).
    rows = ledger.all_entries()
    assert len(rows) == 1
    # 1500 cache-write @ 1h = 1500 * 3.0 * 2.0 / 1e6 = $0.009
    expected = (
        1500 * 3.0 * 2.0  # cache write 1h on sonnet 4.6
        + 1000 * 3.0  # actual input (default actual_input_tokens=1000)
        + 200 * 15.0  # actual output (default 200)
    ) / 1_000_000
    assert rows[0].usd_cost == pytest.approx(expected, abs=1e-12)


def test_count_tokens_includes_tools_when_passed(ledger: Ledger) -> None:
    """Tool definitions can be huge; the pre-flight estimate must include them."""
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    client = _make_mock_client(input_tokens_estimate=5000)

    tools = [
        {
            "name": "lookup",
            "description": "look up a player",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]
    claude_call(
        governor=gov,
        purpose="agent_decision",
        messages=[{"role": "user", "content": "..."}],
        tools=tools,
        client=client,
    )
    count_kwargs = client.messages.count_tokens.call_args.kwargs
    assert count_kwargs["tools"] is tools
