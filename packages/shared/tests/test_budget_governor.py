"""Governor pre/post-flight gating, including the BUF-22 mock-test acceptance."""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.budget import (
    BudgetCaps,
    BudgetExhausted,
    Governor,
    Ledger,
)


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


def test_under_cap_passes_through_and_logs_pre(ledger: Ledger) -> None:
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.05,
    )
    rows = ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].phase == "pre"
    assert rows[0].usd_cost == pytest.approx(0.05)
    assert ticket.row_id == rows[0].id


def test_over_weekly_cap_raises_and_writes_blocked_row(ledger: Ledger) -> None:
    """BUF-22 acceptance: over-cap call raises BudgetExhausted, writes to ledger."""
    # Seed the ledger with $29.99 already spent.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="other",
        phase="post",
        usd_cost=29.99,
    )
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))

    with pytest.raises(BudgetExhausted) as exc_info:
        gov.preflight(
            purpose="patch_intent",
            model="claude-opus-4-7",
            endpoint="messages.create",
            projected_cost_usd=0.05,  # would push us to $30.04, over $30 cap
        )
    err = exc_info.value
    assert err.scope == "weekly"
    assert err.weekly_cap_usd == 30.0
    assert err.projected_cost_usd == pytest.approx(0.05)

    rows = ledger.all_entries()
    assert len(rows) == 2
    blocked = [r for r in rows if r.phase == "blocked"]
    assert len(blocked) == 1
    assert blocked[0].purpose == "patch_intent"
    assert "blocked scope=weekly" in (blocked[0].notes or "")


def test_per_purpose_soft_cap_fires_before_global_cap(ledger: Ledger) -> None:
    """A purpose-cap breach should block even when the global cap has room."""
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="post",
        usd_cost=2.99,
    )
    gov = Governor(
        ledger=ledger,
        caps=BudgetCaps(
            weekly_hard_cap_usd=30.0,
            purpose_caps_usd={"patch_intent": 3.0},
        ),
    )
    with pytest.raises(BudgetExhausted) as exc_info:
        gov.preflight(
            purpose="patch_intent",
            model="claude-opus-4-7",
            endpoint="messages.create",
            projected_cost_usd=0.05,  # would push purpose total to $3.04, over $3 cap
        )
    assert exc_info.value.scope == "purpose"
    assert exc_info.value.weekly_cap_usd == 3.0


def test_purpose_without_cap_only_checks_global(ledger: Ledger) -> None:
    """Purposes not in the soft-cap dict only hit the global cap."""
    gov = Governor(
        ledger=ledger,
        caps=BudgetCaps(
            weekly_hard_cap_usd=30.0,
            purpose_caps_usd={"patch_intent": 3.0},
        ),
    )
    # ``personality`` isn't capped here — should pass even if "personality"
    # spend is high, as long as the global cap has headroom.
    ledger.record(
        endpoint="m",
        model="x",
        purpose="personality",
        phase="post",
        usd_cost=10.0,
    )
    ticket = gov.preflight(
        purpose="personality",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
    )
    assert ticket.row_id is not None


def test_record_post_overwrites_pre_with_actual_usage(ledger: Ledger) -> None:
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
    )
    gov.record_post(
        ticket,
        input_tokens=200,
        output_tokens=100,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        usd_cost=0.0035,  # actual, lower than estimate
        request_id="req_xyz",
    )
    rows = ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].phase == "post"
    assert rows[0].usd_cost == pytest.approx(0.0035)


def test_record_error_marks_pre_row_as_error(ledger: Ledger) -> None:
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
    )
    gov.record_error(ticket, notes="anthropic.APIConnectionError: timeout")
    rows = ledger.all_entries()
    assert rows[0].phase == "error"
    assert "APIConnectionError" in (rows[0].notes or "")
    # Projected cost stays so the cap math doesn't under-attribute the failure.
    assert rows[0].usd_cost == pytest.approx(0.10)


def test_override_disable_caps_skips_enforcement(ledger: Ledger) -> None:
    """Operator break-glass: caps disabled, calls log but don't block."""
    ledger.record(
        endpoint="m",
        model="x",
        purpose="other",
        phase="post",
        usd_cost=29.99,
    )
    gov = Governor(
        ledger=ledger,
        caps=BudgetCaps(
            weekly_hard_cap_usd=30.0,
            override_disable_caps=True,
        ),
    )
    # Even though we'd be over cap, override lets the call through.
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=5.0,
    )
    pre = [e for e in ledger.all_entries() if e.id == ticket.row_id][0]
    assert pre.phase == "pre"
    # The override flag is annotated on the row so audits surface it.
    assert "override_disable_caps" in (pre.notes or "")
