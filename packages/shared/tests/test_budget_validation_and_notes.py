"""Defensive validation + audit-trail preservation.

Covers two review fixes:

1. **Non-negative projection guard** — ``Governor.preflight`` rejects
   negative ``projected_cost_usd`` with ``ValueError`` before any cap
   math. A negative estimate would *reduce* apparent spend in the cap
   check (admitting over-cap traffic) and, if persisted via a ``pre`` row
   that later resolves, grant extra budget on subsequent calls.
2. **record_error preserves preflight notes** — ``Ledger.update_post``
   now *appends* to existing notes instead of replacing them, so error
   annotations don't clobber pre-flight markers like
   ``override_disable_caps=True`` that post-mortems rely on.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.budget import BudgetCaps, Governor, Ledger
from esports_sim.budget.governor import PreflightTicket


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


# ---- non-negative projection guard ----------------------------------------


def test_preflight_rejects_negative_projected_cost(ledger: Ledger) -> None:
    """A negative cost would shrink apparent spend → over-cap admission."""
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    with pytest.raises(ValueError, match="non-negative"):
        gov.preflight(
            purpose="patch_intent",
            model="claude-opus-4-7",
            endpoint="messages.create",
            projected_cost_usd=-0.05,
        )

    # Critically: nothing landed in the ledger. No `pre` row, no
    # `blocked` row — we surfaced the bug before any side effects.
    assert ledger.all_entries() == []


def test_preflight_rejects_negative_even_with_override(ledger: Ledger) -> None:
    """The override skips cap *enforcement*, not the validation guard.

    A negative estimate is a programmer error in the upstream ``estimate_cost``
    call, not something the operator can opt out of. Even in break-glass
    mode we want the bug surfaced, not silently persisted.
    """
    gov = Governor(
        ledger=ledger,
        caps=BudgetCaps(weekly_hard_cap_usd=30.0, override_disable_caps=True),
    )
    with pytest.raises(ValueError, match="non-negative"):
        gov.preflight(
            purpose="patch_intent",
            model="claude-opus-4-7",
            endpoint="messages.create",
            projected_cost_usd=-1.0,
        )
    assert ledger.all_entries() == []


def test_preflight_accepts_zero_projected_cost(ledger: Ledger) -> None:
    """Zero is legal — a valid projection for a (very) small request."""
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.0,
    )
    rows = ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].id == ticket.row_id
    assert rows[0].usd_cost == 0.0


# ---- error-row notes preservation -----------------------------------------


def test_record_error_preserves_override_marker(ledger: Ledger) -> None:
    """The acceptance scenario from the review.

    Break-glass mode marks every admitted call with
    ``override_disable_caps=True``. When such a call fails mid-flight,
    ``record_error`` was overwriting that note with the exception detail,
    losing the audit trail. After the fix the row carries *both*.
    """
    gov = Governor(
        ledger=ledger,
        caps=BudgetCaps(weekly_hard_cap_usd=30.0, override_disable_caps=True),
    )
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
    )

    # Simulate the SDK call raising — the wrapper would call this.
    gov.record_error(ticket, notes="anthropic.APIConnectionError: timeout")

    row = next(r for r in ledger.all_entries() if r.id == ticket.row_id)
    assert row.phase == "error"
    # Both annotations are present, separated by the canonical "; ".
    assert "override_disable_caps=True" in (row.notes or "")
    assert "anthropic.APIConnectionError" in (row.notes or "")


def test_record_error_appends_when_pre_already_had_notes(ledger: Ledger) -> None:
    """Generic version: any pre-flight note is preserved on error."""
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
        notes="run_id=abc123",
    )
    gov.record_error(ticket, notes="anthropic.RateLimitError")

    row = next(r for r in ledger.all_entries() if r.id == ticket.row_id)
    assert "run_id=abc123" in (row.notes or "")
    assert "anthropic.RateLimitError" in (row.notes or "")


def test_record_post_with_no_new_notes_keeps_pre_notes(ledger: Ledger) -> None:
    """``record_post(notes=None)`` (the default) leaves pre notes alone.

    This was already true under the old COALESCE semantics; we re-assert
    after the CASE rewrite to make sure the happy path didn't regress.
    """
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
        notes="run_id=abc123",
    )
    gov.record_post(
        ticket,
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        usd_cost=0.0035,
        request_id="req_xyz",
    )
    row = next(r for r in ledger.all_entries() if r.id == ticket.row_id)
    assert row.notes == "run_id=abc123"


def test_update_post_appends_when_pre_notes_empty_string(ledger: Ledger) -> None:
    """Edge case: empty-string pre notes shouldn't produce ``'; new'``.

    The CASE arm ``notes IS NULL OR notes = ''`` covers it — the new
    note becomes the value rather than getting appended after a leading
    separator.
    """
    # Manually craft an empty-string pre note (no governor wrapper does
    # this today, but defensive coding for anything downstream that does).
    row_id = ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="pre",
        usd_cost=0.10,
        notes="",
    )
    ledger.update_post(
        row_id,
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        usd_cost=0.10,
        request_id=None,
        phase="error",
        notes="anthropic.APIError",
    )
    row = next(r for r in ledger.all_entries() if r.id == row_id)
    assert row.notes == "anthropic.APIError"


def test_record_error_with_no_pre_notes_just_writes_the_error(ledger: Ledger) -> None:
    """When pre-flight had no notes (the common case), the error note
    is the row's first annotation — no leading separator.
    """
    gov = Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))
    ticket = gov.preflight(
        purpose="patch_intent",
        model="claude-opus-4-7",
        endpoint="messages.create",
        projected_cost_usd=0.10,
    )
    # PreflightTicket has the row id; row was created with notes=None.
    assert isinstance(ticket, PreflightTicket)

    gov.record_error(ticket, notes="anthropic.APIError: 500")
    row = next(r for r in ledger.all_entries() if r.id == ticket.row_id)
    assert row.notes == "anthropic.APIError: 500"
