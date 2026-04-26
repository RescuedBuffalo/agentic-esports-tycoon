"""The chokepoint every Claude call goes through (BUF-22, ADR-005).

The governor is intentionally tiny. It owns three concerns:

1. **Pre-flight gating** — given a projected cost, refuse if the rolling
   weekly spend would exceed the hard cap (or the per-purpose soft cap
   for this call's purpose). Raise :class:`BudgetExhausted` and write a
   ``blocked`` row to the ledger.
2. **Pre-flight logging** — write a ``pre`` row before the call so a crash
   mid-call leaves an audit trail.
3. **Post-flight reconciliation** — overwrite the pre row with actual
   usage from the response.

Atomicity: the cap check + the row insert run inside a single SQLite
``BEGIN IMMEDIATE`` transaction (see :meth:`Ledger.serializable_write`).
SQLite's file lock serialises all writers across processes, so two
workers near the cap can't both observe the pre-call spend, both pass
the gate, and both push us over.

Everything else (the SDK call itself, prompt-caching plumbing, retries) is
the wrapper's job; the governor is decoupled so tests can drive it with a
mock client and the real wrapper can stay small.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import timedelta

from esports_sim.budget.caps import BudgetCaps
from esports_sim.budget.errors import BudgetExhausted
from esports_sim.budget.ledger import Ledger, _now_utc


@dataclass
class PreflightTicket:
    """Hand-off between :meth:`Governor.preflight` and :meth:`record_post`.

    Carries the row id we created at pre-flight so the post-call update
    knows which row to reconcile, plus the projected cost we used for the
    cap check (handy when surfacing diagnostics).
    """

    row_id: int
    projected_cost_usd: float


# Internal sentinel returned by the transactional cap check. ``scope`` of
# ``None`` means "under cap, here's the row id"; otherwise the caller
# raises BudgetExhausted with the scope+spend+cap fields.
@dataclass
class _PreflightOutcome:
    row_id: int | None
    scope: str | None
    spend_usd: float
    cap_usd: float


class Governor:
    """Gate + recorder.

    Stateless across calls — the source of truth lives in the SQLite
    ledger, so multiple processes can share one Governor instance via the
    same DB without coordinating.
    """

    def __init__(
        self,
        *,
        ledger: Ledger | None = None,
        caps: BudgetCaps | None = None,
    ) -> None:
        self.ledger = ledger if ledger is not None else Ledger()
        # Honour env-var overrides (NEXUS_BUDGET_WEEKLY_HARD_CAP_USD,
        # NEXUS_BUDGET_DISABLE_CAPS) by default. Callers that want a frozen
        # config — typically tests — pass `caps=` explicitly. Static
        # defaults here would silently ignore the documented break-glass
        # env var, which is exactly the wrong behaviour during incident
        # response.
        self.caps = caps if caps is not None else BudgetCaps.from_env()

    # ---- pre-flight --------------------------------------------------------

    def preflight(
        self,
        *,
        purpose: str,
        model: str,
        endpoint: str,
        projected_cost_usd: float,
        request_id: str | None = None,
        notes: str | None = None,
    ) -> PreflightTicket:
        """Atomically check the caps and write a ``pre`` (or ``blocked``) row.

        Order of checks: per-purpose soft cap *first* (cheaper to fail when
        a runaway is on a single purpose), then the global hard cap. Both
        check ``current_spend + projected_cost`` so we never spend a dollar
        and *then* notice we shouldn't have. The check + insert run inside
        ``BEGIN IMMEDIATE`` so concurrent governors serialise — without
        that lock two callers both reading $29.95 could each commit a $0.10
        call and end the week at $30.15.
        """
        if self.caps.override_disable_caps:
            # Operator disabled enforcement — log it once, loudly, in the
            # row's notes so post-mortems can find every call that slipped
            # through. No transaction needed: with no cap-check there is no
            # cap-check race to lose.
            note = "override_disable_caps=True"
            annotated_notes = f"{notes}; {note}" if notes else note
            row_id = self.ledger.record(
                endpoint=endpoint,
                model=model,
                purpose=purpose,
                phase="pre",
                usd_cost=projected_cost_usd,
                request_id=request_id,
                notes=annotated_notes,
            )
            return PreflightTicket(row_id=row_id, projected_cost_usd=projected_cost_usd)

        # Single transaction. SQLite's BEGIN IMMEDIATE acquires the write
        # lock at start; concurrent processes hitting this method serialise
        # at the file-lock layer.
        with self.ledger.serializable_write() as conn:
            outcome = self._run_caps_in_txn(
                conn,
                purpose=purpose,
                model=model,
                endpoint=endpoint,
                projected_cost_usd=projected_cost_usd,
                request_id=request_id,
                notes=notes,
            )
        # Transaction has committed by here — the blocked-or-pre row is
        # durable. Now react to the outcome.

        if outcome.scope is not None:
            raise BudgetExhausted(
                purpose=purpose,
                weekly_spend_usd=outcome.spend_usd,
                weekly_cap_usd=outcome.cap_usd,
                projected_cost_usd=projected_cost_usd,
                scope=outcome.scope,
            )

        assert outcome.row_id is not None  # invariant from _run_caps_in_txn
        return PreflightTicket(row_id=outcome.row_id, projected_cost_usd=projected_cost_usd)

    def _run_caps_in_txn(
        self,
        conn: sqlite3.Connection,
        *,
        purpose: str,
        model: str,
        endpoint: str,
        projected_cost_usd: float,
        request_id: str | None,
        notes: str | None,
    ) -> _PreflightOutcome:
        """Cap check + row insert. Caller owns the transaction."""
        purpose_cap = self.caps.cap_for(purpose)
        weekly_window_start = _now_utc() - timedelta(days=7)

        # 1. Per-purpose soft cap.
        if purpose_cap is not None:
            purpose_spend = self.ledger._spend_in_window(
                conn, since=weekly_window_start, purpose=purpose
            )
            if purpose_spend + projected_cost_usd > purpose_cap:
                self._record_blocked(
                    conn,
                    endpoint=endpoint,
                    model=model,
                    purpose=purpose,
                    projected_cost_usd=projected_cost_usd,
                    request_id=request_id,
                    scope="purpose",
                    spend=purpose_spend,
                    cap=purpose_cap,
                    notes=notes,
                )
                return _PreflightOutcome(
                    row_id=None,
                    scope="purpose",
                    spend_usd=purpose_spend,
                    cap_usd=purpose_cap,
                )

        # 2. Global weekly hard cap.
        total_spend = self.ledger._spend_in_window(conn, since=weekly_window_start)
        if total_spend + projected_cost_usd > self.caps.weekly_hard_cap_usd:
            self._record_blocked(
                conn,
                endpoint=endpoint,
                model=model,
                purpose=purpose,
                projected_cost_usd=projected_cost_usd,
                request_id=request_id,
                scope="weekly",
                spend=total_spend,
                cap=self.caps.weekly_hard_cap_usd,
                notes=notes,
            )
            return _PreflightOutcome(
                row_id=None,
                scope="weekly",
                spend_usd=total_spend,
                cap_usd=self.caps.weekly_hard_cap_usd,
            )

        # 3. Under cap — write the pre row inside the same lock so a second
        #    concurrent caller can't observe the same spend total and slip
        #    through.
        row_id = self.ledger._insert(
            conn,
            endpoint=endpoint,
            model=model,
            purpose=purpose,
            phase="pre",
            usd_cost=projected_cost_usd,
            request_id=request_id,
            notes=notes,
        )
        return _PreflightOutcome(row_id=row_id, scope=None, spend_usd=0.0, cap_usd=0.0)

    # ---- post-flight -------------------------------------------------------

    def record_post(
        self,
        ticket: PreflightTicket,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        usd_cost: float,
        request_id: str | None,
        notes: str | None = None,
    ) -> None:
        self.ledger.update_post(
            ticket.row_id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            usd_cost=usd_cost,
            request_id=request_id,
            phase="post",
            notes=notes,
        )

    def record_error(
        self,
        ticket: PreflightTicket,
        *,
        notes: str,
    ) -> None:
        """Mark a pre row as ``error`` when the SDK call raises mid-flight.

        Keeps the projected cost in place — the call may have partially
        billed (Anthropic charges for input tokens even on stream-aborted
        responses), so we'd rather over-count than under-count.
        """
        self.ledger.update_post(
            ticket.row_id,
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            usd_cost=ticket.projected_cost_usd,
            request_id=None,
            phase="error",
            notes=notes,
        )

    # ---- helpers -----------------------------------------------------------

    def _record_blocked(
        self,
        conn: sqlite3.Connection,
        *,
        endpoint: str,
        model: str,
        purpose: str,
        projected_cost_usd: float,
        request_id: str | None,
        scope: str,
        spend: float,
        cap: float,
        notes: str | None,
    ) -> None:
        """Insert a ``blocked`` row using the caller's connection."""
        annotation = (
            f"blocked scope={scope} spend=${spend:.4f} cap=${cap:.4f} "
            f"projected=${projected_cost_usd:.4f}"
        )
        self.ledger._insert(
            conn,
            endpoint=endpoint,
            model=model,
            purpose=purpose,
            phase="blocked",
            usd_cost=0.0,  # blocked rows don't count toward spend
            request_id=request_id,
            notes=f"{notes}; {annotation}" if notes else annotation,
        )
