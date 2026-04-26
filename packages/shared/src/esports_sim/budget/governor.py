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

Everything else (the SDK call itself, prompt-caching plumbing, retries) is
the wrapper's job; the governor is decoupled so tests can drive it with a
mock client and the real wrapper can stay small.
"""

from __future__ import annotations

from dataclasses import dataclass

from esports_sim.budget.caps import BudgetCaps, default_caps
from esports_sim.budget.errors import BudgetExhausted
from esports_sim.budget.ledger import Ledger


@dataclass
class PreflightTicket:
    """Hand-off between :meth:`Governor.preflight` and :meth:`record_post`.

    Carries the row id we created at pre-flight so the post-call update
    knows which row to reconcile, plus the projected cost we used for the
    cap check (handy when surfacing diagnostics).
    """

    row_id: int
    projected_cost_usd: float


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
        self.caps = caps if caps is not None else default_caps()

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
        """Check the caps; if OK, write a ``pre`` row and return its id.

        Order of checks: per-purpose soft cap *first* (cheaper to fail when
        a runaway is on a single purpose), then the global hard cap. Both
        check ``current_spend + projected_cost`` so we never spend a dollar
        and *then* notice we shouldn't have.
        """
        if self.caps.override_disable_caps:
            # Operator disabled enforcement — log it once, loudly, in the
            # row's notes so post-mortems can find every call that slipped
            # through.
            note = "override_disable_caps=True"
            notes = f"{notes}; {note}" if notes else note

        # Per-purpose soft cap.
        purpose_cap = self.caps.cap_for(purpose)
        if purpose_cap is not None and not self.caps.override_disable_caps:
            purpose_spend = self.ledger.weekly_spend(purpose=purpose)
            if purpose_spend + projected_cost_usd > purpose_cap:
                self._record_blocked(
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
                raise BudgetExhausted(
                    purpose=purpose,
                    weekly_spend_usd=purpose_spend,
                    weekly_cap_usd=purpose_cap,
                    projected_cost_usd=projected_cost_usd,
                    scope="purpose",
                )

        # Global weekly hard cap.
        if not self.caps.override_disable_caps:
            total_spend = self.ledger.weekly_spend()
            if total_spend + projected_cost_usd > self.caps.weekly_hard_cap_usd:
                self._record_blocked(
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
                raise BudgetExhausted(
                    purpose=purpose,
                    weekly_spend_usd=total_spend,
                    weekly_cap_usd=self.caps.weekly_hard_cap_usd,
                    projected_cost_usd=projected_cost_usd,
                    scope="weekly",
                )

        # All checks passed — write a pre row so a crash mid-call leaves a
        # trace, then return the id for post-flight reconciliation.
        row_id = self.ledger.record(
            endpoint=endpoint,
            model=model,
            purpose=purpose,
            phase="pre",
            usd_cost=projected_cost_usd,
            request_id=request_id,
            notes=notes,
        )
        return PreflightTicket(row_id=row_id, projected_cost_usd=projected_cost_usd)

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
        """Write a ``blocked`` row when a cap fires."""
        annotation = (
            f"blocked scope={scope} spend=${spend:.4f} cap=${cap:.4f} "
            f"projected=${projected_cost_usd:.4f}"
        )
        self.ledger.record(
            endpoint=endpoint,
            model=model,
            purpose=purpose,
            phase="blocked",
            usd_cost=0.0,  # blocked rows don't count toward spend
            request_id=request_id,
            notes=f"{notes}; {annotation}" if notes else annotation,
        )
