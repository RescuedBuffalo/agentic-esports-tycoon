"""Budget-related exceptions."""

from __future__ import annotations


class BudgetExhausted(RuntimeError):
    """Raised when a Claude call would push spend past the hard cap.

    Carries enough context that the caller can surface a clear error
    (or, for batch jobs, log + skip the row) without re-querying the
    ledger.
    """

    def __init__(
        self,
        *,
        purpose: str,
        weekly_spend_usd: float,
        weekly_cap_usd: float,
        projected_cost_usd: float,
        scope: str = "weekly",
    ) -> None:
        self.purpose = purpose
        self.weekly_spend_usd = weekly_spend_usd
        self.weekly_cap_usd = weekly_cap_usd
        self.projected_cost_usd = projected_cost_usd
        # ``scope`` distinguishes the global weekly cap from per-purpose soft
        # caps so callers can decide which alarm to fire.
        self.scope = scope
        super().__init__(
            f"BudgetExhausted ({scope}): purpose={purpose!r} "
            f"spend=${weekly_spend_usd:.2f} cap=${weekly_cap_usd:.2f} "
            f"projected_call=${projected_cost_usd:.4f}"
        )
