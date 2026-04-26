"""Spend caps for the Claude API governor.

The hard cap is enforced (over → :class:`BudgetExhausted`); soft caps are
per-purpose limits enforced the same way. The ratio between them gives us
the buffer the issue calls out: $40/wk effective ceiling, $30/wk hard cap
(blocks before the bill arrives), $10/wk reserve for emergencies that need
a manual override.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Hard cap: when weekly spend exceeds this, every Claude call raises
# BudgetExhausted regardless of purpose. Picked to leave a $10/wk buffer
# under the $40 effective weekly budget per BUF-22.
DEFAULT_WEEKLY_HARD_CAP_USD = 30.00

# Per-purpose soft caps. Values here come from BUF-22's "Per-purpose soft
# caps" examples; everything not listed is unrestricted (bounded only by
# the hard cap above).
DEFAULT_PURPOSE_CAPS_USD: dict[str, float] = {
    "personality": 15.00,
    "patch_intent": 3.00,
}


@dataclass(frozen=True)
class BudgetCaps:
    """Snapshot of the active budget configuration.

    Frozen so tests can pass a deterministic config without worrying about
    later mutation. ``override`` is read on every Governor call so an
    operator can break the glass via env var without code changes.
    """

    weekly_hard_cap_usd: float = DEFAULT_WEEKLY_HARD_CAP_USD
    purpose_caps_usd: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_PURPOSE_CAPS_USD)
    )
    # When True, the governor logs every call but never raises. Used in
    # narrow break-glass scenarios — Anthropic console says we're fine,
    # local ledger is over due to a bug. Drains are still recorded.
    override_disable_caps: bool = False

    @classmethod
    def from_env(cls) -> BudgetCaps:
        """Build caps from process environment variables.

        Recognised variables (all optional):

        * ``NEXUS_BUDGET_WEEKLY_HARD_CAP_USD`` — float, replaces the default
          $30 cap.
        * ``NEXUS_BUDGET_DISABLE_CAPS`` — set to a truthy value (``1``,
          ``true``, ``yes``) to disable enforcement. Use sparingly.
        """
        cap = float(
            os.getenv(
                "NEXUS_BUDGET_WEEKLY_HARD_CAP_USD",
                str(DEFAULT_WEEKLY_HARD_CAP_USD),
            )
        )
        disable = os.getenv("NEXUS_BUDGET_DISABLE_CAPS", "").lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            weekly_hard_cap_usd=cap,
            purpose_caps_usd=dict(DEFAULT_PURPOSE_CAPS_USD),
            override_disable_caps=disable,
        )

    def cap_for(self, purpose: str) -> float | None:
        """Return the soft cap for *purpose*, or ``None`` if unset."""
        return self.purpose_caps_usd.get(purpose)


def default_caps() -> BudgetCaps:
    """Lazy default — equivalent to ``BudgetCaps()`` but explicit at call sites."""
    return BudgetCaps()
