"""Claude API budget ledger + governor (BUF-22, ADR-005).

Every Claude call goes through :func:`claude_call`. The governor checks the
weekly hard cap and any per-purpose soft cap before the request leaves the
process; logs a row to the ledger before *and* after the call (so a crash
mid-call still leaves an audit trail); and surfaces token usage as USD against
the configured pricing table.

Storage: SQLite in WAL mode (per ADR-006). Single-writer append-only — every
process attaches the same file via ``NEXUS_BUDGET_DB``.

Usage::

    from esports_sim.budget import Governor, claude_call, BudgetExhausted
    from esports_sim.budget.caps import default_caps

    gov = Governor(caps=default_caps())
    response = claude_call(
        governor=gov,
        purpose="patch_intent",
        messages=[{"role": "user", "content": "..."}],
        # Anything else messages.create() accepts:
        max_tokens=4096,
        system=[{"type": "text", "text": large_corpus,
                 "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
    )
"""

from esports_sim.budget.caps import BudgetCaps, default_caps
from esports_sim.budget.client import claude_call, claude_stream
from esports_sim.budget.errors import BudgetExhausted
from esports_sim.budget.governor import Governor
from esports_sim.budget.ledger import Ledger, LedgerEntry
from esports_sim.budget.pricing import (
    DEFAULT_MODEL,
    PRICING,
    ModelPricing,
    cost_from_usage,
)
from esports_sim.budget.report import (
    DigestSummary,
    PurposeSummary,
    format_digest,
    summarize_window,
)

__all__ = [
    "BudgetCaps",
    "BudgetExhausted",
    "DEFAULT_MODEL",
    "DigestSummary",
    "Governor",
    "Ledger",
    "LedgerEntry",
    "ModelPricing",
    "PRICING",
    "PurposeSummary",
    "claude_call",
    "claude_stream",
    "cost_from_usage",
    "default_caps",
    "format_digest",
    "summarize_window",
]
