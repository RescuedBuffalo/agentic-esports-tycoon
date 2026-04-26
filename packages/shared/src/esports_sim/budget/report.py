"""Daily/weekly digest of Claude API spend.

Two surfaces:

* :func:`summarize_window` — pure data; tests assert on the summary fields.
* :func:`format_digest` — turns a summary into the human-readable text the
  CLI / cron job prints.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from esports_sim.budget.caps import BudgetCaps, default_caps
from esports_sim.budget.ledger import Ledger


@dataclass(frozen=True)
class PurposeSummary:
    purpose: str
    spend_usd: float
    cap_usd: float | None
    calls: int
    blocked: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int

    @property
    def percent_of_cap(self) -> float | None:
        if self.cap_usd is None or self.cap_usd <= 0:
            return None
        return (self.spend_usd / self.cap_usd) * 100.0


@dataclass(frozen=True)
class DigestSummary:
    """Aggregate view over a rolling window."""

    window_start: datetime
    window_end: datetime
    weekly_spend_usd: float
    weekly_cap_usd: float
    today_spend_usd: float
    by_purpose: list[PurposeSummary] = field(default_factory=list)
    by_model: dict[str, float] = field(default_factory=dict)
    blocked_count: int = 0
    error_count: int = 0

    @property
    def percent_of_weekly_cap(self) -> float:
        if self.weekly_cap_usd <= 0:
            return 0.0
        return (self.weekly_spend_usd / self.weekly_cap_usd) * 100.0


def summarize_window(
    ledger: Ledger,
    *,
    caps: BudgetCaps | None = None,
    now: datetime | None = None,
    window: timedelta = timedelta(days=7),
) -> DigestSummary:
    """Build a :class:`DigestSummary` from the rows newer than ``now-window``."""
    caps = caps or default_caps()
    end = (now or datetime.now(UTC)).astimezone(UTC)
    start = end - window
    today_start = end.replace(hour=0, minute=0, second=0, microsecond=0)

    entries = ledger.entries_since(start)

    weekly_spend = 0.0
    today_spend = 0.0
    by_purpose_spend: dict[str, float] = defaultdict(float)
    by_purpose_calls: dict[str, int] = defaultdict(int)
    by_purpose_blocked: dict[str, int] = defaultdict(int)
    by_purpose_in: dict[str, int] = defaultdict(int)
    by_purpose_out: dict[str, int] = defaultdict(int)
    by_purpose_cache_read: dict[str, int] = defaultdict(int)
    by_model: dict[str, float] = defaultdict(float)
    blocked_count = 0
    error_count = 0

    for e in entries:
        # Blocked rows don't count toward spend (governor sets usd_cost=0)
        # but we still count them so reports surface gating.
        if e.phase == "blocked":
            blocked_count += 1
            by_purpose_blocked[e.purpose] += 1
            continue
        if e.phase == "error":
            error_count += 1
            # error rows carry the projected cost — count them so the cap
            # math doesn't silently under-attribute the failure.
        weekly_spend += e.usd_cost
        if e.timestamp >= today_start:
            today_spend += e.usd_cost
        by_purpose_spend[e.purpose] += e.usd_cost
        by_purpose_calls[e.purpose] += 1
        by_purpose_in[e.purpose] += e.input_tokens
        by_purpose_out[e.purpose] += e.output_tokens
        by_purpose_cache_read[e.purpose] += e.cache_read_input_tokens
        by_model[e.model] += e.usd_cost

    purposes = sorted(by_purpose_spend.keys() | by_purpose_blocked.keys())
    by_purpose = [
        PurposeSummary(
            purpose=p,
            spend_usd=by_purpose_spend[p],
            cap_usd=caps.cap_for(p),
            calls=by_purpose_calls[p],
            blocked=by_purpose_blocked[p],
            input_tokens=by_purpose_in[p],
            output_tokens=by_purpose_out[p],
            cache_read_tokens=by_purpose_cache_read[p],
        )
        for p in purposes
    ]
    # Largest spenders first.
    by_purpose.sort(key=lambda s: s.spend_usd, reverse=True)

    return DigestSummary(
        window_start=start,
        window_end=end,
        weekly_spend_usd=weekly_spend,
        weekly_cap_usd=caps.weekly_hard_cap_usd,
        today_spend_usd=today_spend,
        by_purpose=by_purpose,
        by_model=dict(by_model),
        blocked_count=blocked_count,
        error_count=error_count,
    )


def format_digest(summary: DigestSummary) -> str:
    """Render a :class:`DigestSummary` as plain text for the CLI / email."""
    lines: list[str] = []
    pct = summary.percent_of_weekly_cap
    lines.append(
        f"Last 7 days: ${summary.weekly_spend_usd:.2f} / "
        f"${summary.weekly_cap_usd:.2f} ({pct:.0f}%)"
    )
    lines.append(f"Today:       ${summary.today_spend_usd:.2f}")
    if summary.blocked_count:
        lines.append(f"Blocked calls: {summary.blocked_count}")
    if summary.error_count:
        lines.append(f"Errored calls: {summary.error_count}")

    if summary.by_purpose:
        lines.append("")
        lines.append("By purpose:")
        for ps in summary.by_purpose:
            cap = (
                f"cap ${ps.cap_usd:.2f}, {ps.percent_of_cap:.0f}%"
                if ps.cap_usd is not None and ps.percent_of_cap is not None
                else "no cap"
            )
            extras: list[str] = []
            if ps.blocked:
                extras.append(f"{ps.blocked} blocked")
            extras_str = f" [{', '.join(extras)}]" if extras else ""
            lines.append(
                f"  {ps.purpose:<16} ${ps.spend_usd:>7.2f}  "
                f"({cap}, {ps.calls} calls){extras_str}"
            )

    if summary.by_model:
        lines.append("")
        lines.append("By model:")
        for model, cost in sorted(summary.by_model.items(), key=lambda kv: kv[1], reverse=True):
            lines.append(f"  {model:<28} ${cost:>7.2f}")

    return "\n".join(lines)
