"""Per-model pricing for Claude API calls.

Prices are USD per **million** input/output tokens. Cache-write/read prices
are derived from the published Anthropic pricing rules:

* Cache write (5-minute TTL) = base input price × 1.25
* Cache write (1-hour TTL)   = base input price × 2.00
* Cache read                 = base input price × 0.10

Keep the table in this file as the single source of truth; the governor and
the post-flight cost calculator both read from it. When Anthropic publishes
new prices, update this table — no other code should hardcode rates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Default model for the Nexus pipeline. Per the claude-api skill: use Opus 4.7
# unless the caller explicitly picks something else. Cheap workloads opt down
# to Haiku 4.5 by setting ``model=`` on the call site.
DEFAULT_MODEL = "claude-opus-4-7"


# Cache pricing multipliers — published by Anthropic. Kept here as named
# constants instead of magic numbers so the migration story is greppable.
_CACHE_WRITE_5M_MULT = 1.25
_CACHE_WRITE_1H_MULT = 2.00
_CACHE_READ_MULT = 0.10


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens, broken out by category."""

    input_per_mtok: float
    output_per_mtok: float

    @property
    def cache_write_5m_per_mtok(self) -> float:
        return self.input_per_mtok * _CACHE_WRITE_5M_MULT

    @property
    def cache_write_1h_per_mtok(self) -> float:
        return self.input_per_mtok * _CACHE_WRITE_1H_MULT

    @property
    def cache_read_per_mtok(self) -> float:
        return self.input_per_mtok * _CACHE_READ_MULT


# Pricing table. Sourced from the Anthropic pricing page via the claude-api
# skill (cached 2026-04-15) plus the legacy Opus 4.5 pre-1M pricing for any
# legacy workloads still pinned to that alias.
PRICING: dict[str, ModelPricing] = {
    # Current models (1M context, post-Opus-4.6 pricing).
    "claude-opus-4-7": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-opus-4-6": ModelPricing(input_per_mtok=5.00, output_per_mtok=25.00),
    "claude-sonnet-4-6": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
    "claude-haiku-4-5": ModelPricing(input_per_mtok=1.00, output_per_mtok=5.00),
    # Legacy aliases that may still appear in pinned configs.
    "claude-opus-4-5": ModelPricing(input_per_mtok=15.00, output_per_mtok=75.00),
    "claude-sonnet-4-5": ModelPricing(input_per_mtok=3.00, output_per_mtok=15.00),
}


def get_pricing(model: str) -> ModelPricing:
    """Return pricing for *model*, raising if it is not in the table.

    We intentionally raise rather than fall back to a "guess" — silently
    using the wrong rate would defeat the budget governor's whole purpose.
    """
    try:
        return PRICING[model]
    except KeyError as e:  # pragma: no cover - exercised by tests
        raise KeyError(
            f"No pricing entry for model {model!r}; add it to "
            "esports_sim.budget.pricing.PRICING."
        ) from e


def cost_from_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    cache_write_ttl: str = "5m",
) -> float:
    """Compute USD cost from an Anthropic ``usage`` block.

    Mirrors the wire format of ``Message.usage``: ``input_tokens`` is the
    *uncached* prompt, while ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` carry the cached portions. Adding them naively
    would double-count, so we keep the categories distinct and price each.
    """
    p = get_pricing(model)
    write_rate = p.cache_write_1h_per_mtok if cache_write_ttl == "1h" else p.cache_write_5m_per_mtok
    cost = (
        input_tokens * p.input_per_mtok
        + output_tokens * p.output_per_mtok
        + cache_creation_input_tokens * write_rate
        + cache_read_input_tokens * p.cache_read_per_mtok
    ) / 1_000_000
    return cost


def estimate_cost(
    *,
    model: str,
    input_tokens: int,
    max_output_tokens: int,
    has_cache_control: bool = False,
    cache_write_ttl: str = "5m",
) -> float:
    """Worst-case pre-flight estimate: assume the model writes ``max_tokens``.

    The post-call ``cost_from_usage`` reconciles to the actual usage. Pre-flight
    estimates are intentionally pessimistic so the governor blocks before
    spending — a small over-estimate is fine, an under-estimate is not.

    Cache pricing: when ``has_cache_control`` is True, input tokens are priced
    at the cache-write rate (1.25× base for 5m TTL, 2× for 1h). At
    pre-flight we don't know how the prompt will split between cache-write
    and cache-read tokens — pricing the whole input as a write is
    pessimistic but safe. Underestimating here would let calls slip past
    the cap and post-flight reconciliation would push us over, defeating
    the entire point of a hard rate limiter.
    """
    p = get_pricing(model)
    if has_cache_control:
        input_rate = (
            p.cache_write_1h_per_mtok if cache_write_ttl == "1h" else p.cache_write_5m_per_mtok
        )
    else:
        input_rate = p.input_per_mtok
    return (input_tokens * input_rate + max_output_tokens * p.output_per_mtok) / 1_000_000


def cost_from_usage_obj(model: str, usage: Any, *, cache_write_ttl: str = "5m") -> float:
    """Convenience wrapper accepting an Anthropic ``Message.usage`` object.

    The SDK exposes the fields as attributes, but with optional ones defaulting
    to ``None`` rather than ``0``. Normalise to ints here so callers don't
    have to.
    """
    return cost_from_usage(
        model=model,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        cache_creation_input_tokens=int(getattr(usage, "cache_creation_input_tokens", 0) or 0),
        cache_read_input_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0),
        cache_write_ttl=cache_write_ttl,
    )
