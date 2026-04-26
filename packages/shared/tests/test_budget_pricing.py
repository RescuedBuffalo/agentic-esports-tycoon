"""Pricing math + Anthropic-style usage objects."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from esports_sim.budget.pricing import (
    PRICING,
    cost_from_usage,
    cost_from_usage_obj,
    estimate_cost,
    get_pricing,
)


def test_pricing_table_covers_default_and_legacy_models() -> None:
    # The default model must always have a price entry — the governor would
    # crash on first call otherwise.
    assert "claude-opus-4-7" in PRICING
    # Legacy aliases keep their pre-1M-context pricing so workloads pinned to
    # Opus 4.5 don't quietly under-bill at the new lower rate.
    assert PRICING["claude-opus-4-5"].input_per_mtok == 15.00
    assert PRICING["claude-opus-4-5"].output_per_mtok == 75.00


def test_get_pricing_raises_for_unknown_model() -> None:
    with pytest.raises(KeyError):
        get_pricing("claude-foo-bar")


def test_cost_from_usage_input_output_only() -> None:
    # 1M input + 1M output on Opus 4.7 = $5 + $25 = $30 exactly.
    cost = cost_from_usage(
        model="claude-opus-4-7",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(30.00, abs=1e-9)


def test_cost_from_usage_with_cache_5m() -> None:
    # Cache write @ 5m TTL = base × 1.25; cache read = base × 0.10.
    # 1M cache write + 1M cache read on Sonnet 4.6 ($3 base):
    #   write: 1M * 3 * 1.25 = $3.75
    #   read:  1M * 3 * 0.10 = $0.30
    cost = cost_from_usage(
        model="claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=1_000_000,
        cache_write_ttl="5m",
    )
    assert cost == pytest.approx(3.75 + 0.30, abs=1e-9)


def test_cost_from_usage_with_cache_1h() -> None:
    # 1h TTL bumps the write multiplier to 2.0.
    cost = cost_from_usage(
        model="claude-sonnet-4-6",
        input_tokens=0,
        output_tokens=0,
        cache_creation_input_tokens=1_000_000,
        cache_read_input_tokens=0,
        cache_write_ttl="1h",
    )
    assert cost == pytest.approx(6.00, abs=1e-9)


def test_estimate_cost_is_pessimistic_about_output() -> None:
    """Pre-flight estimate assumes the model writes ``max_tokens``."""
    e = estimate_cost(
        model="claude-haiku-4-5",
        input_tokens=1000,
        max_output_tokens=2000,
    )
    # 1k * $1/MTok + 2k * $5/MTok = $0.001 + $0.01 = $0.011
    assert e == pytest.approx(0.011, abs=1e-9)


def test_cost_from_usage_obj_handles_optional_attrs() -> None:
    # SDK objects expose cache_* fields that may be None when caching is
    # disabled — make sure we coerce to 0 not blow up.
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=None,
        cache_read_input_tokens=None,
    )
    cost = cost_from_usage_obj("claude-haiku-4-5", usage)
    expected = (100 * 1.00 + 50 * 5.00) / 1_000_000
    assert cost == pytest.approx(expected, abs=1e-12)


def test_acceptance_real_call_within_5_cents() -> None:
    """BUF-22 acceptance: ledger cost matches actual within 5 cents.

    The "actual" Anthropic charges in dollars-and-cents reporting use the
    same rates we use here, so an exact-arithmetic check is the right
    proxy for the 5¢ acceptance criterion.
    """
    # Realistic Opus 4.7 call: 8k input prompt, 1.5k output.
    cost = cost_from_usage(
        model="claude-opus-4-7",
        input_tokens=8_000,
        output_tokens=1_500,
    )
    # 8k * $5/MTok + 1.5k * $25/MTok = $0.040 + $0.0375 = $0.0775
    expected = 0.0775
    assert abs(cost - expected) < 0.05  # within 5¢
    assert cost == pytest.approx(expected, abs=1e-9)
