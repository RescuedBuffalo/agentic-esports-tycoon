"""Tests for ``esports_sim.budget.caps`` — env-driven cap configuration (BUF-22).

The governor reads ``BudgetCaps`` on every call, including the
break-glass override flag. Lock down the env-parsing behaviour so a
typo in the env name (or a regression in the truthy-string set)
doesn't silently disable enforcement.
"""

from __future__ import annotations

import pytest
from esports_sim.budget.caps import (
    DEFAULT_PURPOSE_CAPS_USD,
    DEFAULT_WEEKLY_HARD_CAP_USD,
    BudgetCaps,
    default_caps,
)

# --- defaults -------------------------------------------------------------


def test_default_constructor_uses_module_constants() -> None:
    caps = BudgetCaps()
    assert caps.weekly_hard_cap_usd == DEFAULT_WEEKLY_HARD_CAP_USD
    # The default purpose-caps dict is *copied*, not aliased — so a
    # mutation on one ``BudgetCaps`` instance can't bleed into others.
    assert caps.purpose_caps_usd == DEFAULT_PURPOSE_CAPS_USD
    assert caps.purpose_caps_usd is not DEFAULT_PURPOSE_CAPS_USD
    assert caps.override_disable_caps is False


def test_default_caps_helper_matches_default_constructor() -> None:
    assert default_caps() == BudgetCaps()


def test_per_instance_purpose_caps_are_independent() -> None:
    """Mutating one instance's purpose_caps_usd must not leak into another.

    The dataclass uses ``default_factory=lambda: dict(DEFAULT_...)`` to
    avoid the classic shared-default-dict footgun.
    """
    a = BudgetCaps()
    a.purpose_caps_usd["personality"] = 999.0
    b = BudgetCaps()
    assert b.purpose_caps_usd["personality"] == DEFAULT_PURPOSE_CAPS_USD["personality"]


def test_caps_is_frozen() -> None:
    """Frozen so callers can't accidentally mutate the configured cap.

    The class is ``@dataclass(frozen=True)``; reassignment must raise.
    """
    caps = BudgetCaps()
    with pytest.raises(Exception):  # noqa: B017 - dataclasses raises FrozenInstanceError
        caps.weekly_hard_cap_usd = 99.0  # type: ignore[misc]


# --- cap_for() lookup -----------------------------------------------------


def test_cap_for_returns_configured_value() -> None:
    caps = BudgetCaps()
    assert caps.cap_for("personality") == DEFAULT_PURPOSE_CAPS_USD["personality"]


def test_cap_for_unknown_purpose_returns_none() -> None:
    """Unknown purposes have no soft cap and are bounded only by the hard cap.

    ``None`` is the documented "no soft cap" signal — the governor
    treats it as "skip the per-purpose check, fall through to the
    weekly hard cap".
    """
    assert BudgetCaps().cap_for("not-a-real-purpose") is None


# --- from_env() parsing ---------------------------------------------------


def test_from_env_with_no_vars_matches_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", raising=False)
    monkeypatch.delenv("NEXUS_BUDGET_DISABLE_CAPS", raising=False)
    caps = BudgetCaps.from_env()
    assert caps.weekly_hard_cap_usd == DEFAULT_WEEKLY_HARD_CAP_USD
    assert caps.override_disable_caps is False


def test_from_env_overrides_weekly_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", "12.50")
    monkeypatch.delenv("NEXUS_BUDGET_DISABLE_CAPS", raising=False)
    caps = BudgetCaps.from_env()
    assert caps.weekly_hard_cap_usd == pytest.approx(12.50)


def test_from_env_propagates_invalid_float(monkeypatch: pytest.MonkeyPatch) -> None:
    """A garbled env value must fail loud, not silently fall back to the default.

    Prefer a noisy startup error over a quiet "looks fine" config that
    invisibly halves the cap.
    """
    monkeypatch.setenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", "not-a-number")
    with pytest.raises(ValueError):
        BudgetCaps.from_env()


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES", "Yes"])
def test_from_env_truthy_values_disable_caps(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Recognised truthy strings flip the override flag, regardless of case."""
    monkeypatch.setenv("NEXUS_BUDGET_DISABLE_CAPS", value)
    monkeypatch.delenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", raising=False)
    assert BudgetCaps.from_env().override_disable_caps is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "disabled", "FALSE"])
def test_from_env_non_truthy_values_keep_caps_enforced(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Anything outside the canonical truthy set leaves enforcement on.

    Critical for safety: a misspelled "ture" must NOT disable enforcement.
    """
    monkeypatch.setenv("NEXUS_BUDGET_DISABLE_CAPS", value)
    monkeypatch.delenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", raising=False)
    assert BudgetCaps.from_env().override_disable_caps is False


def test_from_env_preserves_purpose_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-purpose soft caps come from the module default; from_env doesn't drop them."""
    monkeypatch.setenv("NEXUS_BUDGET_WEEKLY_HARD_CAP_USD", "100.0")
    caps = BudgetCaps.from_env()
    assert caps.purpose_caps_usd == DEFAULT_PURPOSE_CAPS_USD
