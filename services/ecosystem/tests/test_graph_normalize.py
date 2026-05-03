"""Per-normaliser unit tests (BUF-53).

These are deliberately small and exhaustive — the builder asks only
for ``normalize_column``, but a regression in any single normaliser
would silently corrupt every era's snapshot.
"""

from __future__ import annotations

import numpy as np
import pytest
from ecosystem.graph.normalize import (
    NormalizerError,
    get_normalizer,
    normalize_column,
)

# ---- minmax ---------------------------------------------------------------


def test_minmax_maps_endpoints_to_zero_and_one() -> None:
    out, fit = normalize_column(np.array([1.0, 5.0, 9.0]), normalizer_name="minmax")
    assert out.dtype == np.float32
    assert pytest.approx(out.tolist()) == [0.0, 0.5, 1.0]
    assert fit.params["min"] == 1.0
    assert fit.params["max"] == 9.0


def test_minmax_degenerate_returns_uniform_half() -> None:
    """All-equal column is well-defined: middle of [0, 1]."""
    out, fit = normalize_column(np.array([3.0, 3.0, 3.0]), normalizer_name="minmax")
    assert pytest.approx(out.tolist()) == [0.5, 0.5, 0.5]
    assert fit.params["degenerate"] == 1.0


def test_minmax_clips_unseen_outliers() -> None:
    """Transform-time values outside the fitted range are clipped, not extrapolated."""
    norm = get_normalizer("minmax")
    fit = norm.fit(np.array([0.0, 10.0]))
    out = norm.transform(np.array([-5.0, 15.0]), fit)
    assert out.tolist() == [0.0, 1.0]


# ---- zscore ---------------------------------------------------------------


def test_zscore_centers_to_half_via_logistic() -> None:
    out, _ = normalize_column(np.array([0.0, 0.0, 0.0]), normalizer_name="zscore")
    # Degenerate (zero std) — falls back to 0.5.
    assert pytest.approx(out.tolist()) == [0.5, 0.5, 0.5]


def test_zscore_squashes_to_unit_interval() -> None:
    """Even extreme inputs squash through logistic to (0, 1)."""
    out, _ = normalize_column(
        np.array([-100.0, -1.0, 0.0, 1.0, 100.0]),
        normalizer_name="zscore",
    )
    assert (out >= 0.0).all() and (out <= 1.0).all()
    # Symmetric inputs should bracket 0.5.
    assert out[0] < out[1] < out[2] < out[3] < out[4]


# ---- log1p_minmax ---------------------------------------------------------


def test_log1p_minmax_handles_long_tail() -> None:
    # log1p flattens the tail so the mid-value isn't 0.5.
    out, _ = normalize_column(
        np.array([0.0, 1.0, 100.0, 10000.0]),
        normalizer_name="log1p_minmax",
    )
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)
    # The "1.0" point should be much closer to 0 than to 1.
    assert out[1] < 0.2


def test_log1p_minmax_clamps_negatives_silently() -> None:
    out, _ = normalize_column(
        np.array([-5.0, 0.0, 10.0]),
        normalizer_name="log1p_minmax",
    )
    # Negative becomes 0.0 (post-clamp == post-log1p baseline).
    assert out[0] == pytest.approx(out[1])


# ---- passthrough ----------------------------------------------------------


def test_passthrough_accepts_unit_interval() -> None:
    out, _ = normalize_column(np.array([0.0, 0.5, 1.0]), normalizer_name="passthrough")
    assert pytest.approx(out.tolist()) == [0.0, 0.5, 1.0]


def test_passthrough_rejects_out_of_range_values() -> None:
    with pytest.raises(NormalizerError):
        normalize_column(np.array([0.0, 0.5, 1.5]), normalizer_name="passthrough")


# ---- registry --------------------------------------------------------------


def test_get_normalizer_unknown_raises() -> None:
    with pytest.raises(NormalizerError):
        get_normalizer("not_a_real_normalizer")


# ---- nan handling ---------------------------------------------------------


def test_nan_inputs_are_ignored_in_fit() -> None:
    """NaN values shouldn't shift the fit; downstream fill_policy handles missing."""
    out, fit = normalize_column(
        np.array([1.0, np.nan, 9.0]), normalizer_name="minmax"
    )
    # The two finite values bound the fit. The NaN passes through
    # arithmetic untouched here — the *builder* applies fill_policy on
    # top of normalize_column's output (see test_graph_builder.py).
    assert fit.params["min"] == 1.0
    assert fit.params["max"] == 9.0
    assert np.isnan(out[1])
