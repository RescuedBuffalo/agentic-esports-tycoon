"""Per-column normalisers for the graph builder.

Acceptance: "Feature matrices are fully normalized (no raw un-scaled
columns)." Every column declared in :mod:`ecosystem.graph.schema`
names a normaliser; the builder looks the name up here and applies it.
A column with normaliser ``"passthrough"`` is *not* a raw escape
hatch — it asserts the value is already in the unit interval (typical
for one-hot / boolean indicators) and clips defensively.

Normalisers are stateful per (node_type, column): they fit on the
era's data, then transform. Fit parameters land on the snapshot's
``metadata`` block so a held-out evaluation row can be transformed
later without re-running the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


class NormalizerError(ValueError):
    """Raised when a normaliser is asked to do something impossible."""


# Tolerance for "is this value in [0, 1]?" — passthrough columns get
# clipped to the unit interval if they're within this slack, so a
# float-comparison rounding error doesn't fail validation.
_UNIT_SLACK = 1e-6


@dataclass(slots=True)
class FitParams:
    """The per-column scalar parameters a normaliser learned during fit.

    Stored on the snapshot manifest. ``kind`` is the normaliser name
    so a future loader can re-instantiate without guessing.
    """

    kind: str
    params: dict[str, float] = field(default_factory=dict)

    def to_jsonable(self) -> dict[str, Any]:
        return {"kind": self.kind, "params": dict(self.params)}


class _Normalizer:
    """Internal base class. Each subclass implements ``fit`` + ``transform``.

    Stateless API for the builder: a fresh instance per (node_type,
    column) on every build. The fit / transform separation is what
    makes era-relative normalisation deterministic — fit only sees the
    era's column, transform only sees the era's column.
    """

    name: str

    def fit(self, values: np.ndarray) -> FitParams:  # pragma: no cover - abstract
        raise NotImplementedError

    def transform(  # pragma: no cover - abstract
        self, values: np.ndarray, params: FitParams
    ) -> np.ndarray:
        raise NotImplementedError


class _MinMax(_Normalizer):
    """Linear rescale to ``[0, 1]`` against the era's min/max.

    Degenerate column (all-equal values, or empty input) falls back to
    a uniform 0.5 — preserves the "no raw column" invariant without
    introducing NaNs from a zero-range divide.
    """

    name = "minmax"

    def fit(self, values: np.ndarray) -> FitParams:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return FitParams(self.name, {"min": 0.0, "max": 1.0, "degenerate": 1.0})
        lo = float(finite.min())
        hi = float(finite.max())
        degenerate = 1.0 if hi <= lo else 0.0
        return FitParams(self.name, {"min": lo, "max": hi, "degenerate": degenerate})

    def transform(self, values: np.ndarray, params: FitParams) -> np.ndarray:
        if params.params.get("degenerate", 0.0) >= 1.0:
            return np.full_like(values, 0.5, dtype=np.float64)
        lo = params.params["min"]
        hi = params.params["max"]
        out = (values - lo) / (hi - lo)
        return np.clip(out, 0.0, 1.0)


class _ZScore(_Normalizer):
    """Standardise to mean 0, std 1, then squash through a logistic.

    The logistic squash keeps the output bounded in ``(0, 1)`` so the
    "no raw" invariant holds; the underlying z-score lives in the
    fit params for callers who need to invert.
    """

    name = "zscore"

    def fit(self, values: np.ndarray) -> FitParams:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return FitParams(self.name, {"mean": 0.0, "std": 1.0, "degenerate": 1.0})
        mean = float(finite.mean())
        std = float(finite.std())
        # ``ddof=0`` (population std) — we're describing the era's
        # population, not estimating a sample.
        if std < 1e-9:
            return FitParams(self.name, {"mean": mean, "std": 1.0, "degenerate": 1.0})
        return FitParams(self.name, {"mean": mean, "std": std, "degenerate": 0.0})

    def transform(self, values: np.ndarray, params: FitParams) -> np.ndarray:
        if params.params.get("degenerate", 0.0) >= 1.0:
            return np.full_like(values, 0.5, dtype=np.float64)
        z = (values - params.params["mean"]) / params.params["std"]
        # Stable logistic — clip the input so large |z| doesn't
        # under/overflow exp() in fp64.
        z = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))


class _Log1pMinMax(_Normalizer):
    """``log1p`` then min-max. Use for long-tailed positive counts.

    Negative inputs are clamped to 0 before log1p — long-tailed counts
    are non-negative by construction, so a negative is a data bug we
    quietly correct rather than NaN-propagate.
    """

    name = "log1p_minmax"

    def fit(self, values: np.ndarray) -> FitParams:
        finite = values[np.isfinite(values)]
        clamped = np.clip(finite, 0.0, None)
        if clamped.size == 0:
            return FitParams(self.name, {"min": 0.0, "max": 1.0, "degenerate": 1.0})
        logged = np.log1p(clamped)
        lo = float(logged.min())
        hi = float(logged.max())
        return FitParams(
            self.name,
            {"min": lo, "max": hi, "degenerate": 1.0 if hi <= lo else 0.0},
        )

    def transform(self, values: np.ndarray, params: FitParams) -> np.ndarray:
        if params.params.get("degenerate", 0.0) >= 1.0:
            return np.full_like(values, 0.5, dtype=np.float64)
        clamped = np.clip(values, 0.0, None)
        logged = np.log1p(clamped)
        out = (logged - params.params["min"]) / (params.params["max"] - params.params["min"])
        return np.asarray(np.clip(out, 0.0, 1.0))


class _Passthrough(_Normalizer):
    """Assert + clip to ``[0, 1]``. For one-hots and pre-scaled inputs.

    Slightly out-of-range inputs (within :data:`_UNIT_SLACK`) are
    silently clipped; anything farther out raises so a schema bug
    doesn't sneak through.
    """

    name = "passthrough"

    def fit(self, values: np.ndarray) -> FitParams:
        finite = values[np.isfinite(values)]
        if finite.size and (
            (finite < -_UNIT_SLACK).any() or (finite > 1.0 + _UNIT_SLACK).any()
        ):
            raise NormalizerError(
                f"passthrough column got values outside [0, 1]: "
                f"min={float(finite.min())}, max={float(finite.max())}. "
                f"Either fix the source or pick a real normaliser."
            )
        return FitParams(self.name, {})

    def transform(self, values: np.ndarray, params: FitParams) -> np.ndarray:
        return np.asarray(np.clip(values, 0.0, 1.0))


_NORMALIZERS: dict[str, _Normalizer] = {
    n.name: n
    for n in (_MinMax(), _ZScore(), _Log1pMinMax(), _Passthrough())
}


def get_normalizer(name: str) -> _Normalizer:
    try:
        return _NORMALIZERS[name]
    except KeyError as e:
        raise NormalizerError(
            f"unknown normaliser {name!r}; known: {sorted(_NORMALIZERS)}"
        ) from e


def normalize_column(
    values: np.ndarray,
    *,
    normalizer_name: str,
) -> tuple[np.ndarray, FitParams]:
    """Fit then transform a single column. Returns (out, fit_params).

    The fit params are returned alongside the values so the builder
    can stuff them in the snapshot's metadata; round-tripping a
    held-out row through the same parameters is the only safe way to
    score a player who didn't appear in the training era.
    """
    norm = get_normalizer(normalizer_name)
    fit = norm.fit(values.astype(np.float64, copy=False))
    out = norm.transform(values.astype(np.float64, copy=False), fit)
    # Force fp32 on the way out — matches GraphSnapshot's storage
    # dtype, so the builder doesn't have to re-cast.
    return out.astype(np.float32, copy=False), fit


__all__ = [
    "FitParams",
    "NormalizerError",
    "get_normalizer",
    "normalize_column",
]
