"""Pydantic shapes for the BUF-24 patch-intent extractor.

These are the *only* contract Claude is allowed to return. The extractor
parses ``response.content`` as JSON and validates it through
:class:`PatchIntentResult`; a malformed shape (extra field, out-of-range
score, missing key) raises ``pydantic.ValidationError`` so a model
regression surfaces as a typed test failure rather than a downstream
``KeyError``.

The fields mirror the Systems-spec output shape. Bounds on the 0..1
scores are enforced both here (application boundary) and in the
``patch_intent`` table's CHECK constraints (DB boundary) — defence in
depth so a buggy seed loader can't bypass either layer.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Direction tag on a per-agent / per-map pickrate prediction. ``flat`` is
# the "explicit no-change" signal; the model is asked to use it rather
# than omit an entry it considered, so a downstream "did the model think
# about Chamber?" audit is answerable.
PickrateDirection = Literal["up", "down", "flat"]

# Magnitude buckets. Keep this discrete: a continuous "expected delta"
# would be impressive-sounding and useless — there's no ground truth
# fine enough to score it against. Buckets match the spec's
# ``small`` / ``medium`` / ``large`` rubric.
PickrateMagnitude = Literal["small", "medium", "large"]


class ExpectedPickrateShift(BaseModel):
    """One per-subject pickrate prediction.

    ``subject`` is either an agent name (``"Chamber"``) or a map name
    (``"Bind"``) — the extractor doesn't enforce which kind, since some
    patches couple them ("Chamber's nerf shifts the Bind meta"). The
    consumer uses ``agents_affected`` / ``maps_affected`` on the parent
    result to disambiguate.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    subject: str = Field(min_length=1, max_length=64)
    direction: PickrateDirection
    magnitude: PickrateMagnitude
    # Optional one-line rationale. The parent result's ``reasoning``
    # carries the holistic justification; this is the per-shift
    # micro-justification ("ult cost +1 pushes him out of round 2 buys").
    rationale: str | None = Field(default=None, max_length=240)


class PatchIntentResult(BaseModel):
    """The full BUF-24 classification for one patch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Free-form short label of what Riot was trying to accomplish.
    # Intentionally not an enum — the extractor's value here ("nerf-meta-
    # outlier", "buff-underused-agent", "map-rotation-tweak", ...) is
    # better treated as a search facet than a typed field, and pinning
    # an enum at this layer would require a migration every time a new
    # category emerges.
    primary_intent: str = Field(min_length=1, max_length=64)
    # 0..1 estimate of how much pro-play feedback drove this patch
    # (vs. ranked-ladder data). Bounded both here and in the table's
    # CHECK so a model emitting 1.5 fails fast.
    pro_play_driven_score: float = Field(ge=0.0, le=1.0)
    # Free-form name lists. Validation against the canonical agent /
    # map registry happens downstream; this layer just enforces shape.
    agents_affected: list[str] = Field(default_factory=list)
    maps_affected: list[str] = Field(default_factory=list)
    # ``True`` if anything about the credit economy moved (kill rewards,
    # ability costs, weapon costs, plant/defuse rewards).
    econ_changed: bool
    expected_pickrate_shifts: list[ExpectedPickrateShift] = Field(default_factory=list)
    community_controversy_predicted: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    # The model's rationale for the classification — required so a
    # human spot-check can audit a surprising call.
    reasoning: str = Field(min_length=1)


__all__ = [
    "ExpectedPickrateShift",
    "PatchIntentResult",
    "PickrateDirection",
    "PickrateMagnitude",
]
