"""Pydantic-shape tests for ``esports_sim.patch_intent.schema``.

Covers the contract :class:`PatchIntentResult` enforces against Claude's
output: required fields, bounded floats, frozen instances, and the
shape of :class:`ExpectedPickrateShift`. Drift in any of these surfaces
here as a typed ``ValidationError`` rather than as a runtime crash in
the persistence layer.
"""

from __future__ import annotations

import pytest
from esports_sim.patch_intent import (
    ExpectedPickrateShift,
    PatchIntentResult,
)
from pydantic import ValidationError


def _valid_payload() -> dict:
    """One canonical valid response — base for negative tests below."""
    return {
        "primary_intent": "nerf-meta-outlier",
        "pro_play_driven_score": 0.9,
        "agents_affected": ["Chamber"],
        "maps_affected": [],
        "econ_changed": False,
        "expected_pickrate_shifts": [
            {
                "subject": "Chamber",
                "direction": "down",
                "magnitude": "large",
                "rationale": "Trademark rework + ult cost +1",
            }
        ],
        "community_controversy_predicted": 0.85,
        "confidence": 0.8,
        "reasoning": "Classic 5.12 Chamber nerf — pro play driven.",
    }


def test_valid_payload_round_trips() -> None:
    result = PatchIntentResult.model_validate(_valid_payload())
    assert result.primary_intent == "nerf-meta-outlier"
    assert result.expected_pickrate_shifts[0].subject == "Chamber"
    assert result.expected_pickrate_shifts[0].direction == "down"
    assert result.expected_pickrate_shifts[0].magnitude == "large"


def test_score_above_one_rejected() -> None:
    """0..1 bound on ``pro_play_driven_score`` — model emits 1.5 → fail fast."""
    payload = _valid_payload()
    payload["pro_play_driven_score"] = 1.5
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_negative_confidence_rejected() -> None:
    payload = _valid_payload()
    payload["confidence"] = -0.1
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_extra_field_rejected() -> None:
    """``extra="forbid"`` — a model emitting a field outside the schema is
    a contract regression we want to surface, not silently swallow."""
    payload = _valid_payload()
    payload["surprise_field"] = "wat"
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_missing_required_field_rejected() -> None:
    payload = _valid_payload()
    del payload["primary_intent"]
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_empty_reasoning_rejected() -> None:
    """``reasoning`` is min_length=1 — empty justification defeats the audit."""
    payload = _valid_payload()
    payload["reasoning"] = ""
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_invalid_pickrate_direction_rejected() -> None:
    """``direction`` is a Literal — typo'd value fails validation."""
    payload = _valid_payload()
    payload["expected_pickrate_shifts"] = [
        {"subject": "Chamber", "direction": "downward", "magnitude": "large"}
    ]
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_invalid_pickrate_magnitude_rejected() -> None:
    payload = _valid_payload()
    payload["expected_pickrate_shifts"] = [
        {"subject": "Chamber", "direction": "down", "magnitude": "huge"}
    ]
    with pytest.raises(ValidationError):
        PatchIntentResult.model_validate(payload)


def test_pickrate_shift_rationale_optional() -> None:
    payload = _valid_payload()
    payload["expected_pickrate_shifts"] = [
        {"subject": "Chamber", "direction": "down", "magnitude": "large"}
    ]
    result = PatchIntentResult.model_validate(payload)
    assert result.expected_pickrate_shifts[0].rationale is None


def test_default_empty_lists() -> None:
    """A patch with no agent/map changes lands valid lists, not nulls."""
    payload = _valid_payload()
    del payload["agents_affected"]
    del payload["maps_affected"]
    del payload["expected_pickrate_shifts"]
    result = PatchIntentResult.model_validate(payload)
    assert result.agents_affected == []
    assert result.maps_affected == []
    assert result.expected_pickrate_shifts == []


def test_result_is_frozen() -> None:
    result = PatchIntentResult.model_validate(_valid_payload())
    with pytest.raises(ValidationError):
        result.confidence = 0.1  # type: ignore[misc]


def test_pickrate_shift_is_frozen() -> None:
    shift = ExpectedPickrateShift(subject="Jett", direction="up", magnitude="small")
    with pytest.raises(ValidationError):
        shift.subject = "Sage"  # type: ignore[misc]
