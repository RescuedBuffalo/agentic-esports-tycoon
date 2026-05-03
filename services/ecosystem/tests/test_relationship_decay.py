"""Unit tests for the BUF-26 relationship decay math.

Pure-Python tests — no Postgres, no SQLAlchemy session. The integration
tests for the bootstrap path live alongside the data-pipeline tests
and require ``TEST_DATABASE_URL``.
"""

from __future__ import annotations

import math
import uuid
from datetime import UTC, datetime, timedelta

import pytest
from ecosystem.relationships import (
    DECAY_RATES,
    DecayError,
    decay_edge,
    decay_strength,
    weeks_between,
)
from esports_sim.db.enums import RelationshipEdgeType
from esports_sim.db.models import RelationshipEdge


def _make_edge(
    *,
    edge_type: RelationshipEdgeType = RelationshipEdgeType.TEAMMATE,
    strength: float = 1.0,
    sentiment: float = 1.0,
    last_updated_at: datetime | None = None,
) -> RelationshipEdge:
    """In-memory edge for the pure-Python decay tests."""
    return RelationshipEdge(
        edge_id=uuid.uuid4(),
        src_id=uuid.UUID(int=1),
        dst_id=uuid.UUID(int=2),
        edge_type=edge_type,
        strength=strength,
        sentiment=sentiment,
        last_updated_at=last_updated_at or datetime(2026, 1, 1, tzinfo=UTC),
        extra={},
    )


# ---- DECAY_RATES inventory pin ---------------------------------------------


def test_decay_rates_cover_every_relationship_edge_type() -> None:
    """The dict must have a rate for every enum value — a missing entry
    raises :class:`DecayError` at runtime, which would block the monthly
    job and is a much-worse failure mode than a startup mismatch."""
    assert set(DECAY_RATES) == set(RelationshipEdgeType)


@pytest.mark.parametrize("edge_type", list(RelationshipEdgeType))
def test_every_decay_rate_is_strictly_positive(edge_type: RelationshipEdgeType) -> None:
    """A non-positive rate would either freeze strength (rate=0) or grow
    it back over time (rate<0). Both break the System 07 contract."""
    assert DECAY_RATES[edge_type] > 0.0


# ---- decay_strength regression --------------------------------------------


def test_decay_strength_zero_weeks_is_identity() -> None:
    assert decay_strength(0.7, edge_type=RelationshipEdgeType.TEAMMATE, weeks=0.0) == 0.7


def test_decay_strength_negative_weeks_is_identity() -> None:
    """A clock skew that produces ``weeks < 0`` must not grow strength."""
    assert decay_strength(0.7, edge_type=RelationshipEdgeType.TEAMMATE, weeks=-5.0) == 0.7


def test_decay_strength_zero_strength_stays_zero() -> None:
    assert decay_strength(0.0, edge_type=RelationshipEdgeType.TEAMMATE, weeks=20.0) == 0.0


def test_decay_strength_unknown_edge_type_raises() -> None:
    """Passing a fabricated enum value (e.g. via getattr trickery in a
    test) should fail loudly so a future kind addition without a rate
    update doesn't silently degrade decay behaviour."""

    class _Fake:
        value = "fabricated_kind"

    with pytest.raises(DecayError):
        decay_strength(0.5, edge_type=_Fake(), weeks=1.0)  # type: ignore[arg-type]


def test_decay_strength_rejects_strength_outside_unit_interval() -> None:
    with pytest.raises(DecayError):
        decay_strength(1.5, edge_type=RelationshipEdgeType.TEAMMATE, weeks=1.0)
    with pytest.raises(DecayError):
        decay_strength(-0.1, edge_type=RelationshipEdgeType.TEAMMATE, weeks=1.0)


def test_decay_strength_teammate_20_weeks_acceptance_regression() -> None:
    """BUF-26 acceptance: edge with strength=1.0 and no signal for 20
    weeks decays to the expected exponential value."""
    rate = DECAY_RATES[RelationshipEdgeType.TEAMMATE]
    expected = math.exp(-rate * 20)
    actual = decay_strength(
        1.0,
        edge_type=RelationshipEdgeType.TEAMMATE,
        weeks=20.0,
    )
    # Pin both the closed-form expected value and the numerical
    # constant so a future tweak to ``DECAY_RATES`` produces a
    # localised test failure rather than a slow regression.
    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual == pytest.approx(0.6703200460356393, abs=1e-12)


def test_decay_strength_composes_over_disjoint_intervals() -> None:
    """Two decay applications over (a, b) and (b, c) compose to one
    decay over (a, c). This is what makes the monthly job idempotent
    under jitter — a 4.7-week gap and a 30.3-week gap give the same
    result as one 35-week gap."""
    edge_type = RelationshipEdgeType.RIVAL
    one_shot = decay_strength(1.0, edge_type=edge_type, weeks=12.0)
    composed = decay_strength(
        decay_strength(1.0, edge_type=edge_type, weeks=4.7),
        edge_type=edge_type,
        weeks=7.3,
    )
    assert composed == pytest.approx(one_shot, abs=1e-12)


def test_decay_strength_clamps_to_unit_interval() -> None:
    """The result is always inside ``[0, 1]`` so the column's CHECK
    constraint can never trip from floating-point fuzz."""
    out = decay_strength(1.0, edge_type=RelationshipEdgeType.MENTOR, weeks=520.0)
    assert 0.0 <= out <= 1.0


# ---- decay_edge mutator ---------------------------------------------------


def test_decay_edge_advances_anchor_and_decays_strength() -> None:
    anchor = datetime(2026, 1, 1, tzinfo=UTC)
    now = anchor + timedelta(weeks=20)
    edge = _make_edge(strength=1.0, last_updated_at=anchor)

    decay_edge(edge, now=now)

    rate = DECAY_RATES[RelationshipEdgeType.TEAMMATE]
    assert edge.strength == pytest.approx(math.exp(-rate * 20), abs=1e-12)
    assert edge.last_updated_at == now


def test_decay_edge_leaves_sentiment_alone() -> None:
    """Sentiment is permanent valence; strength decays and sentiment
    survives across the same window."""
    edge = _make_edge(strength=1.0, sentiment=-0.7)
    decay_edge(edge, now=edge.last_updated_at + timedelta(weeks=40))
    assert edge.sentiment == -0.7


def test_decay_edge_is_idempotent_when_now_matches_anchor() -> None:
    edge = _make_edge(strength=0.42)
    decay_edge(edge, now=edge.last_updated_at)
    assert edge.strength == 0.42


def test_decay_edge_does_not_move_anchor_backward() -> None:
    """A clock skew where ``now`` is earlier than ``last_updated_at``
    is a no-op; we never grow strength back from a negative gap."""
    anchor = datetime(2026, 4, 1, tzinfo=UTC)
    earlier = anchor - timedelta(weeks=2)
    edge = _make_edge(strength=0.5, last_updated_at=anchor)

    decay_edge(edge, now=earlier)

    assert edge.strength == 0.5
    assert edge.last_updated_at == anchor


# ---- weeks_between sanity --------------------------------------------------


def test_weeks_between_clamps_negative_intervals_to_zero() -> None:
    a = datetime(2026, 1, 1, tzinfo=UTC)
    b = datetime(2025, 12, 1, tzinfo=UTC)
    assert weeks_between(a, b) == 0.0


def test_weeks_between_returns_fractional_weeks() -> None:
    a = datetime(2026, 1, 1, tzinfo=UTC)
    b = a + timedelta(days=10, hours=12)
    assert weeks_between(a, b) == pytest.approx(10.5 / 7.0, abs=1e-12)
