import pytest
from esports_sim.schemas.events import (
    AnyEvent,
    CashTransferred,
    MatchCompleted,
    MatchScheduled,
    TickAdvanced,
)
from pydantic import TypeAdapter, ValidationError

_adapter = TypeAdapter(AnyEvent)


def test_envelope_required_fields():
    with pytest.raises(ValidationError):
        TickAdvanced(tick=-1, source="sim", new_tick=0)
    with pytest.raises(ValidationError):
        TickAdvanced(tick=0, source="sim", new_tick=-1)


def test_extra_fields_rejected():
    with pytest.raises(ValidationError):
        TickAdvanced(tick=0, source="sim", new_tick=1, bogus=True)


def test_frozen_after_construction():
    e = TickAdvanced(tick=0, source="sim", new_tick=1)
    with pytest.raises(ValidationError):
        e.tick = 5  # type: ignore[misc]


def test_discriminated_union_round_trip():
    e = MatchScheduled(
        tick=10,
        source="sim",
        match_id="m1",
        tournament_id="t1",
        home_team_id="h",
        away_team_id="a",
        scheduled_for_tick=20,
        best_of=3,
    )
    blob = e.model_dump()
    parsed = _adapter.validate_python(blob)
    assert isinstance(parsed, MatchScheduled)
    assert parsed == e


def test_discriminated_union_dispatches_on_kind():
    blob = {
        "tick": 50,
        "source": "sim",
        "kind": "match.completed",
        "match_id": "m1",
        "home_score": 2,
        "away_score": 1,
    }
    parsed = _adapter.validate_python(blob)
    assert isinstance(parsed, MatchCompleted)


def test_unknown_kind_rejected():
    blob = {"tick": 0, "source": "sim", "kind": "match.exploded", "match_id": "m1"}
    with pytest.raises(ValidationError):
        _adapter.validate_python(blob)


def test_cash_transfer_allows_negative_amount():
    # Negative amount is legal (refunds, corrections); only the envelope tick is
    # constrained.
    e = CashTransferred(
        tick=1,
        source="sim",
        from_account="ops",
        to_account="payroll",
        amount_cents=-1000,
    )
    assert e.amount_cents == -1000
