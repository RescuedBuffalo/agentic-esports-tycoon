"""Persistence + scheduler-hook tests for ``esports_sim.patch_intent``.

Two layers:

1. Integration tests against Postgres (skipped on a fresh clone) cover
   ``upsert_patch_intent`` UPSERT semantics and the ``patch_intent``
   table's CHECK constraints.
2. Behavioural tests of ``extract_intent_for_pending`` use an injected
   fake extractor + fake session so the scheduler-hook contract is
   covered without a DB or a Claude call.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from esports_sim.budget import BudgetCaps, BudgetExhausted, Governor, Ledger
from esports_sim.db.models import PatchIntent, PatchNote
from esports_sim.patch_intent import (
    PROMPT_VERSION,
    PatchIntentResult,
    extract_intent_for_pending,
    upsert_patch_intent,
)
from esports_sim.patch_intent.extractor import ExtractionOutcome

# --- shared fixtures ------------------------------------------------------


def _result(
    *,
    primary_intent: str = "nerf-meta-outlier",
    confidence: float = 0.8,
) -> PatchIntentResult:
    return PatchIntentResult(
        primary_intent=primary_intent,
        pro_play_driven_score=0.9,
        agents_affected=["Chamber"],
        maps_affected=[],
        econ_changed=False,
        expected_pickrate_shifts=[
            {"subject": "Chamber", "direction": "down", "magnitude": "large"}
        ],
        community_controversy_predicted=0.85,
        confidence=confidence,
        reasoning="canned",
    )


def _outcome(
    *,
    primary_intent: str = "nerf-meta-outlier",
    confidence: float = 0.8,
    usd_cost: float = 0.05,
) -> ExtractionOutcome:
    return ExtractionOutcome(
        result=_result(primary_intent=primary_intent, confidence=confidence),
        model="claude-opus-4-7",
        prompt_version=PROMPT_VERSION,
        input_tokens=1000,
        output_tokens=200,
        usd_cost=usd_cost,
    )


# --- integration: ``upsert_patch_intent`` and DB constraints --------------


pytestmark_integration = pytest.mark.integration


@pytest.fixture
def patch_note_row(db_session: Any) -> PatchNote:
    """Insert a ``PatchNote`` row the intent test can attach to."""
    note = PatchNote(
        source="playvalorant",
        patch_version="5.12",
        published_at=datetime(2022, 11, 15, tzinfo=UTC),
        raw_html="<html>5.12</html>",
        body_text="Patch 5.12: Chamber adjustments...",
        url="https://playvalorant.com/en-us/news/game-updates/valorant-patch-notes-5-12/",
    )
    db_session.add(note)
    db_session.flush()
    return note


@pytestmark_integration
def test_upsert_inserts_then_updates_in_place(db_session: Any, patch_note_row: PatchNote) -> None:
    first = _outcome(confidence=0.6)
    row, tag = upsert_patch_intent(db_session, patch_note=patch_note_row, outcome=first)
    assert tag == "inserted"
    assert row.confidence == pytest.approx(0.6)
    assert row.primary_intent == "nerf-meta-outlier"

    second = _outcome(confidence=0.9, primary_intent="econ-rebalance")
    row2, tag2 = upsert_patch_intent(db_session, patch_note=patch_note_row, outcome=second)
    assert tag2 == "updated"
    # Same row id — the dedup key matched.
    assert row2.id == row.id
    assert row2.confidence == pytest.approx(0.9)
    assert row2.primary_intent == "econ-rebalance"


@pytestmark_integration
def test_check_constraint_rejects_out_of_range_confidence(
    db_session: Any, patch_note_row: PatchNote
) -> None:
    """The DB-layer 0..1 CHECK is the backstop if a path bypassed Pydantic."""
    from sqlalchemy.exc import IntegrityError

    bad = PatchIntent(
        patch_note_id=patch_note_row.id,
        prompt_version=PROMPT_VERSION,
        model="claude-opus-4-7",
        primary_intent="nerf",
        pro_play_driven_score=0.5,
        agents_affected=[],
        maps_affected=[],
        econ_changed=False,
        expected_pickrate_shifts=[],
        community_controversy_predicted=0.5,
        confidence=1.5,  # out of range
        reasoning="bad",
        input_tokens=1,
        output_tokens=1,
        usd_cost=0.0,
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytestmark_integration
def test_unique_constraint_on_patch_note_and_prompt_version(
    db_session: Any, patch_note_row: PatchNote
) -> None:
    """``(patch_note_id, prompt_version)`` is the dedup key — duplicate inserts fail."""
    from sqlalchemy.exc import IntegrityError

    upsert_patch_intent(db_session, patch_note=patch_note_row, outcome=_outcome())
    # A direct INSERT (bypassing the upsert helper) at the same key must fail.
    duplicate = PatchIntent(
        patch_note_id=patch_note_row.id,
        prompt_version=PROMPT_VERSION,
        model="claude-opus-4-7",
        primary_intent="other",
        pro_play_driven_score=0.5,
        agents_affected=[],
        maps_affected=[],
        econ_changed=False,
        expected_pickrate_shifts=[],
        community_controversy_predicted=0.5,
        confidence=0.5,
        reasoning="dup",
        input_tokens=1,
        output_tokens=1,
        usd_cost=0.0,
    )
    db_session.add(duplicate)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


@pytestmark_integration
def test_distinct_prompt_versions_coexist_for_same_patch(
    db_session: Any, patch_note_row: PatchNote
) -> None:
    """Bumping the prompt version produces a second row, not an UPSERT."""
    upsert_patch_intent(db_session, patch_note=patch_note_row, outcome=_outcome())
    other_version = ExtractionOutcome(
        result=_result(),
        model="claude-opus-4-7",
        prompt_version="v2",
        input_tokens=1,
        output_tokens=1,
        usd_cost=0.0,
    )
    upsert_patch_intent(db_session, patch_note=patch_note_row, outcome=other_version)
    db_session.flush()

    from sqlalchemy import select

    rows = list(
        db_session.execute(
            select(PatchIntent).where(PatchIntent.patch_note_id == patch_note_row.id)
        ).scalars()
    )
    assert len(rows) == 2
    assert {r.prompt_version for r in rows} == {PROMPT_VERSION, "v2"}


# --- scheduler hook: in-memory behavioural tests --------------------------


class _FakePatchNote:
    """Stand-in for a ``PatchNote`` ORM row.

    Just enough surface for the scheduler hook: ``id``, ``body_text``,
    ``patch_version``. The fake session below stores them in a list
    keyed on their UUID so the hook's "exclude already-classified"
    query can be answered without SQLAlchemy.
    """

    def __init__(self, *, patch_version: str, body_text: str = "patch body") -> None:
        self.id = uuid.uuid4()
        self.patch_version = patch_version
        self.body_text = body_text
        self.source = "playvalorant"


class _FakeIntent:
    """Stand-in for ``PatchIntent`` with the columns the scheduler reads."""

    def __init__(self, *, patch_note_id: uuid.UUID, prompt_version: str) -> None:
        self.id = uuid.uuid4()
        self.patch_note_id = patch_note_id
        self.prompt_version = prompt_version


class _FakeSchedulerSession:
    """Minimal session that the scheduler hook can drive end-to-end.

    Stores patch notes + intents in lists. Implements ``execute(stmt)``
    by introspecting the compiled SQL so the hook's two queries (pending
    patches and existing-intent count) work without a real DB.
    """

    def __init__(
        self,
        *,
        patches: Iterable[_FakePatchNote],
        intents: Iterable[_FakeIntent] = (),
    ) -> None:
        self._patches = list(patches)
        self._intents = list(intents)

    def execute(self, stmt: Any) -> Any:
        compiled = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        # Three queries the production code emits — disambiguate on the
        # SELECT projection (FROM clauses overlap because of subqueries
        # and joins).
        if compiled.lstrip().startswith("SELECT patch_intent"):
            # Two callers project patch_intent.*: the per-row upsert
            # SELECT (filters by patch_note_id =) and the
            # existing-count select (filters by prompt_version only,
            # joins patch_note for the source filter).
            # The upsert SELECT filters by a literal patch_note_id ('uuid')
            # — the count SELECT only has ``= patch_note.id`` (the JOIN
            # condition). The trailing apostrophe pins the literal form.
            if "patch_intent.patch_note_id = '" in compiled:
                # Upsert SELECT: pull the literal patch_note_id +
                # prompt_version out and look up exactly one row.
                pid = uuid.UUID(_between(compiled, "patch_intent.patch_note_id = '", "'"))
                pv = _between(compiled, "patch_intent.prompt_version = '", "'")
                match = next(
                    (i for i in self._intents if i.patch_note_id == pid and i.prompt_version == pv),
                    None,
                )
                return _ScalarsResult([match] if match is not None else [])
            # Existing-count select.
            existing = [i for i in self._intents if i.prompt_version == PROMPT_VERSION]
            return _ScalarsResult(existing)
        if compiled.lstrip().startswith("SELECT patch_note"):
            classified_ids = {
                i.patch_note_id for i in self._intents if i.prompt_version == PROMPT_VERSION
            }
            pending = [p for p in self._patches if p.id not in classified_ids]
            return _ScalarsResult(pending)
        raise AssertionError(f"unexpected query: {compiled}")

    def add(self, intent: Any) -> None:
        # The real upsert path inserts a PatchIntent ORM row. The fake
        # session just appends to its list so the scheduler-hook count
        # of "skipped_existing" reflects this pass's writes too.
        self._intents.append(
            _FakeIntent(
                patch_note_id=intent.patch_note_id,
                prompt_version=intent.prompt_version,
            )
        )

    def flush(self) -> None:
        return None


class _ScalarsResult:
    def __init__(self, values: list[Any]) -> None:
        self._values = values

    def scalars(self) -> Any:
        return iter(self._values)

    def scalar_one_or_none(self) -> Any:
        return self._values[0] if self._values else None


def _between(text: str, start: str, end: str) -> str:
    """Return the substring between ``start`` and ``end`` markers."""
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[i:j]


def test_scheduler_hook_calls_extractor_for_each_pending_patch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hook drives ``extract_patch_intent`` once per pending patch
    and writes one row per call."""
    p1 = _FakePatchNote(patch_version="8.04")
    p2 = _FakePatchNote(patch_version="8.05")
    session = _FakeSchedulerSession(patches=[p1, p2])

    calls: list[str] = []

    def _fake_extract(*, governor: Any, patch_notes_text: str, **kwargs: Any) -> ExtractionOutcome:
        calls.append(patch_notes_text)
        return _outcome()

    monkeypatch.setattr(
        "esports_sim.patch_intent.persistence.extract_patch_intent",
        _fake_extract,
    )

    governor = Governor(
        ledger=Ledger(db_path=tmp_path / "b.sqlite"),
        caps=BudgetCaps(weekly_hard_cap_usd=30.0),
    )
    stats = extract_intent_for_pending(session, governor=governor)  # type: ignore[arg-type]

    assert len(calls) == 2
    assert stats.inserted == 2
    assert stats.updated == 0
    assert stats.budget_exhausted == 0


def test_scheduler_hook_skips_already_classified_patches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Idempotency: a pass after every patch has been classified is a no-op."""
    p1 = _FakePatchNote(patch_version="8.04")
    p2 = _FakePatchNote(patch_version="8.05")
    # p1 already has an intent for the current prompt version.
    existing = _FakeIntent(patch_note_id=p1.id, prompt_version=PROMPT_VERSION)
    session = _FakeSchedulerSession(patches=[p1, p2], intents=[existing])

    calls: list[str] = []

    def _fake_extract(*, governor: Any, patch_notes_text: str, **kwargs: Any) -> ExtractionOutcome:
        calls.append(patch_notes_text)
        return _outcome()

    monkeypatch.setattr(
        "esports_sim.patch_intent.persistence.extract_patch_intent",
        _fake_extract,
    )

    governor = Governor(
        ledger=Ledger(db_path=tmp_path / "b.sqlite"),
        caps=BudgetCaps(weekly_hard_cap_usd=30.0),
    )
    stats = extract_intent_for_pending(session, governor=governor)  # type: ignore[arg-type]

    # Only p2 was extracted.
    assert len(calls) == 1
    assert stats.inserted == 1
    assert stats.skipped_existing == 2  # p1 (pre-existing) + p2 (just added)


def test_scheduler_hook_stops_cleanly_on_budget_exhausted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cap hit halts the loop and surfaces the count without raising."""
    p1 = _FakePatchNote(patch_version="8.04")
    p2 = _FakePatchNote(patch_version="8.05")
    session = _FakeSchedulerSession(patches=[p1, p2])

    call_index = {"n": 0}

    def _fake_extract(*, governor: Any, patch_notes_text: str, **kwargs: Any) -> ExtractionOutcome:
        call_index["n"] += 1
        if call_index["n"] == 1:
            return _outcome()
        raise BudgetExhausted(
            purpose="patch_intent",
            weekly_spend_usd=2.99,
            weekly_cap_usd=3.0,
            projected_cost_usd=0.05,
            scope="purpose",
        )

    monkeypatch.setattr(
        "esports_sim.patch_intent.persistence.extract_patch_intent",
        _fake_extract,
    )

    governor = Governor(
        ledger=Ledger(db_path=tmp_path / "b.sqlite"),
        caps=BudgetCaps(weekly_hard_cap_usd=30.0),
    )
    stats = extract_intent_for_pending(session, governor=governor)  # type: ignore[arg-type]

    assert stats.inserted == 1
    assert stats.budget_exhausted == 1


def test_scheduler_hook_respects_limit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """``limit=1`` extracts one patch and stops."""
    p1 = _FakePatchNote(patch_version="8.04")
    p2 = _FakePatchNote(patch_version="8.05")
    p3 = _FakePatchNote(patch_version="8.06")
    session = _FakeSchedulerSession(patches=[p1, p2, p3])

    calls: list[str] = []

    def _fake_extract(*, governor: Any, patch_notes_text: str, **kwargs: Any) -> ExtractionOutcome:
        calls.append(patch_notes_text)
        return _outcome()

    monkeypatch.setattr(
        "esports_sim.patch_intent.persistence.extract_patch_intent",
        _fake_extract,
    )

    governor = Governor(
        ledger=Ledger(db_path=tmp_path / "b.sqlite"),
        caps=BudgetCaps(weekly_hard_cap_usd=30.0),
    )
    stats = extract_intent_for_pending(session, governor=governor, limit=1)  # type: ignore[arg-type]

    assert len(calls) == 1
    assert stats.inserted == 1


# --- DTO sanity -----------------------------------------------------------


def test_patch_intent_dto_round_trips_from_orm_row() -> None:
    """The DTO can be built from an ORM-shaped object via ``from_attributes``."""
    from esports_sim.schemas.dtos import PatchIntentDTO

    fake_row = SimpleNamespace(
        id=uuid.uuid4(),
        patch_note_id=uuid.uuid4(),
        prompt_version="v1",
        model="claude-opus-4-7",
        primary_intent="nerf-meta-outlier",
        pro_play_driven_score=0.9,
        agents_affected=["Chamber"],
        maps_affected=[],
        econ_changed=False,
        expected_pickrate_shifts=[
            {"subject": "Chamber", "direction": "down", "magnitude": "large"}
        ],
        community_controversy_predicted=0.85,
        confidence=0.8,
        reasoning="canned",
        input_tokens=1000,
        output_tokens=200,
        usd_cost=0.05,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    dto = PatchIntentDTO.model_validate(fake_row)
    assert dto.primary_intent == "nerf-meta-outlier"
    assert dto.expected_pickrate_shifts[0]["subject"] == "Chamber"
    # Frozen — instance can't be mutated post-construction.
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        dto.confidence = 0.1  # type: ignore[misc]
