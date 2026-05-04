"""End-to-end transcription worker tests (BUF-21).

Marked ``integration`` because the worker writes through real
``MediaRecord`` / ``Transcript`` rows — pulling untranscribed media,
deleting + re-inserting on a re-transcription, cascading on media
delete are exactly what we're trying to exercise. The
:class:`StubEngine` replaces the faster-whisper backend so CI doesn't
need a GPU or the ML stack; everything below the engine boundary
runs against the real schema migrated by alembic.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from data_pipeline.transcribe import (
    TranscribePendingStats,
    TranscriptionResult,
    TranscriptSegment,
    transcribe_pending,
)
from esports_sim.db.enums import MediaKind
from esports_sim.db.models import (
    MediaRecord,
    Transcript,
    TranscriptChunkEmbedding,
)
from sqlalchemy import select

pytestmark = pytest.mark.integration


# --- test stub --------------------------------------------------------------


class StubEngine:
    """Deterministic :class:`TranscriptionEngine` for the worker tests.

    Returns a canned :class:`TranscriptionResult` derived from the
    file's text contents — enough variation for asserts that need to
    distinguish two media files, no real ML. Records every call so
    tests can assert that "no transcript yet" rows are the only ones
    that get processed by default.
    """

    def __init__(self, *, model_version: str = "stub-large-v3") -> None:
        self._model_version = model_version
        self.calls: list[Path] = []

    @property
    def model_version(self) -> str:
        return self._model_version

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        self.calls.append(audio_path)
        text = audio_path.read_text(encoding="utf-8")
        # Two segments so the per-segment JSONB shape gets exercised.
        head, _, tail = text.partition(" ")
        segments: tuple[TranscriptSegment, ...]
        if tail:
            segments = (
                TranscriptSegment(start=0.0, end=1.0, text=head),
                TranscriptSegment(start=1.0, end=2.0, text=tail),
            )
        else:
            segments = (TranscriptSegment(start=0.0, end=1.0, text=head),)
        return TranscriptionResult(
            language=language or "en",
            model_version=self._model_version,
            segments=segments,
            duration_seconds=2.0,
            wallclock_seconds=0.05,
        )


def _make_media(
    *,
    tmp_path: Path,
    name: str,
    text: str,
    source: str = "twitch_vod",
    kind: MediaKind = MediaKind.AUDIO,
    language: str | None = None,
) -> MediaRecord:
    """Mint a :class:`MediaRecord` whose ``local_path`` is a real text file.

    The stub engine reads the file's text contents so the resulting
    transcript varies per row — letting tests on a multi-row batch
    distinguish which row got which text without involving Whisper.
    """
    audio = tmp_path / f"{name}.txt"
    audio.write_text(text, encoding="utf-8")
    return MediaRecord(
        source=source,
        source_uri=f"https://example.test/{name}",
        local_path=str(audio),
        media_kind=kind,
        language=language,
    )


# --- happy path -------------------------------------------------------------


def test_worker_transcribes_only_untranscribed_media(db_session, tmp_path: Path) -> None:
    """The default pass selects rows that have no transcript and skips the rest.

    Exactly the BUF-21 spec line "where transcript IS NULL" projected
    onto our transcript-as-separate-table schema. Asserting both
    sides — the new row gets processed AND the already-transcribed
    row is left alone — keeps a future refactor of the LEFT JOIN
    from silently re-running the model on every pass.
    """
    fresh = _make_media(tmp_path=tmp_path, name="fresh", text="brand new audio")
    already_done = _make_media(tmp_path=tmp_path, name="done", text="already finished")
    db_session.add_all([fresh, already_done])
    db_session.flush()
    # Pre-seed a transcript on ``already_done`` so the LEFT JOIN
    # filter excludes it.
    db_session.add(
        Transcript(
            media_id=already_done.id,
            language="en",
            model_version="stub-large-v3",
            text="already finished",
            segments=[{"start": 0.0, "end": 1.0, "text": "already finished", "speaker": None}],
            duration_seconds=1.0,
            wallclock_seconds=0.01,
        )
    )
    db_session.flush()

    engine = StubEngine()
    stats = transcribe_pending(session=db_session, engine=engine)

    assert isinstance(stats, TranscribePendingStats)
    assert stats.selected == 1
    assert stats.transcribed == 1
    assert stats.engine_errors == 0
    # Engine was called exactly once — the already-transcribed row
    # never even hit the engine layer.
    assert len(engine.calls) == 1
    assert engine.calls[0] == Path(fresh.local_path)

    transcripts = db_session.execute(select(Transcript)).scalars().all()
    assert len(transcripts) == 2
    fresh_transcript = next(t for t in transcripts if t.media_id == fresh.id)
    assert fresh_transcript.language == "en"
    assert fresh_transcript.model_version == "stub-large-v3"
    assert fresh_transcript.text == "brand new audio"
    assert len(fresh_transcript.segments) == 2


def test_worker_writes_sidecar_json_when_root_configured(db_session, tmp_path: Path) -> None:
    """``sidecar_root`` triggers a per-media ``transcript.json`` write.

    The sidecar shape is the contract a downstream chunk-embedder or
    a manual reviewer reads; assert on its presence + canonical fields
    so a refactor that drops a key surfaces here, not as a missing-
    field crash in the embedder.
    """
    media = _make_media(tmp_path=tmp_path, name="m1", text="hello world")
    db_session.add(media)
    db_session.flush()

    sidecar_root = tmp_path / "sidecars"
    stats = transcribe_pending(
        session=db_session,
        engine=StubEngine(),
        sidecar_root=sidecar_root,
    )

    assert stats.transcribed == 1
    sidecar_path = sidecar_root / str(media.id) / "transcript.json"
    assert sidecar_path.exists()
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    assert payload["media_id"] == str(media.id)
    assert payload["language"] == "en"
    assert payload["model_version"] == "stub-large-v3"
    assert payload["text"] == "hello world"
    assert len(payload["segments"]) == 2
    # Worker also recorded the sidecar path on the row so a downstream
    # join can find the file without re-deriving the path convention.
    transcript = db_session.execute(select(Transcript)).scalar_one()
    assert transcript.transcript_path == str(sidecar_path)


def test_worker_search_via_transcript_text_column(db_session, tmp_path: Path) -> None:
    """Acceptance: ``transcripts searchable via SQL on the transcript column``.

    The BUF-21 spec asks for SQL searchability. ``transcript.text``
    is a plain TEXT column so ``ILIKE`` matches work today; this
    test asserts the contract directly so a future schema change
    that drops the column or moves the search payload into JSONB
    breaks the build instead of silently breaking the acceptance.
    """
    one = _make_media(tmp_path=tmp_path, name="one", text="duelist clutched the round")
    two = _make_media(tmp_path=tmp_path, name="two", text="sentinel held the site")
    db_session.add_all([one, two])
    db_session.flush()

    transcribe_pending(session=db_session, engine=StubEngine())

    matched = (
        db_session.execute(select(Transcript).where(Transcript.text.ilike("%clutched%")))
        .scalars()
        .all()
    )
    assert [t.media_id for t in matched] == [one.id]


# --- idempotency + re-runs --------------------------------------------------


def test_worker_is_a_noop_when_nothing_pending(db_session, tmp_path: Path) -> None:
    """Steady state: every media row already has a transcript → engine never called.

    This is the production loop's most common path — once the corpus
    is caught up, the scheduler's per-cycle pass should make zero
    Whisper calls. Asserting on the engine call counter (rather than
    just the stats) makes that contract explicit.
    """
    media = _make_media(tmp_path=tmp_path, name="m", text="hello")
    db_session.add(media)
    db_session.flush()

    engine = StubEngine()
    transcribe_pending(session=db_session, engine=engine)
    assert len(engine.calls) == 1

    # Second pass is a no-op.
    second = transcribe_pending(session=db_session, engine=engine)
    assert second.selected == 0
    assert second.transcribed == 0
    assert len(engine.calls) == 1


def test_worker_include_existing_replaces_transcript_in_place(db_session, tmp_path: Path) -> None:
    """``include_existing=True`` re-transcribes — the prior row is replaced cleanly.

    The DELETE-then-INSERT replace is what lets a model rotation re-
    run produce a clean ``segments`` JSONB; an UPSERT-in-place would
    risk a stale segment tail if the new model produced fewer
    segments. Asserting on the new ``model_version`` plus a single
    transcript row per media verifies both halves.
    """
    media = _make_media(tmp_path=tmp_path, name="m", text="first take")
    db_session.add(media)
    db_session.flush()

    transcribe_pending(session=db_session, engine=StubEngine(model_version="v1"))

    # Update the file so the second pass produces different text.
    Path(media.local_path).write_text("second take", encoding="utf-8")

    transcribe_pending(
        session=db_session,
        engine=StubEngine(model_version="v2"),
        include_existing=True,
    )

    transcripts = db_session.execute(select(Transcript)).scalars().all()
    assert len(transcripts) == 1
    transcript = transcripts[0]
    assert transcript.model_version == "v2"
    assert transcript.text == "second take"


# --- per-record errors ------------------------------------------------------


def test_worker_skips_missing_audio_files(db_session, tmp_path: Path) -> None:
    """A pruned/evicted local file is logged + counted, not fatal.

    Real-world: a worker host's local cache may evict files between
    when the row was registered and when the worker runs. Skip the
    row (so a later pass after the file is restored picks it up)
    rather than crashing the batch.
    """
    present = _make_media(tmp_path=tmp_path, name="present", text="have audio")
    missing = _make_media(tmp_path=tmp_path, name="missing", text="placeholder")
    Path(missing.local_path).unlink()
    db_session.add_all([present, missing])
    db_session.flush()

    stats = transcribe_pending(session=db_session, engine=StubEngine())

    assert stats.selected == 2
    assert stats.transcribed == 1
    assert stats.file_missing == 1
    transcripts = db_session.execute(select(Transcript)).scalars().all()
    assert [t.media_id for t in transcripts] == [present.id]


class _ExplodingEngine:
    """Engine that always raises — for the per-row error-handling check."""

    @property
    def model_version(self) -> str:
        return "explodes"

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        raise RuntimeError("CTranslate2 said no")


def test_worker_skips_engine_errors_and_continues(db_session, tmp_path: Path) -> None:
    """An engine that crashes on one row doesn't take down the rest of the batch.

    Mirrors the BUF-9 ingestion runner's per-record skip discipline:
    one corrupt audio file should produce a structured ``ENGINE_ERROR``
    log + a counter bump, not abort a 10h batch.
    """
    media = _make_media(tmp_path=tmp_path, name="m", text="audio")
    db_session.add(media)
    db_session.flush()

    stats = transcribe_pending(
        session=db_session,
        engine=_ExplodingEngine(),
    )
    assert stats.selected == 1
    assert stats.transcribed == 0
    assert stats.engine_errors == 1
    assert db_session.execute(select(Transcript)).first() is None


# --- limit + ordering -------------------------------------------------------


def test_worker_respects_limit(db_session, tmp_path: Path) -> None:
    """``limit`` caps the number of rows processed per pass.

    The scheduler passes a bounded number so a crash mid-batch
    doesn't lose more than a manageable chunk; assert the cap
    flows through.
    """
    media: Sequence[MediaRecord] = [
        _make_media(tmp_path=tmp_path, name=f"m{i}", text=f"audio {i}") for i in range(5)
    ]
    db_session.add_all(media)
    db_session.flush()

    stats = transcribe_pending(session=db_session, engine=StubEngine(), limit=2)
    assert stats.selected == 2
    assert stats.transcribed == 2
    assert len(db_session.execute(select(Transcript)).scalars().all()) == 2


# --- BUF-28 FK back-fill ----------------------------------------------------


def test_deleting_media_cascades_to_transcript_chunk_embeddings(db_session, tmp_path: Path) -> None:
    """Migration 0011 wires ``transcript_chunk_embedding.media_id`` → ``media_record.id``.

    BUF-28 (migration 0009) deferred the FK because ``media_record``
    didn't exist yet. BUF-21's migration adds it with ``ON DELETE
    CASCADE`` so a deleted media row drops every embedded chunk
    atomically — the writer-as-cleanup-authority workaround the
    embed module's docstring mentions is no longer load-bearing.

    The test plants a chunk row by hand (not via the BUF-28
    embedder, which would pull torch into CI) and asserts the
    cascade does its job.
    """
    media = _make_media(tmp_path=tmp_path, name="m", text="hello")
    db_session.add(media)
    db_session.flush()

    # 384-dim placeholder vector — pgvector accepts a Python list.
    chunk = TranscriptChunkEmbedding(
        media_id=media.id,
        chunk_idx=0,
        chunk_text="hello",
        embedding=[0.0] * 384,
        model_version="placeholder",
    )
    db_session.add(chunk)
    db_session.flush()

    db_session.delete(media)
    db_session.flush()

    remaining = db_session.execute(select(TranscriptChunkEmbedding)).scalars().all()
    assert remaining == []
