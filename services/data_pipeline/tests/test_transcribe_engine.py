"""Unit tests for the BUF-21 transcription engine layer.

Pure-Python tests against the dataclasses + the Protocol — no
faster-whisper, no Postgres. The integration test
(``test_transcribe_worker``) exercises the worker end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from data_pipeline.transcribe.engine import (
    DEFAULT_MODEL_SIZE,
    FasterWhisperEngine,
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptSegment,
)


def test_transcript_segment_to_jsonable_round_trips_speaker_null() -> None:
    """``speaker`` defaults to ``None`` and survives the JSONable projection.

    The ``Transcript.segments`` JSONB column stores exactly this
    shape, so the worker can ``json.dumps`` it without further
    massaging. Asserting on the null keeps a future change that
    drops the key from breaking downstream readers expecting it.
    """
    seg = TranscriptSegment(start=0.0, end=4.2, text="hello world")
    payload = seg.to_jsonable()

    assert payload == {"start": 0.0, "end": 4.2, "text": "hello world", "speaker": None}


def test_transcription_result_text_joins_segments_with_single_space() -> None:
    """Whisper segment texts come pre-prefixed with a leading space.

    ``TranscriptionResult.text`` strips per-segment then joins with
    a single space — collapsing the leading whitespace Whisper emits
    so the chunker doesn't see ``"  word"`` runs that would
    distort token counts. This is the contract the embedding
    chunker relies on, asserted explicitly so a refactor that swaps
    the join doesn't silently widen the text.
    """
    result = TranscriptionResult(
        language="en",
        model_version="large-v3",
        segments=(
            TranscriptSegment(start=0.0, end=1.0, text=" hello"),
            TranscriptSegment(start=1.0, end=2.0, text=" world"),
            # An empty segment must not produce a leading double-space.
            TranscriptSegment(start=2.0, end=2.0, text="   "),
            TranscriptSegment(start=2.0, end=3.0, text=" again"),
        ),
        duration_seconds=3.0,
        wallclock_seconds=0.1,
    )

    assert result.text == "hello world again"


def test_transcription_result_text_empty_when_no_segments() -> None:
    """An engine that returned zero segments produces empty text.

    The worker still writes the row (the upstream might be a 30s clip
    that is all silence after VAD); empty text is a legitimate
    outcome, not a bug to skip.
    """
    result = TranscriptionResult(
        language="en",
        model_version="large-v3",
        segments=(),
        duration_seconds=0.0,
        wallclock_seconds=0.05,
    )
    assert result.text == ""


def test_default_engine_model_version_is_large_v3() -> None:
    """The constructor's default ``model_size`` is what BUF-21 specifies."""
    engine = FasterWhisperEngine(model=object())  # type: ignore[arg-type]
    assert engine.model_version == DEFAULT_MODEL_SIZE
    assert DEFAULT_MODEL_SIZE == "large-v3"


def test_engine_raises_filenotfound_for_missing_audio(tmp_path: Path) -> None:
    """Missing audio surfaces as a typed ``FileNotFoundError``, not a CT2 abort.

    The worker treats this as "skip + log + counter++" rather than
    "abort the whole batch"; the engine has to raise the typed error
    early (before touching faster-whisper) for that to work.
    """
    engine = FasterWhisperEngine(model=object())  # type: ignore[arg-type]
    missing = tmp_path / "does-not-exist.wav"

    with pytest.raises(FileNotFoundError):
        engine.transcribe(missing)


class _StubEngine:
    """Minimal :class:`TranscriptionEngine` implementation for the protocol check.

    Test-local — the worker tests import their own stub. Kept here so
    the runtime-checkable assertion below has something to bind to.
    """

    @property
    def model_version(self) -> str:
        return "stub-model"

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            language=language or "en",
            model_version=self.model_version,
            segments=(TranscriptSegment(start=0.0, end=1.0, text="ok"),),
            duration_seconds=1.0,
            wallclock_seconds=0.001,
        )


def test_engine_protocol_accepts_minimal_implementation() -> None:
    """Any class with ``model_version`` + ``transcribe`` satisfies the Protocol.

    Documents the surface the worker depends on — bumping the
    Protocol with a new method without updating the worker (or vice
    versa) should fail this assertion at type-check time as well as
    here.
    """
    engine: TranscriptionEngine = _StubEngine()
    assert isinstance(engine, TranscriptionEngine)
    assert engine.model_version == "stub-model"


def test_pending_select_uses_for_update_skip_locked() -> None:
    """The pending-row scan must lock with ``FOR UPDATE SKIP LOCKED``.

    Codex flagged the pre-fix query as racy (PR #29 review): two
    overlapping worker passes could SELECT the same media, both try
    to ``INSERT INTO transcript`` for the same ``media_id``, and the
    second would crash on the PK constraint and roll back its whole
    transaction.

    Asserting on the compiled SQL (rather than driving two real
    sessions concurrently — awkward against the per-test savepoint
    fixture) is the cheapest way to keep the contract: a future
    refactor that drops the lock clause fails here loudly.
    """
    from unittest.mock import MagicMock

    from data_pipeline.transcribe.worker import _select_pending  # noqa: PLC0415
    from sqlalchemy.dialects import postgresql

    captured: dict[str, str] = {}

    def fake_execute(stmt, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["sql"] = str(
            stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
        )
        # Return an object that satisfies ``.scalars().all()`` returning [].
        scalars = MagicMock()
        scalars.all.return_value = []
        result = MagicMock()
        result.scalars.return_value = scalars
        return result

    session = MagicMock()
    session.execute.side_effect = fake_execute

    _select_pending(session, limit=10, include_existing=False)
    assert "FOR UPDATE" in captured["sql"]
    assert "SKIP LOCKED" in captured["sql"]
    # ``OF media_record`` pins the lock target to the parent table
    # only — without it Postgres would try to lock the (NULL)
    # Transcript side of the LEFT JOIN and bail with "FOR UPDATE
    # cannot be applied to the nullable side of an outer join".
    assert "OF media_record" in captured["sql"]

    # Same contract on the include_existing path so a re-transcribe
    # job overlapping with a normal pass doesn't double up either.
    captured.clear()
    _select_pending(session, limit=None, include_existing=True)
    assert "FOR UPDATE" in captured["sql"]
    assert "SKIP LOCKED" in captured["sql"]
