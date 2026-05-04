"""Whisper engine protocol + ``faster-whisper`` default backend (BUF-21).

The :class:`TranscriptionEngine` Protocol is the only surface the
worker depends on — :class:`FasterWhisperEngine` is one
implementation. Tests inject a deterministic stub that returns canned
segments without loading any model weights, so a fresh clone with no
GPU still runs the BUF-21 unit + integration suite green.

Why a Protocol rather than instantiating ``faster-whisper`` directly:

* The default model (``large-v3``) is ~3 GB on disk and ~6 GB of VRAM
  in fp16 — a price worth paying for the production worker on the
  5090, an absurd one for any CI job that touches the worker code.
* Tests want determinism. A real Whisper run is sensitive to thread
  count and seed; a stub that returns ``"hello world"`` no matter
  the input keeps the worker test asserting on the orchestration
  (transcript row written, sidecar emitted, idempotency holds), not
  on the ML.
* ``faster-whisper`` is shipped as an optional extra (see
  ``services/data_pipeline/pyproject.toml`` — the
  ``[transcribe]`` extra). Keeping the default engine behind a lazy
  import means the rest of the data pipeline doesn't pay the
  CTranslate2 + cuDNN install cost.

Defaults match BUF-21's spec: ``large-v3`` model, fp16 on CUDA
(``compute_type="float16"``), Silero VAD enabled to skip silence.
The acceptance criterion is "10h audio in <30 minutes", i.e. ≥20×
realtime — those defaults are what ``faster-whisper`` benchmarks at
on a 5090. Operators can override via the constructor for a
slower-but-cheaper run (e.g., ``model_size="medium"`` on a laptop).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Type-only import: keep ``faster_whisper`` out of the import
    # graph for callers that only need the protocol or a stub
    # engine.
    from faster_whisper import WhisperModel


# Model identity recorded on every ``Transcript`` row when the default
# engine is used. ADR-006 standardises on ``large-v3`` for production
# transcription; rotation is a deliberate operation that bumps every
# row's ``model_version``.
DEFAULT_MODEL_SIZE: str = "large-v3"


@dataclass(frozen=True)
class TranscriptSegment:
    """One timestamped segment in a transcription.

    ``speaker`` is reserved for the future pyannote-driven diarization
    flag BUF-21 mentions — Whisper itself doesn't fill it. Storing it
    on the dataclass (not just the JSONB column) keeps the ABI stable
    when diarization lands so callers don't have to start re-parsing.
    """

    start: float
    end: float
    text: str
    speaker: str | None = None

    def to_jsonable(self) -> dict[str, Any]:
        """Project to a plain dict suitable for JSON / JSONB storage."""
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass(frozen=True)
class TranscriptionResult:
    """What an engine returns for one media file.

    ``duration_seconds`` is the actually-transcribed span (silence
    excluded by VAD); ``wallclock_seconds`` is the model's runtime
    on this file. The two are what BUF-21's throughput acceptance
    ("≥20× realtime") is measured against — the worker writes both
    to the ``Transcript`` row so the dashboard query has its
    numerator and denominator in the same place.
    """

    language: str
    model_version: str
    segments: tuple[TranscriptSegment, ...]
    duration_seconds: float
    wallclock_seconds: float

    @property
    def text(self) -> str:
        """Full transcript as one string — segments joined with spaces.

        ``" ".join`` rather than newlines because the embedding chunker
        (``esports_sim.embeddings.chunker``) re-splits on whitespace
        anyway; preserving newlines would add structure the chunker
        ignores and downstream readers don't rely on.
        """
        return " ".join(segment.text.strip() for segment in self.segments if segment.text.strip())


@runtime_checkable
class TranscriptionEngine(Protocol):
    """Minimal surface a Whisper backend has to expose.

    ``model_version`` is what the worker records on the
    :class:`~esports_sim.db.models.Transcript` row so a future model
    rotation can detect mixed populations. Callers must treat it as
    opaque — comparing it requires the same engine semantics, not
    just the same string.
    """

    @property
    def model_version(self) -> str: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe one audio file. ``language`` may be ``None`` to auto-detect."""


class FasterWhisperEngine:
    """Default :class:`TranscriptionEngine` backed by ``faster-whisper``.

    Lazy-loads the model on first :meth:`transcribe` call so the
    BUF-21 import graph (and any downstream module that pulls
    :mod:`data_pipeline.transcribe`) doesn't pay the CTranslate2 +
    cuDNN cost up-front. A test that stubs out the model can pass
    ``model=stub`` to the constructor and skip the lazy load entirely.

    Defaults are tuned for the production 5090:

    * ``model_size="large-v3"`` — the spec'd model.
    * ``device="cuda"`` + ``compute_type="float16"`` — the throughput
      target ("10h in <30min", ≥20× realtime) only works on GPU. A
      CPU run is supported by passing ``device="cpu"`` and
      ``compute_type="int8"`` but no longer hits the acceptance.
    * ``vad_filter=True`` — Silero VAD via faster-whisper's built-in;
      cuts transcription time ~30% on streams with significant
      silence (BUF-21's "skip silence" line item).
    """

    def __init__(
        self,
        *,
        model_size: str = DEFAULT_MODEL_SIZE,
        device: str = "cuda",
        compute_type: str = "float16",
        vad_filter: bool = True,
        beam_size: int = 5,
        # Pass a pre-loaded model for tests / ahead-of-time loading.
        # When ``None``, the model is constructed lazily on the first
        # ``transcribe`` call.
        model: WhisperModel | None = None,
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._vad_filter = vad_filter
        self._beam_size = beam_size
        self._model: WhisperModel | None = model

    @property
    def model_version(self) -> str:
        # Bare model size is what every operator types into the CLI;
        # decorating it with the device or compute type would force
        # downstream queries to ``LIKE 'large-v3%'`` to find rows.
        return self._model_size

    def _load(self) -> WhisperModel:
        if self._model is not None:
            return self._model
        # Lazy import: pulling ``faster_whisper`` drags in
        # CTranslate2, ctypes-bound cuDNN, and the model weights on
        # first instantiation. The package ships under the
        # ``[transcribe]`` extra; surface a pointed install hint
        # instead of the bare ImportError so an operator missing
        # the extra doesn't have to cross-reference the codebase to
        # find the right install line.
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ImportError(
                "FasterWhisperEngine requires the faster-whisper package, which "
                "ships under the optional `transcribe` extra. Install it with "
                "`uv pip install 'data-pipeline[transcribe]'` "
                "(or pass a pre-loaded model via the constructor)."
            ) from exc

        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        if not audio_path.exists():
            # Surface a typed error rather than letting faster-whisper
            # throw an opaque ``RuntimeError`` from inside the C
            # extension — the worker treats this as "skip + log",
            # not "abort the whole batch".
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        model = self._load()
        wallclock_start = time.perf_counter()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=self._vad_filter,
            beam_size=self._beam_size,
        )
        # ``segments_iter`` is a generator; materialising it is what
        # actually runs the model. Done inside the wallclock window so
        # the recorded time covers the real cost, not just the setup.
        segments = _collect_segments(segments_iter)
        wallclock_seconds = time.perf_counter() - wallclock_start

        # ``info.duration`` is the upstream-reported file length;
        # we want the actually-transcribed span. Sum segment widths
        # to get the post-VAD duration — that's the right numerator
        # for the throughput dashboard.
        transcribed_duration = sum(seg.end - seg.start for seg in segments)

        return TranscriptionResult(
            language=info.language,
            model_version=self.model_version,
            segments=tuple(segments),
            duration_seconds=transcribed_duration,
            wallclock_seconds=wallclock_seconds,
        )


def _collect_segments(segments_iter: Iterable[Any]) -> Sequence[TranscriptSegment]:
    """Project faster-whisper's ``Segment`` namedtuples onto our dataclass.

    Done in a helper so the worker (and tests stubbing the engine)
    don't pull ``faster_whisper`` types into their type signatures.
    """
    out: list[TranscriptSegment] = []
    for seg in segments_iter:
        out.append(
            TranscriptSegment(
                start=float(seg.start),
                end=float(seg.end),
                text=str(seg.text),
                # ``faster-whisper`` doesn't populate speaker today;
                # the diarization flag (BUF-21 note) will land here.
                speaker=None,
            )
        )
    return out
