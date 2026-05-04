"""Local Whisper transcription pipeline (BUF-21).

Public surface::

    TranscriptionEngine          # Protocol every Whisper backend implements
    TranscriptionResult          # what the engine returns for one file
    TranscriptSegment            # one timestamped segment in a result
    FasterWhisperEngine          # default implementation (5090 + faster-whisper)
    transcribe_pending           # batch worker: pulls untranscribed media
    TranscribePendingStats       # outcome counters for one worker pass

The engine is a Protocol so the worker (and tests) can swap in a
deterministic stub without dragging the 3 GB ``faster-whisper`` model
into every CI install. The default :class:`FasterWhisperEngine` lazy-
imports its heavy dependency so the data-pipeline package stays cheap
to import for callers that only need the entity ingest pipeline.
"""

from data_pipeline.transcribe.engine import (
    FasterWhisperEngine,
    TranscriptionEngine,
    TranscriptionResult,
    TranscriptSegment,
)
from data_pipeline.transcribe.worker import (
    TranscribePendingStats,
    transcribe_pending,
)

__all__ = [
    "FasterWhisperEngine",
    "TranscribePendingStats",
    "TranscriptSegment",
    "TranscriptionEngine",
    "TranscriptionResult",
    "transcribe_pending",
]
