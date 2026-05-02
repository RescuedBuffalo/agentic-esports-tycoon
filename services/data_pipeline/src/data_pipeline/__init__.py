"""Valorant-native ingest service: VLR / Riot -> entity graph -> data/.

Stages defined in config/scheduler.yaml. See docs/ARCHITECTURE.md §5.1.

The BUF-9 ingestion framework lives here. Public surface::

    Connector                     # ABC every entity scraper subclasses
    IngestionRecord               # resolver-ready unit an entity connector emits
    RateLimit                     # per-source token-bucket parameters
    TokenBucket                   # the limiter implementation
    run_ingestion                 # one-pass entity orchestrator
    IngestionStats                # outcome counters returned by run_ingestion
    SchemaDriftError              # raise from validate() to skip + log a row
    TransientFetchError           # raise from validate() to skip on a recoverable miss

BUF-83 patch-notes path (parallel ABC, separate orchestrator)::

    PatchNoteConnector            # ABC for document-shaped sources
    PatchNoteRecord               # one patch's projected columns
    PatchNotesStats               # outcome counters
    run_patch_notes_ingestion     # patch-notes orchestrator (UPSERTs PatchNote)
"""

from data_pipeline.connector import Connector, IngestionRecord, RateLimit
from data_pipeline.errors import IngestionError, SchemaDriftError, TransientFetchError
from data_pipeline.patch_notes_runner import (
    PatchNoteConnector,
    PatchNoteRecord,
    PatchNotesStats,
    run_patch_notes_ingestion,
)
from data_pipeline.rate_limiter import TokenBucket
from data_pipeline.runner import IngestionStats, run_ingestion

__version__ = "0.0.1"

__all__ = [
    "Connector",
    "IngestionError",
    "IngestionRecord",
    "IngestionStats",
    "PatchNoteConnector",
    "PatchNoteRecord",
    "PatchNotesStats",
    "RateLimit",
    "SchemaDriftError",
    "TokenBucket",
    "TransientFetchError",
    "run_ingestion",
    "run_patch_notes_ingestion",
]
