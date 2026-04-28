"""Valorant-native ingest service: VLR / Riot / Liquipedia -> entity graph -> data/.

Stages defined in config/scheduler.yaml. See docs/ARCHITECTURE.md §5.1.

The BUF-9 ingestion framework lives here. Public surface::

    Connector            # ABC every scraper subclasses
    IngestionRecord      # resolver-ready unit a connector emits
    RateLimit            # per-source token-bucket parameters
    TokenBucket          # the limiter implementation
    run_ingestion        # one-pass orchestrator
    IngestionStats       # outcome counters returned by run_ingestion
    SchemaDriftError     # raise from validate() to skip + log a row
    TransientFetchError  # raise from validate() to skip on a recoverable miss
"""

from data_pipeline.connector import Connector, IngestionRecord, RateLimit
from data_pipeline.errors import IngestionError, SchemaDriftError, TransientFetchError
from data_pipeline.rate_limiter import TokenBucket
from data_pipeline.runner import IngestionStats, run_ingestion

__version__ = "0.0.1"

__all__ = [
    "Connector",
    "IngestionError",
    "IngestionRecord",
    "IngestionStats",
    "RateLimit",
    "SchemaDriftError",
    "TokenBucket",
    "TransientFetchError",
    "run_ingestion",
]
