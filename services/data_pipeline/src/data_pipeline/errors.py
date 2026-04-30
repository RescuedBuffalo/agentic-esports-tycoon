"""Error taxonomy for the ingestion framework (BUF-9).

The runner distinguishes per-record failures (log + skip the row, keep the
pipeline alive) from fatal failures (propagate, stop the run). Connectors
raise the per-record errors below from inside ``validate``/``transform``;
anything else surfaces as an unhandled exception, which the runner logs
under a distinct event so it stays visible — Systems-spec rule: a single
bad page must not take down a whole crawl.
"""

from __future__ import annotations


class IngestionError(Exception):
    """Base class so callers can ``except IngestionError`` to catch the family."""


class SchemaDriftError(IngestionError):
    """Connector saw a payload that doesn't match its expected schema.

    The runner logs ``SCHEMA_DRIFT`` against the connector and the
    offending row's ``content_hash``, then continues with the next record.
    The raw payload is already persisted by that point, so a maintainer
    can re-run the parser offline against ``raw_record``.
    """


class TransientFetchError(IngestionError):
    """Upstream returned a recoverable error (timeout, 5xx, transient parse).

    Per-record skip; the next scheduled run will retry naturally because
    the row was never deduped (no raw_record was written).
    """


__all__ = ["IngestionError", "SchemaDriftError", "TransientFetchError"]
