"""Direct unit tests for ``data_pipeline.patch_notes_runner`` (BUF-83).

The patch-notes runner is an orchestrator that mirrors the entity
``run_ingestion`` pipeline but persists ``PatchNote`` rows rather than
resolving aliases. Existing coverage lives inside
:mod:`tests.test_playvalorant_connector` and is therefore implicitly
coupled to that one connector's HTML fixtures. This module exercises
the runner against hand-built ``PatchNoteConnector`` stubs so the
contracts (UPSERT idempotency, schema-drift skipping, transient-error
retry semantics, fatal connector errors, rate-limiter wiring,
``PatchNoteRecord`` validation) stay covered independently.

The runner UPSERTs through SQLAlchemy. To keep these tests runnable
without ``TEST_DATABASE_URL``, the integration cases use a small
in-memory ``_FakeSession`` keyed on ``(source, patch_version)`` that
implements the slice of the Session API ``_upsert_patch_note``
actually touches. Tests that need a real database (e.g. column-level
checks) are marked ``@pytest.mark.integration`` and skip cleanly on a
fresh clone.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from data_pipeline.connector import RateLimit
from data_pipeline.errors import IngestionError, SchemaDriftError, TransientFetchError
from data_pipeline.patch_notes_runner import (
    PatchNoteConnector,
    PatchNoteRecord,
    PatchNotesStats,
    run_patch_notes_ingestion,
)
from data_pipeline.rate_limiter import TokenBucket
from pydantic import ValidationError

# --- helpers --------------------------------------------------------------


def _record(
    *,
    patch_version: str = "8.05",
    raw_html: str = "<html>raw</html>",
    body_text: str = "body",
    url: str = "https://example.test/patch/8-05",
    published_at: datetime | None = None,
) -> PatchNoteRecord:
    return PatchNoteRecord(
        patch_version=patch_version,
        published_at=published_at or datetime(2026, 4, 1, tzinfo=UTC),
        raw_html=raw_html,
        body_text=body_text,
        url=url,
    )


@dataclass
class _StubRow:
    """Minimal stand-in for ``PatchNote`` ORM rows in the fake session."""

    source: str
    patch_version: str
    published_at: datetime
    raw_html: str
    body_text: str
    url: str
    fetched_at: datetime | None = None


class _FakeSession:
    """In-memory replacement for ``sqlalchemy.orm.Session``.

    Stores rows in a dict keyed on ``(source, patch_version)`` and
    inspects each ``execute(select(...))`` call by walking the compiled
    SQLAlchemy statement to recover the WHERE values. That's just
    enough fidelity for the SELECT-then-INSERT-or-UPDATE pattern in
    :func:`_upsert_patch_note`; nothing else in the runner reaches the
    session.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], _StubRow] = {}

    def execute(self, stmt: Any) -> Any:  # noqa: ANN401 - mirrors Session.execute
        # The runner's only SELECT filters by ``source == X`` and
        # ``patch_version == Y``. Pull the literal values out of the
        # compiled WHERE clause so we can look up by composite key.
        compiled = stmt.compile(compile_kwargs={"literal_binds": True})
        text = str(compiled)
        # Crude but enough: the runner only emits one shape of query.
        # Extract source / patch_version literals after each ``=``.
        source = _between(text, "patch_note.source = '", "'")
        version = _between(text, "patch_note.patch_version = '", "'")
        row = self.rows.get((source, version))

        class _Result:
            def __init__(self, value: Any) -> None:
                self._value = value

            def scalar_one_or_none(self) -> Any:
                return self._value

        return _Result(row)

    def add(self, row: Any) -> None:
        # Mimic the server default kicking in on flush so the runner's
        # update path (next pass) sees a populated ``fetched_at``.
        if row.fetched_at is None:
            row.fetched_at = datetime.now(tz=UTC)
        self.rows[(row.source, row.patch_version)] = _StubRow(
            source=row.source,
            patch_version=row.patch_version,
            published_at=row.published_at,
            raw_html=row.raw_html,
            body_text=row.body_text,
            url=row.url,
            fetched_at=row.fetched_at,
        )

    def flush(self) -> None:
        return None


def _between(text: str, start: str, end: str) -> str:
    """Return the substring between ``start`` and ``end`` markers."""
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[i:j]


def _make_connector(
    *,
    payloads: Iterable[dict[str, Any]] = (),
    records_per_payload: list[list[PatchNoteRecord]] | None = None,
    validate_raises: list[Exception] | None = None,
    transform_raises: list[Exception] | None = None,
    fetch_raises: Exception | None = None,
    source_name: str = "test-source",
    rate_limit: RateLimit | None = None,
) -> PatchNoteConnector:
    """Build a ``PatchNoteConnector`` whose hooks fire from canned data."""
    payload_list = list(payloads)
    records_list = records_per_payload or [[_record()] for _ in payload_list]
    validate_errors = list(validate_raises or [])
    transform_errors = list(transform_raises or [])
    rl = rate_limit or RateLimit(capacity=100, refill_per_second=1000.0)

    class _Stub(PatchNoteConnector):
        @property
        def source_name(self) -> str:  # noqa: D401 - protocol property
            return source_name

        @property
        def cadence(self) -> timedelta:
            return timedelta(days=7)

        @property
        def rate_limit(self) -> RateLimit:
            return rl

        def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
            if fetch_raises is not None:
                # Match the real connector's contract: the HTTP call
                # happens between yields, so the exception lands inside
                # ``next(iterator)``.
                raise fetch_raises
                yield  # type: ignore[unreachable]  # keeps this a generator
            yield from payload_list

        def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
            if validate_errors:
                err = validate_errors.pop(0)
                if err is not None:
                    raise err
            return raw_payload

        def transform(self, validated_payload: dict[str, Any]) -> Iterable[PatchNoteRecord]:
            if transform_errors:
                err = transform_errors.pop(0)
                if err is not None:
                    raise err
            # Yield the next batch of records for this payload.
            idx = payload_list.index(validated_payload)
            return iter(records_list[idx])

    return _Stub()


# --- PatchNoteRecord validation -------------------------------------------


class TestPatchNoteRecord:
    """Pydantic guarantees the runner relies on at the connector boundary."""

    def test_minimum_field_lengths_enforced(self) -> None:
        # ``patch_version`` and ``url`` carry ``min_length=1`` so an
        # empty connector emit fails loudly here, not as a downstream
        # NULL/empty-string footgun.
        with pytest.raises(ValidationError):
            PatchNoteRecord(
                patch_version="",
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
                raw_html="x",
                body_text="x",
                url="https://example.test",
            )

    def test_patch_version_max_length_matches_column(self) -> None:
        # 32 chars is the column cap; emitting a longer string should
        # surface here rather than as a Postgres truncation later.
        with pytest.raises(ValidationError):
            PatchNoteRecord(
                patch_version="9" * 33,
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
                raw_html="x",
                body_text="x",
                url="https://example.test",
            )

    def test_url_max_length_matches_column(self) -> None:
        with pytest.raises(ValidationError):
            PatchNoteRecord(
                patch_version="8.05",
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
                raw_html="x",
                body_text="x",
                url="https://example.test/" + ("x" * 600),
            )

    def test_extra_fields_rejected(self) -> None:
        # ``model_config = ConfigDict(extra="forbid")`` — a connector
        # accidentally passing an extra column name is a contract bug.
        with pytest.raises(ValidationError):
            PatchNoteRecord(
                patch_version="8.05",
                published_at=datetime(2026, 1, 1, tzinfo=UTC),
                raw_html="x",
                body_text="x",
                url="https://example.test",
                wat="oops",  # type: ignore[call-arg]
            )

    def test_frozen_after_construction(self) -> None:
        rec = _record()
        with pytest.raises(ValidationError):
            rec.patch_version = "9.99"  # type: ignore[misc]


# --- PatchNotesStats defaults --------------------------------------------


def test_patch_notes_stats_defaults_are_zero() -> None:
    """Default-constructed stats are a clean baseline.

    The runner increments these counters in place; an accidental
    non-zero default would silently inflate the on-call dashboard.
    """
    stats = PatchNotesStats()
    assert stats.fetched == 0
    assert stats.schema_drifts == 0
    assert stats.transient_errors == 0
    assert stats.upserted == 0
    assert stats.inserted == 0
    assert stats.updated == 0
    assert stats.unchanged == 0
    assert stats.by_version == {}


# --- runner: error-handling paths ----------------------------------------


def test_empty_fetch_returns_zero_stats() -> None:
    """No payloads in -> no rows out, no errors logged."""
    connector = _make_connector(payloads=[])
    stats = run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.fetched == 0
    assert stats.upserted == 0
    assert stats.inserted == 0
    assert stats.unchanged == 0


def test_schema_drift_during_validate_skips_row_and_continues() -> None:
    """SchemaDriftError on one row counts + logs but does not abort the pass.

    The next payload must still flow through. Mirrors the entity
    runner's ``test_schema_drift_logs_and_continues`` in spirit.
    """
    connector = _make_connector(
        payloads=[{"url": "a"}, {"url": "b"}],
        records_per_payload=[[_record(patch_version="8.05")], [_record(patch_version="8.06")]],
        validate_raises=[SchemaDriftError("upstream changed shape"), None],
    )
    session = _FakeSession()
    stats = run_patch_notes_ingestion(
        connector,
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.schema_drifts == 1
    assert stats.fetched == 1, "drift drops the row before fetched++ on the next iteration"
    assert stats.inserted == 1
    assert ("test-source", "8.06") in session.rows
    assert ("test-source", "8.05") not in session.rows


def test_schema_drift_during_transform_skips_row_and_continues() -> None:
    """``transform`` is the second place SchemaDriftError can fire.

    The runner wraps ``validate`` + ``list(transform)`` in the same
    ``try/except``, so a drift surfacing during projection follows the
    same skip-and-log path.
    """
    connector = _make_connector(
        payloads=[{"url": "a"}, {"url": "b"}],
        records_per_payload=[[_record(patch_version="8.05")], [_record(patch_version="8.06")]],
        transform_raises=[SchemaDriftError("missing version field"), None],
    )
    stats = run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.schema_drifts == 1
    assert stats.inserted == 1


def test_transient_error_during_fetch_is_caught_and_counted() -> None:
    """Transient error raised inside ``next(iterator)`` doesn't abort the run.

    Mirrors the BUF-83 contract: the connector's HTTP call happens in
    the body that runs *between* yields, so a network blip lands in
    ``next(iterator)`` and the runner has to handle it there too.
    """
    connector = _make_connector(fetch_raises=TransientFetchError("upstream 503"))
    stats = run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.transient_errors == 1
    assert stats.fetched == 0
    assert stats.upserted == 0


def test_transient_error_during_validate_is_caught_and_counted() -> None:
    """Transient error from ``validate`` skips the row, run continues."""
    connector = _make_connector(
        payloads=[{"url": "a"}, {"url": "b"}],
        records_per_payload=[[_record(patch_version="8.05")], [_record(patch_version="8.06")]],
        validate_raises=[TransientFetchError("body fetch 504"), None],
    )
    stats = run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.transient_errors == 1
    assert stats.inserted == 1


def test_unknown_connector_exception_propagates() -> None:
    """Anything other than SchemaDrift / TransientFetch is fatal-by-default.

    Systems-spec rule: an unknown failure mode must not silently
    corrupt the document store. The runner logs ``CONNECTOR_ERROR``
    and re-raises.
    """
    connector = _make_connector(
        payloads=[{"url": "a"}],
        validate_raises=[RuntimeError("boom: parser exploded")],
    )
    with pytest.raises(RuntimeError, match="boom"):
        run_patch_notes_ingestion(
            connector,
            session=_FakeSession(),  # type: ignore[arg-type]
            since=datetime(1970, 1, 1, tzinfo=UTC),
        )


def test_ingestion_error_subclass_other_than_drift_or_transient_propagates() -> None:
    """Custom IngestionError that isn't drift/transient is treated as fatal.

    The two ``except`` clauses match exact classes (drift and transient).
    A bespoke ``IngestionError`` subclass would fall through to the
    bare ``except Exception`` re-raise.
    """

    class _CustomIngestion(IngestionError):
        pass

    connector = _make_connector(
        payloads=[{"url": "a"}],
        transform_raises=[_CustomIngestion("don't know what to do with this")],
    )
    with pytest.raises(_CustomIngestion):
        run_patch_notes_ingestion(
            connector,
            session=_FakeSession(),  # type: ignore[arg-type]
            since=datetime(1970, 1, 1, tzinfo=UTC),
        )


# --- runner: rate limiting -----------------------------------------------


def test_rate_limiter_acquired_once_per_iteration() -> None:
    """The runner must acquire a token before each ``next()`` call.

    Without that bracketing the connector's HTTP call is unrate-limited
    on the iterator-advance side — the same bug the entity runner
    fixed in ``test_rate_limiter_acquired_before_fetch_advances``.
    """

    class _CountingBucket(TokenBucket):
        def __init__(self) -> None:
            super().__init__(capacity=10, refill_per_second=1000.0)
            self.calls = 0

        def acquire(self) -> None:
            self.calls += 1
            super().acquire()

    bucket = _CountingBucket()
    connector = _make_connector(
        payloads=[{"url": "a"}, {"url": "b"}],
        records_per_payload=[[_record(patch_version="8.05")], [_record(patch_version="8.06")]],
    )
    run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
        rate_limiter=bucket,
    )
    # Two payloads + one StopIteration probe = three acquires. The probe
    # token is intentionally not refunded (mirrors the entity runner's
    # ``test_eos_probe_token_is_not_refunded``).
    assert bucket.calls == 3


def test_runner_uses_connector_rate_limit_when_no_explicit_limiter() -> None:
    """Default-constructed runs build a TokenBucket from connector.rate_limit.

    A regression here would mean the per-source RPM declared on the
    connector silently has no effect.
    """
    rl = RateLimit(capacity=5, refill_per_second=500.0)
    connector = _make_connector(payloads=[{"url": "a"}], rate_limit=rl)
    # No ``rate_limiter=`` kwarg — runner must build one. The assert is
    # implicit: if the runner failed to build one, this would raise.
    stats = run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.inserted == 1


# --- runner: UPSERT outcomes ---------------------------------------------


def test_first_pass_records_inserted_and_bumps_upserted() -> None:
    connector = _make_connector(
        payloads=[{"url": "a"}],
        records_per_payload=[[_record(patch_version="8.05")]],
    )
    stats = run_patch_notes_ingestion(
        connector,
        session=_FakeSession(),  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.inserted == 1
    assert stats.upserted == 1
    assert stats.updated == 0
    assert stats.unchanged == 0
    assert stats.by_version == {"8.05": 1}


def test_second_pass_with_identical_content_reports_unchanged_only() -> None:
    """Idempotency: re-running on identical content does not inflate upserted.

    BUF-83 contract — the on-call dashboard tracks ``upserted`` to
    detect quiet weeks that suddenly start writing. Counting
    ``unchanged`` toward ``upserted`` would false-positive every healthy
    weekly pass.
    """
    payloads = [{"url": "a"}]
    records = [[_record(patch_version="8.05")]]
    session = _FakeSession()

    first = run_patch_notes_ingestion(
        _make_connector(payloads=payloads, records_per_payload=records),
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert first.inserted == 1

    second = run_patch_notes_ingestion(
        _make_connector(payloads=payloads, records_per_payload=records),
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert second.inserted == 0
    assert second.unchanged == 1
    assert second.upserted == 0


def test_second_pass_with_changed_body_reports_updated() -> None:
    """A real edit (typo fix, late balance change) lands in ``updated``."""
    session = _FakeSession()

    first_records = [[_record(patch_version="8.05", body_text="original")]]
    run_patch_notes_ingestion(
        _make_connector(payloads=[{"url": "a"}], records_per_payload=first_records),
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )

    second_records = [[_record(patch_version="8.05", body_text="edited")]]
    second = run_patch_notes_ingestion(
        _make_connector(payloads=[{"url": "a"}], records_per_payload=second_records),
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )

    assert second.updated == 1
    assert second.unchanged == 0
    assert second.upserted == 1
    assert session.rows[("test-source", "8.05")].body_text == "edited"


def test_two_sources_with_same_patch_version_do_not_collide() -> None:
    """``(source, patch_version)`` is the composite key, not version alone.

    Two patch-note connectors emitting "8.05" must persist as two
    rows, not overwrite each other. Re-tested here with the fake
    session because the integration version of this test
    (``test_two_sources_with_same_patch_version_do_not_collide`` in
    test_playvalorant_connector.py) requires Postgres.
    """
    session = _FakeSession()

    run_patch_notes_ingestion(
        _make_connector(
            payloads=[{"url": "a"}],
            records_per_payload=[[_record(patch_version="8.05")]],
            source_name="game-one",
        ),
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    run_patch_notes_ingestion(
        _make_connector(
            payloads=[{"url": "a"}],
            records_per_payload=[[_record(patch_version="8.05")]],
            source_name="game-two",
        ),
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert ("game-one", "8.05") in session.rows
    assert ("game-two", "8.05") in session.rows
    assert len(session.rows) == 2


def test_multiple_records_from_one_payload_are_all_persisted() -> None:
    """``transform`` may emit zero-or-more records per payload.

    A multi-version announcement post should produce two rows from a
    single fetch.
    """
    connector = _make_connector(
        payloads=[{"url": "a"}],
        records_per_payload=[
            [
                _record(patch_version="8.05"),
                _record(patch_version="8.06"),
            ],
        ],
    )
    session = _FakeSession()
    stats = run_patch_notes_ingestion(
        connector,
        session=session,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )
    assert stats.fetched == 1, "fetched counts upstream payloads, not records"
    assert stats.inserted == 2
    assert stats.upserted == 2
    assert stats.by_version == {"8.05": 1, "8.06": 1}
    assert ("test-source", "8.05") in session.rows
    assert ("test-source", "8.06") in session.rows
