"""End-to-end ingestion pipeline tests (BUF-9).

Marked ``integration`` because the pipeline writes through real
``RawRecord`` / ``StagingRecord`` / ``EntityAlias`` rows — the resolver
and dedup paths are exactly what we're trying to exercise. The
``FakeConnector`` replaces the upstream HTTP layer; everything below
the connector boundary runs against the real BUF-6 schema.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
import structlog
from data_pipeline import (
    IngestionRecord,
    RateLimit,
    SchemaDriftError,
    TokenBucket,
    run_ingestion,
)
from esports_sim.db.enums import (
    EntityType,
    Platform,
    ReviewStatus,
    StagingStatus,
)
from esports_sim.db.models import (
    AliasReviewQueue,
    Entity,
    EntityAlias,
    RawRecord,
    StagingRecord,
)
from pipeline_fixtures import FakeConnector, make_player_payload
from sqlalchemy import select

pytestmark = pytest.mark.integration

# A long-ago watermark so connectors yield everything; the runner just
# passes ``since`` through to ``connector.fetch``.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# --- end-to-end happy path --------------------------------------------------


def test_full_pipeline_creates_canonical_and_staging_rows(db_session) -> None:
    """One brand-new player runs all the way through.

    Asserts each downstream side-effect: the raw blob landed under its
    content_hash, the resolver minted a canonical, the alias was
    inserted under the connector's platform, and the staging row was
    written as ``processed`` carrying the new canonical_id.
    """
    connector = FakeConnector(
        payloads=[make_player_payload(platform_id="vlr-tenz", platform_name="TenZ")]
    )

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    assert stats.fetched == 1
    assert stats.processed == 1
    assert stats.duplicates == 0
    assert stats.schema_drifts == 0

    raw = db_session.execute(select(RawRecord)).scalars().all()
    assert len(raw) == 1
    assert raw[0].source == "fake"

    aliases = db_session.execute(select(EntityAlias)).scalars().all()
    assert len(aliases) == 1
    alias = aliases[0]
    assert alias.platform is Platform.VLR
    assert alias.platform_id == "vlr-tenz"
    assert alias.confidence == 1.0

    staging = db_session.execute(select(StagingRecord)).scalars().all()
    assert len(staging) == 1
    sr = staging[0]
    assert sr.status is StagingStatus.PROCESSED
    assert sr.canonical_id == alias.canonical_id
    assert stats.by_status["created"] == 1


def test_pipeline_dedups_repeated_payload(db_session) -> None:
    """Two fetches of the same payload only persist one raw + one alias."""
    payload = make_player_payload(platform_id="vlr-tenz", platform_name="TenZ")
    connector = FakeConnector(payloads=[payload, payload])

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    assert stats.fetched == 1
    assert stats.duplicates == 1
    assert stats.processed == 1

    assert len(db_session.execute(select(RawRecord)).scalars().all()) == 1
    assert len(db_session.execute(select(EntityAlias)).scalars().all()) == 1
    assert len(db_session.execute(select(StagingRecord)).scalars().all()) == 1


def test_pipeline_handles_multi_record_transform(db_session) -> None:
    """One match payload yielding several player records all flow through.

    Real connectors will commonly explode an upstream document into
    multiple :class:`IngestionRecord` rows (a match -> ten players).
    Make sure each record gets its own resolver call and staging row.
    """

    def transform_match(payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        for player in payload["players"]:
            yield IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id=player["id"],
                platform_name=player["name"],
                payload=player,
            )

    connector = FakeConnector(
        payloads=[
            {
                "match_id": "m-1",
                "players": [
                    {"id": "vlr-1", "name": "TenZ"},
                    {"id": "vlr-2", "name": "Sacy"},
                    {"id": "vlr-3", "name": "Zekken"},
                ],
            }
        ],
        transform_fn=transform_match,
    )

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)
    assert stats.processed == 3
    assert len(db_session.execute(select(EntityAlias)).scalars().all()) == 3


def test_pipeline_routes_pending_review_to_review_status(db_session) -> None:
    """Resolver PENDING -> staging row with status=review and null canonical."""
    # Seed players so the new handle lands in the [0.70, 0.90) review band.
    e1 = Entity(entity_type=EntityType.PLAYER)
    e2 = Entity(entity_type=EntityType.PLAYER)
    db_session.add_all([e1, e2])
    db_session.flush()
    db_session.add_all(
        [
            EntityAlias(
                canonical_id=e1.canonical_id,
                platform=Platform.VLR,
                platform_id="vlr-sentinel",
                platform_name="Sentinel",
                confidence=1.0,
            ),
            EntityAlias(
                canonical_id=e2.canonical_id,
                platform=Platform.VLR,
                platform_id="vlr-sinatraa",
                platform_name="Sinatraa",
                confidence=1.0,
            ),
        ]
    )
    db_session.flush()

    # The connector represents the second crawler (Liquipedia) seeing
    # "Sentinal" — close to "Sentinel" but not auto-merge close.
    connector = FakeConnector(
        payloads=[make_player_payload(platform_id="liq-sentinal", platform_name="Sentinal")],
        platform=Platform.LIQUIPEDIA,
    )
    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    assert stats.processed == 1
    assert stats.by_status.get("pending") == 1

    sr = db_session.execute(select(StagingRecord)).scalars().one()
    assert sr.status is StagingStatus.REVIEW
    assert sr.canonical_id is None

    review = db_session.execute(select(AliasReviewQueue)).scalars().one()
    assert review.platform_id == "liq-sentinal"
    assert review.status is ReviewStatus.PENDING


# --- schema drift -----------------------------------------------------------


def test_schema_drift_logs_and_continues(db_session, caplog) -> None:
    """BUF-9 acceptance: drift in one record must not stop the run.

    Three payloads — middle one is malformed and the connector raises
    :class:`SchemaDriftError`. The runner should log ``SCHEMA_DRIFT``,
    persist the raw row anyway (replay-friendly), skip staging for that
    row only, and process the other two normally.
    """

    def picky_validate(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("malformed"):
            raise SchemaDriftError("missing required field 'platform_id'")
        return payload

    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="Alpha"),
            {"malformed": True, "junk": "no schema"},
            make_player_payload(platform_id="vlr-3", platform_name="Gamma"),
        ],
        validate_fn=picky_validate,
    )

    # Wire structlog through stdlib logging so caplog captures it. Default
    # pytest caplog only sees stdlib loggers; ``render_to_log_kwargs`` makes
    # the BoundLogger emit through the stdlib logger of the same name.
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.stdlib.render_to_log_kwargs,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    caplog.set_level(logging.WARNING, logger="data_pipeline.ingestion")

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    assert stats.fetched == 3  # raw rows persisted for all three
    assert stats.schema_drifts == 1
    assert stats.processed == 2
    # Two staging rows; the malformed payload contributed none.
    assert len(db_session.execute(select(StagingRecord)).scalars().all()) == 2

    drift_records = [
        r
        for r in caplog.records
        if "SCHEMA_DRIFT" in r.getMessage() or getattr(r, "code", None) == "SCHEMA_DRIFT"
    ]
    assert drift_records, "expected at least one SCHEMA_DRIFT log line; got: " + ", ".join(
        r.getMessage() for r in caplog.records
    )


def test_unexpected_connector_exception_logs_connector_error_and_propagates(
    db_session, caplog
) -> None:
    """Regression for a Codex review finding.

    A connector error that isn't ``IngestionError`` (e.g., ``KeyError``
    from a parser regression) must surface a structured
    ``CONNECTOR_ERROR`` log line — carrying the bound ``source`` and
    ``content_hash`` context — *before* the exception propagates and
    aborts the run. Without that, postmortems lose the failing-payload
    trail even though the run correctly stopped.
    """

    def buggy_validate(payload: dict[str, Any]) -> dict[str, Any]:
        # Mimic a parser regression: a previously-stable field has gone
        # missing and the connector hasn't been updated to handle it.
        return {"normalised": payload["nonexistent_field"]}

    connector = FakeConnector(
        payloads=[make_player_payload(platform_id="vlr-1", platform_name="A")],
        validate_fn=buggy_validate,
    )

    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.EventRenamer("event"),
            structlog.stdlib.render_to_log_kwargs,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )
    caplog.set_level(logging.ERROR, logger="data_pipeline.ingestion")

    with pytest.raises(KeyError):
        run_ingestion(connector, session=db_session, since=_EPOCH)

    connector_error_records = [
        r
        for r in caplog.records
        if "CONNECTOR_ERROR" in r.getMessage() or getattr(r, "code", None) == "CONNECTOR_ERROR"
    ]
    assert (
        connector_error_records
    ), "expected a CONNECTOR_ERROR log line before the exception propagated; got: " + ", ".join(
        r.getMessage() for r in caplog.records
    )


# --- rate limiting wired through ------------------------------------------


def test_runner_consults_rate_limiter_per_record(db_session) -> None:
    """Every ``fetch`` yield should trip the bucket exactly once.

    Use a real :class:`TokenBucket` with capacity 100 (so it never
    blocks) but a counting wrapper so the test can assert the runner
    actually called it.
    """
    real_bucket = TokenBucket(capacity=100, refill_per_second=100.0)
    calls = {"n": 0}

    class CountingBucket:
        def acquire(self) -> None:
            calls["n"] += 1
            real_bucket.acquire()

    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="A"),
            make_player_payload(platform_id="vlr-2", platform_name="B"),
            make_player_payload(platform_id="vlr-3", platform_name="C"),
        ],
        rate_limit=RateLimit(capacity=10, refill_per_second=10.0),
    )

    run_ingestion(
        connector,
        session=db_session,
        since=_EPOCH,
        rate_limiter=CountingBucket(),  # type: ignore[arg-type]
    )

    assert calls["n"] == 3


def test_runner_uses_connector_rate_limit_when_no_explicit_limiter(db_session) -> None:
    """Runner builds its own bucket from ``connector.rate_limit`` if not given.

    Sanity check; we don't measure timing, just that the call doesn't
    error out and stats reflect the run.
    """
    connector = FakeConnector(
        payloads=[make_player_payload(platform_id="vlr-1", platform_name="A")],
        rate_limit=RateLimit(capacity=2, refill_per_second=2.0),
    )
    stats = run_ingestion(connector, session=db_session, since=_EPOCH)
    assert stats.processed == 1


# --- since-watermark contract ---------------------------------------------


def test_runner_passes_since_to_connector(db_session) -> None:
    """``since`` is the connector's contract for "what's new" — pass it through."""
    connector = FakeConnector(payloads=[])
    since = datetime(2026, 4, 1, tzinfo=UTC)
    run_ingestion(connector, session=db_session, since=since)
    assert connector.fetch_calls == [since]


def test_cadence_metadata_round_trips() -> None:
    """The scheduler-facing ``cadence`` property is preserved through the ABC."""
    connector = FakeConnector(payloads=[], cadence=timedelta(minutes=15))
    assert connector.cadence == timedelta(minutes=15)


# --- rate-limit ordering regression (Codex P1) ----------------------------


def test_rate_limiter_acquired_before_fetch_advances(db_session) -> None:
    """``acquire`` must run *before* each ``next()`` on the fetch iterator.

    Regression for a Codex review finding: the runner used to call
    ``rate_limiter.acquire()`` inside the loop body, which fires *after*
    Python has already advanced the generator and run the connector's
    upstream HTTP code. The fix moves acquisition between iterator
    advances. We verify by snapshotting the limiter's acquire-count
    just before each yield and asserting the count is at least
    ``yield_index + 1`` — i.e., a token had already been pulled by
    that point.
    """
    bucket = TokenBucket(capacity=10, refill_per_second=1000.0)
    acquire_count = {"n": 0}
    real_acquire = bucket.acquire

    def counting_acquire() -> None:
        acquire_count["n"] += 1
        real_acquire()

    bucket.acquire = counting_acquire  # type: ignore[method-assign]

    yield_observations: list[tuple[int, int]] = []

    class OrderTracker(FakeConnector):
        def fetch(self, since):  # type: ignore[no-untyped-def]
            for i, payload in enumerate(self._payloads):
                # Snapshot acquire-count before yielding. If the runner
                # is correct, the limiter has already deducted a token
                # by this moment — i.e., ``acquire_count`` >= i+1.
                yield_observations.append((i, acquire_count["n"]))
                yield payload

    connector = OrderTracker(
        payloads=[
            make_player_payload(platform_id=f"vlr-{i}", platform_name=f"P{i}") for i in range(3)
        ],
    )

    run_ingestion(connector, session=db_session, since=_EPOCH, rate_limiter=bucket)

    assert yield_observations, "fetch never yielded; ordering check is moot"
    for yield_idx, observed_acquire_count in yield_observations:
        assert observed_acquire_count >= yield_idx + 1, (
            f"yield {yield_idx} happened with acquire_count={observed_acquire_count}; "
            "limiter must run before the iterator advances"
        )


# --- per-record errors raised from transform (Codex P1) -------------------


def test_schema_drift_raised_from_transform_does_not_abort_run(db_session) -> None:
    """Regression for a Codex review finding.

    ``errors.py`` documents ``SchemaDriftError`` as a per-record error
    raised from ``validate`` *or* ``transform``. The runner used to
    catch it only around ``validate``, so a ``transform`` that raised
    drift would escape and abort the entire pass. The fix wraps the
    materialised ``transform(...)`` call in the same handler.
    """

    def transform_with_drift(payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        if payload.get("trigger_drift"):
            raise SchemaDriftError("transform: required key 'platform_id' missing")
        yield IngestionRecord(
            entity_type=EntityType.PLAYER,
            platform_id=payload["platform_id"],
            platform_name=payload["platform_name"],
            payload=payload,
        )

    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="Alpha"),
            {"trigger_drift": True, "junk": True},
            make_player_payload(platform_id="vlr-3", platform_name="Gamma"),
        ],
        transform_fn=transform_with_drift,
    )

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    # All three raw payloads landed (raw-first persistence happens before
    # validate/transform), but only the two well-formed ones produced a
    # staging row. The drifting middle payload bumps the schema_drifts
    # counter and the run continues.
    assert stats.fetched == 3
    assert stats.schema_drifts == 1
    assert stats.processed == 2
    assert len(db_session.execute(select(StagingRecord)).scalars().all()) == 2


def test_partial_transform_drop_on_mid_iteration_drift(db_session) -> None:
    """A transform that yields N records and *then* raises drift drops all N.

    Resolver writes are payload-level: once the connector signals the
    payload is malformed, half-translated records would corrupt the
    staging table. The runner materialises ``transform``'s output so
    the drift is caught synchronously and no record from the bad
    payload reaches the resolver.
    """

    def transform_partial(payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        for player in payload["players"]:
            if player.get("malformed"):
                raise SchemaDriftError("player record missing 'id'")
            yield IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id=player["id"],
                platform_name=player["name"],
                payload=player,
            )

    connector = FakeConnector(
        payloads=[
            {
                "match_id": "m-1",
                "players": [
                    {"id": "vlr-1", "name": "TenZ"},
                    {"id": "vlr-2", "name": "Sacy"},
                    {"malformed": True},
                    {"id": "vlr-4", "name": "Zekken"},
                ],
            }
        ],
        transform_fn=transform_partial,
    )

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)
    assert stats.fetched == 1
    assert stats.schema_drifts == 1
    # No staging rows for the broken payload — the eager-list materialisation
    # means we only commit a payload's records once the whole transform
    # succeeds.
    assert stats.processed == 0
    assert len(db_session.execute(select(StagingRecord)).scalars().all()) == 0


# --- transient errors retain retryability (Codex P2) ----------------------


def test_transient_error_does_not_persist_raw(db_session) -> None:
    """Regression for a Codex review finding.

    ``TransientFetchError`` is documented as a recoverable per-record
    skip whose ``content_hash`` must NOT land in ``raw_record``;
    otherwise the next scheduled run dedups the same payload and the
    temporary upstream blip becomes a permanent drop. The runner used
    to persist raw before validate, breaking that contract.
    """
    from data_pipeline import TransientFetchError

    def flaky_validate(payload: dict[str, Any]) -> dict[str, Any]:
        if payload.get("flaky"):
            raise TransientFetchError("upstream returned 503 mid-parse")
        return payload

    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="Alpha"),
            {"flaky": True, "platform_id": "vlr-2", "platform_name": "Beta"},
            make_player_payload(platform_id="vlr-3", platform_name="Gamma"),
        ],
        validate_fn=flaky_validate,
    )

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    assert stats.transient_errors == 1
    assert stats.processed == 2
    # Two raw rows landed (the well-formed payloads); the transient one
    # left no audit trail so the next run can re-fetch it.
    raw_rows = db_session.execute(select(RawRecord)).scalars().all()
    assert len(raw_rows) == 2


def test_transient_error_payload_can_succeed_on_retry(db_session) -> None:
    """The same payload that hit a transient error must retry cleanly.

    Simulates two ingestion passes against the same FakeConnector
    state. First pass: validate raises ``TransientFetchError`` for one
    payload. Second pass: the upstream has recovered, validate passes,
    the payload reaches the resolver and writes a staging row.
    """
    from data_pipeline import TransientFetchError

    fail_once = {"done": False}

    def recovering_validate(payload: dict[str, Any]) -> dict[str, Any]:
        if payload["platform_id"] == "vlr-flaky" and not fail_once["done"]:
            fail_once["done"] = True
            raise TransientFetchError("first attempt: timeout")
        return payload

    payloads = [make_player_payload(platform_id="vlr-flaky", platform_name="Flaky")]

    first_run = run_ingestion(
        FakeConnector(payloads=payloads, validate_fn=recovering_validate),
        session=db_session,
        since=_EPOCH,
    )
    assert first_run.transient_errors == 1
    assert first_run.processed == 0
    assert len(db_session.execute(select(RawRecord)).scalars().all()) == 0

    second_run = run_ingestion(
        FakeConnector(payloads=payloads, validate_fn=recovering_validate),
        session=db_session,
        since=_EPOCH,
    )
    # Second pass observes the same content_hash but no prior raw row,
    # so dedup doesn't fire — validate succeeds, resolver runs, staging
    # row appears.
    assert second_run.transient_errors == 0
    assert second_run.processed == 1
    assert len(db_session.execute(select(StagingRecord)).scalars().all()) == 1


# --- unexpected resolver exceptions propagate (Codex P1) ------------------


def test_unexpected_resolver_exception_is_not_swallowed(db_session, monkeypatch) -> None:
    """Regression for a Codex review finding.

    ``_process_record`` used to ``except Exception`` over the resolver
    call, downgrading every failure — including non-recoverable DB
    faults — to a logged per-record skip. The fix narrows the catch to
    :class:`IngestionError`; anything else propagates so the
    fatal-by-default rule holds.
    """
    from data_pipeline import runner

    def boom(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        # Simulate the kind of fault ``resolve_entity`` would explicitly
        # re-raise — e.g., a unique-constraint violation that the
        # savepoint couldn't recover from.
        raise RuntimeError("alias unique violation recovery: post-conflict lookup missed")

    monkeypatch.setattr(runner, "resolve_entity", boom)

    connector = FakeConnector(
        payloads=[make_player_payload(platform_id="vlr-x", platform_name="X")]
    )

    with pytest.raises(RuntimeError, match="post-conflict lookup missed"):
        run_ingestion(connector, session=db_session, since=_EPOCH)


def test_ingestion_error_from_resolver_is_still_skipped(db_session, monkeypatch) -> None:
    """The narrow ``except IngestionError`` branch still works — only
    ``IngestionError`` and its subclasses get the per-record skip."""
    from data_pipeline import IngestionError, runner

    def soft_fail(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise IngestionError("connector-side validate leaked an error")

    monkeypatch.setattr(runner, "resolve_entity", soft_fail)

    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="A"),
            make_player_payload(platform_id="vlr-2", platform_name="B"),
        ]
    )

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)
    # Both records hit the soft-error branch; the run completes.
    assert stats.resolver_errors == 2
    assert stats.processed == 0


# --- source-scoped dedup (Codex P1) --------------------------------------


def test_dedup_is_scoped_by_connector_source(db_session) -> None:
    """Two connectors emitting byte-identical JSON each persist a raw row.

    Regression for a Codex review finding: the dedup hash used to ignore
    ``source``, so the second connector's payload would collide on
    ``content_hash`` with the first's and be silently dropped before
    the resolver ran. The fix folds ``source`` into the hash input.
    """
    payload = make_player_payload(platform_id="shared-id", platform_name="Shared")

    connector_vlr = FakeConnector(
        payloads=[payload],
        source_name="vlr",
        platform=Platform.VLR,
    )
    connector_liq = FakeConnector(
        payloads=[payload],
        source_name="liquipedia",
        platform=Platform.LIQUIPEDIA,
    )

    stats_vlr = run_ingestion(connector_vlr, session=db_session, since=_EPOCH)
    stats_liq = run_ingestion(connector_liq, session=db_session, since=_EPOCH)

    assert stats_vlr.processed == 1
    assert stats_vlr.duplicates == 0
    # Second connector must not be deduped against the first.
    assert stats_liq.processed == 1
    assert stats_liq.duplicates == 0

    raw_rows = db_session.execute(select(RawRecord)).scalars().all()
    assert len(raw_rows) == 2
    assert {r.source for r in raw_rows} == {"vlr", "liquipedia"}
    # Hashes must differ — that's what makes the second insert legal under
    # the global ``content_hash`` unique constraint.
    assert raw_rows[0].content_hash != raw_rows[1].content_hash


def test_same_source_still_dedups_on_repeat(db_session) -> None:
    """Source-scoping mustn't break within-source dedup.

    A single connector that yields the same payload twice should still
    only write one raw row — that's the point of the content_hash
    fingerprint in the first place.
    """
    payload = make_player_payload(platform_id="vlr-tenz", platform_name="TenZ")
    connector = FakeConnector(payloads=[payload, payload])

    stats = run_ingestion(connector, session=db_session, since=_EPOCH)

    assert stats.fetched == 1
    assert stats.duplicates == 1
    assert len(db_session.execute(select(RawRecord)).scalars().all()) == 1


# --- token refund on EOS (Codex P2) --------------------------------------


def test_eos_probe_token_is_not_refunded(db_session) -> None:
    """The runner must not refund the EOS-probe token.

    Regression for a Codex P1 finding: paginated connectors legitimately
    perform an HTTP call inside the final ``next()`` (fetching an empty
    page, or polling for "is there new data?") and then return without
    yielding. Refunding the token in that case would let repeated empty
    polls bypass the limiter and exceed provider quotas. The trade-off
    is that materialised-list connectors waste one token per pass —
    acceptable in exchange for never under-charging real polls.

    With capacity=3 and two consumed items, the bucket should hold
    exactly zero tokens after the run (item 1, item 2, EOS probe = 3
    acquires; refill rate is set negligibly low so wall-clock time
    can't restore tokens during the test).
    """
    bucket = TokenBucket(capacity=3, refill_per_second=0.001)
    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="Alpha"),
            make_player_payload(platform_id="vlr-2", platform_name="Beta"),
        ],
    )

    run_ingestion(connector, session=db_session, since=_EPOCH, rate_limiter=bucket)

    # 3 acquires happened (item 1, item 2, EOS probe); none were refunded.
    # The bucket is exhausted within the bounds of the slow refill.
    assert bucket.try_acquire() is False


def test_limiter_minimal_contract_is_just_acquire(db_session) -> None:
    """The runner only requires ``acquire`` from a limiter shim.

    Tests, instrumentation wrappers, and alternative limiter
    implementations should be able to implement just one method.
    Verifying this explicitly guards against future code (e.g.,
    additional bookkeeping calls in ``_rate_limited``) re-introducing
    a wider implicit contract.
    """
    real_bucket = TokenBucket(capacity=10, refill_per_second=10.0)
    acquire_calls = {"n": 0}

    class AcquireOnlyLimiter:
        """Deliberately exposes only ``acquire``."""

        def acquire(self) -> None:
            acquire_calls["n"] += 1
            real_bucket.acquire()

    connector = FakeConnector(
        payloads=[
            make_player_payload(platform_id="vlr-1", platform_name="A"),
            make_player_payload(platform_id="vlr-2", platform_name="B"),
        ]
    )

    stats = run_ingestion(
        connector,
        session=db_session,
        since=_EPOCH,
        rate_limiter=AcquireOnlyLimiter(),  # type: ignore[arg-type]
    )

    assert stats.processed == 2
    # 2 yields + 1 EOS probe = 3 acquires. Pagination correctness
    # demands we charge for the probe; see ``_rate_limited`` docstring.
    assert acquire_calls["n"] == 3
