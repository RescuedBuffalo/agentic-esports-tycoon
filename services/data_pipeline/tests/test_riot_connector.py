"""Tests for the Riot API connector (BUF-82).

Unit cases drive the connector with a fake ``http_get`` so the test
suite is offline-safe — no real Riot API key required. The integration
case at the bottom only runs when ``TEST_DATABASE_URL`` is set; that
gate keeps a fresh clone green while still exercising the full
``run_ingestion`` -> ``raw_record`` / ``staging_record`` / ``entity_alias``
chain when a Postgres is available.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from data_pipeline import (
    IngestionRecord,
    SchemaDriftError,
    TransientFetchError,
    run_ingestion,
)
from data_pipeline.connectors.riot import RiotConnector, _parse_retry_after
from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import EntityAlias, RawRecord, StagingRecord
from sqlalchemy import select

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "riot"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / name).read_text())


# Watermark older than every fixture match so ``fetch`` yields the lot.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)

# Matches ``match-001`` (1714e9 ms) but not ``match-old`` (1700e9 ms).
# Used by the since-watermark test below.
_BETWEEN_OLD_AND_NEW = datetime(2024, 4, 1, tzinfo=UTC)


# --- fake HTTP plumbing ----------------------------------------------------


class _FakeHttp:
    """Tiny URL-routed fake.

    Each test wires a route map: ``{path_substring: response_dict}`` (or
    a callable for stateful behaviour). The connector's ``_get`` reads
    ``status_code``, ``headers``, and ``json`` keys exactly the way the
    real httpx-backed factory shapes them.
    """

    def __init__(self, routes: dict[str, Any]) -> None:
        # Values are either a literal response dict or a callable
        # ``(url, params) -> response``; we don't bother typing the
        # union because tests aren't part of the strict-mypy surface.
        self._routes = routes
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((url, dict(params)))
        for needle, resp in self._routes.items():
            if needle in url:
                if callable(resp):
                    return resp(url, params)  # type: ignore[no-any-return]
                return copy.deepcopy(resp)  # type: ignore[no-any-return]
        # Surface the unmatched URL clearly — easier to diagnose than a
        # KeyError from the connector trying to unpack a missing key.
        raise AssertionError(f"FakeHttp: no route configured for {url}")


def _ok(json_body: dict[str, Any]) -> dict[str, Any]:
    return {"status_code": 200, "headers": {}, "json": json_body}


def _ok_match() -> dict[str, Any]:
    return _ok({"match": _load_fixture("match.json")})


def _matchlist_response() -> dict[str, Any]:
    return _ok(_load_fixture("matchlist.json"))


def _make_connector(
    *,
    http_get: Any,
    seed_puuids: list[str] | None = None,
    sleeper: Any = None,
) -> RiotConnector:
    """Helper: build a RiotConnector with a no-op default sleeper.

    Tests that care about the sleep trace pass their own list-appender
    sleeper; tests that don't care use this one and never observe it.
    """

    def default_sleeper(_seconds: float) -> None:
        return None

    return RiotConnector(
        seed_puuids=seed_puuids or ["PUUID-SEED-AAAA"],
        http_get=http_get,
        sleeper=sleeper or default_sleeper,
    )


# --- metadata & construction ----------------------------------------------


def test_metadata_properties_match_spec() -> None:
    connector = _make_connector(http_get=_FakeHttp({}))
    assert connector.source_name == "riot_api"
    assert connector.platform is Platform.RIOT_API
    assert connector.entity_types == (EntityType.PLAYER,)
    assert connector.cadence.days == 1
    # Conservative bucket: ~10 rpm steady, burst 20.
    assert connector.rate_limit.capacity == 20
    assert connector.rate_limit.refill_per_second == pytest.approx(20.0 / 120.0)


def test_constructor_rejects_empty_seed_list() -> None:
    with pytest.raises(ValueError, match="at least one seed PUUID"):
        RiotConnector(seed_puuids=[], http_get=_FakeHttp({}))


# --- transform / validate -------------------------------------------------


def test_transform_yields_one_record_per_player() -> None:
    """A 10-player match payload produces exactly 10 IngestionRecords.

    Each record carries the player's PUUID as ``platform_id`` and the
    Riot identity tag ``gameName#tagLine`` as ``platform_name``.
    """
    connector = _make_connector(http_get=_FakeHttp({}))
    payload: dict[str, Any] = {
        "match_id": "match-001",
        "puuid_seed": "PUUID-SEED-AAAA",
        "match": _load_fixture("match.json"),
    }

    records: list[IngestionRecord] = list(connector.transform(connector.validate(payload)))

    assert len(records) == 10
    ids = [r.platform_id for r in records]
    names = [r.platform_name for r in records]
    assert "PUUID-PLAYER-01" in ids
    assert "Alpha#NA1" in names
    # All ten platform_ids are distinct.
    assert len(set(ids)) == 10
    # Each record's payload is shaped for downstream stat extraction.
    sample = next(r for r in records if r.platform_id == "PUUID-PLAYER-01")
    assert sample.payload["match_id"] == "match-001"
    assert "match_info" in sample.payload
    assert sample.payload["this_player"]["puuid"] == "PUUID-PLAYER-01"
    assert "rounds" in sample.payload
    assert "roster_puuids" in sample.payload
    assert len(sample.payload["roster_puuids"]) == 10


def test_transform_drops_players_missing_identity() -> None:
    """Anonymised player blocks (no gameName/tagLine) are skipped, not raised.

    The Riot API ships these for dev accounts and customs; treating them
    as schema drift would falsely flag every customs match. The
    connector logs a warning and emits records only for the identifiable
    players.
    """
    connector = _make_connector(http_get=_FakeHttp({}))
    fixture = _load_fixture("match.json")
    fixture["players"][0].pop("gameName")
    payload = {"match_id": "match-001", "puuid_seed": "X", "match": fixture}

    records = list(connector.transform(connector.validate(payload)))
    assert len(records) == 9  # 10 minus the anonymised one
    assert "PUUID-PLAYER-01" not in [r.platform_id for r in records]


def test_validate_raises_schema_drift_on_missing_match_info() -> None:
    connector = _make_connector(http_get=_FakeHttp({}))
    fixture = _load_fixture("match.json")
    fixture.pop("matchInfo")
    payload = {"match_id": "x", "puuid_seed": "y", "match": fixture}

    with pytest.raises(SchemaDriftError, match="matchInfo"):
        connector.validate(payload)


def test_validate_raises_schema_drift_on_missing_players() -> None:
    connector = _make_connector(http_get=_FakeHttp({}))
    fixture = _load_fixture("match.json")
    fixture.pop("players")
    payload = {"match_id": "x", "puuid_seed": "y", "match": fixture}

    with pytest.raises(SchemaDriftError, match="players"):
        connector.validate(payload)


def test_validate_raises_schema_drift_on_non_dict_match_block() -> None:
    """``match`` field present but not a dict — drift, preserve raw for triage."""
    connector = _make_connector(http_get=_FakeHttp({}))
    with pytest.raises(SchemaDriftError, match="match"):
        connector.validate({"match_id": "x", "puuid_seed": "y", "match": "not a dict"})


# --- fetch: 429 / Retry-After / 5xx ---------------------------------------


def test_fetch_raises_transient_fetch_error_on_429() -> None:
    """A 429 on the matchlist endpoint short-circuits the seed.

    The runner skips the row without persisting raw, so the next pass
    retries the same seed — that's the contract.
    """
    sleeps: list[float] = []
    http = _FakeHttp(
        {
            "matchlists/by-puuid": {
                "status_code": 429,
                "headers": {"Retry-After": "2"},
                "json": {"status": {"message": "Rate limit exceeded"}},
            }
        }
    )
    connector = _make_connector(http_get=http, sleeper=sleeps.append)

    with pytest.raises(TransientFetchError, match="429"):
        list(connector.fetch(_EPOCH))

    # ``Retry-After: 2`` was honoured before raising.
    assert sleeps == [2.0]


def test_fetch_caps_retry_after_defensively() -> None:
    """A garbage Retry-After value is capped, not honoured verbatim.

    A buggy or malicious upstream that sends Retry-After: 1000000 must
    not wedge the run for a million seconds.
    """
    sleeps: list[float] = []
    http = _FakeHttp(
        {
            "matchlists/by-puuid": {
                "status_code": 429,
                "headers": {"Retry-After": "1000000"},
                "json": None,
            }
        }
    )
    connector = _make_connector(http_get=http, sleeper=sleeps.append)
    with pytest.raises(TransientFetchError):
        list(connector.fetch(_EPOCH))
    assert sleeps == [120.0]  # the module's _MAX_RETRY_AFTER_SECONDS


def test_fetch_handles_missing_retry_after_header() -> None:
    """No Retry-After header — no sleep, but still TransientFetchError."""
    sleeps: list[float] = []
    http = _FakeHttp(
        {"matchlists/by-puuid": {"status_code": 429, "headers": {}, "json": None}}
    )
    connector = _make_connector(http_get=http, sleeper=sleeps.append)
    with pytest.raises(TransientFetchError):
        list(connector.fetch(_EPOCH))
    assert sleeps == []  # nothing slept; bucket alone gates the next attempt


def test_fetch_raises_transient_on_5xx() -> None:
    http = _FakeHttp(
        {"matchlists/by-puuid": {"status_code": 503, "headers": {}, "json": None}}
    )
    connector = _make_connector(http_get=http)
    with pytest.raises(TransientFetchError, match="503"):
        list(connector.fetch(_EPOCH))


def test_fetch_raises_schema_drift_on_400() -> None:
    """Non-rate-limit, non-5xx errors are payload bugs — preserve via SchemaDrift."""
    http = _FakeHttp(
        {"matchlists/by-puuid": {"status_code": 404, "headers": {}, "json": None}}
    )
    connector = _make_connector(http_get=http)
    with pytest.raises(SchemaDriftError, match="404"):
        list(connector.fetch(_EPOCH))


def test_fetch_translates_unknown_http_get_exception_to_transient() -> None:
    """Raw network errors from the http_get shim become TransientFetchError.

    A connect-timeout shouldn't burn the retry slot — the runner needs
    to skip without persisting raw so the next pass re-tries.
    """

    def boom(url: str, params: dict[str, Any]) -> dict[str, Any]:
        raise OSError("connect timeout")

    connector = _make_connector(http_get=boom)
    with pytest.raises(TransientFetchError, match="connect timeout"):
        list(connector.fetch(_EPOCH))


# --- fetch: since watermark ----------------------------------------------


def test_fetch_skips_matches_at_or_before_since() -> None:
    """``gameStartTimeMillis <= since`` -> match is filtered before the GET.

    The matchlist contains three matches; with ``since`` set between
    ``match-old`` and the two newer entries, we should never call the
    detail endpoint for ``match-old``.
    """
    http = _FakeHttp(
        {
            "matchlists/by-puuid": _matchlist_response(),
            "matches/match-001": _ok_match(),
            "matches/match-002": _ok_match(),
        }
    )
    connector = _make_connector(http_get=http)

    payloads = list(connector.fetch(_BETWEEN_OLD_AND_NEW))

    assert {p["match_id"] for p in payloads} == {"match-001", "match-002"}
    # ``match-old`` was never requested.
    assert all("match-old" not in url for url, _ in http.calls)


def test_fetch_filters_at_inclusive_boundary() -> None:
    """A match whose timestamp equals ``since`` is excluded (strict >).

    The runner uses ``since`` as the high-water mark of the previous
    pass; anything at or before it is by definition already known.
    """
    http = _FakeHttp(
        {
            "matchlists/by-puuid": _matchlist_response(),
            "matches/match-001": _ok_match(),
            "matches/match-002": _ok_match(),
        }
    )
    connector = _make_connector(http_get=http)
    # 1714e9 ms == match-001's start. Equal => filtered.
    boundary = datetime.fromtimestamp(1_714_000_000, tz=UTC)
    payloads = list(connector.fetch(boundary))
    assert {p["match_id"] for p in payloads} == {"match-002"}


def test_fetch_skips_matchlist_entries_without_match_id() -> None:
    """A malformed matchlist entry skips that match — doesn't abort the seed."""
    matchlist = _load_fixture("matchlist.json")
    matchlist["history"].append({"gameStartTimeMillis": 1_716_000_000_000})  # no matchId
    http = _FakeHttp(
        {
            "matchlists/by-puuid": _ok(matchlist),
            "matches/match-001": _ok_match(),
            "matches/match-002": _ok_match(),
            "matches/match-old": _ok_match(),
        }
    )
    connector = _make_connector(http_get=http)
    # ``_EPOCH`` -> include all three with valid matchIds; ignore the 4th
    payloads = list(connector.fetch(_EPOCH))
    assert {p["match_id"] for p in payloads} == {"match-001", "match-002", "match-old"}


# --- _parse_retry_after edge cases ---------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, 0.0),
        ("", 0.0),
        ("not-a-number", 0.0),
        ("-5", 0.0),
        ("0", 0.0),
        ("1.5", 1.5),
        ("60", 60.0),
        ("999999", 120.0),
    ],
)
def test_parse_retry_after_handles_edge_cases(value: Any, expected: float) -> None:
    assert _parse_retry_after(value) == expected


# --- end-to-end with run_ingestion (skipped without TEST_DATABASE_URL) ----


@pytest.mark.integration
def test_full_pipeline_creates_one_staging_row_per_player(db_session) -> None:  # type: ignore[no-untyped-def]
    """Drive ``run_ingestion`` against a real DB session.

    Asserts the contract end-to-end: one IngestionRecord per player ->
    one alias + one staging row per player, all linked to the same
    canonical entity for that player.
    """
    http = _FakeHttp(
        {
            "matchlists/by-puuid": _matchlist_response(),
            "matches/match-001": _ok_match(),
            "matches/match-002": _ok_match(),
        }
    )
    connector = _make_connector(http_get=http)

    stats = run_ingestion(connector, session=db_session, since=_BETWEEN_OLD_AND_NEW)

    # Two matches, ten players each => 20 staging rows + 20 aliases.
    # (Both matches use the same player roster in the fixture, so
    # players 1..10 each get two staging rows but only one alias —
    # alias upsert keys on (platform, platform_id).)
    staging = db_session.execute(select(StagingRecord)).scalars().all()
    assert len(staging) == 20

    aliases = db_session.execute(select(EntityAlias)).scalars().all()
    assert len(aliases) == 10
    assert all(a.platform is Platform.RIOT_API for a in aliases)
    assert all(a.confidence == 1.0 for a in aliases)
    assert {a.platform_id for a in aliases} == {f"PUUID-PLAYER-{i:02d}" for i in range(1, 11)}

    raw = db_session.execute(select(RawRecord)).scalars().all()
    assert len(raw) == 2  # one raw row per match
    assert all(r.source == "riot_api" for r in raw)

    assert stats.processed == 20
    assert stats.fetched == 2


@pytest.mark.integration
def test_pipeline_idempotent_on_re_run(db_session) -> None:  # type: ignore[no-untyped-def]
    """Same match, second pass — dedup via content_hash, no new staging rows."""

    def matchlist_response_factory() -> dict[str, Any]:
        return _matchlist_response()

    http = _FakeHttp(
        {
            "matchlists/by-puuid": matchlist_response_factory(),
            "matches/match-001": _ok_match(),
            "matches/match-002": _ok_match(),
        }
    )
    connector = _make_connector(http_get=http)

    first = run_ingestion(connector, session=db_session, since=_BETWEEN_OLD_AND_NEW)
    assert first.processed == 20

    # Re-run: brand new connector, same fake http. The content_hash dedup
    # in the runner should skip every payload.
    http2 = _FakeHttp(
        {
            "matchlists/by-puuid": matchlist_response_factory(),
            "matches/match-001": _ok_match(),
            "matches/match-002": _ok_match(),
        }
    )
    connector2 = _make_connector(http_get=http2)
    second = run_ingestion(connector2, session=db_session, since=_BETWEEN_OLD_AND_NEW)
    assert second.duplicates == 2
    assert second.processed == 0

    # Still the original 20 staging rows, 10 aliases.
    staging = db_session.execute(select(StagingRecord)).scalars().all()
    assert len(staging) == 20
    aliases = db_session.execute(select(EntityAlias)).scalars().all()
    assert len(aliases) == 10


