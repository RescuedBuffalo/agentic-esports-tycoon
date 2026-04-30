"""Unit + integration tests for the Liquipedia connector (BUF-11).

The unit tests drive the connector with a canned ``http_get`` callable
and JSON fixtures from ``tests/fixtures/liquipedia/`` — no network, no
``httpx`` dependency at runtime. They exercise the full
``fetch -> validate -> transform`` pipeline plus the per-record_type
shape checks.

The single integration test (``rebrand_extends_alias_with_new_valid_from``)
is skipped without ``TEST_DATABASE_URL``; it seeds a "Sentinels" entity
via ``resolve_entity`` and feeds the connector a rebrand payload to
prove the same canonical_id is reused with a fresh alias row.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from data_pipeline.connectors.liquipedia import (
    DEFAULT_BASE_URL,
    LIQUIPEDIA_CADENCE,
    LIQUIPEDIA_RATE_LIMIT,
    LiquipediaConnector,
    _extend_alias_for_rebrand,
)
from data_pipeline.errors import SchemaDriftError, TransientFetchError
from esports_sim.db.enums import EntityType, Platform

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "liquipedia"

# Long-ago watermark so the connector always queries with a stable
# ``since``; the unit tests don't actually depend on the value, but the
# rebrand integration test wants something tz-aware.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _load(name: str) -> Any:
    with (_FIXTURES / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _routed_http_get(routes: dict[str, Any]) -> Callable[[str], Any]:
    """Build an ``http_get`` that returns canned responses by URL suffix.

    ``routes`` keys are URL suffixes (so tests can ignore the base) and
    values are the JSON body that endpoint should return. A missing
    route raises so a test that fetches an unexpected URL fails loud.
    """

    def _get(url: str) -> Any:
        for suffix, body in routes.items():
            if url.endswith(suffix) or suffix in url:
                return body
        raise AssertionError(f"unexpected URL fetched in test: {url}")

    return _get


# --- metadata properties --------------------------------------------------


def test_metadata_properties_match_ticket() -> None:
    """Source name, platform, entity types, cadence, rate limit are all wired
    per the BUF-11 ticket and the BUF-9 contract."""
    conn = LiquipediaConnector(http_get=lambda url: {})
    assert conn.source_name == "liquipedia"
    assert conn.platform is Platform.LIQUIPEDIA
    assert conn.entity_types == (
        EntityType.PLAYER,
        EntityType.TEAM,
        EntityType.COACH,
        EntityType.TOURNAMENT,
    )
    assert conn.cadence == LIQUIPEDIA_CADENCE
    # 30 req/min == 0.5 tokens/sec; capacity 1 prevents bursting.
    assert conn.rate_limit == LIQUIPEDIA_RATE_LIMIT
    assert conn.rate_limit.capacity == 1
    assert conn.rate_limit.refill_per_second == 0.5


# --- fetch surface --------------------------------------------------------


def test_fetch_yields_tagged_payloads_for_each_endpoint() -> None:
    """Each seeded slug + the list endpoints emit a ``record_type`` payload."""
    routes = {
        "/player/tenz": _load("player.json"),
        "/team/sentinels": _load("team.json"),
        "/coach/kaplan": _load("coach.json"),
        "/tournaments": _load("tournaments.json"),
        "/transfers": _load("transfers.json"),
    }
    conn = LiquipediaConnector(
        player_slugs=["tenz"],
        team_slugs=["sentinels"],
        coach_slugs=["kaplan"],
        http_get=_routed_http_get(routes),
    )
    payloads = list(conn.fetch(_EPOCH))

    record_types = [p["record_type"] for p in payloads]
    # 1 player + 1 team + 1 coach + 2 tournaments + 2 transfers = 7
    assert record_types == [
        "player",
        "team",
        "coach",
        "tournament",
        "tournament",
        "transfer",
        "transfer",
    ]


def test_fetch_uses_since_in_transfer_query() -> None:
    """The transfer endpoint is queried with ``date_gte`` derived from ``since``.

    The runner passes a watermark; we check the connector forwards it
    into the query string Liquipedia expects.
    """
    seen_urls: list[str] = []

    def _get(url: str) -> Any:
        seen_urls.append(url)
        if "transfers" in url:
            return {"items": []}
        if "tournaments" in url:
            return []
        return {}

    conn = LiquipediaConnector(http_get=_get)
    list(conn.fetch(_EPOCH))

    transfer_calls = [u for u in seen_urls if "/transfers" in u]
    assert len(transfer_calls) == 1
    assert "date_gte=2026-01-01" in transfer_calls[0]


def test_fetch_skips_one_failed_slug_and_continues_to_the_rest() -> None:
    """One ``TransientFetchError`` mid-fetch must not abort the run.

    The runner's per-record error handling only fires inside
    ``validate``/``transform``; a failure raised during iterator
    advancement (i.e. the ``http_get`` call before the next ``yield``)
    propagates and skips every remaining slug. The connector now
    catches and downgrades to "log + skip" so the rest of the pass
    still completes.
    """

    def _get(url: str) -> Any:
        if "/player/broken" in url:
            raise TransientFetchError("upstream 502")
        if "/player/tenz" in url:
            return _load("player.json")
        if "/transfers" in url:
            return {"items": []}
        if "/tournaments" in url:
            return []
        return {}

    conn = LiquipediaConnector(
        player_slugs=["broken", "tenz"],
        http_get=_get,
    )
    payloads = list(conn.fetch(_EPOCH))
    record_types = [p["record_type"] for p in payloads]

    # ``broken`` was skipped; ``tenz`` still flowed through.
    assert "player" in record_types
    assert sum(1 for rt in record_types if rt == "player") == 1


def test_fetch_skips_envelope_drift_without_aborting_run() -> None:
    """An unknown list-envelope shape downgrades to a per-endpoint skip.

    When Liquipedia changes a tournaments envelope from ``[...]`` to,
    say, ``{"results": [...]}``, ``_iter_list`` raises
    :class:`SchemaDriftError`. The connector logs that as envelope
    drift and skips just that endpoint — the per-slug pulls and the
    transfer envelope still complete.
    """

    def _get(url: str) -> Any:
        if "/tournaments" in url:
            # Drifted envelope: not list, not {"items": [...]}
            return {"results": [{"slug": "ignored", "name": "ignored"}]}
        if "/transfers" in url:
            return {"items": []}
        if "/player/tenz" in url:
            return _load("player.json")
        return {}

    conn = LiquipediaConnector(
        player_slugs=["tenz"],
        http_get=_get,
    )
    payloads = list(conn.fetch(_EPOCH))
    record_types = [p["record_type"] for p in payloads]

    # Tournament envelope drifted -> 0 tournament payloads.
    # Player still flows; transfers envelope is well-formed (empty).
    assert record_types.count("tournament") == 0
    assert record_types.count("player") == 1


def test_iter_list_raises_schema_drift_on_unknown_envelope() -> None:
    """``_iter_list`` is the canonical envelope-shape gate.

    Two drifted shapes the regression test covers:

    * a dict without ``items`` (e.g. Liquipedia rewraps as ``results``);
    * a non-list, non-dict scalar.
    """
    from data_pipeline.connectors.liquipedia import _iter_list

    with pytest.raises(SchemaDriftError):
        list(_iter_list({"results": [{"slug": "x"}]}))
    with pytest.raises(SchemaDriftError):
        list(_iter_list("not a list"))


# --- validate -------------------------------------------------------------


@pytest.mark.parametrize(
    ("record_type", "fixture", "missing_key"),
    [
        ("player", "player.json", "name"),
        ("team", "team.json", "slug"),
        ("coach", "coach.json", "name"),
        ("tournament", "tournaments.json", "name"),
    ],
)
def test_validate_raises_schema_drift_on_missing_required_keys(
    record_type: str, fixture: str, missing_key: str
) -> None:
    """validate() rejects payloads with required fields stripped out."""
    raw = _load(fixture)
    if isinstance(raw, list):
        raw = raw[0]
    raw.pop(missing_key, None)
    payload = {"record_type": record_type, "slug": raw.get("slug", ""), "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    with pytest.raises(SchemaDriftError, match=missing_key):
        conn.validate(payload)


def test_validate_rejects_unknown_record_type() -> None:
    conn = LiquipediaConnector(http_get=lambda url: {})
    with pytest.raises(SchemaDriftError, match="unknown record_type"):
        conn.validate({"record_type": "match", "data": {"slug": "x", "name": "y"}})


def test_validate_rejects_non_dict_data() -> None:
    """``data`` must be a dict — a stray list or string is drift."""
    conn = LiquipediaConnector(http_get=lambda url: {})
    with pytest.raises(SchemaDriftError, match="missing dict 'data'"):
        conn.validate({"record_type": "player", "data": "oops not a dict"})


def test_validate_rejects_transfer_missing_date() -> None:
    """Transfers carry six required keys; missing any raises drift."""
    transfer = _load("transfers.json")["items"][0].copy()
    del transfer["date"]
    payload = {"record_type": "transfer", "id": transfer["id"], "data": transfer}
    conn = LiquipediaConnector(http_get=lambda url: {})
    with pytest.raises(SchemaDriftError, match="date"):
        conn.validate(payload)


def test_validate_passes_well_formed_payload() -> None:
    """Sanity check — a fixture-shaped payload validates cleanly."""
    raw = _load("player.json")
    payload = {"record_type": "player", "slug": raw["slug"], "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    assert conn.validate(payload) == payload


# --- transform per record type -------------------------------------------


def test_transform_player_record() -> None:
    raw = _load("player.json")
    payload = {"record_type": "player", "slug": raw["slug"], "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    records = list(conn.transform(payload))
    assert len(records) == 1
    rec = records[0]
    assert rec.entity_type is EntityType.PLAYER
    assert rec.platform_id == "tenz"
    assert rec.platform_name == "TenZ"
    assert rec.payload == raw


def test_transform_team_record() -> None:
    raw = _load("team.json")
    payload = {"record_type": "team", "slug": raw["slug"], "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    [rec] = list(conn.transform(payload))
    assert rec.entity_type is EntityType.TEAM
    assert rec.platform_id == "sentinels"
    assert rec.platform_name == "Sentinels"


def test_transform_coach_record() -> None:
    raw = _load("coach.json")
    payload = {"record_type": "coach", "slug": raw["slug"], "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    [rec] = list(conn.transform(payload))
    assert rec.entity_type is EntityType.COACH
    assert rec.platform_id == "kaplan"
    assert rec.platform_name == "Kaplan"


def test_transform_tournament_record() -> None:
    raw = _load("tournaments.json")[0]
    payload = {"record_type": "tournament", "slug": raw["slug"], "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    [rec] = list(conn.transform(payload))
    assert rec.entity_type is EntityType.TOURNAMENT
    assert rec.platform_id == "vct-2026-americas-stage-1"
    assert rec.platform_name == "VCT 2026: Americas Stage 1"


def test_transform_team_uses_previous_slug_for_rebrand() -> None:
    """A team payload with ``previous_slug`` emits a record under the OLD slug.

    This is the alias-extension shortcut: feeding the resolver the
    previous slug as ``platform_id`` makes its exact-alias lookup match
    the existing canonical, so the new display name lands as a fresh
    alias row on the same entity rather than minting a new one.
    """
    raw = _load("team_rebrand.json")
    payload = {"record_type": "team", "slug": raw["slug"], "data": raw}
    conn = LiquipediaConnector(http_get=lambda url: {})
    [rec] = list(conn.transform(payload))
    assert rec.platform_id == "sentinels"  # NOT "team-sentinels-esports"
    assert rec.platform_name == "Team Sentinels Esports"


# --- transform: transfers fan out to player + team -----------------------


def test_transform_transfer_emits_player_and_team_records() -> None:
    """One transfer event -> one PLAYER record + one TEAM record.

    Both carry the same ``roster_change`` blob under ``payload`` so a
    later RosterChange backfill can join them without reaching across
    record boundaries.
    """
    transfer = _load("transfers.json")["items"][0]
    payload = {"record_type": "transfer", "id": transfer["id"], "data": transfer}
    conn = LiquipediaConnector(http_get=lambda url: {})
    records = list(conn.transform(payload))

    assert len(records) == 2

    player_rec, team_rec = records
    assert player_rec.entity_type is EntityType.PLAYER
    assert player_rec.platform_id == "zekken"
    assert player_rec.platform_name == "Zekken"

    assert team_rec.entity_type is EntityType.TEAM
    assert team_rec.platform_id == "evil-geniuses"
    assert team_rec.platform_name == "Evil Geniuses"

    # Same roster_change blob on both sides — that's the contract.
    assert player_rec.payload["roster_change"] == team_rec.payload["roster_change"]
    rc = player_rec.payload["roster_change"]
    assert rc["source"] == "liquipedia"
    assert rc["id"] == "transfer-2026-04-15-zekken"
    assert rc["from_team_slug"] == "sentinels"
    assert rc["to_team_slug"] == "evil-geniuses"
    assert rc["date"] == "2026-04-15"


def test_transform_transfer_handles_missing_optional_fields() -> None:
    """``from_team_slug``/``role`` are optional — a free-agent signing
    has no ``from_team_*`` and a non-positional move can lack ``role``.
    """
    transfer = {
        "id": "free-agent-signing-1",
        "player_slug": "newpro",
        "player_name": "NewPro",
        "to_team_slug": "team-x",
        "to_team_name": "Team X",
        "date": "2026-04-20",
    }
    payload = {"record_type": "transfer", "id": transfer["id"], "data": transfer}
    conn = LiquipediaConnector(http_get=lambda url: {})
    records = list(conn.transform(payload))
    assert len(records) == 2
    for rec in records:
        assert rec.payload["roster_change"]["from_team_slug"] is None
        assert rec.payload["roster_change"]["role"] is None


# --- default factory uses httpx (smoke) ----------------------------------


def test_default_http_get_factory_imports_httpx() -> None:
    """The default factory wires ``httpx`` lazily.

    We don't want to actually hit the network in unit tests; just check
    that constructing without an ``http_get`` doesn't raise (i.e.
    ``httpx`` is installed) and produces a callable.
    """
    conn = LiquipediaConnector(
        player_slugs=[],
        include_tournaments=False,
        include_transfers=False,
    )
    # No slugs and the list endpoints disabled -> fetch yields nothing,
    # so we never actually call out. Just exercise the constructor +
    # list() so the default factory builds without ever touching the
    # network.
    assert list(conn.fetch(_EPOCH)) == []


def test_default_http_get_translates_5xx_to_transient() -> None:
    """5xx status codes raise :class:`TransientFetchError`.

    Stubs out ``httpx.get`` so the test stays offline; verifies the
    factory's contract that 5xx is recoverable while 4xx is not.
    """
    import httpx
    from data_pipeline.connectors.liquipedia import _default_http_get_factory

    class _StubResponse:
        def __init__(self, status: int) -> None:
            self.status_code = status
            self.text = "boom"

        def json(self) -> Any:
            return {}

    def fake_get_5xx(url: str, timeout: float = 10.0) -> Any:
        return _StubResponse(503)

    real_get = httpx.get
    httpx.get = fake_get_5xx  # type: ignore[assignment]
    try:
        get = _default_http_get_factory()
        with pytest.raises(TransientFetchError):
            get("https://example.invalid/anything")
    finally:
        httpx.get = real_get  # type: ignore[assignment]


def test_default_http_get_translates_network_error_to_transient() -> None:
    """``httpx.HTTPError`` (timeout, DNS, etc.) becomes ``TransientFetchError``."""
    import httpx
    from data_pipeline.connectors.liquipedia import _default_http_get_factory

    def fake_get_raises(url: str, timeout: float = 10.0) -> Any:
        raise httpx.ConnectError("name resolution failed")

    real_get = httpx.get
    httpx.get = fake_get_raises  # type: ignore[assignment]
    try:
        get = _default_http_get_factory()
        with pytest.raises(TransientFetchError):
            get("https://example.invalid/anything")
    finally:
        httpx.get = real_get  # type: ignore[assignment]


# --- base URL + DEFAULT_BASE_URL -----------------------------------------


def test_base_url_overrides_default() -> None:
    """The connector accepts a custom ``base_url`` for staging environments."""
    seen: list[str] = []

    def _get(url: str) -> Any:
        seen.append(url)
        return {"slug": "tenz", "name": "TenZ"}

    conn = LiquipediaConnector(
        player_slugs=["tenz"],
        include_tournaments=False,
        include_transfers=False,
        http_get=_get,
        base_url="https://staging.example/api",
    )
    list(conn.fetch(_EPOCH))
    assert seen == ["https://staging.example/api/player/tenz"]


def test_default_base_url_is_liquipedia_v1() -> None:
    """Sanity: the production base URL is the Valorant v1 tree."""
    assert DEFAULT_BASE_URL.startswith("https://liquipedia.net/valorant/api/")


# --- rebrand integration test (TEST_DATABASE_URL required) ---------------


@pytest.mark.integration
def test_rebrand_reuses_canonical_id_and_does_not_create_new_entity(db_session) -> None:
    """Rebrand acceptance regression (Systems-spec System 03).

    The Systems-spec rule is "do NOT create new entity; extend alias
    table with ``valid_from``". With the current BUF-7 resolver this
    materialises as: feed the rebrand record under the ``previous_slug``
    so the resolver's exact-alias path matches and returns the existing
    canonical, instead of fuzzy-falling-through into a brand-new entity.

    The "two aliases with distinct valid_from" half of the spec depends
    on the resolver gaining display-name update support — see the test
    docstring + PR description's Out of scope note. For BUF-11 the
    regression we lock down here is the half that's deliverable today:
    the same ``canonical_id`` is reused after a rebrand, and the
    ``Entity`` table grows by exactly zero.
    """
    from data_pipeline import run_ingestion
    from esports_sim.db.models import Entity, EntityAlias
    from esports_sim.resolver import ResolutionStatus, resolve_entity
    from sqlalchemy import func, select

    # 1. Seed Sentinels as the canonical team via the resolver.
    seed_result = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )
    assert seed_result.status is ResolutionStatus.CREATED
    seed_canonical = seed_result.canonical_id
    assert seed_canonical is not None
    db_session.flush()

    entity_count_before = db_session.execute(select(func.count()).select_from(Entity)).scalar_one()

    # Pre-flight: the helper finds the existing alias for the previous slug.
    found = _extend_alias_for_rebrand(
        db_session,
        previous_slug="sentinels",
        new_name="Team Sentinels Esports",
    )
    assert found is not None
    assert found.canonical_id == seed_canonical

    # 2. Feed the connector a fake rebrand response and run the pipeline.
    rebrand_payload = _load("team_rebrand.json")
    routes = {"/team/team-sentinels-esports": rebrand_payload}
    conn = LiquipediaConnector(
        team_slugs=["team-sentinels-esports"],
        include_tournaments=False,
        include_transfers=False,
        http_get=_routed_http_get(routes),
    )
    stats = run_ingestion(conn, session=db_session, since=_EPOCH)
    assert stats.processed == 1
    # Resolver should have MATCHED on the existing (LIQUIPEDIA, sentinels)
    # alias — no new entity, no fuzzy auto-merge churn.
    assert stats.by_status.get("matched") == 1

    # 3. Same canonical_id reused; entity table did not grow.
    entity_count_after = db_session.execute(select(func.count()).select_from(Entity)).scalar_one()
    assert entity_count_after == entity_count_before

    # The alias under (LIQUIPEDIA, sentinels) still maps to the original
    # canonical — that's the "no new entity" half of the rebrand rule.
    alias = db_session.execute(
        select(EntityAlias).where(
            EntityAlias.platform == Platform.LIQUIPEDIA,
            EntityAlias.platform_id == "sentinels",
        )
    ).scalar_one()
    assert alias.canonical_id == seed_canonical


@pytest.mark.integration
def test_extend_alias_for_rebrand_returns_none_for_unknown_slug(db_session) -> None:
    """The helper returns None when no alias exists yet — so the connector
    can fall through to the resolver's CREATED path for a brand-new team."""
    result = _extend_alias_for_rebrand(
        db_session,
        previous_slug="never-seen-this-slug",
        new_name="Whatever",
    )
    assert result is None
