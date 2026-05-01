"""Tests for the BUF-8 Liquipedia seed.

Two layers:

* Unit tests against an in-memory routed ``http_get`` cover the
  discovery → profile fan-out, social-alias projection, and manifest
  writing without touching Postgres.
* Integration tests (``-m integration``, gated on
  ``TEST_DATABASE_URL``) prove the resolver chokepoint is wired up
  correctly: canonical rows + Liquipedia aliases at confidence 1.0,
  social aliases at 0.95 against the same canonical, and re-running
  the seed adds zero new rows (the BUF-8 idempotency acceptance).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from data_pipeline.errors import TransientFetchError
from data_pipeline.seeds.liquipedia import (
    SeedManifest,
    _insert_social_alias_idempotent,
    seed_from_liquipedia,
)
from esports_sim.db.enums import EntityType, Platform

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "liquipedia"


def _load(name: str) -> Any:
    with (_FIXTURES / name).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _full_routes() -> dict[str, Any]:
    """Standard two-page-player + one-page-team/coach/tournament fixture set.

    The discovery endpoints page through ``?cursor=p2`` for players to
    cover the multi-page path; the rest are single-page. Each profile
    has socials so the manifest exercises every counter bucket.
    """
    return {
        # Player discovery: page 1 (cursor=None) advertises a next_cursor
        # of "p2"; page 2 advertises no next_cursor and stops the walk.
        "/players?cursor=p2": _load("players_discovery_page2.json"),
        "/players": _load("players_discovery_page1.json"),
        # Team / coach / tournament discovery: single-page envelopes.
        "/teams": _load("teams_discovery.json"),
        "/coaches": _load("coaches_discovery.json"),
        "/tournaments": _load("tournaments.json"),
        # Profile fetches.
        "/player/tenz": _load("player.json"),
        "/player/sacy": _load("player_with_socials.json"),
        "/player/zekken": _load("player_zekken.json"),
        "/team/sentinels": _load("team_with_socials.json"),
        "/coach/kaplan": _load("coach.json"),
    }


def _routed_get(routes: dict[str, Any]) -> Callable[[str], Any]:
    """Return a ``http_get`` that resolves URLs by best-suffix match.

    Cursor-bearing paths (e.g. ``/players?cursor=p2``) MUST be matched
    before their cursor-less ancestor so paginated walks resolve
    correctly. We sort routes by descending length so the longest
    suffix wins, which is enough to disambiguate.
    """
    sorted_routes = sorted(routes.items(), key=lambda kv: -len(kv[0]))

    def _get(url: str) -> Any:
        for suffix, body in sorted_routes:
            if url.endswith(suffix):
                return body
        raise AssertionError(f"unexpected URL fetched in test: {url}")

    return _get


# --- discovery + manifest -------------------------------------------------


def test_seed_yields_per_type_counters_in_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest carries one bucket per entity type, with discovered>=created.

    Pure-function test: no DB; we patch ``resolve_entity`` to always
    succeed with CREATED so the manifest reflects the discovery shape.
    """
    import uuid

    from data_pipeline.seeds import liquipedia as seed_mod
    from esports_sim.resolver import ResolutionStatus, ResolveResult

    def fake_resolve(session: Any, **kwargs: Any) -> ResolveResult:
        return ResolveResult(
            status=ResolutionStatus.CREATED,
            canonical_id=uuid.uuid4(),
            confidence=1.0,
        )

    monkeypatch.setattr(seed_mod, "resolve_entity", fake_resolve)

    # Patch the alias inserter too — we have no DB; just record outcomes
    # so the social-counter assertions still mean something.
    inserted: list[tuple[Platform, str]] = []

    def fake_insert(session: Any, *, canonical_id: Any, platform: Platform, handle: str) -> str:
        inserted.append((platform, handle))
        return "inserted"

    monkeypatch.setattr(seed_mod, "_insert_social_alias_idempotent", fake_insert)

    class _NullSession:
        def flush(self) -> None:
            return None

    manifest = seed_from_liquipedia(
        _NullSession(),
        http_get=_routed_get(_full_routes()),
        base_url="https://liquipedia.test/api",
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )

    assert isinstance(manifest, SeedManifest)
    assert manifest.seed_date == "2026-04-30"
    # Three players over two pages, one team, one coach, two tournaments
    # in the fixture.
    assert manifest.by_type[EntityType.PLAYER.value].discovered == 3
    assert manifest.by_type[EntityType.PLAYER.value].created == 3
    assert manifest.by_type[EntityType.TEAM.value].discovered == 1
    assert manifest.by_type[EntityType.TEAM.value].created == 1
    assert manifest.by_type[EntityType.COACH.value].discovered == 1
    assert manifest.by_type[EntityType.COACH.value].created == 1
    assert manifest.by_type[EntityType.TOURNAMENT.value].discovered == 2
    assert manifest.by_type[EntityType.TOURNAMENT.value].created == 2
    assert manifest.total_canonical_created == 7

    # Twitter handles seen for: sacy, zekken (players), sentinels (team) = 3.
    # Twitch handles seen for:  sacy (player), sentinels (team)         = 2.
    assert manifest.socials[Platform.TWITTER.value].inserted == 3
    assert manifest.socials[Platform.TWITCH.value].inserted == 2


def test_seed_writes_manifest_to_seeds_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The manifest lands at ``seeds/liquipedia_seed_{date}.json``."""
    import uuid

    from data_pipeline.seeds import liquipedia as seed_mod
    from esports_sim.resolver import ResolutionStatus, ResolveResult

    monkeypatch.setattr(
        seed_mod,
        "resolve_entity",
        lambda *a, **kw: ResolveResult(
            status=ResolutionStatus.CREATED,
            canonical_id=uuid.uuid4(),
            confidence=1.0,
        ),
    )
    monkeypatch.setattr(
        seed_mod,
        "_insert_social_alias_idempotent",
        lambda *a, **kw: "inserted",
    )

    class _NullSession:
        def flush(self) -> None:
            return None

    seed_from_liquipedia(
        _NullSession(),
        http_get=_routed_get(_full_routes()),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    out = tmp_path / "liquipedia_seed_2026-04-30.json"
    assert out.exists()
    body = json.loads(out.read_text(encoding="utf-8"))
    assert body["seed_date"] == "2026-04-30"
    assert body["by_type"]["player"]["created"] == 3


def test_seed_paginates_player_discovery_via_next_cursor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``next_cursor`` is followed until the response omits one."""
    seen_urls: list[str] = []

    def _get(url: str) -> Any:
        seen_urls.append(url)
        return _routed_get(_full_routes())(url)

    import uuid

    from data_pipeline.seeds import liquipedia as seed_mod
    from esports_sim.resolver import ResolutionStatus, ResolveResult

    monkeypatch.setattr(
        seed_mod,
        "resolve_entity",
        lambda *a, **kw: ResolveResult(
            status=ResolutionStatus.CREATED,
            canonical_id=uuid.uuid4(),
            confidence=1.0,
        ),
    )
    monkeypatch.setattr(seed_mod, "_insert_social_alias_idempotent", lambda *a, **kw: "inserted")

    class _NullSession:
        def flush(self) -> None:
            return None

    seed_from_liquipedia(
        _NullSession(),
        http_get=_get,
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )

    # Page 1 is fetched without ?cursor=, page 2 with ?cursor=p2.
    player_disc = [u for u in seen_urls if u.endswith("/players") or "/players?cursor=" in u]
    assert any(u.endswith("/players") for u in player_disc)
    assert any("/players?cursor=p2" in u for u in player_disc)


def test_seed_skips_transient_profile_fetch_and_continues(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One TransientFetchError on a profile must not abort the seed."""
    import uuid

    from data_pipeline.seeds import liquipedia as seed_mod
    from esports_sim.resolver import ResolutionStatus, ResolveResult

    def _get(url: str) -> Any:
        if url.endswith("/player/tenz"):
            raise TransientFetchError("upstream 503")
        return _routed_get(_full_routes())(url)

    monkeypatch.setattr(
        seed_mod,
        "resolve_entity",
        lambda *a, **kw: ResolveResult(
            status=ResolutionStatus.CREATED,
            canonical_id=uuid.uuid4(),
            confidence=1.0,
        ),
    )
    monkeypatch.setattr(seed_mod, "_insert_social_alias_idempotent", lambda *a, **kw: "inserted")

    class _NullSession:
        def flush(self) -> None:
            return None

    manifest = seed_from_liquipedia(
        _NullSession(),
        http_get=_get,
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    counters = manifest.by_type[EntityType.PLAYER.value]
    assert counters.transient_errors == 1
    # 3 discovered minus 1 transient = 2 still resolved.
    assert counters.created == 2


def test_seed_skips_profile_missing_required_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A profile missing required ``name`` is logged as schema drift, skipped."""
    import uuid

    from data_pipeline.seeds import liquipedia as seed_mod
    from esports_sim.resolver import ResolutionStatus, ResolveResult

    routes = _full_routes()
    routes["/player/tenz"] = {"slug": "tenz"}  # missing "name"

    monkeypatch.setattr(
        seed_mod,
        "resolve_entity",
        lambda *a, **kw: ResolveResult(
            status=ResolutionStatus.CREATED,
            canonical_id=uuid.uuid4(),
            confidence=1.0,
        ),
    )
    monkeypatch.setattr(seed_mod, "_insert_social_alias_idempotent", lambda *a, **kw: "inserted")

    class _NullSession:
        def flush(self) -> None:
            return None

    manifest = seed_from_liquipedia(
        _NullSession(),
        http_get=_routed_get(routes),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    counters = manifest.by_type[EntityType.PLAYER.value]
    assert counters.schema_drifts == 1
    assert counters.created == 2  # tenz dropped; sacy + zekken survived


# --- integration: real DB writes -----------------------------------------


@pytest.mark.integration
def test_seed_creates_canonical_entities_and_socials(db_session, tmp_path: Path) -> None:
    """End-to-end: discovery → profile → resolver → social alias.

    Asserts the BUF-8 surface contract: every profile produces a
    canonical entity + Liquipedia alias at confidence 1.0, every social
    handle on the profile produces an alias at 0.95 against the same
    canonical, and the manifest carries the right counts.
    """
    from esports_sim.db.models import Entity, EntityAlias
    from sqlalchemy import select

    manifest = seed_from_liquipedia(
        db_session,
        http_get=_routed_get(_full_routes()),
        base_url="https://liquipedia.test/api",
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    db_session.flush()

    # Three players + one team + one coach + two tournaments = 7 entities.
    entity_count = db_session.execute(select(Entity)).scalars().all()
    assert len(entity_count) == 7

    # Profile aliases land at 1.0 against LIQUIPEDIA.
    liq_aliases = (
        db_session.execute(select(EntityAlias).where(EntityAlias.platform == Platform.LIQUIPEDIA))
        .scalars()
        .all()
    )
    assert len(liq_aliases) == 7
    assert all(a.confidence == 1.0 for a in liq_aliases)

    # Sacy + Zekken + Sentinels each have a Twitter handle at 0.95.
    twitter_aliases = (
        db_session.execute(select(EntityAlias).where(EntityAlias.platform == Platform.TWITTER))
        .scalars()
        .all()
    )
    assert len(twitter_aliases) == 3
    assert all(a.confidence == 0.95 for a in twitter_aliases)

    # Sacy + Sentinels each have a Twitch handle at 0.95.
    twitch_aliases = (
        db_session.execute(select(EntityAlias).where(EntityAlias.platform == Platform.TWITCH))
        .scalars()
        .all()
    )
    assert len(twitch_aliases) == 2
    assert all(a.confidence == 0.95 for a in twitch_aliases)

    # Manifest matches the DB.
    assert manifest.by_type[EntityType.PLAYER.value].created == 3
    assert manifest.socials[Platform.TWITTER.value].inserted == 3
    assert manifest.socials[Platform.TWITCH.value].inserted == 2

    # Each social alias points at the canonical it was scraped from.
    sacy_canon = db_session.execute(
        select(EntityAlias.canonical_id).where(
            EntityAlias.platform == Platform.LIQUIPEDIA,
            EntityAlias.platform_id == "sacy",
        )
    ).scalar_one()
    sacy_twitter = db_session.execute(
        select(EntityAlias).where(
            EntityAlias.platform == Platform.TWITTER,
            EntityAlias.platform_id == "Sacyzin",
        )
    ).scalar_one()
    assert sacy_twitter.canonical_id == sacy_canon


@pytest.mark.integration
def test_seed_idempotent_rerun_adds_no_rows(db_session, tmp_path: Path) -> None:
    """BUF-8 acceptance: a second run produces zero new entities or aliases.

    The manifest's ``inserted`` counter for each social platform pegs
    to zero on the second run, and the ``existing`` counter records
    every handle that was already there from pass one. The same
    invariant holds for canonical rows: the per-type ``matched`` count
    equals what ``created`` was on the first run.
    """
    from esports_sim.db.models import Entity, EntityAlias
    from sqlalchemy import func, select

    routes = _full_routes()

    seed_from_liquipedia(
        db_session,
        http_get=_routed_get(routes),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    db_session.flush()
    entity_count_after_first = db_session.execute(
        select(func.count()).select_from(Entity)
    ).scalar_one()
    alias_count_after_first = db_session.execute(
        select(func.count()).select_from(EntityAlias)
    ).scalar_one()

    second = seed_from_liquipedia(
        db_session,
        http_get=_routed_get(routes),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    db_session.flush()

    entity_count_after_second = db_session.execute(
        select(func.count()).select_from(Entity)
    ).scalar_one()
    alias_count_after_second = db_session.execute(
        select(func.count()).select_from(EntityAlias)
    ).scalar_one()

    assert entity_count_after_second == entity_count_after_first
    assert alias_count_after_second == alias_count_after_first

    # Manifest from pass two: no new canonical rows, no new social
    # aliases, every profile resolves as MATCHED.
    assert second.by_type[EntityType.PLAYER.value].created == 0
    assert second.by_type[EntityType.PLAYER.value].matched == 3
    assert second.socials[Platform.TWITTER.value].inserted == 0
    assert second.socials[Platform.TWITTER.value].existing == 3
    assert second.socials[Platform.TWITCH.value].inserted == 0
    assert second.socials[Platform.TWITCH.value].existing == 2


# --- rebrand alias extension --------------------------------------------


def _routes_with_rebranded_team() -> dict[str, Any]:
    """Discovery + profile routes where the team carries ``previous_slug``.

    Reuses the connector test fixture ``team_rebrand.json`` so the
    seed test and the connector test share one source of truth on
    rebrand payload shape. Discovery advertises only the new slug —
    that's the production scenario, where the operator has already
    run the seed once and the team has since rebranded.
    """
    return {
        "/players": {"items": [], "next_cursor": None},
        "/teams": {"items": [{"slug": "team-sentinels-esports"}], "next_cursor": None},
        "/coaches": {"items": [], "next_cursor": None},
        "/tournaments": [],
        "/team/team-sentinels-esports": _load("team_rebrand.json"),
    }


@pytest.mark.integration
def test_seed_rebrand_extends_alias_chain_with_new_slug(db_session, tmp_path: Path) -> None:
    """Round 2 regression: a profile with ``previous_slug`` registers BOTH slugs.

    Without the fix, the seed resolved on ``previous_slug`` only and
    never persisted the new slug as an alias — so a later record
    carrying only ``team-sentinels-esports`` would fuzzy-fall-through
    into a fork. With the fix, both ``(LIQUIPEDIA, sentinels)`` and
    ``(LIQUIPEDIA, team-sentinels-esports)`` map to the same canonical.
    """
    from esports_sim.db.models import Entity, EntityAlias
    from sqlalchemy import select

    # Pre-seed the canonical under the old slug so the rebrand path
    # exercises the MATCHED branch — this is the realistic scenario
    # (the seed was run before the team rebranded).
    pre = resolve_entity_unprefixed(db_session)
    db_session.flush()

    routes = _routes_with_rebranded_team()
    manifest = seed_from_liquipedia(
        db_session,
        http_get=_routed_get(routes),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    db_session.flush()

    # Exactly one team canonical; alias chain = OLD slug + NEW slug.
    team_aliases = (
        db_session.execute(
            select(EntityAlias).where(
                EntityAlias.platform == Platform.LIQUIPEDIA,
                EntityAlias.platform_id.in_(["sentinels", "team-sentinels-esports"]),
            )
        )
        .scalars()
        .all()
    )
    assert len(team_aliases) == 2
    canonical_ids = {a.canonical_id for a in team_aliases}
    assert canonical_ids == {pre.canonical_id}

    # The new alias's valid_from carries the rebrand effective date
    # (parsed from the profile's ``renamed_at`` field).
    new_alias = next(a for a in team_aliases if a.platform_id == "team-sentinels-esports")
    assert new_alias.valid_from.isoformat().startswith("2026-04-01")

    # Manifest counter pegged.
    assert manifest.by_type[EntityType.TEAM.value].rebrands_registered == 1
    assert manifest.by_type[EntityType.TEAM.value].rebrands_existing == 0

    # Sanity: still one team entity, no fork.
    team_entities = (
        db_session.execute(select(Entity).where(Entity.entity_type == EntityType.TEAM))
        .scalars()
        .all()
    )
    assert len(team_entities) == 1


@pytest.mark.integration
def test_seed_rebrand_idempotent_on_rerun(db_session, tmp_path: Path) -> None:
    """Re-running the seed on the same rebrand profile adds no new aliases."""
    from esports_sim.db.models import EntityAlias
    from sqlalchemy import func, select

    resolve_entity_unprefixed(db_session)
    db_session.flush()

    routes = _routes_with_rebranded_team()
    seed_from_liquipedia(
        db_session,
        http_get=_routed_get(routes),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    db_session.flush()
    alias_count_after_first = db_session.execute(
        select(func.count()).select_from(EntityAlias)
    ).scalar_one()

    second = seed_from_liquipedia(
        db_session,
        http_get=_routed_get(routes),
        seeds_dir=tmp_path,
        today=date(2026, 4, 30),
    )
    db_session.flush()

    alias_count_after_second = db_session.execute(
        select(func.count()).select_from(EntityAlias)
    ).scalar_one()
    assert alias_count_after_second == alias_count_after_first
    assert second.by_type[EntityType.TEAM.value].rebrands_registered == 0
    assert second.by_type[EntityType.TEAM.value].rebrands_existing == 1


def resolve_entity_unprefixed(db_session: Any) -> Any:
    """Helper: pre-seed the rebrand source canonical via the resolver chokepoint."""
    from esports_sim.resolver import resolve_entity

    return resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )


# --- _parse_renamed_at ---------------------------------------------------


def test_parse_renamed_at_handles_bare_date() -> None:
    """ISO date strings (no time, no tz) become UTC midnight."""
    from data_pipeline.seeds.liquipedia import _parse_renamed_at

    parsed = _parse_renamed_at("2026-04-01")
    assert parsed.year == 2026
    assert parsed.month == 4
    assert parsed.day == 1
    assert parsed.hour == 0
    assert parsed.tzinfo is not None


def test_parse_renamed_at_falls_back_on_garbage() -> None:
    """Unparseable input degrades to ``datetime.now(UTC)`` rather than raising."""
    from data_pipeline.seeds.liquipedia import _parse_renamed_at

    parsed = _parse_renamed_at("not-a-date")
    assert parsed.tzinfo is not None  # tz-aware fallback


def test_parse_renamed_at_handles_none() -> None:
    """A missing ``renamed_at`` field falls back to ``now``."""
    from data_pipeline.seeds.liquipedia import _parse_renamed_at

    parsed = _parse_renamed_at(None)
    assert parsed.tzinfo is not None


# --- existing social-alias unit test -------------------------------------


@pytest.mark.integration
def test_insert_social_alias_idempotent_returns_existing_on_duplicate(
    db_session,
) -> None:
    """The savepointed insert returns ``"existing"`` on a unique-key collision."""
    from esports_sim.db.models import Entity

    entity = Entity(entity_type=EntityType.PLAYER)
    db_session.add(entity)
    db_session.flush()

    first = _insert_social_alias_idempotent(
        db_session,
        canonical_id=entity.canonical_id,
        platform=Platform.TWITTER,
        handle="dup_handle",
    )
    db_session.flush()
    second = _insert_social_alias_idempotent(
        db_session,
        canonical_id=entity.canonical_id,
        platform=Platform.TWITTER,
        handle="dup_handle",
    )

    assert first == "inserted"
    assert second == "existing"
