"""Tests for the BUF-85 VLR per-match player participation scraper.

The parser unit tests run without a database — they exercise the
HTML-state machine against canned ``/match/<id>`` fixtures. The
end-to-end ``scrape_vlr_match_players`` test is gated on
``TEST_DATABASE_URL`` (it consumes the ``db_session`` fixture in
``conftest.py``), so a fresh clone with no Postgres still has a
green ``uv run pytest``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from data_pipeline.connectors.vlr import (
    USER_AGENT,
    VLR_BASE_URL,
    _RobotsCache,
)
from data_pipeline.connectors.vlr_match import (
    ParsedPlayerStat,
    parse_match_page,
    scrape_vlr_match_players,
)
from data_pipeline.errors import TransientFetchError
from data_pipeline.rate_limiter import TokenBucket
from esports_sim.db.enums import EntityType, Platform
from sqlalchemy import select

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "vlr"


def _read_fixture(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


def _stub_robots(allow_all: bool = True) -> _RobotsCache:
    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=lambda _url: "")
    cache._loaded = True
    cache._disallows = [] if allow_all else ["/"]
    return cache


def _make_fetcher(mapping: dict[str, str]) -> Callable[[str], str]:
    return lambda url: mapping[url]


def _free_bucket() -> TokenBucket:
    """Effectively unlimited bucket for tests — pace is checked in BUF-10's tests."""
    return TokenBucket(capacity=100, refill_per_second=100.0)


# --- parser ----------------------------------------------------------------


def test_parser_extracts_one_row_per_player_per_map() -> None:
    rows = parse_match_page(_read_fixture("match_300001.html"))
    # 3 players on map 1 + 2 players on map 2 = 5 total rows.
    assert len(rows) == 5
    # Each row carries a vlr_player_id, vlr_game_id, and display_name.
    for row in rows:
        assert row.vlr_player_id, "vlr_player_id must be present"
        assert row.vlr_game_id, "vlr_game_id must be present"
        assert row.display_name


def test_parser_links_player_to_correct_map() -> None:
    rows = parse_match_page(_read_fixture("match_300001.html"))
    by_map: dict[str, set[str]] = {}
    for row in rows:
        by_map.setdefault(row.vlr_game_id, set()).add(row.vlr_player_id)
    assert by_map == {
        "g-100001": {"9", "729", "2329"},
        "g-100002": {"9", "2329"},
    }


def test_parser_extracts_team_side_from_row_class() -> None:
    rows = parse_match_page(_read_fixture("match_300001.html"))
    # TenZ + Zekken on team1 of g-100001; Asuna on team2 of g-100001.
    by_player_map = {(r.vlr_game_id, r.vlr_player_id): r.team_side for r in rows}
    assert by_player_map[("g-100001", "9")] == "team1"
    assert by_player_map[("g-100001", "729")] == "team1"
    assert by_player_map[("g-100001", "2329")] == "team2"


def test_parser_extracts_agent_from_image_title() -> None:
    rows = parse_match_page(_read_fixture("match_300001.html"))
    by_player_map = {(r.vlr_game_id, r.vlr_player_id): r.agent for r in rows}
    # Different agents per map should round-trip through the parser.
    assert by_player_map[("g-100001", "9")] == "jett"
    assert by_player_map[("g-100002", "9")] == "chamber"
    assert by_player_map[("g-100001", "2329")] == "raze"
    assert by_player_map[("g-100002", "2329")] == "jett"


def test_parser_extracts_headline_stats_in_correct_columns() -> None:
    """The position-based mapping must put ACS in ACS, K in K, etc."""
    rows = parse_match_page(_read_fixture("match_300001.html"))
    tenz_map1 = next(
        r for r in rows if r.vlr_game_id == "g-100001" and r.vlr_player_id == "9"
    )
    assert tenz_map1.rating == pytest.approx(1.32)
    assert tenz_map1.acs == pytest.approx(285.0)
    assert tenz_map1.kills == 22
    assert tenz_map1.deaths == 11
    assert tenz_map1.assists == 5
    assert tenz_map1.kast_pct == pytest.approx(75.0)
    assert tenz_map1.adr == pytest.approx(155.4)
    assert tenz_map1.hs_pct == pytest.approx(35.0)
    assert tenz_map1.first_kills == 5
    assert tenz_map1.first_deaths == 2


def test_parser_handles_negative_diff_and_percent_signs() -> None:
    """Stats like KD diff ship as ``+11`` / ``-1`` and KAST as ``75%``."""
    rows = parse_match_page(_read_fixture("match_300001.html"))
    asuna_map1 = next(
        r for r in rows if r.vlr_game_id == "g-100001" and r.vlr_player_id == "2329"
    )
    # Negative diff kept on the unmapped column wouldn't surface here,
    # but kast/hs need the % stripping path.
    assert asuna_map1.kast_pct == pytest.approx(68.0)
    assert asuna_map1.hs_pct == pytest.approx(28.0)


def test_parser_skips_player_anchors_outside_stats_block() -> None:
    """Header anchors are not player rows.

    ``match_300001.html`` puts Sentinels/100 Thieves header anchors
    above the ``vm-stats-game`` blocks. They must not land in the
    parsed player list.
    """
    rows = parse_match_page(_read_fixture("match_300001.html"))
    # Only player ids from the fixture's stat tables.
    parsed_ids = {row.vlr_player_id for row in rows}
    assert parsed_ids == {"9", "729", "2329"}


def test_parser_drops_rows_with_unexpected_cell_count() -> None:
    """A row with a wrong column count should not silently mis-attribute stats.

    The parser logs the drift event per row and continues — so the
    surrounding well-formed rows remain in the output.
    """
    drifted = """
    <html><body>
      <div class="vm-stats-game" data-game-id="g-broken">
        <table class="wf-table-inset mod-overview">
          <tbody>
            <tr class="mod-t1">
              <td class="mod-player"><a href="/player/9/tenz">TenZ</a></td>
              <td class="mod-agents"><img title="Jett"/></td>
              <td class="mod-stat">1.32</td>
              <td class="mod-stat">285</td>
            </tr>
            <tr class="mod-t1">
              <td class="mod-player"><a href="/player/729/zekken">Zekken</a></td>
              <td class="mod-agents"><img title="Sova"/></td>
              <td class="mod-stat">1.05</td>
              <td class="mod-stat">220</td>
              <td class="mod-stat">17</td>
              <td class="mod-stat">14</td>
              <td class="mod-stat">7</td>
              <td class="mod-stat">+3</td>
              <td class="mod-stat">70%</td>
              <td class="mod-stat">141.0</td>
              <td class="mod-stat">22%</td>
              <td class="mod-stat">3</td>
              <td class="mod-stat">3</td>
              <td class="mod-stat">0</td>
            </tr>
          </tbody>
        </table>
      </div>
    </body></html>
    """
    rows = parse_match_page(drifted)
    # First row drops on cell-count mismatch; the second row (full
    # 14-column shape) survives.
    assert [r.vlr_player_id for r in rows] == ["729"]


# --- scraper (integration) -------------------------------------------------


pytestmark_integration = pytest.mark.integration


def _seed_match(
    db_session: Any,
    *,
    vlr_match_id: str,
    map_payloads: list[dict[str, Any]],
) -> dict[str, uuid.UUID]:
    """Seed one match + N map_results so the scraper has FK targets.

    Returns a mapping ``vlr_game_id -> map_result_id`` so tests can
    join back to the canonical UUIDs the scraper writes against.
    """
    from esports_sim.db.models import MapResult, Match

    match = Match(
        match_id=uuid.uuid4(),
        vlr_match_id=vlr_match_id,
        match_date=datetime(2026, 4, 26, 18, tzinfo=UTC),
    )
    db_session.add(match)
    db_session.flush()

    map_ids: dict[str, uuid.UUID] = {}
    for payload in map_payloads:
        mr = MapResult(
            map_result_id=uuid.uuid4(),
            match_id=match.match_id,
            vlr_game_id=payload["vlr_game_id"],
            vlr_map_id=payload.get("vlr_map_id", 1),
            team1_rounds=13,
            team2_rounds=10,
            team1_atk_rounds=7,
            team1_def_rounds=6,
            team2_atk_rounds=4,
            team2_def_rounds=6,
            team1_rating=1.10,
            team2_rating=0.95,
            team1_stats={},
            team2_stats={},
        )
        db_session.add(mr)
        map_ids[payload["vlr_game_id"]] = mr.map_result_id
    db_session.flush()
    return map_ids


@pytestmark_integration
def test_scrape_writes_player_match_stat_rows(db_session) -> None:
    """End-to-end happy path: every parsed player lands one stat row."""
    from esports_sim.db.models import EntityAlias, PlayerMatchStat

    map_ids = _seed_match(
        db_session,
        vlr_match_id="m-300001",
        map_payloads=[
            {"vlr_game_id": "g-100001", "vlr_map_id": 1},
            {"vlr_game_id": "g-100002", "vlr_map_id": 2},
        ],
    )

    url = f"{VLR_BASE_URL}/match/m-300001"
    fetcher = _make_fetcher({url: _read_fixture("match_300001.html")})

    stats = scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-300001"],
        page_fetcher=fetcher,
        rate_limiter=_free_bucket(),
        robots_cache=_stub_robots(),
    )

    assert stats.matches_fetched == 1
    assert stats.players_parsed == 5
    assert stats.players_inserted == 5
    assert stats.players_existing == 0

    rows = db_session.execute(select(PlayerMatchStat)).scalars().all()
    assert len(rows) == 5

    # Every alias was minted under the namespaced platform_id.
    aliases = (
        db_session.execute(
            select(EntityAlias.platform_id).where(EntityAlias.platform == Platform.VLR)
        )
        .scalars()
        .all()
    )
    expected_ids = {"player-9", "player-729", "player-2329"}
    assert expected_ids.issubset(set(aliases))

    # Each stat row resolves to the right map_result via vlr_game_id.
    by_pair = {(row.map_result_id, row.source_player_id) for row in rows}
    assert (map_ids["g-100001"], "9") in by_pair
    assert (map_ids["g-100002"], "9") in by_pair
    assert (map_ids["g-100002"], "2329") in by_pair


@pytestmark_integration
def test_rerun_is_idempotent(db_session) -> None:
    """Re-running the scraper for the same match must add zero rows."""
    from esports_sim.db.models import PlayerMatchStat

    _seed_match(
        db_session,
        vlr_match_id="m-300001",
        map_payloads=[
            {"vlr_game_id": "g-100001"},
            {"vlr_game_id": "g-100002"},
        ],
    )
    url = f"{VLR_BASE_URL}/match/m-300001"
    fetcher = _make_fetcher({url: _read_fixture("match_300001.html")})

    first = scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-300001"],
        page_fetcher=fetcher,
        rate_limiter=_free_bucket(),
        robots_cache=_stub_robots(),
    )
    assert first.players_inserted == 5

    # Second pass: the in-memory ``existing_stat_keys`` pre-load picks
    # up every (map_result_id, entity_id) row from the first pass and
    # skips before the savepoint round-trip.
    second = scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-300001"],
        page_fetcher=fetcher,
        rate_limiter=_free_bucket(),
        robots_cache=_stub_robots(),
    )
    assert second.players_inserted == 0
    assert second.players_existing == 5

    rows = db_session.execute(select(PlayerMatchStat)).scalars().all()
    # Schema-level dedup also confirmed: only 5 rows total even though
    # the scraper ran twice.
    assert len(rows) == 5


@pytestmark_integration
def test_player_aliases_reused_across_maps_and_matches(db_session) -> None:
    """Same vlr_player_id across maps must resolve to one canonical_id."""
    from esports_sim.db.models import Entity, PlayerMatchStat

    _seed_match(
        db_session,
        vlr_match_id="m-300001",
        map_payloads=[
            {"vlr_game_id": "g-100001"},
            {"vlr_game_id": "g-100002"},
        ],
    )
    url = f"{VLR_BASE_URL}/match/m-300001"
    fetcher = _make_fetcher({url: _read_fixture("match_300001.html")})

    scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-300001"],
        page_fetcher=fetcher,
        rate_limiter=_free_bucket(),
        robots_cache=_stub_robots(),
    )

    # TenZ (vlr id 9) appears on both maps. The two stat rows must
    # share one entity_id.
    rows = (
        db_session.execute(
            select(PlayerMatchStat).where(PlayerMatchStat.source_player_id == "9")
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert rows[0].entity_id == rows[1].entity_id

    # And the canonical entity is a PLAYER.
    entity = db_session.execute(
        select(Entity).where(Entity.canonical_id == rows[0].entity_id)
    ).scalar_one()
    assert entity.entity_type is EntityType.PLAYER


@pytestmark_integration
def test_skip_when_map_result_missing(db_session) -> None:
    """A scraped game id with no matching map_result skips with a warning."""
    from esports_sim.db.models import PlayerMatchStat

    # Seed only one of the two maps from the fixture.
    _seed_match(
        db_session,
        vlr_match_id="m-300001",
        map_payloads=[{"vlr_game_id": "g-100001"}],
    )
    url = f"{VLR_BASE_URL}/match/m-300001"
    fetcher = _make_fetcher({url: _read_fixture("match_300001.html")})

    stats = scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-300001"],
        page_fetcher=fetcher,
        rate_limiter=_free_bucket(),
        robots_cache=_stub_robots(),
    )
    # 5 players parsed total (3 on g-100001, 2 on g-100002).
    # The 2 on the un-seeded g-100002 are skipped.
    assert stats.players_parsed == 5
    assert stats.players_skipped_no_map == 2
    assert stats.players_inserted == 3

    rows = db_session.execute(select(PlayerMatchStat)).scalars().all()
    # Only the 3 players from the seeded map landed.
    assert len(rows) == 3


@pytestmark_integration
def test_fetcher_failure_skips_match_and_continues(db_session) -> None:
    """A transient fetch error on one match must not abort the run."""
    from esports_sim.db.models import PlayerMatchStat

    _seed_match(
        db_session,
        vlr_match_id="m-300001",
        map_payloads=[{"vlr_game_id": "g-100001"}, {"vlr_game_id": "g-100002"}],
    )
    _seed_match(
        db_session,
        vlr_match_id="m-300002",
        map_payloads=[{"vlr_game_id": "g-100099"}],
    )

    bad_url = f"{VLR_BASE_URL}/match/m-300002"

    def selective_fetcher(url: str) -> str:
        if url == bad_url:
            raise TransientFetchError("503 simulated")
        return _read_fixture("match_300001.html")

    stats = scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-300002", "m-300001"],
        page_fetcher=selective_fetcher,
        rate_limiter=_free_bucket(),
        robots_cache=_stub_robots(),
    )
    assert stats.matches_fetched == 1
    assert stats.matches_skipped_fetch == 1
    # The good match's 5 players still landed.
    rows = db_session.execute(select(PlayerMatchStat)).scalars().all()
    assert len(rows) == 5


@pytestmark_integration
def test_rate_limiter_paces_fetch(db_session) -> None:
    """A capacity=1 / refill=20rpm bucket must throttle each fetch.

    With three matches and the BUF-85 ``20 req/min`` limit, two of
    the three calls each pay one 3-second wait. The first call is
    free (full bucket), so the fake clock should advance by at least
    ``2 * 3.0`` seconds across the run.
    """

    class _FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def time(self) -> float:
            return self.now

        def sleep(self, seconds: float) -> None:
            if seconds > 0:
                self.now += seconds

    _seed_match(
        db_session,
        vlr_match_id="m-pace-1",
        map_payloads=[{"vlr_game_id": "g-100001"}, {"vlr_game_id": "g-100002"}],
    )
    _seed_match(
        db_session,
        vlr_match_id="m-pace-2",
        map_payloads=[{"vlr_game_id": "g-200001"}],
    )
    _seed_match(
        db_session,
        vlr_match_id="m-pace-3",
        map_payloads=[{"vlr_game_id": "g-300001"}],
    )

    clock = _FakeClock()
    bucket = TokenBucket(
        capacity=1,
        refill_per_second=20.0 / 60.0,
        clock=clock.time,
        sleeper=clock.sleep,
    )

    def fetcher(_url: str) -> str:
        # Same fixture for every match; only the first one has the
        # matching vlr_game_ids, the others get "no map result"
        # skips. We're measuring rate-limit pacing, not insertion.
        return _read_fixture("match_300001.html")

    scrape_vlr_match_players(
        db_session,
        vlr_match_ids=["m-pace-1", "m-pace-2", "m-pace-3"],
        page_fetcher=fetcher,
        rate_limiter=bucket,
        robots_cache=_stub_robots(),
    )

    # Two waits at 3 seconds each — first call was free.
    assert clock.now >= 2 * 3.0 - 1e-9


# --- ParsedPlayerStat dataclass --------------------------------------------


def test_parsed_player_stat_is_frozen_with_default_extra() -> None:
    """The dataclass is value-typed (hashable, frozen) with an empty default extra."""
    row = ParsedPlayerStat(
        vlr_game_id="g-1",
        vlr_player_id="9",
        display_name="TenZ",
        team_side="team1",
        agent="jett",
        rating=1.0,
        acs=200.0,
        kills=15,
        deaths=10,
        assists=5,
        kast_pct=70.0,
        adr=140.0,
        hs_pct=30.0,
        first_kills=2,
        first_deaths=1,
    )
    assert row.extra == {}
    # Frozen dataclass — assignment must raise FrozenInstanceError.
    with pytest.raises(AttributeError):
        row.vlr_player_id = "X"  # type: ignore[misc]
