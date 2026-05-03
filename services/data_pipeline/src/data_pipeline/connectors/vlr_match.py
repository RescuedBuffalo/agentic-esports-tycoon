"""VLR.gg per-match player participation scraper (BUF-85).

Fills in the player layer the BUF-8 v2 CSV bootstrap deferred. The
seed populated ``match`` + ``map_result`` rows with stable
``vlr_match_id`` / ``vlr_game_id`` anchors but had no roster
information — the CSV's ``Team1Game1..Team2Game5`` columns are
recent-form match-id history, not lineups. Per-map rosters live on
the per-match profile page at ``/match/<id>``, which this module
scrapes.

Pipeline shape:

    list of vlr_match_ids -> rate-limited Playwright fetch
        -> parse rosters/stats per map
        -> create-or-get canonical PLAYER entity per (Platform.VLR, vlr_player_id)
        -> upsert player_match_stat row keyed on (map_result_id, entity_id)

The module deliberately bypasses the steady-state
:func:`~data_pipeline.runner.run_ingestion` flow: that orchestrator
only writes ``raw_record`` and ``staging_record``, and the BUF-85
schema lands typed columns directly on ``player_match_stat``. The
seed module (:mod:`data_pipeline.seeds.vlr`) follows the same
shape — direct create-or-get on ``(Platform.VLR, platform_id)``,
with ``EntityAlias`` namespaced via :func:`vlr_alias_platform_id`
so VLR's overlapping per-resource id spaces (player vs. team vs.
event) cannot collide on the alias unique constraint.

Idempotency is the load-bearing property: a re-scrape over the
same match list is a no-op. The unique constraint
``uq_player_match_stat_map_result_entity`` is the schema-level
guarantee; a ``begin_nested`` savepoint plus an ``IntegrityError``
catch handles the race between two concurrent scrapers.

Out of scope:

* The ``StagingRecord`` write the steady-state runner would do. The
  BUF-85 acceptance criteria is the typed ``player_match_stat`` row;
  raw page HTML is preserved by the caller via ``page_fetcher``
  logging if needed (we do not stash it in ``raw_record`` because
  reprocessing a per-match page is cheap relative to a full crawl).
* ``RoundResult`` per-round granularity. VLR's ``/match/`` page only
  exposes per-map aggregates; the per-round Riot-API parity ticket
  (RescuedBuffalo/agentic-esports-tycoon#2) feeds the same table
  with its own extractor.
* Backfill orchestration. Callers pass a ``vlr_match_ids`` iterable;
  a configurable backfill range is just ``select(Match.vlr_match_id)``
  with a ``where(Match.match_date > ...)`` and lives in the operator
  CLI, not in this module.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any

from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import Entity, EntityAlias, MapResult, PlayerMatchStat
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data_pipeline.connectors.vlr import (
    USER_AGENT,
    VLR_BASE_URL,
    PageFetcher,
    _RobotsCache,
    vlr_alias_platform_id,
)
from data_pipeline.errors import SchemaDriftError, TransientFetchError
from data_pipeline.rate_limiter import TokenBucket

logger = logging.getLogger(__name__)


# Same 20 req/min budget the rest of the VLR connector pays — kept
# consistent so a parallel run of both does not exceed the upstream
# politeness limit at the host level.
_VLR_TOKEN_REFILL_PER_SECOND: float = 20.0 / 60.0
_VLR_TOKEN_CAPACITY: int = 1


# Position-indexed columns inside the ``mod-overview`` stat row, after
# the player and agent cells. VLR's column order has been stable across
# the last several site revisions; if it ever shifts, the parser raises
# :class:`SchemaDriftError` rather than silently mis-attributing stats.
_STAT_ORDER: tuple[str, ...] = (
    "rating",
    "acs",
    "kills",
    "deaths",
    "assists",
    "kd_diff",  # ignored on the model; kept for position alignment
    "kast_pct",
    "adr",
    "hs_pct",
    "first_kills",
    "first_deaths",
    "fk_fd_diff",  # ignored on the model; kept for position alignment
)

# Column count in the VLR ``mod-overview`` per-map stat row. 14 = player
# + agent + 12 stat cells. A row with a different count is treated as
# upstream drift and the row is skipped (logged) rather than silently
# attributing stats to the wrong column.
_EXPECTED_CELL_COUNT: int = 14


_PLAYER_HREF_RE = re.compile(r"^/player/(\d+)/([^/?#]+)")


@dataclass(frozen=True)
class ParsedPlayerStat:
    """One per-map player row extracted from a VLR ``/match/<id>`` page.

    ``vlr_game_id`` is the upstream id the seed already used as the
    ``map_result.vlr_game_id`` anchor; ``vlr_player_id`` is what
    :func:`vlr_alias_platform_id` namespaces with ``"player-"`` to
    produce the alias ``platform_id``. Stats are kept as the
    ``Optional[float|int]`` shape the model column expects — empty
    cells become ``None`` so a forfeit/walkover row preserves the
    absence rather than coercing to zero.
    """

    vlr_game_id: str
    vlr_player_id: str
    display_name: str
    team_side: str | None
    agent: str | None
    rating: float | None
    acs: float | None
    kills: int | None
    deaths: int | None
    assists: int | None
    kast_pct: float | None
    adr: float | None
    hs_pct: float | None
    first_kills: int | None
    first_deaths: int | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VlrMatchScrapeStats:
    """Counters from one :func:`scrape_vlr_match_players` invocation.

    Each per-row outcome maps to exactly one counter so totals add up;
    ``skipped_*`` buckets are non-fatal drops that the operator can
    triage from the structured-log warnings without re-running.
    """

    matches_seen: int = 0
    matches_fetched: int = 0
    matches_skipped_fetch: int = 0
    matches_skipped_drift: int = 0
    players_parsed: int = 0
    players_inserted: int = 0
    players_existing: int = 0
    players_skipped_no_map: int = 0


# --- HTML parser ----------------------------------------------------------


class _MatchPageParser(HTMLParser):
    """State machine for the ``/match/<id>`` rendered DOM.

    Tracks three pieces of state during the walk:

    * The currently-open ``vm-stats-game`` block's ``data-game-id``,
      maintained as a stack so a nested ``</div>`` from an inner
      element does not pop the outer block. The same stack-frame-per-
      open-tag discipline that :class:`_AnchorCollector` uses;
      diverging here would silently mis-attribute every player row
      below the first inner ``</div>``.
    * The current ``<tr>`` row's accumulator: team side (from
      ``mod-t1`` / ``mod-t2`` classes on the row or the player cell),
      player anchor (vlr_player_id + display name), agent name (from
      the first ``<img title="...">`` inside the row), and the list
      of stat cells in order.
    * Per-cell text accumulation. Each ``<td>`` contributes one entry
      to the row's cell list; we strip whitespace at finalize time.

    The parser is forgiving about which ``vm-stats-game`` div the
    ``data-game-id`` lives on — VLR has historically attached it to
    either the outer wrapper or an inner card, so we walk every open
    ``vm-stats-game`` ancestor and use the topmost id we see. Only
    rows that resolve to *both* a vlr_player_id and a vlr_game_id
    are emitted; everything else is treated as page chrome and
    silently dropped (a player anchor in the global header isn't a
    map row).
    """

    _VOID_ELEMENTS = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        # Stack of (tag, game_id_or_none) — one frame per open non-void
        # element so closing tags pop the correct frame.
        self._tag_stack: list[tuple[str, str | None]] = []
        # Current row state — None when we're not inside a player
        # row's <tr>. We do not greedily start a row on every <tr>
        # because the page also contains scoreboard/header tables;
        # rows are committed only if a player anchor is seen inside.
        self._row: dict[str, Any] | None = None
        # Per-cell text buffer. Reset on each <td>.
        self._cell_buffer: list[str] = []
        self._in_cell = False
        self._row_team_side: str | None = None
        self.rows: list[ParsedPlayerStat] = []

    def _current_game_id(self) -> str | None:
        # Walk the stack from the top, returning the closest non-None
        # game id. Multiple nested ``vm-stats-game`` blocks would be a
        # pathological VLR layout, but keeping the lookup explicit
        # means an inner block does override an outer one.
        for _tag, game_id in reversed(self._tag_stack):
            if game_id is not None:
                return game_id
        return None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: v for k, v in attrs if v is not None}
        classes = set((attr_map.get("class") or "").split())

        # Track the game-id when entering a vm-stats-game block. Other
        # ``data-game-id`` attributes elsewhere on the page are
        # ignored — only the wrapping block sets the context.
        game_id_to_push: str | None = None
        if "vm-stats-game" in classes:
            game_id_to_push = attr_map.get("data-game-id")

        if tag in self._VOID_ELEMENTS:
            # Void elements never push frames — they have no closing
            # tag and would leak onto the stack. Their attribute
            # handling (e.g. agent <img>) happens here in starttag.
            if tag == "img" and self._in_cell and self._row is not None:
                # Agent name comes from the img's title attribute (the
                # tooltip VLR renders); fall back to alt for older
                # snapshots that used alt instead.
                title = attr_map.get("title") or attr_map.get("alt")
                if title and self._row.get("agent") is None:
                    self._row["agent"] = title.strip().lower() or None
            return

        self._tag_stack.append((tag, game_id_to_push))

        if tag == "tr" and self._current_game_id() is not None:
            # Start a new candidate row. Whether it actually becomes a
            # player row depends on whether we see a /player/ anchor
            # before the closing </tr>.
            self._row = {
                "game_id": self._current_game_id(),
                "vlr_player_id": None,
                "display_name": None,
                "agent": None,
                "cells": [],
            }
            self._row_team_side = None
            if "mod-t1" in classes:
                self._row_team_side = "team1"
            elif "mod-t2" in classes:
                self._row_team_side = "team2"
            return

        if tag == "td" and self._row is not None:
            self._in_cell = True
            self._cell_buffer = []
            # Some VLR layouts put the team side on the player cell
            # rather than the row. Pick it up either way.
            if self._row_team_side is None:
                if "mod-t1" in classes:
                    self._row_team_side = "team1"
                elif "mod-t2" in classes:
                    self._row_team_side = "team2"
            return

        if tag == "a" and self._in_cell and self._row is not None:
            href = attr_map.get("href", "")
            match = _PLAYER_HREF_RE.match(href)
            if match and self._row["vlr_player_id"] is None:
                self._row["vlr_player_id"] = match.group(1)
                # Slug is fallback display_name if the anchor text is
                # empty — same logic the BUF-10 parser uses.
                self._row["_player_slug"] = match.group(2)
            return

    def handle_endtag(self, tag: str) -> None:
        if tag in self._VOID_ELEMENTS:
            return

        if tag == "td" and self._in_cell and self._row is not None:
            text = "".join(self._cell_buffer).strip()
            self._row["cells"].append(text)
            self._in_cell = False
            self._cell_buffer = []
            # The first cell holds the player name (anchor text). Cache
            # it once so a later cell with the same text doesn't
            # overwrite. ``len(cells) == 1`` is the just-finished
            # player cell.
            if (
                len(self._row["cells"]) == 1
                and self._row.get("display_name") is None
                and self._row.get("vlr_player_id") is not None
            ):
                # The cell text is the anchor's display name; fall
                # back to the slug if the anchor was empty.
                slug_fallback = self._row.get("_player_slug") or ""
                self._row["display_name"] = text or slug_fallback

        if tag == "tr" and self._row is not None:
            row = self._row
            self._row = None
            try:
                if row.get("vlr_player_id") and row.get("game_id"):
                    try:
                        parsed = _row_to_parsed_stat(row, team_side=self._row_team_side)
                    except SchemaDriftError as exc:
                        # One malformed row should not abort the whole
                        # page — losing nine valid players to one drift
                        # event would defeat the BUF-85 backfill. Log
                        # and skip; the orchestrator's per-page
                        # ``schema_drift`` counter increments only when
                        # the page itself is unparseable.
                        logger.warning(
                            "vlr_match.row_drift",
                            extra={
                                "vlr_game_id": row.get("game_id"),
                                "vlr_player_id": row.get("vlr_player_id"),
                                "detail": str(exc),
                            },
                        )
                    else:
                        self.rows.append(parsed)
            finally:
                self._row_team_side = None

        # Pop the matching frame. Well-formed HTML: top of stack
        # matches; misnested HTML walks back to the most recent open
        # tag of the same name and closes everything above it. The
        # inheritance-stack discipline mirrors :class:`_AnchorCollector`.
        if self._tag_stack and self._tag_stack[-1][0] == tag:
            self._tag_stack.pop()
            return
        for index in range(len(self._tag_stack) - 1, -1, -1):
            if self._tag_stack[index][0] == tag:
                del self._tag_stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_buffer.append(data)


def _row_to_parsed_stat(row: dict[str, Any], *, team_side: str | None) -> ParsedPlayerStat:
    """Project an accumulated row dict into a :class:`ParsedPlayerStat`.

    Stat cells beyond the player + agent columns map onto
    :data:`_STAT_ORDER` by position. A row whose cell count diverges
    from :data:`_EXPECTED_CELL_COUNT` is treated as schema drift —
    silently re-mapping by position would attribute, say, ACS into
    the rating column.
    """
    cells: list[str] = row["cells"]
    if len(cells) != _EXPECTED_CELL_COUNT:
        raise SchemaDriftError(
            f"vlr /match row for player {row.get('vlr_player_id')!r} has "
            f"{len(cells)} cells, expected {_EXPECTED_CELL_COUNT}"
        )
    # cells[0] = player (handled), cells[1] = agent (handled via img).
    stat_cells = cells[2:]
    stat_map: dict[str, str] = dict(zip(_STAT_ORDER, stat_cells, strict=True))
    return ParsedPlayerStat(
        vlr_game_id=row["game_id"],
        vlr_player_id=row["vlr_player_id"],
        display_name=row.get("display_name") or row.get("_player_slug") or "",
        team_side=team_side,
        agent=row.get("agent"),
        rating=_parse_float(stat_map["rating"]),
        acs=_parse_float(stat_map["acs"]),
        kills=_parse_int(stat_map["kills"]),
        deaths=_parse_int(stat_map["deaths"]),
        assists=_parse_int(stat_map["assists"]),
        kast_pct=_parse_percent(stat_map["kast_pct"]),
        adr=_parse_float(stat_map["adr"]),
        hs_pct=_parse_percent(stat_map["hs_pct"]),
        first_kills=_parse_int(stat_map["first_kills"]),
        first_deaths=_parse_int(stat_map["first_deaths"]),
    )


def parse_match_page(html: str) -> list[ParsedPlayerStat]:
    """Public entry point: rendered HTML -> list of player rows.

    Caller is expected to have already obtained the HTML via
    Playwright (or an injected fake in tests). Returns a flat list
    across every map on the page; the consumer keys on
    ``vlr_game_id`` to bucket per-map.
    """
    parser = _MatchPageParser()
    parser.feed(html)
    parser.close()
    return list(parser.rows)


# --- numeric parsers ------------------------------------------------------


def _parse_float(value: str) -> float | None:
    candidate = (value or "").strip()
    if not candidate or candidate == "-":
        return None
    # Strip a leading + so "+3" parses; bare "-" is the upstream
    # null sentinel and was handled above.
    if candidate.startswith("+"):
        candidate = candidate[1:]
    try:
        return float(candidate)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    f = _parse_float(value)
    if f is None:
        return None
    if not float(f).is_integer():
        return None
    return int(f)


def _parse_percent(value: str) -> float | None:
    """Parse ``"75%"`` into ``75.0``; bare numerics pass through."""
    candidate = (value or "").strip()
    if not candidate or candidate == "-":
        return None
    if candidate.endswith("%"):
        candidate = candidate[:-1]
    return _parse_float(candidate)


# --- scraper orchestrator -------------------------------------------------


def scrape_vlr_match_players(
    session: Session,
    *,
    vlr_match_ids: Iterable[str],
    page_fetcher: PageFetcher,
    rate_limiter: TokenBucket | None = None,
    robots_cache: _RobotsCache | None = None,
    base_url: str = VLR_BASE_URL,
    user_agent: str = USER_AGENT,
) -> VlrMatchScrapeStats:
    """Backfill ``player_match_stat`` for every match in ``vlr_match_ids``.

    Reads each ``/match/<id>`` page through ``page_fetcher`` (a
    Playwright shim in production, an injected fake in tests),
    parses player rows per map, resolves canonical PLAYER entities
    via direct create-or-get (bypassing fuzzy — the rationale matches
    :func:`seed_from_vlr_csv`'s decision tree), and upserts
    ``player_match_stat`` rows keyed on ``(map_result_id, entity_id)``.

    Idempotency rests on three layers:

    1. The unique constraint ``uq_player_match_stat_map_result_entity``
       at the schema level; the migration owns the guarantee.
    2. A per-row in-memory check before the insert to avoid the
       savepoint round-trip when the row is already known.
    3. A ``begin_nested`` savepoint around the actual insert so a
       race with a concurrent scraper resolves to a clean
       ``IntegrityError`` rather than aborting the outer transaction.

    Caller owns the transaction. The scraper ``flush``es as it goes
    so existing-row checks within a run see fresh inserts; it does
    not ``commit``.
    """
    rate_limiter = rate_limiter or TokenBucket(
        capacity=_VLR_TOKEN_CAPACITY,
        refill_per_second=_VLR_TOKEN_REFILL_PER_SECOND,
    )
    robots_cache = robots_cache or _RobotsCache(base_url, user_agent=user_agent)
    stats = VlrMatchScrapeStats()

    # Pre-load existing PLAYER aliases so the per-row create-or-get is
    # an O(1) hashmap probe instead of a SELECT per id. Only PLAYER
    # rows are relevant; the BUF-85 scraper writes nothing else.
    existing_alias_canonical = _load_existing_player_aliases(session)

    # Pre-load (vlr_game_id -> map_result_id) for every map in the
    # configured match list. A scrape-time miss means the seed never
    # ingested that match — we log + skip rather than mint a phantom
    # map_result row, since the seed is the canonical writer for that
    # column.
    map_id_by_game_id = _load_map_results_for_matches(session, vlr_match_ids)

    # Pre-load existing (map_result_id, entity_id) pairs so the
    # idempotent re-run path doesn't hit the DB at all per row. We
    # query in one shot rather than per-row.
    existing_stat_keys = _load_existing_stat_keys(session, list(map_id_by_game_id.values()))

    for vlr_match_id in vlr_match_ids:
        stats.matches_seen += 1
        url = _match_page_url(base_url, vlr_match_id)
        if not robots_cache.allows(url):
            logger.info("vlr_match.skip_disallowed", extra={"url": url})
            continue

        rate_limiter.acquire()
        try:
            html = page_fetcher(url)
        except TransientFetchError as exc:
            logger.warning(
                "vlr_match.transient_fetch_error",
                extra={"url": url, "vlr_match_id": vlr_match_id, "detail": str(exc)},
            )
            stats.matches_skipped_fetch += 1
            continue
        except Exception as exc:
            # Match-page failures are kept non-fatal so a single
            # broken page doesn't abort a 1000-match backfill. The
            # operator triages from the structured warnings + the
            # final stats counters.
            logger.warning(
                "vlr_match.connector_error",
                extra={
                    "url": url,
                    "vlr_match_id": vlr_match_id,
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                },
            )
            stats.matches_skipped_fetch += 1
            continue

        try:
            parsed_rows = parse_match_page(html)
        except SchemaDriftError as exc:
            logger.warning(
                "vlr_match.schema_drift",
                extra={"url": url, "vlr_match_id": vlr_match_id, "detail": str(exc)},
            )
            stats.matches_skipped_drift += 1
            continue
        stats.matches_fetched += 1

        for parsed in parsed_rows:
            stats.players_parsed += 1
            map_result_id = map_id_by_game_id.get(parsed.vlr_game_id)
            if map_result_id is None:
                # Seeded matches must own their map_result rows; a
                # scrape-time miss usually means the seed never ran
                # for this match. Skip rather than mint a
                # ``map_result`` row from the scraper — the seed is
                # the documented owner of that column.
                stats.players_skipped_no_map += 1
                logger.warning(
                    "vlr_match.no_map_result",
                    extra={
                        "vlr_match_id": vlr_match_id,
                        "vlr_game_id": parsed.vlr_game_id,
                        "vlr_player_id": parsed.vlr_player_id,
                    },
                )
                continue

            entity_id = _resolve_player_canonical(
                session,
                vlr_player_id=parsed.vlr_player_id,
                display_name=parsed.display_name,
                existing=existing_alias_canonical,
            )

            if (map_result_id, entity_id) in existing_stat_keys:
                stats.players_existing += 1
                continue

            inserted = _insert_player_match_stat(
                session,
                parsed=parsed,
                map_result_id=map_result_id,
                entity_id=entity_id,
            )
            if inserted:
                existing_stat_keys.add((map_result_id, entity_id))
                stats.players_inserted += 1
            else:
                stats.players_existing += 1

        session.flush()

    logger.info(
        "vlr_match.done matches_fetched=%d players_inserted=%d players_existing=%d",
        stats.matches_fetched,
        stats.players_inserted,
        stats.players_existing,
    )
    return stats


# --- internals ------------------------------------------------------------


def _match_page_url(base_url: str, vlr_match_id: str) -> str:
    """Compose the ``/match/<id>`` URL the scraper hits."""
    return f"{base_url.rstrip('/')}/match/{urllib.parse.quote(vlr_match_id, safe='')}"


def _load_existing_player_aliases(session: Session) -> dict[str, uuid.UUID]:
    """Map every existing ``platform-namespaced player id -> canonical_id``.

    Loaded once before the per-row resolve so the inner loop is a
    hashmap probe rather than a SELECT per id. Mirrors the seed's
    pre-load pattern.
    """
    rows = session.execute(
        select(EntityAlias.platform_id, EntityAlias.canonical_id).where(
            EntityAlias.platform == Platform.VLR,
            EntityAlias.platform_id.like("player-%"),
        )
    ).all()
    return {platform_id: canonical_id for platform_id, canonical_id in rows}


def _load_map_results_for_matches(
    session: Session, vlr_match_ids: Iterable[str]
) -> dict[str, uuid.UUID]:
    """Return a dict of ``vlr_game_id -> map_result_id`` for every map of every input match.

    Materialising the input ids once means we can query in a single
    round-trip with a JOIN, and a per-map FK lookup during the scrape
    is a hashmap probe.
    """
    ids = list(vlr_match_ids)
    if not ids:
        return {}
    # Lazy import to avoid a top-level circular with Match.
    from esports_sim.db.models import Match

    rows = session.execute(
        select(MapResult.vlr_game_id, MapResult.map_result_id)
        .join(Match, Match.match_id == MapResult.match_id)
        .where(Match.vlr_match_id.in_(ids))
    ).all()
    return {vlr_game_id: map_result_id for vlr_game_id, map_result_id in rows}


def _load_existing_stat_keys(
    session: Session, map_result_ids: list[uuid.UUID]
) -> set[tuple[uuid.UUID, uuid.UUID]]:
    """Pre-load ``{(map_result_id, entity_id), ...}`` for every map in scope.

    Lets the per-player insert path skip the savepoint round-trip on
    re-runs, which is what makes a 1000-match idempotent re-scrape
    cheap. Out-of-scope map ids would just bloat the set without
    helping, so we restrict to the maps the current call touches.
    """
    if not map_result_ids:
        return set()
    rows = session.execute(
        select(PlayerMatchStat.map_result_id, PlayerMatchStat.entity_id).where(
            PlayerMatchStat.map_result_id.in_(map_result_ids)
        )
    ).all()
    return {(map_result_id, entity_id) for map_result_id, entity_id in rows}


def _resolve_player_canonical(
    session: Session,
    *,
    vlr_player_id: str,
    display_name: str,
    existing: dict[str, uuid.UUID],
) -> uuid.UUID:
    """Create-or-get the canonical PLAYER entity for one VLR player id.

    Bypasses the fuzzy resolver intentionally — VLR's numeric ids are
    immutable and unique per player; running fuzzy on display names
    would risk auto-merging two distinct players who share a handle
    (a documented historic problem with names like "Dgzin"). The
    direct (Platform.VLR, ``vlr_alias_platform_id`` namespaced) path
    is what the resolver's exact-alias lookup would do anyway in the
    no-fuzzy path.

    Race-safety: a savepoint scopes the insert; a concurrent
    inserter that beat us to the alias row produces an
    ``IntegrityError`` on the unique ``(platform, platform_id)``
    constraint, which we catch and degrade to "existing".
    """
    namespaced_id = vlr_alias_platform_id(EntityType.PLAYER, vlr_player_id)
    cached = existing.get(namespaced_id)
    if cached is not None:
        return cached

    try:
        with session.begin_nested():
            entity = Entity(entity_type=EntityType.PLAYER)
            session.add(entity)
            session.flush()  # populate canonical_id
            session.add(
                EntityAlias(
                    canonical_id=entity.canonical_id,
                    platform=Platform.VLR,
                    platform_id=namespaced_id,
                    platform_name=display_name or namespaced_id,
                    confidence=1.0,
                )
            )
            session.flush()
    except IntegrityError as exc:
        if "uq_entity_alias_platform_platform_id" not in str(exc):
            raise
        existing_canonical = session.execute(
            select(EntityAlias.canonical_id).where(
                EntityAlias.platform == Platform.VLR,
                EntityAlias.platform_id == namespaced_id,
            )
        ).scalar_one()
        existing[namespaced_id] = existing_canonical
        return existing_canonical

    existing[namespaced_id] = entity.canonical_id
    return entity.canonical_id


def _insert_player_match_stat(
    session: Session,
    *,
    parsed: ParsedPlayerStat,
    map_result_id: uuid.UUID,
    entity_id: uuid.UUID,
) -> bool:
    """Insert one row; return True on success, False on idempotent conflict.

    Wrapping the add+flush in a ``begin_nested`` savepoint means a
    concurrent scraper that already inserted the same key triggers
    an ``IntegrityError`` on flush, which we catch and translate
    into the documented "already-exists" outcome — same pattern the
    seed module uses for its alias create-or-get.
    """
    try:
        with session.begin_nested():
            row = PlayerMatchStat(
                map_result_id=map_result_id,
                entity_id=entity_id,
                source="vlr",
                source_player_id=parsed.vlr_player_id,
                team_side=parsed.team_side,
                agent=parsed.agent,
                rating=parsed.rating,
                acs=parsed.acs,
                kills=parsed.kills,
                deaths=parsed.deaths,
                assists=parsed.assists,
                kast_pct=parsed.kast_pct,
                adr=parsed.adr,
                hs_pct=parsed.hs_pct,
                first_kills=parsed.first_kills,
                first_deaths=parsed.first_deaths,
                extra=dict(parsed.extra),
            )
            session.add(row)
            session.flush()
    except IntegrityError as exc:
        if "uq_player_match_stat_map_result_entity" not in str(exc):
            raise
        return False
    return True


def iter_match_ids_from_db(session: Session) -> Iterator[str]:
    """Convenience producer: every ``vlr_match_id`` in the ``match`` table.

    The CLI/operator script default. Wrap in a date filter for a
    backfill range. Kept here (rather than at the call site) so the
    scraper module is self-contained for tests that want to drive
    the full backfill loop with a real DB.
    """
    from esports_sim.db.models import Match

    rows = session.execute(select(Match.vlr_match_id).order_by(Match.match_date.desc())).scalars()
    yield from rows


__all__ = [
    "ParsedPlayerStat",
    "VlrMatchScrapeStats",
    "iter_match_ids_from_db",
    "parse_match_page",
    "scrape_vlr_match_players",
]
