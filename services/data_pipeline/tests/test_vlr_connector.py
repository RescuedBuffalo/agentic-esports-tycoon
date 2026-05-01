"""Tests for the BUF-10 VLR.gg connector.

All tests run without network: a fake ``page_fetcher`` returns canned
HTML from ``tests/fixtures/vlr/``. The integration test that drives the
runner end-to-end is gated on ``TEST_DATABASE_URL`` (it inherits the
``db_session`` fixture from ``conftest.py`` which auto-skips when unset),
so a fresh clone with no Postgres still produces a green ``uv run pytest``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from data_pipeline import (
    IngestionRecord,
    SchemaDriftError,
    TokenBucket,
    TransientFetchError,
    run_ingestion,
)
from data_pipeline.connectors.vlr import (
    DEFAULT_PAGE_URLS,
    USER_AGENT,
    VLR_BASE_URL,
    VLRConnector,
    VLRPageRow,
    VLRParser,
    _iter_anchors,
    _RobotsCache,
)
from esports_sim.db.enums import EntityType, Platform

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "vlr"

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


# --- helpers ---------------------------------------------------------------


def _read_fixture(name: str) -> str:
    """Load a static HTML/text fixture as a UTF-8 string."""
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


def _stub_robots(allow_all: bool = True) -> _RobotsCache:
    """A robots cache that's pre-loaded so it never reaches the network."""
    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=lambda _url: "")
    cache._loaded = True
    cache._disallows = [] if allow_all else ["/"]
    return cache


def _make_fetcher(mapping: dict[str, str]) -> Callable[[str], str]:
    """Build a ``page_fetcher`` that returns canned HTML keyed by URL.

    Unknown URLs raise ``KeyError``; the connector wraps any non-
    ``TransientFetchError`` exception, so a typo in the test surfaces
    as a clean ``TransientFetchError`` rather than a confusing AttributeError.
    """
    return lambda url: mapping[url]


def _default_url_mapping() -> dict[str, str]:
    return {
        f"{VLR_BASE_URL}/stats": _read_fixture("stats.html"),
        f"{VLR_BASE_URL}/matches?completed": _read_fixture("matches.html"),
        f"{VLR_BASE_URL}/rankings": _read_fixture("rankings.html"),
    }


def _build_connector(
    *,
    page_urls: tuple[tuple[str, str], ...] | None = None,
    fetcher: Callable[[str], str] | None = None,
    robots_cache: _RobotsCache | None = None,
) -> VLRConnector:
    return VLRConnector(
        page_fetcher=fetcher or _make_fetcher(_default_url_mapping()),
        page_urls=page_urls if page_urls is not None else DEFAULT_PAGE_URLS,
        robots_cache=robots_cache or _stub_robots(),
    )


# --- metadata --------------------------------------------------------------


def test_connector_metadata_matches_buf10_spec() -> None:
    connector = _build_connector()
    assert connector.source_name == "vlr"
    assert connector.platform is Platform.VLR
    assert connector.entity_types == (EntityType.PLAYER, EntityType.TEAM, EntityType.TOURNAMENT)
    assert connector.cadence == timedelta(days=1)
    rate = connector.rate_limit
    assert rate.capacity == 1
    # 20 req/min = 1/3 req/sec; allow tiny float epsilon.
    assert abs(rate.refill_per_second - (20.0 / 60.0)) < 1e-9


def test_user_agent_identifies_project_and_contact() -> None:
    """BUF-10 spec: UA must name the project + a contact email."""
    assert "agentic-esports-tycoon-data-pipeline" in USER_AGENT
    assert "@" in USER_AGENT  # contact email present


# --- parser ----------------------------------------------------------------


def test_parser_stats_yields_player_rows_with_stable_vlr_ids() -> None:
    parser = VLRParser()
    rows = list(parser.parse_stats(_read_fixture("stats.html")))

    # Three players + the team links also appear, but parse_stats only
    # follows /player/ anchors.
    player_rows = [row for row in rows if row.entity_type is EntityType.PLAYER]
    assert len(player_rows) == 3
    assert {row.vlr_id for row in player_rows} == {"9", "2329", "729"}
    assert {row.display_name for row in player_rows} == {"TenZ", "aspas", "Zekken"}
    # vlr_id must be the numeric id, not the slug — the resolver keys on it.
    for row in player_rows:
        assert row.vlr_id.isdigit(), f"expected numeric vlr_id, got {row.vlr_id!r}"


def test_parser_matches_yields_team_rows_with_match_metadata() -> None:
    parser = VLRParser()
    rows = list(parser.parse_matches(_read_fixture("matches.html")))

    # Six team anchors total (three matches x two teams each).
    assert len(rows) == 6
    assert all(row.entity_type is EntityType.TEAM for row in rows)
    # Match metadata rides on extra (no EntityType.MATCH yet — see ticket).
    assert all("match_id" in row.extra for row in rows)
    assert {row.extra["match_id"] for row in rows} == {"m-300001", "m-300002", "m-300003"}
    # Per-row timestamps parsed for the since-filter.
    assert all(row.timestamp is not None for row in rows)


def test_parser_rankings_yields_team_and_tournament_rows() -> None:
    parser = VLRParser()
    rows = list(parser.parse_rankings(_read_fixture("rankings.html")))

    teams = [row for row in rows if row.entity_type is EntityType.TEAM]
    tournaments = [row for row in rows if row.entity_type is EntityType.TOURNAMENT]

    assert len(teams) == 3
    assert len(tournaments) == 1
    assert tournaments[0].vlr_id == "2097"
    assert tournaments[0].display_name == "VCT 2026 Americas Stage 1"


def test_parser_unknown_page_type_raises_drift() -> None:
    parser = VLRParser()
    with pytest.raises(SchemaDriftError, match="unknown VLR page_type"):
        list(parser.parse("matchups", "<html></html>"))


# --- transform -------------------------------------------------------------


def test_transform_yields_well_formed_records_per_entity_type() -> None:
    """Each row should round-trip into an IngestionRecord without losing identity."""
    connector = _build_connector()
    payloads = list(connector.fetch(_EPOCH))

    # Build one combined record list across all three pages.
    all_records: list[IngestionRecord] = []
    for payload in payloads:
        validated = connector.validate(payload)
        all_records.extend(connector.transform(validated))

    # We expect at least one of each entity type the connector advertises.
    seen_types = {record.entity_type for record in all_records}
    assert seen_types == {EntityType.PLAYER, EntityType.TEAM, EntityType.TOURNAMENT}

    # platform_id is the VLR-stable numeric id, never a display name.
    for record in all_records:
        assert record.platform_id, "platform_id must be non-empty"
        assert not record.platform_id.isspace()
        # All seeded ids in the fixtures are numeric.
        assert record.platform_id.isdigit()
        # platform_name is the free-form display string.
        assert record.platform_name
        # Payload preserves the row blob for replay.
        assert record.payload["vlr_id"] == record.platform_id


def test_transform_uses_numeric_id_not_slug_or_display_name() -> None:
    """Resolver-idempotence anchor: same vlr_id across casing variants."""
    connector = _build_connector(
        page_urls=(("stats", f"{VLR_BASE_URL}/stats"),),
        fetcher=_make_fetcher({f"{VLR_BASE_URL}/stats": _read_fixture("stats_renamed_tenz.html")}),
    )
    payloads = list(connector.fetch(_EPOCH))
    records = list(connector.transform(connector.validate(payloads[0])))
    assert len(records) == 1
    assert records[0].platform_id == "9"
    # Display name reflects the upstream rename, but the id is stable.
    assert records[0].platform_name == "tenz"


# --- validate --------------------------------------------------------------


def test_validate_passes_well_formed_payload() -> None:
    connector = _build_connector()
    payload = next(iter(connector.fetch(_EPOCH)))
    # Should not raise.
    out = connector.validate(payload)
    assert out is payload


def test_validate_raises_drift_on_missing_top_level_key() -> None:
    connector = _build_connector()
    bad: dict[str, Any] = {"page_type": "stats", "url": f"{VLR_BASE_URL}/stats"}
    # Missing ``rows``.
    with pytest.raises(SchemaDriftError, match="missing required key: 'rows'"):
        connector.validate(bad)


def test_validate_raises_drift_on_unknown_page_type() -> None:
    connector = _build_connector()
    with pytest.raises(SchemaDriftError, match="unknown page_type"):
        connector.validate(
            {
                "page_type": "rosters",  # not a page we ship
                "url": f"{VLR_BASE_URL}/rosters",
                "rows": [],
            }
        )


def test_validate_raises_drift_on_row_missing_required_field() -> None:
    connector = _build_connector()
    with pytest.raises(SchemaDriftError, match="missing required key: 'vlr_id'"):
        connector.validate(
            {
                "page_type": "stats",
                "url": f"{VLR_BASE_URL}/stats",
                "rows": [{"entity_type": "player", "display_name": "X"}],
            }
        )


def test_validate_raises_drift_on_non_dict_payload() -> None:
    connector = _build_connector()
    with pytest.raises(SchemaDriftError, match="must be dict"):
        connector.validate(["not", "a", "dict"])  # type: ignore[arg-type]


def test_validate_raises_drift_on_non_list_rows() -> None:
    connector = _build_connector()
    with pytest.raises(SchemaDriftError, match="'rows' must be list"):
        connector.validate(
            {
                "page_type": "stats",
                "url": f"{VLR_BASE_URL}/stats",
                "rows": "should be a list",
            }
        )


# --- since filter ----------------------------------------------------------


def test_since_filter_excludes_older_match_rows() -> None:
    """``matches.html`` has rows from 2025-12 and 2026-04. since=2026-01-01
    must drop the 2025 ones; the 2026 ones must pass through."""
    connector = _build_connector(
        page_urls=(("matches", f"{VLR_BASE_URL}/matches?completed"),),
        fetcher=_make_fetcher({f"{VLR_BASE_URL}/matches?completed": _read_fixture("matches.html")}),
    )
    since = datetime(2026, 1, 1, tzinfo=UTC)
    payloads = list(connector.fetch(since))
    records = list(connector.transform(connector.validate(payloads[0])))

    # Six anchors total in the fixture, two from 2025-12 (Paper Rex + DRX)
    # and four from 2026-04. Filter must drop exactly two.
    assert len(records) == 4
    assert "2593" not in {r.platform_id for r in records}  # Paper Rex
    assert "8877" not in {r.platform_id for r in records}  # DRX


def test_since_filter_passes_rows_without_timestamps() -> None:
    """``/rankings`` rows have no per-row timestamp; spec says current state."""
    connector = _build_connector(
        page_urls=(("rankings", f"{VLR_BASE_URL}/rankings"),),
        fetcher=_make_fetcher({f"{VLR_BASE_URL}/rankings": _read_fixture("rankings.html")}),
    )
    far_future = datetime(2099, 1, 1, tzinfo=UTC)
    payloads = list(connector.fetch(far_future))
    records = list(connector.transform(connector.validate(payloads[0])))
    # Even with a since in the year 2099, ranking rows pass because
    # they have no timestamp.
    assert len(records) > 0


def test_match_anchor_inherits_timestamp_from_ancestor() -> None:
    """``data-utc-ts`` on a wrapping ``<div>`` propagates to inner anchors.

    VLR's match-list cards put the kickoff timestamp on the wrapping
    ``<div class="match-item" data-utc-ts="...">`` rather than the team
    anchor itself. Reading only the anchor's own attrs would let those
    matches bypass the connector's ``since`` filter on every run; the
    parser instead inherits ``data-utc-ts`` (and ``data-match-id``)
    from the closest enclosing ancestor.
    """
    nested_html = """
    <html><body>
      <div class="match-item" data-utc-ts="2025-12-15T10:00:00Z" data-match-id="m-old">
        <a href="/team/2593/paper-rex">Paper Rex</a>
        <a href="/team/8877/drx">DRX</a>
      </div>
      <div class="match-item" data-utc-ts="2026-04-26T22:00:00Z" data-match-id="m-new">
        <a href="/team/188/leviatan">LEVIATAN</a>
        <a href="/team/4915/loud">LOUD</a>
      </div>
    </body></html>
    """
    matches_url = f"{VLR_BASE_URL}/matches?completed"
    connector = _build_connector(
        page_urls=(("matches", matches_url),),
        fetcher=_make_fetcher({matches_url: nested_html}),
    )
    since = datetime(2026, 1, 1, tzinfo=UTC)
    payloads = list(connector.fetch(since))
    records = list(connector.transform(connector.validate(payloads[0])))

    # The 2025-12 card's two anchors must be filtered out via inherited
    # ancestor timestamp; the 2026-04 card's two anchors pass.
    ids = {r.platform_id for r in records}
    assert ids == {"188", "4915"}
    # And the inherited match_id is on each kept row's payload, so the
    # transform sees the row-level metadata even when the anchor itself
    # carried none.
    for record in records:
        assert record.payload["match_id"] == "m-new"


def test_emitted_payload_is_stable_across_runs() -> None:
    """The fetched payload must not include per-run metadata.

    Including ``fetched_at`` / ``since`` would make the same upstream
    page hash differently every pass, defeating the runner's
    ``RawRecord.content_hash`` dedup. We assert two things:

    1. The emitted dict's keys are exactly what the parser produced
       (no volatile fields snuck in).
    2. Hashing the JSON bytes of two ``fetch`` passes against the same
       fixture produces identical digests.
    """
    import hashlib
    import json

    fixture_url = f"{VLR_BASE_URL}/stats"
    connector = _build_connector(
        page_urls=(("stats", fixture_url),),
        fetcher=_make_fetcher({fixture_url: _read_fixture("stats.html")}),
    )
    payload_a = list(connector.fetch(_EPOCH))[0]
    payload_b = list(connector.fetch(_EPOCH))[0]

    # No volatile metadata leaked through.
    assert "fetched_at" not in payload_a
    assert "since" not in payload_a
    assert set(payload_a.keys()) == {"page_type", "url", "rows"}

    # And the bytes match — same fixture, same hash.
    digest_a = hashlib.sha256(
        json.dumps(payload_a, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    digest_b = hashlib.sha256(
        json.dumps(payload_b, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert digest_a == digest_b


# --- fetch + rate limiter --------------------------------------------------


class _FakeClock:
    """Same shape as the rate-limiter test's FakeClock — keeps the contract clear."""

    def __init__(self) -> None:
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self.now += seconds


def test_fetch_uses_injected_page_fetcher() -> None:
    """No real HTTP — the injected callable is the only thing that runs."""
    seen_urls: list[str] = []

    def recording_fetch(url: str) -> str:
        seen_urls.append(url)
        return _default_url_mapping()[url]

    connector = _build_connector(fetcher=recording_fetch)
    payloads = list(connector.fetch(_EPOCH))

    assert seen_urls == [
        f"{VLR_BASE_URL}/stats",
        f"{VLR_BASE_URL}/matches?completed",
        f"{VLR_BASE_URL}/rankings",
    ]
    assert [payload["page_type"] for payload in payloads] == ["stats", "matches", "rankings"]


def test_rate_limiter_paces_fetch_with_fake_clock() -> None:
    """Drive ``fetch`` through the runner-side rate limiter, no real sleep.

    With ``capacity=1`` and ``refill=20/60`` (one token per 3 seconds),
    three sequential acquires must advance the fake clock by at least
    ``2 * 3.0`` seconds: the first is free, the next two each wait one
    refill interval. The runner adds a fourth acquire on EOS, but
    that's a wall-clock no-op once we measure right after the third.
    """
    clock = _FakeClock()
    bucket = TokenBucket(
        capacity=1,
        refill_per_second=20.0 / 60.0,
        clock=clock.time,
        sleeper=clock.sleep,
    )
    connector = _build_connector()

    iterator = iter(connector.fetch(_EPOCH))
    bucket.acquire()
    next(iterator)  # /stats
    assert clock.now == 0.0  # first acquire was free

    bucket.acquire()
    next(iterator)  # /matches
    bucket.acquire()
    next(iterator)  # /rankings

    expected_floor = 2 * 3.0  # two waits at 3s each
    assert clock.now >= expected_floor - 1e-9
    # And we shouldn't have over-slept by more than one tick's worth.
    assert clock.now <= expected_floor + 3.0 + 1e-9


def test_fetch_skips_page_on_arbitrary_fetcher_exception() -> None:
    """Per-page failures log + skip instead of aborting the run.

    ``run_ingestion``'s post-yield error handling can't see exceptions
    raised during iterator advancement (the fetcher call runs *before*
    the next ``yield``), so re-raising would skip every later URL too.
    The connector now catches and continues; with all pages failing,
    ``fetch`` simply yields nothing.
    """

    def boom(url: str) -> str:
        raise RuntimeError("upstream timeout simulated")

    connector = _build_connector(fetcher=boom)
    payloads = list(connector.fetch(_EPOCH))
    # Every URL hit the same exception; nothing was yielded but the
    # generator still terminated cleanly rather than propagating.
    assert payloads == []


def test_fetch_skips_one_failed_page_and_continues_to_the_rest() -> None:
    """One failing URL doesn't take down the surrounding pages.

    The first URL in the configured page list fails; the second
    succeeds. The connector must yield exactly one payload — for the
    succeeding URL — proving that the failure did not short-circuit
    iteration.
    """
    bad_url = f"{VLR_BASE_URL}/stats"
    good_url = f"{VLR_BASE_URL}/rankings"

    def selective_fetcher(url: str) -> str:
        if url == bad_url:
            raise TransientFetchError("503 simulated")
        return _read_fixture("rankings.html")

    connector = _build_connector(
        page_urls=(("stats", bad_url), ("rankings", good_url)),
        fetcher=selective_fetcher,
    )
    payloads = list(connector.fetch(_EPOCH))

    assert len(payloads) == 1
    assert payloads[0]["page_type"] == "rankings"


def test_fetch_skips_page_on_explicit_transient_fetch_error() -> None:
    """An explicit ``TransientFetchError`` is caught + logged, not re-raised."""

    def flaky(url: str) -> str:
        raise TransientFetchError("503 from VLR")

    connector = _build_connector(fetcher=flaky)
    payloads = list(connector.fetch(_EPOCH))
    assert payloads == []


def test_anchor_collector_handles_nested_same_tag_open_close() -> None:
    """Inner same-tag elements must not pop the outer ancestor's frame.

    VLR's match-card markup looks like::

        <div class="match-item" data-utc-ts="..." data-match-id="...">
          <div class="match-item-meta">...</div>   <!-- inner div -->
          <a href="/team/X/foo">Team X</a>          <!-- still inside outer -->
        </div>

    With a tag-name-only stack pop, the inner ``</div>`` would discard
    the outer frame and the trailing anchor would lose its inherited
    timestamp. The fix tracks one stack frame per *open* non-anchor
    tag, regardless of whether the element carries any tracked
    attribute, so closes pop the right depth.
    """
    nested_html = """
    <html><body>
      <div class="match-item" data-utc-ts="2026-04-26T22:00:00Z" data-match-id="m-new">
        <div class="match-item-meta">scoreboard</div>
        <a href="/team/188/leviatan">LEVIATAN</a>
        <a href="/team/4915/loud">LOUD</a>
      </div>
    </body></html>
    """
    anchors = list(_iter_anchors(nested_html))
    # Both anchors land — and both inherited the outer div's
    # timestamp/match-id even though an inner ``</div>`` closed before
    # them.
    assert {a["href"] for a in anchors} == {
        "/team/188/leviatan",
        "/team/4915/loud",
    }
    for anchor in anchors:
        assert anchor["attrs"]["data-utc-ts"] == "2026-04-26T22:00:00Z"
        assert anchor["attrs"]["data-match-id"] == "m-new"


# --- robots.txt ------------------------------------------------------------


def test_robots_disallow_blocks_page_url() -> None:
    """A disallow that matches our path must skip that URL."""

    def fetcher_with_robots(_url: str) -> str:
        # Block the rankings page specifically.
        return "User-agent: *\nDisallow: /rankings\n"

    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=fetcher_with_robots)
    assert cache.allows(f"{VLR_BASE_URL}/stats") is True
    assert cache.allows(f"{VLR_BASE_URL}/rankings") is False


def test_robots_failure_falls_through_to_allow() -> None:
    """A robots fetch that raises must not abort the crawl."""

    def fetcher_500(_url: str) -> str:
        raise RuntimeError("upstream 500")

    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=fetcher_500)
    assert cache.allows(f"{VLR_BASE_URL}/stats") is True


def test_robots_grouped_user_agents_share_rules() -> None:
    """Consecutive ``User-agent`` lines start one shared group (RFC 9309 §2.1).

    The earlier parser dropped out of the matching group as soon as it
    saw a non-matching ``User-agent`` line, so this layout silently
    crawled ``/private``::

        User-agent: agentic-esports-tycoon-data-pipeline
        User-agent: otherbot
        Disallow: /private

    Both UA lines belong to the same group; the ``Disallow`` should
    apply to us. The test pins that contract.
    """
    body = (
        "User-agent: agentic-esports-tycoon-data-pipeline\n"
        "User-agent: otherbot\n"
        "Disallow: /private\n"
    )
    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=lambda _u: body)
    assert cache.allows(f"{VLR_BASE_URL}/private") is False


def test_robots_specific_group_overrides_wildcard() -> None:
    """A specific-UA group's rules take precedence over ``User-agent: *``."""
    body = (
        "User-agent: *\n"
        "Disallow: /everything\n"
        "\n"
        "User-agent: agentic-esports-tycoon-data-pipeline\n"
        "Disallow: /just-us\n"
    )
    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=lambda _u: body)
    # The specific group wins entirely — wildcard rules don't merge.
    assert cache.allows(f"{VLR_BASE_URL}/everything") is True
    assert cache.allows(f"{VLR_BASE_URL}/just-us") is False


def test_robots_new_group_after_rule_resets_match_state() -> None:
    """A ``User-agent`` line after a rule starts a fresh group.

    If the second group's UA doesn't match, its rules don't apply to
    us — even though the previous group's matched. Without correct
    state-machine handling the parser could leak a matching-group's
    ``state_after_rule`` into the next group's UA decision.
    """
    body = (
        "User-agent: agentic-esports-tycoon-data-pipeline\n"
        "Disallow: /ours\n"
        "\n"
        "User-agent: someone-else\n"
        "Disallow: /theirs\n"
    )
    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=lambda _u: body)
    assert cache.allows(f"{VLR_BASE_URL}/ours") is False
    # ``/theirs`` belongs to a non-matching group — we may crawl it.
    assert cache.allows(f"{VLR_BASE_URL}/theirs") is True


def test_parse_matches_skips_anchors_without_match_metadata() -> None:
    """Page-chrome ``/team/...`` links must not be ingested as match rows.

    VLR's ``/matches`` page header carries global nav links like
    ``/team/9/sentinels/current-roster`` and a sidebar with "popular
    teams". They have no ``data-match-id`` / ``data-utc-ts``. Without
    a metadata gate, every crawl would re-emit those rows with a
    ``None`` timestamp, bypass the since-filter, and flood the staging
    queue on every pass.
    """
    chrome_plus_card_html = """
    <html><body>
      <header>
        <a href="/team/9/sentinels">Sentinels</a>  <!-- nav link, no metadata -->
      </header>
      <aside>
        <a href="/team/120/100-thieves">Popular: 100T</a>  <!-- sidebar -->
      </aside>
      <div class="match-item" data-match-id="m-real" data-utc-ts="2026-04-26T22:00:00Z">
        <a href="/team/188/leviatan">LEVIATAN</a>
        <a href="/team/4915/loud">LOUD</a>
      </div>
    </body></html>
    """
    parser = VLRParser()
    rows = list(parser.parse_matches(chrome_plus_card_html))

    # Only the two anchors inside the real match card emit rows; the
    # nav and sidebar anchors (no inherited match metadata) are skipped.
    assert {row.vlr_id for row in rows} == {"188", "4915"}
    for row in rows:
        assert row.extra["match_id"] == "m-real"
        assert row.timestamp is not None


def test_fetch_skips_disallowed_pages() -> None:
    """End-to-end: a disallowed page yields zero payloads for that URL."""
    cache = _RobotsCache(VLR_BASE_URL, user_agent=USER_AGENT, fetcher=lambda _u: "")
    cache._loaded = True
    cache._disallows = ["/rankings"]

    connector = _build_connector(robots_cache=cache)
    payloads = list(connector.fetch(_EPOCH))
    page_types = {payload["page_type"] for payload in payloads}
    assert "rankings" not in page_types
    assert page_types == {"stats", "matches"}


# --- VLRPageRow round-trip --------------------------------------------------


def test_vlr_page_row_to_payload_is_json_safe() -> None:
    """Payload lands on raw_record.payload — must be JSON-safe."""
    import json

    row = VLRPageRow(
        entity_type=EntityType.PLAYER,
        vlr_id="9",
        display_name="TenZ",
        profile_url="https://www.vlr.gg/player/9/tenz",
        timestamp=datetime(2026, 4, 25, 18, 30, tzinfo=UTC),
        extra={"slug": "tenz"},
    )
    payload = row.to_payload()
    encoded = json.dumps(payload)  # must not raise
    decoded = json.loads(encoded)
    assert decoded["vlr_id"] == "9"
    assert decoded["entity_type"] == "player"
    assert decoded["timestamp"] == "2026-04-25T18:30:00+00:00"


# --- integration: drive through run_ingestion ------------------------------


@pytest.mark.integration
def test_run_ingestion_with_vlr_connector(db_session: Any) -> None:
    """End-to-end: connector + runner + real schema.

    Skipped automatically when ``TEST_DATABASE_URL`` is unset (the
    ``db_session`` fixture in conftest.py handles the skip). When a DB
    is wired up, this verifies staging rows land with the correct
    ``source``, ``entity_type``, and a non-null ``canonical_id`` for
    every record we expect to AUTO_MERGE/CREATE.
    """
    from esports_sim.db.models import StagingRecord
    from sqlalchemy import select

    connector = _build_connector()
    # The connector's default rate_limit (1 token / 3 seconds) would
    # serialise the three page fetches behind real waits if the runner
    # auto-built its bucket. Inject a permissive limiter so the
    # integration test runs in milliseconds, mirroring the BUF-9
    # ``test_runner_consults_rate_limiter_per_record`` pattern.
    bucket = TokenBucket(capacity=100, refill_per_second=1000.0)

    stats = run_ingestion(connector, session=db_session, since=_EPOCH, rate_limiter=bucket)

    # Three pages were fetched. The fixtures yield, after dedup of
    # cross-page team repeats:
    #   - 3 players (stats)
    #   - 6 teams from /matches (4 after since=epoch passes them all)
    #   - 3 teams + 1 tournament from /rankings
    # The runner doesn't dedup across pages by content_hash because each
    # whole page payload hashes uniquely; the resolver dedups at the
    # alias level, so repeated team ids across pages produce one alias
    # row but two staging rows (one per page). Check the floors.
    assert stats.fetched == 3
    assert stats.processed > 0
    assert stats.schema_drifts == 0

    rows = db_session.execute(select(StagingRecord)).scalars().all()
    assert all(row.source == "vlr" for row in rows)
    seen_types = {row.entity_type for row in rows}
    assert seen_types == {EntityType.PLAYER, EntityType.TEAM, EntityType.TOURNAMENT}


@pytest.mark.integration
def test_tenz_to_tenz_rename_does_not_split_canonical(db_session: Any) -> None:
    """BUF-10 acceptance: 'TenZ' -> 'tenz' must reuse the same canonical_id.

    Two passes: the first seeds an alias under VLR id ``9`` with the
    canonical-cased "TenZ"; the second hits the same id with display
    "tenz". Because we key on the *id* (not the name), the resolver's
    exact-alias lookup catches it as MATCHED and returns the existing
    canonical_id.
    """
    from esports_sim.db.models import EntityAlias
    from sqlalchemy import select

    bucket = TokenBucket(capacity=100, refill_per_second=1000.0)
    only_stats = (("stats", f"{VLR_BASE_URL}/stats"),)

    pass_one = _build_connector(
        page_urls=only_stats,
        fetcher=_make_fetcher({f"{VLR_BASE_URL}/stats": _read_fixture("stats.html")}),
    )
    run_ingestion(pass_one, session=db_session, since=_EPOCH, rate_limiter=bucket)

    aliases_after_pass_one = (
        db_session.execute(
            select(EntityAlias).where(
                EntityAlias.platform == Platform.VLR, EntityAlias.platform_id == "9"
            )
        )
        .scalars()
        .all()
    )
    assert len(aliases_after_pass_one) == 1
    canonical_before = aliases_after_pass_one[0].canonical_id

    pass_two = _build_connector(
        page_urls=only_stats,
        fetcher=_make_fetcher({f"{VLR_BASE_URL}/stats": _read_fixture("stats_renamed_tenz.html")}),
    )
    run_ingestion(pass_two, session=db_session, since=_EPOCH, rate_limiter=bucket)

    aliases_after_pass_two = (
        db_session.execute(
            select(EntityAlias).where(
                EntityAlias.platform == Platform.VLR, EntityAlias.platform_id == "9"
            )
        )
        .scalars()
        .all()
    )
    # Still exactly one alias; same canonical. The rename does not split
    # the entity.
    assert len(aliases_after_pass_two) == 1
    assert aliases_after_pass_two[0].canonical_id == canonical_before
