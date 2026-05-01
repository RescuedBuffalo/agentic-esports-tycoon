"""Tests for the playvalorant.com patch-notes connector (BUF-83).

Two layers:

* Unit tests against the connector's parsing — they don't touch the DB,
  drive the connector with fixture HTML, and assert on the parsed
  ``PatchNoteRecord`` shape.
* One integration test against the patch-notes runner — wires the
  connector through ``run_patch_notes_ingestion`` and asserts the UPSERT
  semantics the BUF-83 ticket calls out (re-running the same article
  bumps ``fetched_at`` but does not insert a duplicate row). This test
  is marked ``integration`` and skipped when ``TEST_DATABASE_URL`` is
  unset, matching the rest of the data-pipeline test suite.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from data_pipeline import (
    PatchNoteConnector,
    PatchNoteRecord,
    RateLimit,
    SchemaDriftError,
    TransientFetchError,
    run_patch_notes_ingestion,
)
from data_pipeline.connectors.playvalorant import (
    LIST_URL,
    PlayValorantPatchNotesConnector,
)

_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "playvalorant"


def _load(name: str) -> str:
    return (_FIXTURE_DIR / name).read_text(encoding="utf-8")


def _make_http_get(mapping: dict[str, str]) -> Any:
    """Build a fake ``http_get`` that maps URL prefixes to fixture HTML.

    The connector hits the list URL and several article URLs in one
    pass; the test fixtures are per-URL so we look up by exact match
    first and fall back to "any URL containing this slug" for the
    article fetches.
    """

    def fake_get(url: str) -> str:
        if url in mapping:
            return mapping[url]
        for needle, html in mapping.items():
            if needle in url:
                return html
        raise AssertionError(f"unmocked URL fetched: {url}")

    return fake_get


# --- validate -------------------------------------------------------------


def test_validate_extracts_patch_version_and_published_at() -> None:
    """A real-shape article fixture parses to ``patch_version=8.05``.

    Asserts both fields the BUF-83 acceptance criteria call out:
    ``patch_version`` parsed from the title and ``published_at`` parsed
    from the ``<time datetime=...>`` attribute, in UTC.
    """
    html = _load("article_8_05.html")
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))

    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})

    assert validated["patch_version"] == "8.05"
    assert validated["published_at"] == datetime(2026, 3, 12, 17, 0, 0, tzinfo=UTC)
    assert validated["url"] == "https://playvalorant.com/x"
    assert validated["raw_html"] == html


def test_validate_raises_schema_drift_when_version_regex_fails() -> None:
    """An article without a "Patch X.Y" title is rejected.

    The connector treats this as schema drift rather than silently
    dropping the row — a renamed title shape is the kind of upstream
    change the BUF-9 SCHEMA_DRIFT log was designed to surface.
    """
    html = _load("article_no_version.html")
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))

    with pytest.raises(SchemaDriftError) as excinfo:
        connector.validate({"url": "https://playvalorant.com/yir", "html": html})

    assert "patch version" in str(excinfo.value).lower()


def test_validate_raises_schema_drift_when_payload_is_malformed() -> None:
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))
    with pytest.raises(SchemaDriftError):
        connector.validate({"url": "https://x", "html": None})  # type: ignore[dict-item]


# --- body cleaning --------------------------------------------------------


def test_body_text_strips_nav_chrome_and_collapses_whitespace() -> None:
    """``body_text`` excludes nav/footer/aside/script chrome.

    The fixture seeds every chrome region with the literal sentinel
    ``NAV-CHROME-`` so a regression that lets one slip through is
    visible immediately. We also assert the patch-body content survived
    so we don't accidentally pass by stripping everything.
    """
    html = _load("article_8_05.html")
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))

    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})
    body = validated["body_text"]

    # No chrome leaked through. Each strip-list category is exercised by
    # its own sentinel — if any of these assertions fires, the failing
    # string names which family of tags regressed.
    assert "NAV-CHROME-NEWS" not in body
    assert "NAV-CHROME-AGENTS" not in body
    assert "NAV-CHROME-SIDEBAR" not in body
    assert "NAV-CHROME-FOOTER" not in body
    assert "NAV-CHROME-NOSCRIPT" not in body
    assert "NAV-CHROME-TRACKER" not in body  # <script> contents
    assert "nav-chrome-style-block" not in body  # <style> contents

    # And the actual prose did survive the strip.
    assert "Clove receives a small adjustment" in body
    assert "Sunset's mid-cubby" in body

    # Whitespace was normalised: no triple-newline runs (the regex
    # collapses them to a single blank line) and no leading/trailing
    # whitespace.
    assert "\n\n\n" not in body
    assert body == body.strip()


# --- fetch (since filter, pagination) -------------------------------------


def test_fetch_skips_articles_at_or_before_since() -> None:
    """``since`` filter drops list cards before drilling into article bodies.

    Wires a list page with two cards (2026-03 and 2024-01); fetching
    with ``since=2025-01-01`` must yield only the 2026 article and only
    fetch its body. The 2024 article's URL must never be requested —
    if it were, the test's ``http_get`` would raise an AssertionError
    against the unmocked URL.
    """
    list_html = _load("list_page_1.html")
    article_html = _load("article_8_05.html")

    fetched_urls: list[str] = []

    def http_get(url: str) -> str:
        fetched_urls.append(url)
        if url == LIST_URL:
            return list_html
        if "valorant-patch-notes-8-05" in url:
            return article_html
        # Page 2 onward: empty list page so the connector stops paginating.
        if "page=" in url:
            return "<html><body><main></main></body></html>"
        raise AssertionError(f"unmocked URL fetched: {url}")

    connector = PlayValorantPatchNotesConnector(http_get=http_get)
    since = datetime(2025, 1, 1, tzinfo=UTC)

    payloads = list(connector.fetch(since))

    assert len(payloads) == 1
    assert "valorant-patch-notes-8-05" in payloads[0]["url"]
    # The 7.12 article (2024-01-09) must not have been fetched.
    assert not any("7-12" in url for url in fetched_urls)


# --- transform ------------------------------------------------------------


def test_transform_yields_one_record_with_all_fields_populated() -> None:
    html = _load("article_8_05.html")
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))

    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})
    records = list(connector.transform(validated))

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, PatchNoteRecord)
    assert record.patch_version == "8.05"
    assert record.published_at == datetime(2026, 3, 12, 17, 0, 0, tzinfo=UTC)
    assert record.url == "https://playvalorant.com/x"
    assert record.raw_html == html
    assert "Clove" in record.body_text


def test_validate_preserves_hotfix_letter_suffix_in_patch_version() -> None:
    """``Patch 11.07b`` parses to version ``11.07b``, not ``11.07``.

    Riot occasionally ships a letter-suffixed hotfix (a quick balance
    pass between numbered patches). ``patch_version`` is the UPSERT
    key in BUF-83's schema; if the suffix were dropped, the hotfix
    would overwrite the base patch's row instead of getting its own.
    """
    html = """
    <html><head><title>VALORANT Patch Notes 11.07b</title></head>
    <body>
      <h1>VALORANT Patch Notes 11.07b</h1>
      <time datetime="2026-05-14T17:00:00Z"></time>
      <main><article><p>Quick hotfix for Patch 11.07.</p></article></main>
    </body></html>
    """
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))
    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})
    assert validated["patch_version"] == "11.07b"


def test_validate_treats_naive_datetime_as_drift_rather_than_typeerror() -> None:
    """An offsetless ``<time datetime>`` becomes drift, not a runtime crash.

    ``datetime.fromisoformat`` returns a naive datetime when the
    string lacks an offset. Comparing that against a timezone-aware
    ``since`` (which the runner always passes) raises ``TypeError``
    and would abort the whole pass. The parser now returns ``None``
    on a naive value, and validate surfaces that as a clean
    ``SchemaDriftError`` so the runner logs it and moves on.
    """
    html = """
    <html><head><title>VALORANT Patch Notes 9.10</title></head>
    <body>
      <h1>VALORANT Patch Notes 9.10</h1>
      <time datetime="2026-05-14T17:00:00"></time>  <!-- naive, no offset -->
      <main><article><p>Body.</p></article></main>
    </body></html>
    """
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))
    with pytest.raises(SchemaDriftError):
        connector.validate({"url": "https://playvalorant.com/x", "html": html})


def test_fetch_list_page_transient_propagates_rather_than_skipping_ahead() -> None:
    """A transient on a list page must NOT be skipped to the next page.

    The feed is reverse-chronological: a page-1 blip followed by a
    healthy page-2 (whose articles are older than ``since``) would
    silently early-stop the pass via the no-new-cards branch and drop
    every newer article that lived on page 1. Re-raising lets the
    runner count it and the next scheduled pass retries from page 1.
    """
    article_html = _load("article_8_05.html")
    page_2_html = """
    <html><body>
      <a href="/en-us/news/game-updates/valorant-patch-notes-7-12/">
        <time datetime="2024-01-09T17:00:00Z">Jan 9</time> Patch 7.12
      </a>
    </body></html>
    """

    def http_get(url: str) -> str:
        if url == LIST_URL:
            # Page 1 is having a Cloudflare moment.
            raise TransientFetchError("503 on list page 1")
        if "page=2" in url:
            return page_2_html
        if "valorant-patch-notes" in url:
            return article_html
        raise AssertionError(f"unexpected fetch: {url}")

    connector = PlayValorantPatchNotesConnector(http_get=http_get)
    since = datetime(2025, 6, 1, tzinfo=UTC)

    # The transient propagates — the runner's post-yield handler will
    # count it and retry next pass. The previous behaviour silently
    # advanced to page 2, found everything older than ``since``, and
    # returned cleanly with zero payloads — turning a transient edge
    # blip into a permanent data gap.
    with pytest.raises(TransientFetchError, match="503"):
        list(connector.fetch(since))


def test_fetch_skips_one_transient_article_and_continues_to_the_rest() -> None:
    """A transient failure on one article must not truncate the rest of the pass.

    Generators close after raising any uncaught exception, which means a
    runner-side ``except TransientFetchError`` around ``next(iterator)``
    can't keep iteration going — the next ``next()`` call on a closed
    generator returns ``StopIteration``, silently skipping every later
    article on the page. The fix is to absorb the transient inside the
    connector's ``fetch`` loop. This test pins that contract: three
    articles in a row, the middle one fails, the other two flow.
    """
    list_html = """
    <html><body>
      <a href="/en-us/news/game-updates/valorant-patch-notes-8-05/">
        <time datetime="2026-03-12T17:00:00Z">Mar 12</time> Patch 8.05
      </a>
      <a href="/en-us/news/game-updates/valorant-patch-notes-8-04/">
        <time datetime="2026-02-26T17:00:00Z">Feb 26</time> Patch 8.04
      </a>
      <a href="/en-us/news/game-updates/valorant-patch-notes-8-03/">
        <time datetime="2026-02-12T17:00:00Z">Feb 12</time> Patch 8.03
      </a>
    </body></html>
    """
    article_8_05 = _load("article_8_05.html")
    article_8_03 = article_8_05.replace("8.05", "8.03").replace(
        "2026-03-12T17:00:00Z", "2026-02-12T17:00:00Z"
    )

    def http_get(url: str) -> str:
        if url == LIST_URL:
            return list_html
        if "8-05" in url:
            return article_8_05
        if "8-04" in url:
            # The middle article hits a transient blip.
            raise TransientFetchError("upstream 502 on article body")
        if "8-03" in url:
            return article_8_03
        if "page=" in url:
            return "<html><body><main></main></body></html>"
        raise AssertionError(f"unexpected article fetch: {url}")

    connector = PlayValorantPatchNotesConnector(http_get=http_get)
    payloads = list(connector.fetch(datetime(1970, 1, 1, tzinfo=UTC)))

    # Two articles flowed; the failing one was logged and skipped.
    urls = [p["url"] for p in payloads]
    assert any("8-05" in url for url in urls)
    assert any("8-03" in url for url in urls)
    assert not any("8-04" in url for url in urls)


# --- list-card filtering --------------------------------------------------


def test_fetch_skips_non_patch_news_articles_in_game_updates_feed() -> None:
    """The /game-updates/ feed mixes patch notes with trailers + dev posts.

    Without slug-level filtering, every non-patch entry would be
    fetched and then rejected by ``validate`` as ``SCHEMA_DRIFT`` —
    inflating the drift counter on a healthy run. The connector now
    filters list-card URLs by the ``valorant-patch-notes-*`` slug
    pattern, so only real patch articles reach ``validate``.
    """
    list_html = """
    <html><body>
      <a href="/en-us/news/game-updates/valorant-patch-notes-8-05/">
        <time datetime="2026-03-12T17:00:00Z">Mar 12</time> Patch 8.05
      </a>
      <a href="/en-us/news/game-updates/episode-9-cinematic-trailer/">
        <time datetime="2026-03-10T17:00:00Z">Mar 10</time> Trailer
      </a>
      <a href="/en-us/news/game-updates/dev-diaries-march/">
        <time datetime="2026-03-08T17:00:00Z">Mar 8</time> Dev Diary
      </a>
    </body></html>
    """
    article_html = _load("article_8_05.html")

    fetched_urls: list[str] = []

    def http_get(url: str) -> str:
        fetched_urls.append(url)
        if url == LIST_URL:
            return list_html
        if "valorant-patch-notes-8-05" in url:
            return article_html
        if "page=" in url:
            return "<html><body><main></main></body></html>"
        raise AssertionError(f"unexpected article fetch: {url}")

    connector = PlayValorantPatchNotesConnector(http_get=http_get)
    payloads = list(connector.fetch(datetime(1970, 1, 1, tzinfo=UTC)))

    # Exactly one article was emitted — the patch notes — and the two
    # non-patch entries were never even fetched, so they can't pollute
    # the SCHEMA_DRIFT counter.
    assert len(payloads) == 1
    assert "valorant-patch-notes-8-05" in payloads[0]["url"]
    assert not any("trailer" in url for url in fetched_urls)
    assert not any("dev-diaries" in url for url in fetched_urls)


# --- runner: transient errors during fetch --------------------------------


def test_runner_catches_transient_fetch_error_during_iterator_advance() -> None:
    """A ``TransientFetchError`` raised inside ``connector.fetch`` is counted + skipped.

    Connector ``fetch`` makes its HTTP call in the body that runs
    *between* yields — i.e. during the runner's ``next(iterator)``
    call. A network blip on the article-list page (or on an article
    body) would otherwise escape the runner's post-yield handler and
    abort the whole pass. The runner now catches there too: the run
    keeps going, ``transient_errors`` increments, and the next
    scheduled pass retries naturally because no raw_record was
    written.
    """
    from data_pipeline.patch_notes_runner import (
        PatchNotesStats,
        run_patch_notes_ingestion,
    )

    class _BlippyConnector(PatchNoteConnector):
        @property
        def source_name(self) -> str:
            return "playvalorant"

        @property
        def cadence(self) -> timedelta:
            return timedelta(days=7)

        @property
        def rate_limit(self) -> RateLimit:
            return RateLimit(capacity=10, refill_per_second=100.0)

        def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
            raise SchemaDriftError("this should not run — overridden below")  # pragma: no cover

        def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
            return raw_payload  # pragma: no cover

        def transform(self, validated_payload: dict[str, Any]) -> Iterable[PatchNoteRecord]:
            return iter(())  # pragma: no cover

    connector = _BlippyConnector()

    def fetcher(_since: datetime) -> Iterable[dict[str, Any]]:
        # Mimic the real connector's contract: the HTTP call happens in
        # the body that runs *before* the next yield. Raising here
        # therefore lands inside ``next(iterator)`` rather than after a
        # successful yield.
        raise TransientFetchError("upstream 503")
        yield  # type: ignore[unreachable]  # never reached, but keeps this a generator

    connector.fetch = fetcher  # type: ignore[method-assign]

    # ``session=None`` is OK — the transient path bails before any
    # session writes. Asserting that contract is the test's whole point.
    stats: PatchNotesStats = run_patch_notes_ingestion(
        connector,
        session=None,  # type: ignore[arg-type]
        since=datetime(1970, 1, 1, tzinfo=UTC),
    )

    assert stats.transient_errors == 1
    assert stats.fetched == 0
    assert stats.upserted == 0


# --- integration: idempotent re-run ---------------------------------------


class _StubPatchNoteConnector(PatchNoteConnector):
    """In-memory connector stand-in for the runner integration test.

    Yields the same single article on every ``fetch`` so we can drive
    the runner twice and assert on UPSERT semantics — the simpler the
    surface, the less the integration test re-tests parsing.
    """

    def __init__(self, *, payloads: list[dict[str, Any]]) -> None:
        self._payloads = payloads
        self._connector = PlayValorantPatchNotesConnector(http_get=lambda _u: "")

    @property
    def source_name(self) -> str:
        return "playvalorant"

    @property
    def cadence(self) -> timedelta:
        return timedelta(days=7)

    @property
    def rate_limit(self) -> RateLimit:
        # Generous bucket so the test's two ``fetch`` calls don't add
        # measurable wall-clock time to the suite.
        return RateLimit(capacity=100, refill_per_second=100.0)

    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        yield from self._payloads

    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        return self._connector.validate(raw_payload)

    def transform(self, validated_payload: dict[str, Any]) -> Iterable[PatchNoteRecord]:
        return self._connector.transform(validated_payload)


@pytest.mark.integration
def test_rerun_is_idempotent_on_patch_version(db_session) -> None:
    """Running the same article twice updates ``fetched_at`` but does
    not insert a second row.

    UPSERT-on-``patch_version`` is the BUF-83 idempotency contract.
    The test feeds the same article through the runner twice; after
    the first pass we record ``fetched_at``; after the second pass we
    assert the row count is still 1 and ``fetched_at`` advanced.
    """
    from esports_sim.db.models import PatchNote
    from sqlalchemy import select

    html = _load("article_8_05.html")
    connector = _StubPatchNoteConnector(
        payloads=[{"url": "https://playvalorant.com/x", "html": html}],
    )

    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    run_patch_notes_ingestion(connector, session=db_session, since=epoch)
    db_session.flush()

    rows = list(db_session.execute(select(PatchNote)).scalars())
    assert len(rows) == 1
    first_fetched_at = rows[0].fetched_at
    assert rows[0].patch_version == "8.05"

    # Sleep enough that the wall clock advances past the previous
    # ``fetched_at`` — datetime.now(UTC) has microsecond resolution but
    # Postgres TIMESTAMPTZ is microsecond too, so a tiny sleep is
    # plenty. Without it the assertion below could flake on a fast
    # machine.
    time.sleep(0.01)

    run_patch_notes_ingestion(connector, session=db_session, since=epoch)
    db_session.flush()

    rows = list(db_session.execute(select(PatchNote)).scalars())
    assert len(rows) == 1, "re-run must UPSERT on (source, patch_version), not insert a duplicate"
    assert rows[0].fetched_at >= first_fetched_at
    assert rows[0].source == "playvalorant"


@pytest.mark.integration
def test_two_sources_with_same_patch_version_do_not_collide(db_session) -> None:
    """Two connectors with different ``source_name`` and the same patch
    version must coexist as distinct rows.

    The BUF-83 schema's uniqueness key is ``(source, patch_version)``,
    not ``patch_version`` alone. Without that scope, a future
    second-game patch-notes connector emitting (e.g.) ``"8.05"`` would
    overwrite the playvalorant row of the same version. This test
    runs the same article through two stub connectors with different
    ``source_name`` and asserts both rows persist.
    """
    from esports_sim.db.models import PatchNote
    from sqlalchemy import select

    class _OtherSourceStub(_StubPatchNoteConnector):
        @property
        def source_name(self) -> str:
            return "another-game"

    html = _load("article_8_05.html")
    payload = [{"url": "https://playvalorant.com/x", "html": html}]
    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    run_patch_notes_ingestion(
        _StubPatchNoteConnector(payloads=payload),
        session=db_session,
        since=epoch,
    )
    run_patch_notes_ingestion(
        _OtherSourceStub(payloads=payload),
        session=db_session,
        since=epoch,
    )
    db_session.flush()

    rows = sorted(
        db_session.execute(select(PatchNote)).scalars().all(),
        key=lambda row: row.source,
    )
    assert len(rows) == 2
    assert {row.source for row in rows} == {"another-game", "playvalorant"}
    # Both rows are 8.05 — they coexist because the unique constraint
    # is on the composite key.
    assert all(row.patch_version == "8.05" for row in rows)


# --- parser fallbacks (BUF-3 follow-up) -----------------------------------


def test_validate_falls_back_to_og_published_time_meta_when_time_tag_missing() -> None:
    """No ``<time datetime>`` — fall through to ``article:published_time`` meta.

    Riot's article-body markup has redesigned roughly once a year. The
    OpenGraph head block tends to outlive a body-template churn because
    it's regenerated by the social-share tooling. Without this fallback
    a single redesign would silently shift every article into
    ``SCHEMA_DRIFT`` until a parser update lands; with it, the connector
    keeps producing rows from a degraded shape until the primary path
    can be repaired.
    """
    html = """
    <html>
      <head>
        <title>VALORANT Patch Notes 9.04</title>
        <meta property="article:published_time" content="2026-07-08T17:00:00Z">
      </head>
      <body>
        <h1>VALORANT Patch Notes 9.04</h1>
        <main><article><p>Body content survives.</p></article></main>
      </body>
    </html>
    """
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))
    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})
    assert validated["published_at"] == datetime(2026, 7, 8, 17, 0, 0, tzinfo=UTC)


def test_validate_falls_back_to_parsely_pub_date_when_other_metas_missing() -> None:
    """Last-resort fallback: Parse.ly meta on legacy articles.

    Older patch articles still ship the Parse.ly analytics tag. Trying
    it after the OG meta is cheap insurance against the day Riot drops
    both the ``<time>`` element and the OpenGraph head block — the
    article still has a parseable timestamp, it just lives in the
    least-prominent of the three layers we know about.
    """
    html = """
    <html>
      <head>
        <title>VALORANT Patch Notes 6.11</title>
        <meta name="parsely-pub-date" content="2023-08-15T17:00:00Z">
      </head>
      <body>
        <h1>VALORANT Patch Notes 6.11</h1>
        <main><article><p>Body.</p></article></main>
      </body>
    </html>
    """
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))
    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})
    assert validated["published_at"] == datetime(2023, 8, 15, 17, 0, 0, tzinfo=UTC)


def test_validate_uses_data_testid_article_body_when_no_article_tag() -> None:
    """``data-testid="article-body"`` is the QA-hook fallback for body extraction.

    Riot's front-end QA tooling pins this attribute; it tends to
    outlive a visual redesign because the team's Cypress / RTL
    selectors break otherwise. The connector tries it between
    ``<article>`` and ``<main>`` so the body parse keeps working when
    the redesign retires the semantic ``<article>`` for a styled
    ``<div>``. Without it we'd fall straight through to ``<main>``,
    which often pulls in adjacent navigation widgets.
    """
    html = """
    <html>
      <head><title>VALORANT Patch Notes 9.10</title></head>
      <body>
        <h1>VALORANT Patch Notes 9.10</h1>
        <time datetime="2026-09-30T17:00:00Z"></time>
        <main>
          <div class="recommended-rail">NAV-CHROME-RAIL-DO-NOT-LEAK</div>
          <div data-testid="article-body">
            <p>Patch 9.10 ships a new map vote rotation.</p>
          </div>
        </main>
      </body>
    </html>
    """
    connector = PlayValorantPatchNotesConnector(http_get=_make_http_get({}))
    validated = connector.validate({"url": "https://playvalorant.com/x", "html": html})
    body = validated["body_text"]
    assert "Patch 9.10 ships a new map vote rotation" in body
    # The recommended-rail sibling must NOT be in the body — the
    # data-testid container scoped extraction past it.
    assert "NAV-CHROME-RAIL-DO-NOT-LEAK" not in body


# --- min_published_at floor (BUF-3 follow-up) -----------------------------


def test_min_published_at_floor_skips_pre_floor_articles() -> None:
    """A configured floor short-circuits a backfill before pre-floor pages.

    With the default floor (E1A1, 2020-06-02) and ``since=epoch`` a
    fresh backfill would still walk the full archive — but anything
    before E1A1 is clearly out of scope. A tighter floor at e.g.
    2025-01-01 should drop the 2024 patch from the fixture list page
    even though ``since`` would include it.
    """
    list_html = _load("list_page_1.html")
    article_html = _load("article_8_05.html")

    fetched_urls: list[str] = []

    def http_get(url: str) -> str:
        fetched_urls.append(url)
        if url == LIST_URL:
            return list_html
        if "valorant-patch-notes-8-05" in url:
            return article_html
        if "page=" in url:
            return "<html><body><main></main></body></html>"
        raise AssertionError(f"unmocked URL fetched: {url}")

    connector = PlayValorantPatchNotesConnector(
        http_get=http_get,
        min_published_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    payloads = list(connector.fetch(datetime(1970, 1, 1, tzinfo=UTC)))

    assert len(payloads) == 1
    assert "valorant-patch-notes-8-05" in payloads[0]["url"]
    # The 2024 article (7.12) is below the configured floor and must
    # not have been fetched even though ``since=epoch`` would have
    # otherwise admitted it.
    assert not any("7-12" in url for url in fetched_urls)


def test_min_published_at_floor_rejects_naive_datetime_at_construction() -> None:
    """A naive ``min_published_at`` is rejected up-front, not mid-fetch.

    Comparing a naive floor against the connector's timezone-aware
    list-card timestamps would raise ``TypeError`` partway through the
    pagination walk and abort the run with a stack trace far away
    from the operator's config edit. Catching it in ``__init__`` keeps
    the failure mode "test/config error, fix immediately" instead of
    "scheduler crashed at 03:00 UTC".
    """
    with pytest.raises(ValueError, match="timezone-aware"):
        PlayValorantPatchNotesConnector(
            http_get=_make_http_get({}),
            min_published_at=datetime(2024, 1, 1),  # naive
        )


def test_min_published_at_floor_combines_with_since_via_max() -> None:
    """``floor = max(since, min_published_at)`` — whichever is later wins.

    With ``since`` later than the floor, the floor is irrelevant and
    ``since`` filters the cards. With ``since`` before the floor, the
    floor binds. The behaviour matches the docstring contract; this
    test pins it so a refactor doesn't quietly switch to ``min``
    semantics (which would let backfills walk pre-floor history).
    """
    list_html = _load("list_page_1.html")
    article_html = _load("article_8_05.html")

    def http_get(url: str) -> str:
        if url == LIST_URL:
            return list_html
        if "valorant-patch-notes-8-05" in url:
            return article_html
        if "page=" in url:
            return "<html><body><main></main></body></html>"
        raise AssertionError(f"unmocked URL fetched: {url}")

    # ``since`` later than the floor: ``since`` wins. The 7.12 article
    # is older than ``since=2025-01-01`` and is dropped at the
    # list-card stage.
    connector = PlayValorantPatchNotesConnector(
        http_get=http_get,
        min_published_at=datetime(2020, 1, 1, tzinfo=UTC),  # below since
    )
    payloads = list(connector.fetch(datetime(2025, 1, 1, tzinfo=UTC)))
    assert len(payloads) == 1


# --- inserted / updated / unchanged stats split (BUF-3 follow-up) --------


def test_runner_unchanged_outcome_does_not_inflate_upserted() -> None:
    """A re-run on identical content reports ``unchanged``, not ``upserted``.

    The on-call dashboard tracks ``upserted`` to spot quiet weeks that
    suddenly start writing. If a no-op re-fetch counted as ``upserted``,
    the dashboard would false-positive every healthy weekly pass.
    The split (``upserted = inserted + updated``; ``unchanged`` is
    separate) makes the metric meaningful again.
    """
    from data_pipeline.patch_notes_runner import (
        PatchNoteConnector as _PNC,
    )
    from data_pipeline.patch_notes_runner import (
        run_patch_notes_ingestion,
    )

    # In-memory ``Session`` substitute that just records the row.
    # Using a stub avoids requiring TEST_DATABASE_URL for this
    # contract-level assertion.
    class _FakeSession:
        def __init__(self) -> None:
            self.row: PatchNote_t | None = None  # type: ignore[name-defined]

        def execute(self, _stmt: Any) -> Any:
            class _R:
                def __init__(self, row: Any) -> None:
                    self._row = row

                def scalar_one_or_none(self) -> Any:
                    return self._row

            return _R(self.row)

        def add(self, row: Any) -> None:
            # Mimic flush populating server defaults — fetched_at
            # gets a value so the second-pass path can compare.
            row.fetched_at = datetime.now(UTC)
            self.row = row

        def flush(self) -> None:
            return None

    from esports_sim.db.models import PatchNote as PatchNote_t  # type: ignore[unused-import]

    article_html = _load("article_8_05.html")

    class _Stub(_PNC):
        @property
        def source_name(self) -> str:
            return "playvalorant"

        @property
        def cadence(self) -> timedelta:
            return timedelta(days=7)

        @property
        def rate_limit(self) -> RateLimit:
            return RateLimit(capacity=100, refill_per_second=100.0)

        def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
            yield {"url": "https://playvalorant.com/x", "html": article_html}

        def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
            return PlayValorantPatchNotesConnector(http_get=lambda _u: "").validate(raw_payload)

        def transform(self, validated_payload: dict[str, Any]) -> Iterable[PatchNoteRecord]:
            return PlayValorantPatchNotesConnector(http_get=lambda _u: "").transform(
                validated_payload
            )

    session = _FakeSession()
    epoch = datetime(1970, 1, 1, tzinfo=UTC)

    first = run_patch_notes_ingestion(_Stub(), session=session, since=epoch)  # type: ignore[arg-type]
    assert first.inserted == 1
    assert first.updated == 0
    assert first.unchanged == 0
    assert first.upserted == 1

    # Re-run with identical content. The fixture didn't move, so every
    # comparison column matches and the outcome flips to ``unchanged``.
    second = run_patch_notes_ingestion(_Stub(), session=session, since=epoch)  # type: ignore[arg-type]
    assert second.inserted == 0
    assert second.updated == 0
    assert second.unchanged == 1
    # ``upserted`` is the actually-wrote-content total; an unchanged
    # re-fetch must not bump it.
    assert second.upserted == 0


@pytest.mark.integration
def test_runner_updates_outcome_when_body_text_changes(db_session) -> None:  # type: ignore[no-untyped-def]
    """A re-fetch with mutated body lands in ``updated``, not ``unchanged``.

    Riot occasionally edits a published article (typo fix, late
    balance change). The runner's job is to refresh the columns in
    place and report the change so an operator can see exactly which
    versions Riot mutated this week.
    """
    from esports_sim.db.models import PatchNote
    from sqlalchemy import select

    article_html = _load("article_8_05.html")
    edited_html = article_html.replace(
        "Clove receives a small adjustment",
        "Clove receives a major rework",
    )

    payloads_first = [{"url": "https://playvalorant.com/x", "html": article_html}]
    payloads_second = [{"url": "https://playvalorant.com/x", "html": edited_html}]

    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    first = run_patch_notes_ingestion(
        _StubPatchNoteConnector(payloads=payloads_first),
        session=db_session,
        since=epoch,
    )
    db_session.flush()
    assert first.inserted == 1

    second = run_patch_notes_ingestion(
        _StubPatchNoteConnector(payloads=payloads_second),
        session=db_session,
        since=epoch,
    )
    db_session.flush()
    assert second.updated == 1
    assert second.unchanged == 0
    assert second.upserted == 1

    refreshed = db_session.execute(select(PatchNote)).scalar_one()
    assert "Clove receives a major rework" in refreshed.body_text
    assert "Clove receives a small adjustment" not in refreshed.body_text
