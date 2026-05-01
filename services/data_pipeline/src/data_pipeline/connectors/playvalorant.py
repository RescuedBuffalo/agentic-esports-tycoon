"""playvalorant.com patch-notes scraper (BUF-83).

Fetches patch articles from ``https://playvalorant.com/en-us/news/game-updates/``,
extracts ``patch_version`` / ``published_at`` / clean ``body_text``, and
yields :class:`PatchNoteRecord`s for the runner to UPSERT.

Design constraints:

* The connector is a pure parser around two HTTP GETs (article-list and
  article-body). HTTP is injected as a callable so tests feed local HTML
  fixtures rather than hitting the network.
* The list page is paginated with ``?page=N``. We walk pages until one
  contains zero new articles (after applying the ``since`` filter) — the
  upstream site's pagination is dense enough that "all articles older
  than ``since``" is the correct stop condition. A hard cap on pages is
  in place so a markup change doesn't loop forever.
* Body cleaning strips standard chrome (``<nav>``, ``<footer>``,
  ``<aside>``, ad iframes, ``<script>``/``<style>``). The strip list is
  documented in :func:`_clean_body_text` so a future site re-skin shows
  up as a single edit point.

Out of scope (per the BUF-83 ticket):

* Robots.txt parsing (we use the same User-Agent as BUF-10's scrapers).
* Backfill driver — the connector exposes ``cadence`` and the operator
  runs a one-shot manual pass to seed history from 2020-06-02 onwards.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from data_pipeline.connector import RateLimit
from data_pipeline.errors import SchemaDriftError, TransientFetchError
from data_pipeline.patch_notes_runner import PatchNoteConnector, PatchNoteRecord

# Per-article / per-list-page transient failures are caught inside
# ``fetch`` (generator-close semantics — see ``fetch`` docstring); we
# log them here so an operator can correlate the skip with the
# upstream blip without having to inspect ``stats.transient_errors``
# alone.
_logger = logging.getLogger("data_pipeline.connectors.playvalorant")

# The article-list page; ``?page=N`` paginates further back. Older pages
# eventually 404 — the connector treats that as end-of-list rather than
# an error.
LIST_URL = "https://playvalorant.com/en-us/news/game-updates/"

# Hard cap on pagination so a markup regression can't drive the connector
# into an infinite loop. Patch cadence is roughly monthly across acts —
# 200 pages comfortably covers everything from Episode 1 Act 1 (2020-06-02)
# forward with several years of headroom.
_MAX_LIST_PAGES = 200

# Default historical floor: VALORANT's E1A1 launch (2020-06-02). The
# connector's ``fetch`` filters articles published on-or-before
# ``max(since, min_published_at)`` so a fresh backfill can pass
# ``since=epoch`` without grinding through pre-launch garbage that
# would otherwise SCHEMA_DRIFT (older news section, unrelated formats).
# Operators can tighten this floor (e.g. to "from patch 5.0 onward")
# via the constructor without touching code.
_DEFAULT_MIN_PUBLISHED_AT = datetime(2020, 6, 2, tzinfo=UTC)

# ``Patch X.YY`` (with optional patch-letter, e.g. "Patch 5.04 ") is the
# title format Riot has used since Episode 1. Riot's <title> tag actually
# reads "VALORANT Patch Notes 8.05" — so we allow an optional " Notes"
# token between the literal "Patch" and the version. The regex is also
# tolerant of leading/trailing whitespace and case differences.
#
# Hotfix suffixes: Riot occasionally ships a letter-suffixed patch
# (e.g. "Patch 11.07b") between numbered drops to deliver a quick
# balance hotfix. The trailing ``[a-z]?`` captures that letter so the
# suffixed article gets its own ``patch_version`` row instead of
# UPSERT-overwriting the base ``11.07`` patch — the BUF-83 schema's
# ``patch_version`` column is the dedup key.
_PATCH_VERSION_RE = re.compile(
    r"\bPatch(?:\s+Notes)?\s+(\d+\.\d+(?:\.\d+)?[a-z]?)",
    re.IGNORECASE,
)

# HTTP fetcher contract: takes a URL, returns the rendered HTML body.
# Tests inject a fake that maps URLs to fixture strings; production wires
# in a real ``httpx.get(...).text`` (with timeout + User-Agent).
HttpGet = Callable[[str], str]

# Tag families to delete before extracting text. ``script`` / ``style``
# leak code into ``get_text``; ``nav`` / ``footer`` / ``aside`` are the
# standard semantic-chrome containers; ``iframe`` covers ad embeds.
# ``noscript`` carries fallback ad markup in some Riot pages.
_STRIP_TAGS = ("script", "style", "nav", "footer", "aside", "iframe", "noscript")

# Whitespace cleanup: collapse runs of spaces/tabs to a single space, then
# collapse runs of blank lines to two newlines (preserves paragraph
# breaks in the patch-note prose).
_WS_RUN = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n\s*", re.MULTILINE)

# Patch-note slugs on playvalorant.com follow ``valorant-patch-notes-X-YY``.
# The /news/game-updates/ feed also includes trailers, dev-team posts, and
# announcement blogs; their slugs do NOT contain ``patch-notes``. Filtering
# at the list-card level avoids fetching every announcement and then
# logging a SCHEMA_DRIFT when version extraction fails on it. The pattern
# is permissive (case-insensitive, allows additional suffix tokens) so a
# minor URL-format tweak doesn't immediately silently lose patches.
_PATCH_NOTE_SLUG_RE = re.compile(r"valorant[-_]patch[-_]notes", re.IGNORECASE)


class PlayValorantPatchNotesConnector(PatchNoteConnector):
    """Scrape playvalorant.com game-updates for patch notes.

    Construction takes an injected ``http_get`` so the same code path
    runs against fixtures in tests and against ``httpx`` in production.
    Defaulting ``http_get`` to ``None`` and lazy-binding it to a real
    ``httpx`` call would couple test setup to httpx; explicit injection
    is cleaner.
    """

    def __init__(
        self,
        *,
        http_get: HttpGet,
        list_url: str = LIST_URL,
        max_list_pages: int = _MAX_LIST_PAGES,
        min_published_at: datetime = _DEFAULT_MIN_PUBLISHED_AT,
    ) -> None:
        """Construct a playvalorant patch-notes connector.

        ``min_published_at`` is a hard archival floor evaluated as
        ``max(since, min_published_at)`` inside ``fetch``. The default
        is VALORANT's E1A1 launch (2020-06-02), the earliest date we
        could plausibly find a real patch article on. Operators
        tighten this when they only want a sub-window — e.g. setting
        it to 2022-06-22 to align with the BUF-3 acceptance criterion
        ("from patch 5.0 onward") for a downstream BUF-24 backfill
        that doesn't need the full pre-5.0 history.

        Why a floor on top of the runner's ``since`` watermark: the
        watermark moves forward as runs complete and is the right
        knob for *incremental* polling. The floor is a static
        configuration that survives a watermark reset (e.g. someone
        runs ``since=EPOCH`` to rebuild from scratch); without it a
        backfill would drill into every ``/news/game-updates/``
        article all the way back to whenever Riot first published
        non-patch news under that path.
        """
        if min_published_at.tzinfo is None:
            # Naive ``min_published_at`` would compare unequally
            # against the timezone-aware list-card timestamps and
            # raise ``TypeError`` mid-fetch. Reject at construction
            # time so the failure mode is "test/config error", not
            # "scheduler crashed at 03:00 UTC".
            raise ValueError("min_published_at must be timezone-aware")
        self._http_get = http_get
        self._list_url = list_url
        self._max_list_pages = max_list_pages
        self._min_published_at = min_published_at

    # -- metadata -----------------------------------------------------------

    @property
    def source_name(self) -> str:
        return "playvalorant"

    @property
    def cadence(self) -> timedelta:
        # Weekly poll: Riot ships patches every two weeks during a normal
        # act and on hotfix days otherwise. Polling weekly keeps the
        # freshness margin (``cadence * 2`` = 14 days) inside the slowest
        # release cycle without spamming a static site.
        return timedelta(days=7)

    @property
    def rate_limit(self) -> RateLimit:
        # Static CDN-backed site, but we're polite: 1 req/s, no burst.
        # Real-world traffic is ~1 list page + ~5 article fetches per run
        # so this is well below any plausible limit.
        return RateLimit(capacity=1, refill_per_second=1.0)

    # -- fetch --------------------------------------------------------------

    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        """Walk the article-list pages, yielding article HTML envelopes.

        Articles whose list-card metadata reports a ``published_at`` at or
        before ``since`` are skipped *before* we drill into the article
        body — the BUF-83 ticket calls this out explicitly to save
        fetches on the incremental cadence.

        We stop paginating when a full page contains no new articles
        (everything is at-or-before ``since``). That works because the
        list is reverse-chronological; once we've passed ``since`` the
        rest of the archive is older still.

        Per-article transient failures are absorbed inside the
        generator (log + skip + continue): we know exactly what we
        missed (one article body) and the rest of the page's articles
        are still discoverable from the same already-loaded list HTML.
        Generator-close semantics make a "raise and let the runner
        catch it" approach truncate the rest of the pass, so per-
        article skipping has to happen here.

        Per-list-page transient failures **propagate**. Skipping a
        failed list page and continuing to the next was tempting but
        wrong: the feed is reverse-chronological, so a transient blip
        on page 1 followed by a healthy page 2 (whose articles are
        already older than ``since``) would trigger the early-stop
        below and silently drop every newer article that lived on the
        failed page. We re-raise instead so the runner increments
        ``transient_errors`` and the next scheduled pass retries from
        page 1 — the only safe outcome when we can't tell whether
        we've crossed the ``since`` watermark yet.

        ``min_published_at`` is layered on top of ``since`` as
        ``floor = max(since, self._min_published_at)``. This protects
        a fresh-from-scratch backfill (``since=EPOCH``) from grinding
        the pre-launch / non-patch tail of the feed.
        """
        floor = max(since, self._min_published_at)
        for page in range(1, self._max_list_pages + 1):
            page_url = self._list_url if page == 1 else f"{self._list_url}?page={page}"
            try:
                list_html = self._http_get(page_url)
            except TransientFetchError:
                # See docstring: skip-ahead would silently drop newer
                # articles when a healthy older page early-stops the
                # pass. Re-raise so the runner counts + retries.
                _logger.warning(
                    "playvalorant.list_page_transient_error",
                    extra={"page_url": page_url},
                )
                raise
            except Exception as exc:  # pragma: no cover - safety net
                # Anything else is fatal-by-default; wrap the URL in for
                # operator context.
                raise RuntimeError(f"playvalorant list fetch failed for {page_url}") from exc

            cards = list(_parse_article_cards(list_html, base_url=self._list_url))
            if not cards:
                # Empty page = end of archive (or 404 turned into empty by
                # the ``http_get`` impl). Either way, stop.
                return

            new_cards = [
                card
                for card in cards
                if card["published_at"] is None or card["published_at"] > floor
            ]
            if not new_cards:
                # All articles on this page are at-or-before ``since`` —
                # because the list is reverse-chronological, the rest of
                # the archive is older still.
                return

            for card in new_cards:
                article_url = card["url"]
                try:
                    article_html = self._http_get(article_url)
                except TransientFetchError as exc:
                    # Per-article transient: log + skip + continue. Do
                    # NOT propagate — see the docstring for why
                    # generator-close semantics make that fatal for
                    # the rest of the pass.
                    _logger.warning(
                        "playvalorant.article_transient_error",
                        extra={"article_url": article_url, "detail": str(exc)},
                    )
                    continue
                yield {
                    "url": article_url,
                    "html": article_html,
                    # Pass through the list-card date as a hint; the body
                    # parse re-extracts ``published_at`` authoritatively
                    # from ``<time datetime=...>``.
                    "list_published_at": (
                        card["published_at"].isoformat()
                        if card["published_at"] is not None
                        else None
                    ),
                }

    # -- validate -----------------------------------------------------------

    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Parse the article HTML and confirm we have all three required fields.

        Raises :class:`SchemaDriftError` when:

        * The patch-version regex doesn't match (title shape changed).
        * No ``<time datetime=...>`` is present (publication metadata
          moved or was removed).
        * The article body container is empty after cleaning.
        """
        url = raw_payload.get("url")
        html = raw_payload.get("html")
        if not isinstance(url, str) or not isinstance(html, str):
            raise SchemaDriftError("playvalorant: payload missing 'url' or 'html'")

        soup = BeautifulSoup(html, "html.parser")

        title = _extract_title(soup)
        version_match = _PATCH_VERSION_RE.search(title) if title else None
        if version_match is None:
            raise SchemaDriftError(
                f"playvalorant: could not extract patch version from title {title!r}"
            )
        patch_version = version_match.group(1)

        published_at = _extract_published_at(soup)
        if published_at is None:
            raise SchemaDriftError("playvalorant: could not find <time datetime=...> on article")

        body_text = _clean_body_text(soup)
        if not body_text:
            raise SchemaDriftError("playvalorant: article body is empty after cleaning")

        return {
            "url": url,
            "raw_html": html,
            "patch_version": patch_version,
            "published_at": published_at,
            "body_text": body_text,
        }

    # -- transform ----------------------------------------------------------

    def transform(self, validated_payload: dict[str, Any]) -> Iterable[PatchNoteRecord]:
        yield PatchNoteRecord(
            patch_version=validated_payload["patch_version"],
            published_at=validated_payload["published_at"],
            raw_html=validated_payload["raw_html"],
            body_text=validated_payload["body_text"],
            url=validated_payload["url"],
        )


# --- HTML parsing helpers --------------------------------------------------


def _parse_article_cards(
    list_html: str,
    *,
    base_url: str,
) -> Iterable[dict[str, Any]]:
    """Yield ``{"url": str, "published_at": datetime | None}`` per article card.

    The exact CSS selector for an article card has churned across Riot
    site refreshes. We use a generous fallback chain: any ``<a>`` whose
    ``href`` looks like an article path (contains ``/news/game-updates/``
    and ends in something other than the list root) is a candidate; we
    look for the card's ``<time datetime=...>`` ancestor sibling for the
    published-at hint.
    """
    soup = BeautifulSoup(list_html, "html.parser")

    seen: set[str] = set()
    for anchor in soup.find_all("a", href=True):
        if not isinstance(anchor, Tag):
            continue
        href = anchor.get("href")
        if not isinstance(href, str):
            continue
        if "/news/game-updates/" not in href:
            continue
        # Skip the list-page link itself.
        if href.rstrip("/").endswith("/game-updates"):
            continue
        # The /game-updates/ feed mixes patch notes with trailers, dev
        # posts, and announcements. Filtering by slug here means
        # ``validate`` only ever sees real patch articles — non-patch
        # entries would otherwise fail version extraction and inflate
        # ``SCHEMA_DRIFT`` counts on every healthy run.
        if not _PATCH_NOTE_SLUG_RE.search(href):
            continue
        absolute = urljoin(base_url, href)
        if absolute in seen:
            continue
        seen.add(absolute)

        # The list-card's published-at is best-effort: if the markup
        # doesn't expose it on the card, the article-body parse will
        # supply it authoritatively. ``None`` here means "we can't
        # pre-filter; fetch the article and decide later".
        time_tag = anchor.find("time")
        if time_tag is None:
            # Walk up to the nearest ancestor that contains a <time>.
            ancestor = anchor.parent
            while ancestor is not None and time_tag is None:
                if isinstance(ancestor, Tag):
                    time_tag = ancestor.find("time")
                ancestor = ancestor.parent if isinstance(ancestor, Tag) else None

        published_at: datetime | None = None
        if isinstance(time_tag, Tag):
            datetime_attr = time_tag.get("datetime")
            if isinstance(datetime_attr, str):
                published_at = _parse_iso8601(datetime_attr)

        yield {"url": absolute, "published_at": published_at}


def _extract_title(soup: BeautifulSoup) -> str | None:
    """Extract the article title, preferring the ``<h1>`` over ``<title>``.

    ``<title>`` typically includes site-suffix noise (e.g. " | VALORANT")
    that ``<h1>`` doesn't, but Riot's templates are inconsistent enough
    that we fall back if the page doesn't render an ``<h1>`` server-side.
    """
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        text = h1.get_text(strip=True)
        if text:
            return text
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        return title_tag.get_text(strip=True)
    return None


def _extract_published_at(soup: BeautifulSoup) -> datetime | None:
    """Find the article's publish timestamp via a layered selector chain.

    Strategies tried in order; first parseable hit wins:

    1. ``<time datetime="...">`` — Riot's current semantic-HTML template.
       Trust the attribute over the human-readable text content.
    2. ``<meta property="article:published_time" content="...">`` —
       OpenGraph metadata. Survives template churn because Riot's
       social-share head block is regenerated independently of the
       article-body markup.
    3. ``<meta name="parsely-pub-date" content="...">`` — Parse.ly
       analytics tag still present on older articles; cheap belt-and-
       braces against a future redesign that drops both of the above.

    Returning ``None`` (rather than raising) keeps this helper composable
    — ``validate`` is the single place that turns "no timestamp" into a
    :class:`SchemaDriftError`. Naive datetimes (no offset) also degrade
    to ``None`` per :func:`_parse_iso8601` so the runner's offset-aware
    ``since`` comparison can't trip a ``TypeError``.

    The OpenGraph + Parse.ly fallbacks are layered fallbacks for layout
    drift, not new acceptance criteria — the primary contract is still
    ``<time datetime>``.
    """
    for time_tag in soup.find_all("time"):
        if not isinstance(time_tag, Tag):
            continue
        attr = time_tag.get("datetime")
        if isinstance(attr, str):
            parsed = _parse_iso8601(attr)
            if parsed is not None:
                return parsed

    # CSS attribute selectors with ``:`` in the value (e.g.
    # ``article:published_time``) must be quoted — soupsieve treats the
    # bare colon as a pseudo-class boundary otherwise. The other selectors
    # are quoted for consistency.
    for selector, attr_name in (
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="parsely-pub-date"]', "content"),
    ):
        meta = soup.select_one(selector)
        if isinstance(meta, Tag):
            value = meta.get(attr_name)
            if isinstance(value, str):
                parsed = _parse_iso8601(value)
                if parsed is not None:
                    return parsed

    return None


def _parse_iso8601(value: str) -> datetime | None:
    """Lenient ISO-8601 parser for the ``datetime`` attribute.

    ``fromisoformat`` accepts the ``YYYY-MM-DDTHH:MM:SSZ`` shape on
    Python 3.12+ as long as we replace the trailing ``Z`` with
    ``+00:00``; we do that explicitly rather than relying on platform
    quirks.
    """
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # ``datetime.fromisoformat`` returns a naive value when the
        # source string had no offset (e.g. ``"2026-03-12T17:00:00"``).
        # We later compare these against a timezone-aware ``since`` in
        # ``fetch`` (``card["published_at"] > since``); mixing naive
        # and aware datetimes raises ``TypeError`` and would abort the
        # whole pass. Treat a missing offset as "we don't know" and
        # return ``None`` so the caller's drift / since-filter paths
        # decide deterministically. Riot's templates always include
        # an offset on the ``<time datetime>`` attribute, so this is
        # genuinely a drift-shaped event rather than a normal value.
        return None
    return parsed


def _clean_body_text(soup: BeautifulSoup) -> str:
    """Strip chrome and return whitespace-normalised article prose.

    Strip list (documented in ``_STRIP_TAGS``):

    * ``<script>`` / ``<style>`` — code that ``get_text`` would otherwise
      inline.
    * ``<nav>`` / ``<footer>`` / ``<aside>`` — semantic chrome.
    * ``<iframe>`` — Riot embeds video and ads via iframes.
    * ``<noscript>`` — fallback ad markup on some pages.

    After tag removal we extract text with paragraph-aware separators,
    then collapse whitespace runs and blank-line runs so the resulting
    body is one paragraph per double newline — easy for BUF-24 patch
    intent extraction to consume.
    """
    # ``decompose`` mutates the tree, so operate on a copy if we ever
    # need the soup elsewhere. Here the caller doesn't, so mutate-in-
    # place is fine.
    for tag_name in _STRIP_TAGS:
        for el in soup.find_all(tag_name):
            if isinstance(el, Tag):
                el.decompose()

    # Prefer the article body if it's marked up; otherwise try the
    # front-end team's stable QA hook (``data-testid="article-body"`` —
    # often outlives a visual-template redesign because Cypress / RTL
    # selectors are coupled to it), then ``<main>``, and finally fall
    # back to the document root. The QA-hook fallback is a layered
    # defence against a redesign that retires ``<article>`` for a
    # styled ``<div>`` — without it we'd silently get the whole page
    # body wrapped in chrome-stripping (which usually still works, but
    # produces noisier ``body_text`` until a parser update lands).
    container: Tag | BeautifulSoup
    article = soup.find("article")
    testid_body = soup.select_one('[data-testid="article-body"]')
    main = soup.find("main")
    if isinstance(article, Tag):
        container = article
    elif isinstance(testid_body, Tag):
        container = testid_body
    elif isinstance(main, Tag):
        container = main
    else:
        container = soup

    raw = container.get_text(separator="\n", strip=True)
    # Collapse runs of horizontal whitespace, then blank lines.
    collapsed = _WS_RUN.sub(" ", raw)
    collapsed = _BLANK_LINES.sub("\n\n", collapsed)
    return collapsed.strip()


__all__ = ["PlayValorantPatchNotesConnector"]
