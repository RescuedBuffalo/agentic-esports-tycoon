"""VLR.gg connector — headless-browser scrape of players, teams, tournaments (BUF-10).

VLR.gg has no official public API; the user-facing pages are JS-rendered
React, so a plain HTTP+BeautifulSoup scrape returns a shell with no
stats data. We therefore drive a headless Chromium via Playwright and
extract structured rows from the rendered DOM.

The connector keeps two things deliberately separate:

* **Page fetching** — a ``page_fetcher: Callable[[str], str]`` that takes
  a URL and returns rendered HTML. The default factory (lazy) wires
  Playwright; tests inject a synchronous fake that returns canned HTML.
  This is the dependency-injection seam that lets ``uv run pytest`` pass
  on a fresh clone with no browser binaries installed.

* **Parsing** — a ``parser`` that turns rendered HTML into rows. The
  default :class:`VLRParser` understands the three pages we currently
  scrape (``/stats``, ``/matches?completed``, ``/rankings``); tests can
  swap in a stub when they want to assert on the orchestrator alone.

Acceptance scenarios (BUF-10):

* Daily pull of last 7 days produces >=500 staging records — exercised
  manually against live VLR; the unit tests verify the contract, not the
  volume.
* Resolver integration handles "TenZ" -> "tenz" without splitting the
  canonical entity. The ``platform_id`` we emit is the *VLR-stable id*
  from the page URL (``/player/123/tenz`` -> ``"123"``), not the
  display name, so the resolver's exact-alias lookup catches the second
  pass via :class:`~esports_sim.resolver.ResolutionStatus.MATCHED` even
  if VLR re-cases the handle. (See :func:`_extract_player_id_from_url`.)
* Per-run summary counters land in :class:`~data_pipeline.runner.IngestionStats`
  (the runner already emits them; the connector contributes the
  ``fetched`` side via its ``fetch`` generator).
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from html.parser import HTMLParser
from typing import Any

from esports_sim.db.enums import EntityType, Platform

from data_pipeline.connector import Connector, IngestionRecord, RateLimit
from data_pipeline.errors import SchemaDriftError, TransientFetchError

# Project + contact identifier the BUF-10 spec requires we put on the
# wire. Polite scrapers identify themselves; an opaque UA gets banned
# faster and gives the upstream operator no avenue to flag us.
USER_AGENT: str = (
    "agentic-esports-tycoon-data-pipeline/0.1 (+contact: aidan@buffalo-studios.example)"
)

# Base URL kept as a module constant so tests and the live runner agree
# on the host the parser expects, without a separate config file.
VLR_BASE_URL: str = "https://www.vlr.gg"

# Pages we crawl on each daily pass. Ordering is deterministic so the
# rate-limited fetch sequence is reproducible from logs.
DEFAULT_PAGE_URLS: tuple[tuple[str, str], ...] = (
    ("stats", f"{VLR_BASE_URL}/stats"),
    ("matches", f"{VLR_BASE_URL}/matches?completed"),
    ("rankings", f"{VLR_BASE_URL}/rankings"),
)

# Per-page-type required keys. Splitting the validation table out lets a
# future page (events, teams, etc.) plug in without touching the
# ``validate`` body.
_REQUIRED_PAGE_KEYS: dict[str, tuple[str, ...]] = {
    "stats": ("page_type", "url", "rows"),
    "matches": ("page_type", "url", "rows"),
    "rankings": ("page_type", "url", "rows"),
}

logger = logging.getLogger(__name__)


# --- public types -----------------------------------------------------------


PageFetcher = Callable[[str], str]
"""Sync callable: URL -> rendered HTML. Injected so tests skip Playwright."""


@dataclass(frozen=True)
class VLRPageRow:
    """One parsed row from a VLR page.

    The connector keeps the raw shape on the payload (so reprocessing
    against ``raw_record`` is possible later) and pulls the resolver-
    keying fields up to dedicated attributes. ``vlr_id`` is the stable
    numeric/slug id parsed from the page URL — *never* the display name.

    ``timestamp`` is set when the page exposes one (``/matches`` shows a
    completed-at; ``/stats`` shows nothing). ``None`` means "current
    state, no since-filter applies", used by ``/rankings``.
    """

    entity_type: EntityType
    vlr_id: str
    display_name: str
    profile_url: str
    timestamp: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe shape that lands on ``raw_record.payload`` rows."""
        return {
            "entity_type": self.entity_type.value,
            "vlr_id": self.vlr_id,
            "display_name": self.display_name,
            "profile_url": self.profile_url,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            **self.extra,
        }


# --- robots.txt -------------------------------------------------------------


class _RobotsCache:
    """Tiny one-shot ``robots.txt`` checker.

    We don't use :mod:`urllib.robotparser` directly because its file
    fetch happens at construction; the BUF-10 spec wants robots fetched
    *once* on first ``fetch`` so a constructor that runs at import time
    in a test (where the network isn't reachable) doesn't reach out.
    The cache lazily loads on the first ``allows`` call and falls back
    to "allow" on a fetch failure — the polite UA + rate-limit are the
    primary citizenship guards; an unreachable robots.txt is no reason
    to drop the whole crawl.
    """

    def __init__(
        self,
        base_url: str,
        *,
        user_agent: str,
        fetcher: Callable[[str], str] | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._user_agent = user_agent
        self._fetcher = fetcher or _http_get
        self._loaded = False
        self._disallows: list[str] = []

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True  # set first; any failure path below is a no-op grant
        try:
            body = self._fetcher(f"{self._base_url}/robots.txt")
        except Exception:
            logger.warning("robots.txt fetch failed; defaulting to allow", exc_info=True)
            return
        self._disallows = list(_parse_disallows(body, user_agent=self._user_agent))

    def allows(self, url: str) -> bool:
        """True iff ``url`` is not under any disallowed prefix."""
        self._ensure_loaded()
        path = urllib.parse.urlparse(url).path or "/"
        return not any(path.startswith(prefix) for prefix in self._disallows)


def _parse_disallows(body: str, *, user_agent: str) -> Iterable[str]:
    """Yield ``Disallow:`` paths that apply to ``user_agent``.

    Implements just enough of RFC 9309 / Google's spec to honour both
    a project-specific ``User-agent: agentic-esports-tycoon-data-pipeline``
    block (if VLR ever adds one) and the generic ``User-agent: *`` block.
    Comments, blank lines, and unsupported directives (Crawl-delay,
    Sitemap, etc.) are dropped — :mod:`urllib.robotparser` overcomplicates
    this for what's effectively a 50-line file.
    """
    ua_lower = user_agent.lower()
    in_block_for_us = False
    seen_specific = False
    pending_disallows: list[str] = []
    star_disallows: list[str] = []

    for raw in body.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            in_block_for_us = False
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()

        if key == "user-agent":
            agent = value.lower()
            # Match either an exact prefix (e.g. our project token) or
            # the wildcard ``*``. We prefer a specific block over ``*``.
            if agent != "*" and agent in ua_lower:
                in_block_for_us = True
                seen_specific = True
                pending_disallows = []
            elif agent == "*" and not seen_specific:
                in_block_for_us = True
            else:
                in_block_for_us = False
        elif key == "disallow" and in_block_for_us:
            if value:
                if seen_specific:
                    pending_disallows.append(value)
                else:
                    star_disallows.append(value)

    yield from (pending_disallows or star_disallows)


def _http_get(url: str) -> str:
    """Plain HTTPS fetcher used only for robots.txt.

    We deliberately don't pull in ``requests`` for this one call — the
    stdlib does fine and keeps the dependency surface honest. The fetch
    is read-only, single-shot, and outside the rate-limited fetch loop.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        decoded: str = response.read().decode("utf-8", errors="replace")
        return decoded


# --- parser -----------------------------------------------------------------


_PLAYER_URL_RE = re.compile(r"/player/(\d+)/([^/?#]+)")
_TEAM_URL_RE = re.compile(r"/team/(\d+)/([^/?#]+)")
_EVENT_URL_RE = re.compile(r"/event/(\d+)/([^/?#]+)")


def _extract_player_id_from_url(href: str) -> tuple[str, str] | None:
    """Pull (vlr_id, slug) out of ``/player/<id>/<slug>``.

    Returns the *id* — not the slug — as the stable platform id. The
    BUF-10 acceptance scenario "TenZ -> tenz" rests on this: VLR may
    re-case the slug between crawls, but the numeric id is immutable,
    so the resolver's exact-alias lookup keys hit on the second pass.
    """
    match = _PLAYER_URL_RE.search(href)
    if not match:
        return None
    return match.group(1), match.group(2)


def _extract_team_id_from_url(href: str) -> tuple[str, str] | None:
    match = _TEAM_URL_RE.search(href)
    if not match:
        return None
    return match.group(1), match.group(2)


def _extract_event_id_from_url(href: str) -> tuple[str, str] | None:
    match = _EVENT_URL_RE.search(href)
    if not match:
        return None
    return match.group(1), match.group(2)


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    """Parse VLR's ISO-ish timestamp strings, naive-aware-safe.

    VLR variously emits ``2026-04-22T18:30:00Z``, ``2026-04-22 18:30``,
    or pure ISO 8601. We accept the first two by normalising into the
    pattern :func:`datetime.fromisoformat` understands; anything else
    becomes ``None`` and the row falls through as "no since filter".
    """
    if not value:
        return None
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        # VLR's "naive" timestamps are UTC in practice; without this
        # assumption a ``since`` comparison against an aware datetime
        # would raise TypeError at runtime.
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


class _AnchorCollector(HTMLParser):
    """Walk an HTML document and yield ``(href, link_text)`` pairs.

    We use the stdlib parser rather than BeautifulSoup to keep
    ``data_pipeline`` free of an extra runtime dependency — the parsing
    we need is anchor-shape recognition, not full DOM traversal.

    For each anchor we capture both its own ``data-*`` attributes *and*
    the closest enclosing ``data-utc-ts`` / ``data-match-id`` from any
    ancestor open at the time the anchor is encountered. VLR's match
    cards put the completion timestamp on the wrapping ``<div
    class="match-item">`` rather than the team anchor, so reading the
    anchor in isolation would always see a ``None`` timestamp and let
    every match bypass the connector's ``since`` filter on every run.
    Inheriting from ancestors fixes that without forcing a full DOM
    traversal — we only track the two attributes the parser actually
    consumes.
    """

    # The two ancestor data-attrs we propagate down into anchors. Add
    # more here only when a parser actually needs to read them; every
    # extra key inflates per-tag bookkeeping for every page.
    _INHERITED_DATA_ATTRS = ("data-utc-ts", "data-match-id")

    # HTML void elements (no closing tag). We never push frames for
    # these — they'd accumulate forever because there's no
    # ``handle_endtag`` call to pop them. Source: WHATWG spec.
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
        self.anchors: list[dict[str, Any]] = []
        self._current: dict[str, Any] | None = None
        # One frame per currently-open non-anchor, non-void element.
        # Pushed unconditionally on ``handle_starttag`` (even when the
        # element carries no inherited attrs) so the stack mirrors the
        # actual open-tag depth — essential to correctly handle nested
        # same-tag elements like ``<div data-utc-ts="X"><div>...</div>
        # <a>row</a></div>``. Without per-open-tag bookkeeping, the
        # inner ``</div>`` would pop the outer frame and the trailing
        # anchor would lose its inherited timestamp. Each frame stores
        # the inherited attrs it contributes (empty dict when the
        # element carried no tracked attrs).
        self._inherited_stack: list[tuple[str, dict[str, str]]] = []

    def _current_inherited(self) -> dict[str, str]:
        """Flatten the active inheritance stack into one attr map.

        Later (more deeply-nested) frames win — that mirrors normal CSS
        / DOM "innermost ancestor" semantics. Empty when no enclosing
        element carries a tracked attribute.
        """
        merged: dict[str, str] = {}
        for _tag, frame in self._inherited_stack:
            merged.update(frame)
        return merged

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: v for k, v in attrs if v is not None}
        if tag != "a":
            # Void elements have no end tag — pushing a frame for them
            # would leak onto the stack indefinitely. Skip them entirely
            # since they can't carry our two tracked attrs in any
            # meaningful structural position anyway.
            if tag in self._VOID_ELEMENTS:
                return
            inherited = {
                key: attr_map[key] for key in self._INHERITED_DATA_ATTRS if key in attr_map
            }
            # Push unconditionally (even with an empty ``inherited`` dict).
            # The stack frame represents "this element is currently open"
            # so its closing tag pops the right frame. Without this, an
            # inner ``<div>`` with no tracked attrs and an outer
            # ``<div data-utc-ts="X">`` would share one stack frame and
            # the inner close would discard the outer's inherited attrs.
            self._inherited_stack.append((tag, inherited))
            return
        href = attr_map.get("href")
        if not href:
            return
        # Inherit ancestor data-attrs *only* when the anchor itself
        # doesn't already carry the same key — explicit beats inherited
        # so a row that does decorate the anchor keeps its own value.
        for key, value in self._current_inherited().items():
            attr_map.setdefault(key, value)
        self._current = {"href": href, "attrs": attr_map, "text_parts": []}

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        # XHTML self-closing form ``<tag/>``. Don't push — there's no
        # corresponding end tag, and treating the element as if it had
        # children would shift the whole inheritance stack.
        return

    def handle_endtag(self, tag: str) -> None:
        if tag == "a":
            if self._current is None:
                return
            text = "".join(self._current["text_parts"]).strip()
            anchor = {
                "href": self._current["href"],
                "text": text,
                "attrs": self._current["attrs"],
            }
            self.anchors.append(anchor)
            self._current = None
            return
        if tag in self._VOID_ELEMENTS:
            # Defensive: void elements shouldn't reach ``handle_endtag``
            # at all (they have no closing form), but if a malformed
            # document includes one, ignore it rather than corrupting
            # the stack.
            return
        # Well-formed HTML: the topmost frame matches this closing tag.
        # Pop it.
        if self._inherited_stack and self._inherited_stack[-1][0] == tag:
            self._inherited_stack.pop()
            return
        # Misnested HTML (e.g. ``<div><span></div></span>``): walk down
        # to find the most recent matching open tag and pop it together
        # with everything above it (treating the intervening elements
        # as implicitly closed). This mirrors browsers' lenient parsing
        # without leaving phantom frames on the stack.
        for index in range(len(self._inherited_stack) - 1, -1, -1):
            if self._inherited_stack[index][0] == tag:
                del self._inherited_stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if self._current is not None:
            self._current["text_parts"].append(data)


@dataclass
class VLRParser:
    """Default rendered-HTML -> :class:`VLRPageRow` parser.

    Each ``parse_*`` method consumes one page's HTML and yields zero or
    more rows. Returning iterables (not lists) keeps memory tame on
    high-volume pages and lets ``fetch`` apply its since-filter lazily.

    The parser is split out from :class:`VLRConnector` so tests and
    future schema-drift fixes can swap a stub in without touching the
    orchestration logic.
    """

    base_url: str = VLR_BASE_URL

    def parse(self, page_type: str, html: str) -> Iterable[VLRPageRow]:
        """Dispatch on ``page_type`` to the right parser method."""
        if page_type == "stats":
            return self.parse_stats(html)
        if page_type == "matches":
            return self.parse_matches(html)
        if page_type == "rankings":
            return self.parse_rankings(html)
        # Unknown page types are reported as drift rather than silently
        # producing zero rows — a config typo upstream would otherwise
        # take ages to notice.
        raise SchemaDriftError(f"unknown VLR page_type: {page_type!r}")

    def parse_stats(self, html: str) -> Iterable[VLRPageRow]:
        """``/stats`` is a leaderboard of player profiles.

        We yield one PLAYER row per ``<a href="/player/<id>/<slug>">``
        anchor. ``/stats`` doesn't expose a per-row timestamp, so the
        ``since`` filter passes everything through (it's "current
        state", which the BUF-10 spec explicitly allows).
        """
        for anchor in _iter_anchors(html):
            ids = _extract_player_id_from_url(anchor["href"])
            if not ids:
                continue
            vlr_id, slug = ids
            display_name = anchor["text"] or slug
            yield VLRPageRow(
                entity_type=EntityType.PLAYER,
                vlr_id=vlr_id,
                display_name=display_name,
                profile_url=urllib.parse.urljoin(self.base_url, anchor["href"]),
                extra={"slug": slug},
            )

    def parse_matches(self, html: str) -> Iterable[VLRPageRow]:
        """``/matches?completed`` is a list of finished matches.

        We yield one TEAM row per ``<a href="/team/<id>/<slug>">``
        anchor and stash the match metadata (id, completed_at, the
        opposing team) on ``extra`` because :class:`EntityType.MATCH`
        doesn't exist in the schema yet (out of scope per BUF-10 ticket
        notes).

        Per-row timestamps come from a sibling ``data-utc-ts`` (or the
        anchor's own ``data-utc-ts``) so the connector's ``since``
        filter can drop already-seen completed matches without us
        having to remember the last-processed match id.
        """
        for anchor in _iter_anchors(html):
            ids = _extract_team_id_from_url(anchor["href"])
            if not ids:
                continue
            vlr_id, slug = ids
            display_name = anchor["text"] or slug
            ts = _parse_iso_timestamp(anchor["attrs"].get("data-utc-ts"))
            extra: dict[str, Any] = {"slug": slug}
            match_id = anchor["attrs"].get("data-match-id")
            if match_id:
                extra["match_id"] = match_id
            yield VLRPageRow(
                entity_type=EntityType.TEAM,
                vlr_id=vlr_id,
                display_name=display_name,
                profile_url=urllib.parse.urljoin(self.base_url, anchor["href"]),
                timestamp=ts,
                extra=extra,
            )

    def parse_rankings(self, html: str) -> Iterable[VLRPageRow]:
        """``/rankings`` is a regional standings page.

        Yields one TEAM row per anchor (we don't try to differentiate
        regional pools here; the resolver handles the per-region
        canonical entity) plus one TOURNAMENT row per ``/event/`` link
        — the page header on every ranking lists the active event.
        """
        for anchor in _iter_anchors(html):
            team_ids = _extract_team_id_from_url(anchor["href"])
            if team_ids:
                vlr_id, slug = team_ids
                display_name = anchor["text"] or slug
                yield VLRPageRow(
                    entity_type=EntityType.TEAM,
                    vlr_id=vlr_id,
                    display_name=display_name,
                    profile_url=urllib.parse.urljoin(self.base_url, anchor["href"]),
                    extra={"slug": slug},
                )
                continue
            event_ids = _extract_event_id_from_url(anchor["href"])
            if event_ids:
                vlr_id, slug = event_ids
                display_name = anchor["text"] or slug
                yield VLRPageRow(
                    entity_type=EntityType.TOURNAMENT,
                    vlr_id=vlr_id,
                    display_name=display_name,
                    profile_url=urllib.parse.urljoin(self.base_url, anchor["href"]),
                    extra={"slug": slug},
                )


def _iter_anchors(html: str) -> Iterable[dict[str, Any]]:
    parser = _AnchorCollector()
    parser.feed(html)
    parser.close()
    return parser.anchors


# --- connector --------------------------------------------------------------


class VLRConnector(Connector):
    """:class:`Connector` for VLR.gg via headless browser.

    Construction is HTTP-free: nothing reaches out until :meth:`fetch`.
    That matters for tests (they construct without a network) and for
    any future scheduler that imports the connector list at startup.
    """

    def __init__(
        self,
        *,
        page_fetcher: PageFetcher | None = None,
        parser: VLRParser | None = None,
        page_urls: Sequence[tuple[str, str]] = DEFAULT_PAGE_URLS,
        robots_cache: _RobotsCache | None = None,
        base_url: str = VLR_BASE_URL,
        user_agent: str = USER_AGENT,
    ) -> None:
        # ``page_fetcher`` is None until fetch() runs so importing the
        # module from ``ruff`` / ``mypy`` / a fresh test process does
        # not trigger a Playwright import. The factory is invoked
        # lazily in ``_get_fetcher``.
        self._page_fetcher = page_fetcher
        self._parser = parser or VLRParser(base_url=base_url)
        self._page_urls = tuple(page_urls)
        self._base_url = base_url
        self._user_agent = user_agent
        self._robots = robots_cache or _RobotsCache(base_url, user_agent=user_agent)

    # --- Connector ABC properties --------------------------------------

    @property
    def source_name(self) -> str:
        return "vlr"

    @property
    def platform(self) -> Platform:
        return Platform.VLR

    @property
    def entity_types(self) -> tuple[EntityType, ...]:
        # No EntityType.MATCH yet (see module docstring + ticket out-of-
        # scope). Match metadata rides on TEAM rows' ``extra`` instead.
        return (EntityType.PLAYER, EntityType.TEAM, EntityType.TOURNAMENT)

    @property
    def cadence(self) -> timedelta:
        return timedelta(days=1)

    @property
    def rate_limit(self) -> RateLimit:
        # 20 req/min steady state, no burst tolerance. Keeping
        # ``capacity=1`` means each crawl pays the full 3-second gap
        # between page fetches, which is what "polite, no official API"
        # bakes into the spec.
        return RateLimit(capacity=1, refill_per_second=20.0 / 60.0)

    # --- Connector ABC methods -----------------------------------------

    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        """Yield one page payload per crawl URL allowed by robots.txt.

        Implementation note: the runner gates each ``next()`` through
        the rate limiter, so each yield costs one token. We therefore
        keep one HTTP roundtrip per yield — splitting a page across
        multiple yields would over-consume the bucket.

        Per-page fetch failures are caught here and **logged + skipped**
        rather than propagated. ``run_ingestion``'s post-yield error
        handling can't see exceptions raised during iterator advancement
        (the fetcher call runs *before* the next ``yield``), so a
        timeout on one VLR page would otherwise abort the whole daily
        crawl and skip every later page. We log a structured warning
        so the failure stays observable, then continue with the next
        URL. Subsequent runs retry naturally because no ``raw_record``
        is written for a failed fetch.
        """
        fetcher = self._get_fetcher()
        for page_type, url in self._page_urls:
            if not self._robots.allows(url):
                logger.info("vlr.skip_disallowed", extra={"url": url})
                continue
            try:
                rendered = fetcher(url)
            except TransientFetchError as exc:
                # Recoverable upstream miss — log and move on. The next
                # scheduled run will retry; nothing was persisted.
                logger.warning(
                    "vlr.transient_fetch_error",
                    extra={
                        "url": url,
                        "page_type": page_type,
                        "detail": str(exc),
                    },
                )
                continue
            except Exception as exc:
                # Unknown fetch failure (parser regression, browser
                # crash, etc.). We deliberately keep the run alive —
                # log enough context for triage and continue. A
                # systematic failure will show up across many URLs in
                # the same run, which is the signal a maintainer
                # actually needs.
                logger.warning(
                    "vlr.connector_error",
                    extra={
                        "url": url,
                        "page_type": page_type,
                        "error_type": type(exc).__name__,
                        "detail": str(exc),
                    },
                )
                continue
            rows = list(self._parser.parse(page_type, rendered))
            # Apply the since-filter here, before emitting the payload.
            # We deliberately do NOT stash ``fetched_at`` / ``since`` on
            # the yielded dict: the runner hashes payload bytes for
            # dedup, and any per-run metadata would make the same
            # upstream page hash differently every pass — defeating
            # ``RawRecord.content_hash`` replay detection. Filtering
            # rows here keeps the emitted payload a pure content
            # snapshot. ``RawRecord.fetched_at`` already records the
            # crawl time at the row level via its server default.
            kept_rows = [
                row.to_payload() for row in rows if row.timestamp is None or row.timestamp > since
            ]
            yield {
                "page_type": page_type,
                "url": url,
                "rows": kept_rows,
            }

    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Shape-check the page payload.

        Schema drift symptoms we look for:

        * top-level keys missing (the ``fetch`` contract);
        * unknown ``page_type`` (someone added a URL but not a parser);
        * a ``rows`` element that isn't a dict, or is missing the
          required per-row keys.

        We deliberately do *not* validate the *content* of each row
        (timestamps in the future, etc.) — that's the resolver's
        problem, and we'd rather log one drift event for a structural
        change than 500 spurious ones for a value-level oddity.
        """
        if not isinstance(raw_payload, dict):
            raise SchemaDriftError(f"vlr payload must be dict, got {type(raw_payload).__name__}")

        page_type = raw_payload.get("page_type")
        if page_type not in _REQUIRED_PAGE_KEYS:
            raise SchemaDriftError(f"vlr payload has unknown page_type: {page_type!r}")

        for key in _REQUIRED_PAGE_KEYS[page_type]:
            if key not in raw_payload:
                raise SchemaDriftError(f"vlr {page_type} payload missing required key: {key!r}")

        rows = raw_payload["rows"]
        if not isinstance(rows, list):
            raise SchemaDriftError(
                f"vlr {page_type} payload 'rows' must be list, got {type(rows).__name__}"
            )
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise SchemaDriftError(
                    f"vlr {page_type} row[{index}] must be dict, got {type(row).__name__}"
                )
            for required in ("entity_type", "vlr_id", "display_name"):
                if required not in row:
                    raise SchemaDriftError(
                        f"vlr {page_type} row[{index}] missing required key: {required!r}"
                    )
        return raw_payload

    def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        """Project rows into resolver inputs.

        ``platform_id`` is the VLR-stable id (parsed from the page URL
        and copied onto ``vlr_id``), *not* the display name. The
        resolver's exact-alias lookup keys on it for idempotence; using
        the display name would split the canonical row the moment VLR
        re-cased "TenZ" -> "tenz".

        The since-filter is applied during ``fetch`` (before the
        payload is hashed for dedup), so this method just projects
        whatever rows the validated payload contains.
        """
        for row in validated_payload["rows"]:
            yield IngestionRecord(
                entity_type=EntityType(row["entity_type"]),
                platform_id=row["vlr_id"],
                platform_name=row["display_name"],
                payload=row,
            )

    # --- internals -----------------------------------------------------

    def _get_fetcher(self) -> PageFetcher:
        """Lazy-init the Playwright-backed page fetcher.

        Importing ``playwright`` at module-import time would force every
        test process — even the unit tests that inject a fake fetcher —
        to install browser binaries. The lazy factory means the import
        only happens on a real crawl.
        """
        if self._page_fetcher is None:
            self._page_fetcher = _build_playwright_fetcher(user_agent=self._user_agent)
        return self._page_fetcher


def _build_playwright_fetcher(*, user_agent: str) -> PageFetcher:
    """Default factory wiring a sync-Playwright Chromium.

    Kept inline (not in tests) because production code is allowed to
    depend on Playwright. Tests pass an explicit ``page_fetcher`` and
    never hit this code path. The factory itself is small enough to
    inline rather than splitting into a separate ``_playwright.py``
    module.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise RuntimeError(
            "playwright is required for the live VLR connector; install with"
            " `uv sync` and `python -m playwright install chromium`."
        ) from exc

    def fetch(url: str) -> str:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(user_agent=user_agent)
                page = context.new_page()
                # ``networkidle`` waits until the page's React hydration
                # has finished firing XHRs. Without it the rendered HTML
                # is a skeleton and the parser yields zero rows.
                page.goto(url, wait_until="networkidle", timeout=30_000)
                return page.content()
            finally:
                browser.close()

    return fetch


__all__ = [
    "DEFAULT_PAGE_URLS",
    "PageFetcher",
    "USER_AGENT",
    "VLRConnector",
    "VLRPageRow",
    "VLRParser",
    "VLR_BASE_URL",
]
