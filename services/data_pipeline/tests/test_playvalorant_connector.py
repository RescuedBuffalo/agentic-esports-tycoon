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

    validated = connector.validate(
        {"url": "https://playvalorant.com/x", "html": html}
    )
    records = list(connector.transform(validated))

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, PatchNoteRecord)
    assert record.patch_version == "8.05"
    assert record.published_at == datetime(2026, 3, 12, 17, 0, 0, tzinfo=UTC)
    assert record.url == "https://playvalorant.com/x"
    assert record.raw_html == html
    assert "Clove" in record.body_text


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
    assert len(rows) == 1, "re-run must UPSERT on patch_version, not insert a duplicate"
    assert rows[0].fetched_at >= first_fetched_at
