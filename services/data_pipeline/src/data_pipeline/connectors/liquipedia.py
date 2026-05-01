"""Liquipedia REST connector (BUF-11).

Liquipedia is the second-highest-priority canonical source after Riot
(BUF-12). Weekly pull of rosters, transfers, tournaments, and coaches —
treated as truth for conflict resolution.

Endpoints (Valorant tree)::

    /valorant/player/{slug}
    /valorant/team/{slug}
    /valorant/coach/{slug}
    /valorant/tournaments
    /valorant/transfers?date_gte={since}

Rate limit: **30 req/min**, configured via the BUF-9 ``RateLimit``
contract as ``capacity=1, refill_per_second=0.5`` so the runner's token
bucket gates each upstream call cleanly.

Transfers and rebrands are the two pieces of behaviour that diverge from
a "fetch + flatten" connector:

* Transfer events are projected into TWO :class:`IngestionRecord`s — one
  PLAYER (the moving player) and one TEAM (the destination team) — each
  with the transfer event embedded under ``payload.roster_change``. The
  Systems-spec entity model has no ``ROSTER_CHANGE`` type yet (Phase 1's
  Relationship Engine is BUF-?? and out of scope here); doubling the
  record this way keeps the transfer audit trail visible on both sides
  of the move and is forward-compatible with a dedicated
  ``RosterChange`` table.
* Team rebrands carry a ``previous_slug`` field on Liquipedia. Rather
  than minting a new canonical (which the fuzzy resolver would do for a
  dissimilar rename like "Sentinels" -> "Team Sentinels Esports"), the
  connector emits the IngestionRecord under the **previous** slug as
  ``platform_id`` so the resolver's exact-alias path matches it. The new
  display name lands as ``platform_name`` on a fresh alias row with a
  later ``valid_from`` — the alias-extension behaviour the Systems-spec
  System 03 requires. See :func:`_extend_alias_for_rebrand` for the
  small helper that bookkeeps a rebrand event into the connector's own
  log so a maintainer can spot-check rename history later.

Out of scope:

* Real ``httpx`` retries with backoff. The default factory wires plain
  ``httpx.get``; transient failures bubble out as
  :class:`TransientFetchError` and the runner skips them. A proper
  retry policy is a follow-up when we have observed flakiness in
  production.
* ``RosterChange`` table (Phase 1 Relationship Engine).
* Tournament-result schema beyond name / start / end dates.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any

from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import EntityAlias
from sqlalchemy import select
from sqlalchemy.orm import Session

from data_pipeline.connector import Connector, IngestionRecord, RateLimit
from data_pipeline.errors import SchemaDriftError, TransientFetchError

# Structured-logging surface for fetch-time failures the runner can't
# see (see ``_safe_fetch``). The runner's ``IngestionStats`` only
# counts errors raised post-yield; logging here is the per-record
# visibility a maintainer needs when one slug fails out of a hundred.
_logger = logging.getLogger("data_pipeline.liquipedia")

# Type alias for the injectable HTTP client. Tests pass a callable that
# returns a canned dict; production wires the default factory below
# which dispatches against ``httpx.get``. Keeping the surface this thin
# means we don't drag ``httpx`` into the connector's type signature and
# every test stays mock-free.
HttpGet = Callable[[str], Any]

# Liquipedia exposes Valorant data under a fixed prefix; centralising it
# here means the rest of the file just talks paths.
DEFAULT_BASE_URL = "https://liquipedia.net/valorant/api/v1"

# 30 requests per minute = 0.5 tokens/sec; capacity 1 disables bursting
# so a misbehaving caller can't run a 30-call burst and then idle. The
# runner's ``_rate_limited`` brackets each fetch ``next()`` with one
# acquire, so this directly matches the documented quota.
LIQUIPEDIA_RATE_LIMIT = RateLimit(capacity=1, refill_per_second=0.5)

# Weekly cadence per the ticket — the scheduler reads this hint from
# ``connector.cadence`` to decide when to re-invoke ``run_ingestion``.
LIQUIPEDIA_CADENCE = timedelta(days=7)

# Minimum keys per record_type. ``validate`` raises SchemaDriftError when
# any of these are missing. Kept module-level so tests can import them.
_REQUIRED_PLAYER_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_TEAM_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_COACH_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_TOURNAMENT_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_TRANSFER_KEYS: frozenset[str] = frozenset(
    {"id", "player_slug", "player_name", "to_team_slug", "to_team_name", "date"}
)


def _default_http_get_factory() -> HttpGet:
    """Build the production HTTP getter (lazy ``httpx`` import).

    Importing ``httpx`` only here lets the unit tests run without the
    dependency installed and keeps the connector module's import-time
    surface minimal — the BUF-9 framework imports the connector class
    eagerly from the registry, but tests pass their own ``http_get``
    so the network library is never touched.
    """
    import httpx

    def _get(url: str) -> Any:
        try:
            response = httpx.get(url, timeout=10.0)
        except httpx.HTTPError as exc:
            # Network / parse / timeout: recoverable, the runner will
            # log + skip and the next scheduled pass retries.
            raise TransientFetchError(f"liquipedia fetch failed: {exc}") from exc
        if response.status_code >= 500:
            raise TransientFetchError(f"liquipedia 5xx: {response.status_code} for {url}")
        if response.status_code >= 400:
            # 4xx is a permanent miss — let it surface as a connector
            # error rather than masquerading as transient.
            raise RuntimeError(f"liquipedia {response.status_code} for {url}: {response.text!r}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise TransientFetchError(f"liquipedia JSON parse failed for {url}: {exc}") from exc
        return payload

    return _get


class LiquipediaConnector(Connector):
    """Weekly pull of Liquipedia data for the canonical-resolver pipeline.

    Construct with explicit ``player_slugs`` / ``team_slugs`` / ``coach_slugs``
    seed lists; the connector walks each per-slug endpoint plus the
    tournaments and transfers list endpoints. Tests pass tiny lists +
    a canned ``http_get`` so the full ``fetch -> validate -> transform``
    flow runs offline.

    The connector does **not** discover slugs autonomously — Liquipedia
    has no "list all players" endpoint, and a Phase 1 task is "ingest
    the slug list from the Riot region rosters and feed it here". For
    BUF-11's acceptance test (10 known transfers) seed lists are enough.
    """

    def __init__(
        self,
        *,
        player_slugs: Iterable[str] = (),
        team_slugs: Iterable[str] = (),
        coach_slugs: Iterable[str] = (),
        include_tournaments: bool = True,
        include_transfers: bool = True,
        http_get: HttpGet | None = None,
        base_url: str = DEFAULT_BASE_URL,
    ) -> None:
        self._player_slugs = tuple(player_slugs)
        self._team_slugs = tuple(team_slugs)
        self._coach_slugs = tuple(coach_slugs)
        self._include_tournaments = include_tournaments
        self._include_transfers = include_transfers
        # Lazily build the default getter — tests that pass their own
        # ``http_get`` never trigger the ``httpx`` import.
        self._http_get: HttpGet = http_get if http_get is not None else _default_http_get_factory()
        self._base_url = base_url.rstrip("/")

    # --- BUF-9 metadata properties ----------------------------------------

    @property
    def source_name(self) -> str:
        return "liquipedia"

    @property
    def platform(self) -> Platform:
        return Platform.LIQUIPEDIA

    @property
    def entity_types(self) -> tuple[EntityType, ...]:
        return (EntityType.PLAYER, EntityType.TEAM, EntityType.COACH, EntityType.TOURNAMENT)

    @property
    def cadence(self) -> timedelta:
        return LIQUIPEDIA_CADENCE

    @property
    def rate_limit(self) -> RateLimit:
        return LIQUIPEDIA_RATE_LIMIT

    # --- BUF-9 lifecycle hooks -------------------------------------------

    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        """Yield raw upstream payloads, tagged with ``record_type``.

        Generator-shaped so the runner can rate-limit per-yield. Each
        payload carries a ``record_type`` discriminator and an inline
        ``data`` blob; ``transform`` switches on the discriminator to
        decide which :class:`IngestionRecord`s to emit.

        Per-slug HTTP failures (``TransientFetchError`` from a 5xx, an
        ``httpx.HTTPError`` from a timeout, a list-endpoint envelope
        drift) are caught here and **logged + skipped** rather than
        propagated. ``run_ingestion`` only catches errors raised inside
        ``validate``/``transform`` — anything thrown during iterator
        advancement (i.e. during the next ``self._http_get`` call before
        ``yield``) would abort the whole pass and skip every remaining
        slug. Catching here keeps one rate-limited / timeout / drifted
        endpoint from taking down the rest of the weekly run.
        """
        for slug in self._player_slugs:
            data = self._safe_fetch(
                f"{self._base_url}/player/{slug}",
                record_type="player",
                identifier=slug,
            )
            if data is not None:
                yield {"record_type": "player", "slug": slug, "data": data}
        for slug in self._team_slugs:
            data = self._safe_fetch(
                f"{self._base_url}/team/{slug}",
                record_type="team",
                identifier=slug,
            )
            if data is not None:
                yield {"record_type": "team", "slug": slug, "data": data}
        for slug in self._coach_slugs:
            data = self._safe_fetch(
                f"{self._base_url}/coach/{slug}",
                record_type="coach",
                identifier=slug,
            )
            if data is not None:
                yield {"record_type": "coach", "slug": slug, "data": data}
        if self._include_tournaments:
            tournaments = self._safe_fetch(
                f"{self._base_url}/tournaments",
                record_type="tournaments_envelope",
                identifier="all",
            )
            if tournaments is not None:
                yield from self._yield_envelope_items(
                    tournaments, record_type="tournament", id_key="slug"
                )
        if self._include_transfers:
            since_iso = since.date().isoformat()
            transfers = self._safe_fetch(
                f"{self._base_url}/transfers?date_gte={since_iso}",
                record_type="transfers_envelope",
                identifier=since_iso,
            )
            if transfers is not None:
                yield from self._yield_envelope_items(
                    transfers, record_type="transfer", id_key="id"
                )

    def _safe_fetch(self, url: str, *, record_type: str, identifier: str) -> Any | None:
        """Run one ``http_get`` and absorb genuinely transient failures.

        Returns ``None`` when the call fails transiently; the caller
        treats ``None`` as "skip this record". Failures are logged with
        enough context (record_type + identifier + url) for a
        postmortem.

        Only ``TransientFetchError`` is caught here. Anything else —
        ``RuntimeError`` from a 4xx in :func:`_default_http_get_factory`
        (bad base URL, expired key, deleted slug returning 404), an
        ``ImportError`` from a broken ``httpx`` install, an
        ``AttributeError`` from a code regression — is treated as a
        permanent / programming failure that must surface loudly. Round 1
        caught ``Exception`` here, which silently turned 401/403/404s
        into per-record skips and let the run finish with no ingested
        data — exactly the kind of production outage we want to fail
        fast on. Permanent errors propagate out of the generator and the
        runner's ``CONNECTOR_ERROR`` path takes over.
        """
        try:
            return self._http_get(url)
        except TransientFetchError as exc:
            _logger.warning(
                "liquipedia.transient_fetch_error url=%s record_type=%s id=%s detail=%s",
                url,
                record_type,
                identifier,
                exc,
            )
            return None

    def _yield_envelope_items(
        self,
        envelope: Any,
        *,
        record_type: str,
        id_key: str,
    ) -> Iterable[dict[str, Any]]:
        """Iterate a list endpoint, surfacing envelope drift cleanly.

        ``_iter_list`` raises :class:`SchemaDriftError` for envelope
        shapes other than ``list`` or ``{"items": [...]}``; we catch and
        log here so the unknown shape is observable but the rest of the
        pass keeps running.
        """
        try:
            items = list(_iter_list(envelope))
        except SchemaDriftError as exc:
            _logger.warning(
                "liquipedia.envelope_drift record_type=%s detail=%s",
                record_type,
                exc,
            )
            return
        for item in items:
            yield {
                "record_type": record_type,
                id_key: item.get(id_key, ""),
                "data": item,
            }

    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Shape check by ``record_type``. Raise :class:`SchemaDriftError` on miss.

        We deliberately don't normalise the payload here — the BUF-9
        contract is "validate is a gate, transform is the projector".
        Returning the input unchanged keeps that split clean.
        """
        record_type = raw_payload.get("record_type")
        data = raw_payload.get("data")
        if not isinstance(data, dict):
            raise SchemaDriftError(
                f"liquipedia payload missing dict 'data' (record_type={record_type!r})"
            )

        if not isinstance(record_type, str):
            raise SchemaDriftError(f"liquipedia: unknown record_type {record_type!r}")
        required = _REQUIRED_KEYS_BY_TYPE.get(record_type)
        if required is None:
            raise SchemaDriftError(f"liquipedia: unknown record_type {record_type!r}")
        missing = required - data.keys()
        if missing:
            raise SchemaDriftError(
                f"liquipedia {record_type} payload missing required keys: " f"{sorted(missing)!r}"
            )
        return raw_payload

    def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        """Project a validated upstream payload into resolver inputs.

        Dispatch on ``record_type``. Player/team/coach/tournament each
        yield exactly one record; ``transfer`` yields TWO (player +
        destination team) so the roster_change blob is durable on both
        sides of the move until BUF-?? introduces a proper
        ``RosterChange`` table.
        """
        record_type = validated_payload["record_type"]
        data = validated_payload["data"]

        if record_type == "player":
            yield self._player_record(data)
            return
        if record_type == "team":
            yield self._team_record(data)
            return
        if record_type == "coach":
            yield self._coach_record(data)
            return
        if record_type == "tournament":
            yield self._tournament_record(data)
            return
        if record_type == "transfer":
            yield from self._transfer_records(data)
            return
        # Unreachable: validate() rejects unknown record_types. Defensive
        # raise so a future record_type added to fetch() but not transform()
        # surfaces loudly rather than silently dropping records.
        raise SchemaDriftError(f"liquipedia: no transform branch for {record_type!r}")

    # --- per-record_type projections --------------------------------------

    def _player_record(self, data: dict[str, Any]) -> IngestionRecord:
        slug = data["slug"]
        # If Liquipedia advertises a previous_slug for this player (rare
        # for players, common for teams), prefer it as the platform_id so
        # the resolver matches the existing alias rather than minting a
        # new canonical. The new display name lands as platform_name on
        # the existing entity's alias row.
        platform_id = data.get("previous_slug") or slug
        return IngestionRecord(
            entity_type=EntityType.PLAYER,
            platform_id=platform_id,
            platform_name=data["name"],
            payload=data,
        )

    def _team_record(self, data: dict[str, Any]) -> IngestionRecord:
        slug = data["slug"]
        platform_id = data.get("previous_slug") or slug
        return IngestionRecord(
            entity_type=EntityType.TEAM,
            platform_id=platform_id,
            platform_name=data["name"],
            payload=data,
        )

    def _coach_record(self, data: dict[str, Any]) -> IngestionRecord:
        slug = data["slug"]
        platform_id = data.get("previous_slug") or slug
        return IngestionRecord(
            entity_type=EntityType.COACH,
            platform_id=platform_id,
            platform_name=data["name"],
            payload=data,
        )

    def _tournament_record(self, data: dict[str, Any]) -> IngestionRecord:
        return IngestionRecord(
            entity_type=EntityType.TOURNAMENT,
            platform_id=data["slug"],
            platform_name=data["name"],
            payload=data,
        )

    def _transfer_records(self, data: dict[str, Any]) -> Iterable[IngestionRecord]:
        """Project a transfer event into player + destination team records.

        Each carries the same ``roster_change`` blob under ``payload``.
        That blob is what the Phase 1 Relationship Engine will read when
        it backfills its ``RosterChange`` table — emitting the same dict
        on both sides means a future migration doesn't have to reach
        across two records to reconstruct the event.
        """
        # The roster_change blob is the canonical event shape downstream
        # consumers should read; we copy the upstream fields verbatim
        # plus a ``source`` tag so a later RosterChange backfill can pick
        # it out of payload without reverse-engineering the upstream
        # schema.
        roster_change = {
            "source": "liquipedia",
            "id": data["id"],
            "player_slug": data["player_slug"],
            "player_name": data["player_name"],
            "from_team_slug": data.get("from_team_slug"),
            "from_team_name": data.get("from_team_name"),
            "to_team_slug": data["to_team_slug"],
            "to_team_name": data["to_team_name"],
            "role": data.get("role"),
            "date": data["date"],
        }

        yield IngestionRecord(
            entity_type=EntityType.PLAYER,
            platform_id=data["player_slug"],
            platform_name=data["player_name"],
            payload={**data, "roster_change": roster_change},
        )
        yield IngestionRecord(
            entity_type=EntityType.TEAM,
            platform_id=data["to_team_slug"],
            platform_name=data["to_team_name"],
            payload={**data, "roster_change": roster_change},
        )


def _extend_alias_for_rebrand(
    session: Session,
    *,
    previous_slug: str,
    new_name: str,
) -> EntityAlias | None:
    """Pre-resolver lookup for a Liquipedia rebrand record.

    Returns the existing alias row keyed by ``(LIQUIPEDIA, previous_slug)``
    if one exists, ``None`` otherwise. Callers (the connector and the
    rebrand integration test) use this to confirm a rebrand is in fact
    a rename of a known canonical before emitting an IngestionRecord —
    the actual alias-extension write happens inside ``resolve_entity``,
    which the runner invokes one record later. The "extension" is then
    just the resolver inserting a new alias row under the same
    ``canonical_id`` because it auto-merges (or matches) on the
    previous slug.

    Why a helper instead of folding this into ``transform``: the ticket
    spec calls it out explicitly so a maintainer can read off the
    rebrand path in one place, and so a test can assert the lookup
    behaviour without driving the whole connector. The function deliberately
    does NOT perform a write — the resolver remains the single sanctioned
    writer of alias rows. The connector only adjusts the platform_id /
    platform_name pair it hands to the resolver, which is enough to
    extend (rather than fork) the alias chain.

    ``new_name`` is accepted for symmetry with the call site and so a
    future revision can log the (previous_slug -> new_name) mapping;
    today the function is purely a read.
    """
    stmt = select(EntityAlias).where(
        EntityAlias.platform == Platform.LIQUIPEDIA,
        EntityAlias.platform_id == previous_slug,
    )
    return session.execute(stmt).scalar_one_or_none()


def _iter_list(value: Any) -> Iterable[dict[str, Any]]:
    """Yield dict items from a list or ``{"items": [...]}`` envelope.

    Liquipedia list endpoints are not perfectly consistent: tournaments
    sometimes ship as a bare array, transfers as ``{"items": [...]}``.
    Centralising the shape-tolerance keeps ``fetch`` linear.

    Anything that doesn't fit either shape (e.g. Liquipedia silently
    re-wraps as ``{"results": [...]}``) raises
    :class:`SchemaDriftError` so the connector can log a structured
    drift event rather than dropping the whole list silently. The
    caller in ``fetch`` catches and downgrades to a per-endpoint skip
    so the run keeps going.

    Element-shape drift is treated the same way: list entries that
    aren't dicts (e.g. an upstream change that ships ``[{"slug": ...},
    "free-form-tag", {"slug": ...}]``) raise drift instead of being
    silently filtered out. Filtering would let partial data loss
    happen with no observable signal — the explicit raise keeps it on
    the SCHEMA_DRIFT path where ``nexus validate`` and an operator
    will see it.
    """
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                raise SchemaDriftError(
                    "liquipedia list entry must be dict, got " f"{type(item).__name__}"
                )
            yield item
        return
    if isinstance(value, dict):
        items = value.get("items")
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict):
                    raise SchemaDriftError(
                        "liquipedia list entry must be dict, got " f"{type(item).__name__}"
                    )
                yield item
            return
        raise SchemaDriftError(
            f"liquipedia list envelope missing 'items' (top-level keys: {sorted(value.keys())!r})"
        )
    raise SchemaDriftError(
        f"liquipedia list envelope must be list or {{'items': [...]}}, got {type(value).__name__}"
    )


# Lookup table for ``validate``. Module-level so tests can import the
# expected keys per record_type without monkeying with the connector.
_REQUIRED_KEYS_BY_TYPE: dict[str, frozenset[str]] = {
    "player": _REQUIRED_PLAYER_KEYS,
    "team": _REQUIRED_TEAM_KEYS,
    "coach": _REQUIRED_COACH_KEYS,
    "tournament": _REQUIRED_TOURNAMENT_KEYS,
    "transfer": _REQUIRED_TRANSFER_KEYS,
}


__all__ = [
    "DEFAULT_BASE_URL",
    "HttpGet",
    "LIQUIPEDIA_CADENCE",
    "LIQUIPEDIA_RATE_LIMIT",
    "LiquipediaConnector",
]
