"""``seed_from_liquipedia()`` — bootstrap canonical entities (BUF-8).

One-shot Liquipedia crawl run once per environment to populate the
canonical entity table before any incremental scraper goes live. The
``LiquipediaConnector`` (BUF-11) is the steady-state weekly pull that
keeps the store fresh; this module is the genesis pass.

Decision tree per entity:

* ``/{kind}s?cursor=`` discovery walks return slugs (no fetch quota
  spent on profile data we'd then discard if the slug is bad).
* For each slug, ``/{kind}/{slug}`` is the profile fetch.
* ``resolve_entity`` is the single sanctioned writer of canonical +
  Liquipedia alias rows — same chokepoint the connector uses, so the
  seed and the steady-state pipeline can never disagree on what a
  canonical row looks like.
* Twitter / Twitch handles on the profile become aliases at confidence
  0.95 against the same canonical (Liquipedia curates these but a
  typo is plausible). Inserts go through a savepointed
  ``_insert_social_alias_idempotent`` so a re-run never duplicates.

Idempotency contract: the second invocation must add zero rows and
produce a manifest that distinguishes "newly created" from "already
present" — the manifest is what proves we satisfied BUF-8's
acceptance bullet without a manual DB diff.

Out of scope:

* No ``RawRecord`` / ``StagingRecord`` writes. The seed predates the
  steady-state pipeline; its only purpose is to populate canonical +
  alias. Audit replay of the seed itself lives on the manifest, not in
  the staging tables (which are for incremental traffic).
* No fuzzy retry. If the resolver lands a slug on the review queue
  (because its name happens to fuzzy-match an existing entity in the
  same type), the seed lets that PENDING outcome stand and increments
  ``review_queued`` rather than minting a forced-CREATE — the human
  reviewer is the right authority for that ambiguous case.
* No retry of ``TransientFetchError``. One transient miss skips the
  one slug; the operator re-runs the seed and idempotency takes care
  of the rest.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import EntityAlias
from esports_sim.resolver import (
    RebrandConflictError,
    ResolutionStatus,
    handle_rebrand,
    parse_renamed_at,
    resolve_entity,
)
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data_pipeline.connectors.liquipedia import (
    DEFAULT_BASE_URL,
    HttpGet,
    _default_http_get_factory,
    _iter_list,
)
from data_pipeline.errors import SchemaDriftError, TransientFetchError

_logger = logging.getLogger("data_pipeline.seeds.liquipedia")

# Where the manifest file lands. Operators can override; default is the
# top-level ``seeds/`` directory the README pins as the seed-artifact
# location.
DEFAULT_SEEDS_DIR = Path("seeds")

# Confidence convention from the BUF-8 spec.
#
# 1.0 — the Liquipedia profile slug *is* the canonical Liquipedia
#       identity. The resolver's CREATED path already inserts at 1.0;
#       this constant exists so the social-alias path below has a
#       symmetric reference.
_PROFILE_ALIAS_CONFIDENCE = 1.0
# 0.95 — Liquipedia editors curate twitter/twitch handles but a typo
#        on a freshly-edited profile is plausible. Below 1.0 so a future
#        verification job (manual or automated) can lift the score
#        without overlapping the profile-alias band.
_SOCIAL_ALIAS_CONFIDENCE = 0.95

# Map a Liquipedia profile-page field name to the ``Platform`` we
# project the handle into. Only twitter + twitch — those are the two
# the BUF-8 spec calls out explicitly. Adding instagram/youtube later
# requires a Platform enum value (and the ALTER TYPE migration
# ``packages/shared/src/esports_sim/db/enums.py`` warns about) before
# this map should grow, otherwise a SQLAlchemy enum cast fails.
_SOCIAL_FIELD_PLATFORMS: dict[str, Platform] = {
    "twitter": Platform.TWITTER,
    "twitch": Platform.TWITCH,
}

# Discovery endpoint plurals. Liquipedia uses ``coaches`` not ``coachs``;
# we map explicitly rather than naively pluralising so a future kind
# (``staff``? ``analyst``?) doesn't silently 404 against ``staffs``.
_DISCOVERY_PLURAL: dict[str, str] = {
    "player": "players",
    "team": "teams",
    "coach": "coaches",
}

# Required keys per profile shape — duplicated from the connector's
# private ``_REQUIRED_*`` frozensets so the seed doesn't reach into the
# connector's internal name. Re-exporting them from the connector would
# let a downstream consumer rebind validate without realising the seed
# leaned on the same set.
_REQUIRED_PLAYER_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_TEAM_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_COACH_KEYS: frozenset[str] = frozenset({"slug", "name"})
_REQUIRED_TOURNAMENT_KEYS: frozenset[str] = frozenset({"slug", "name"})


@dataclass
class _TypeCounters:
    """Per-entity-type running totals for the manifest.

    ``review_queued`` and ``schema_drifts`` are skip outcomes; the rest
    add up to the number of profiles successfully resolved.
    """

    discovered: int = 0
    created: int = 0
    matched: int = 0
    auto_merged: int = 0
    review_queued: int = 0
    schema_drifts: int = 0
    transient_errors: int = 0
    # Rebrand profiles (those carrying ``previous_slug``) where the
    # seed extended the alias chain with the NEW slug after the
    # resolver matched on the old one. Idempotent on re-runs: a
    # second pass leaves this at zero because the new alias already
    # exists. Tracking it lets the operator prove from the manifest
    # that the rebrand-extension path actually fired the first time.
    rebrands_registered: int = 0
    rebrands_existing: int = 0


@dataclass
class _SocialCounters:
    """Per-platform totals for social aliases.

    ``inserted`` is the new-row count this run; ``existing`` is the
    idempotent-no-op count (a re-run pegs ``inserted=0`` and
    ``existing=`` whatever was inserted last time).
    """

    inserted: int = 0
    existing: int = 0
    # Cross-canonical conflicts: this run tried to attach the handle
    # to canonical X, but it was already pinned to canonical Y. The
    # seed logs + counts these rather than aborting; an operator can
    # run an audit pass to disambiguate.
    conflicts: int = 0


@dataclass
class SeedManifest:
    """Auditable record of one ``seed_from_liquipedia`` invocation.

    Persisted to ``{seeds_dir}/liquipedia_seed_{seed_date}.json`` so an
    operator can diff manifests across runs and prove the BUF-8
    acceptance numbers (>=1000 players, >=100 teams, >=20 tournaments)
    held the first time and that subsequent re-runs added zero rows.
    """

    seed_date: str
    started_at: str
    finished_at: str
    base_url: str
    by_type: dict[str, _TypeCounters] = field(default_factory=dict)
    socials: dict[str, _SocialCounters] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        # ``asdict`` walks dataclass fields recursively; ``_TypeCounters``
        # and ``_SocialCounters`` flatten cleanly into nested dicts.
        return asdict(self)

    @property
    def total_canonical_created(self) -> int:
        return sum(c.created for c in self.by_type.values())


def seed_from_liquipedia(
    session: Session,
    *,
    http_get: HttpGet | None = None,
    base_url: str = DEFAULT_BASE_URL,
    seeds_dir: Path | None = None,
    today: date | None = None,
    write_manifest: bool = True,
) -> SeedManifest:
    """Bootstrap the canonical entity store from Liquipedia.

    The walk order is deterministic (players → teams → coaches →
    tournaments) so a re-run produces the same manifest event sequence
    in the structured logs. Each entity type's discovery walk paginates
    via ``?cursor=`` until the response carries no ``next_cursor``;
    each slug then fans out to a profile fetch and a single
    ``resolve_entity`` call.

    Caller owns the transaction. The seed ``flush``es as it goes (so
    socials inserted for slug N can race against slug N+1 cleanly) but
    does not ``commit``. Wrap in ``session.begin()`` to get all-or-
    nothing semantics; the operator script default is "commit at end
    of seed".

    The ``http_get`` callable is the same shape the connector takes —
    a function that accepts a URL and returns the parsed JSON body.
    Tests pass a routed dict; production uses the lazy ``httpx``
    factory the connector module exports.
    """
    base = base_url.rstrip("/")
    get = http_get if http_get is not None else _default_http_get_factory()
    seed_date = today or datetime.now(UTC).date()
    started_at = datetime.now(UTC)

    # Pre-seed the manifest with one counters bucket per declared entity
    # type. Always present (even at zero) makes the manifest easier for
    # ``jq`` queries and prevents missing-key errors when a later
    # acceptance script reads the file.
    manifest = SeedManifest(
        seed_date=seed_date.isoformat(),
        started_at=started_at.isoformat(),
        finished_at="",  # filled after the walk
        base_url=base,
        by_type={
            EntityType.PLAYER.value: _TypeCounters(),
            EntityType.TEAM.value: _TypeCounters(),
            EntityType.COACH.value: _TypeCounters(),
            EntityType.TOURNAMENT.value: _TypeCounters(),
        },
        socials={p.value: _SocialCounters() for p in _SOCIAL_FIELD_PLATFORMS.values()},
    )

    # Players + teams + coaches all share the discovery → profile fan-out
    # shape; only the tournament path differs (tournaments come back
    # fully-formed in the discovery envelope, no per-slug profile fetch).
    _seed_profile_kind(
        session,
        get=get,
        base=base,
        kind="player",
        entity_type=EntityType.PLAYER,
        required_keys=_REQUIRED_PLAYER_KEYS,
        manifest=manifest,
        capture_socials=True,
    )
    _seed_profile_kind(
        session,
        get=get,
        base=base,
        kind="team",
        entity_type=EntityType.TEAM,
        required_keys=_REQUIRED_TEAM_KEYS,
        manifest=manifest,
        capture_socials=True,
    )
    _seed_profile_kind(
        session,
        get=get,
        base=base,
        kind="coach",
        entity_type=EntityType.COACH,
        required_keys=_REQUIRED_COACH_KEYS,
        manifest=manifest,
        # Coach profiles rarely carry socials and the BUF-8 spec only
        # mentions player + team handles; flip on later if/when the
        # data is there.
        capture_socials=False,
    )
    _seed_tournaments(
        session,
        get=get,
        base=base,
        manifest=manifest,
    )

    manifest.finished_at = datetime.now(UTC).isoformat()

    if write_manifest:
        target_dir = seeds_dir if seeds_dir is not None else DEFAULT_SEEDS_DIR
        _persist_manifest(manifest, target_dir)

    _logger.info(
        "liquipedia_seed.done base_url=%s players_created=%d teams_created=%d "
        "coaches_created=%d tournaments_created=%d",
        base,
        manifest.by_type[EntityType.PLAYER.value].created,
        manifest.by_type[EntityType.TEAM.value].created,
        manifest.by_type[EntityType.COACH.value].created,
        manifest.by_type[EntityType.TOURNAMENT.value].created,
    )
    return manifest


# --- per-kind walkers -----------------------------------------------------


def _seed_profile_kind(
    session: Session,
    *,
    get: HttpGet,
    base: str,
    kind: str,
    entity_type: EntityType,
    required_keys: frozenset[str],
    manifest: SeedManifest,
    capture_socials: bool,
) -> None:
    """Discover slugs of one ``kind`` and resolve each profile."""
    counters = manifest.by_type[entity_type.value]
    for slug in _walk_discovery_slugs(get, base=base, kind=kind, counters=counters):
        counters.discovered += 1
        profile = _safe_get(get, f"{base}/{kind}/{slug}", kind=kind, identifier=slug)
        if profile is None:
            counters.transient_errors += 1
            continue
        if not isinstance(profile, dict):
            _logger.warning(
                "liquipedia_seed.schema_drift kind=%s slug=%s detail=non-dict profile",
                kind,
                slug,
            )
            counters.schema_drifts += 1
            continue
        missing = required_keys - profile.keys()
        if missing:
            _logger.warning(
                "liquipedia_seed.schema_drift kind=%s slug=%s missing=%s",
                kind,
                slug,
                sorted(missing),
            )
            counters.schema_drifts += 1
            continue

        canonical_id = _resolve_profile(
            session,
            entity_type=entity_type,
            profile=profile,
            counters=counters,
            kind=kind,
        )
        if canonical_id is None or not capture_socials:
            continue
        _attach_social_aliases(
            session,
            canonical_id=canonical_id,
            profile=profile,
            manifest=manifest,
        )


def _seed_tournaments(
    session: Session,
    *,
    get: HttpGet,
    base: str,
    manifest: SeedManifest,
) -> None:
    """Walk the tournaments envelope.

    Tournaments come back fully-formed in the list response — no
    per-slug profile fetch — so the discovery loop also resolves each
    item. ``next_cursor`` is honoured the same way as for profile
    discovery.
    """
    counters = manifest.by_type[EntityType.TOURNAMENT.value]
    for item in _walk_envelope(get, base=base, kind="tournaments", counters=counters):
        counters.discovered += 1
        if not isinstance(item, dict):
            _logger.warning("liquipedia_seed.schema_drift kind=tournament detail=non-dict item")
            counters.schema_drifts += 1
            continue
        missing = _REQUIRED_TOURNAMENT_KEYS - item.keys()
        if missing:
            _logger.warning(
                "liquipedia_seed.schema_drift kind=tournament slug=%s missing=%s",
                item.get("slug", "<unknown>"),
                sorted(missing),
            )
            counters.schema_drifts += 1
            continue
        _resolve_profile(
            session,
            entity_type=EntityType.TOURNAMENT,
            profile=item,
            counters=counters,
            kind="tournament",
        )


# --- discovery walks ------------------------------------------------------


def _walk_discovery_slugs(
    get: HttpGet,
    *,
    base: str,
    kind: str,
    counters: _TypeCounters,
) -> Iterable[str]:
    """Yield each slug the ``/{kind}s`` discovery endpoint advertises.

    Liquipedia paginates list endpoints with ``?cursor=`` and a
    ``next_cursor`` field on the response. ``_walk_envelope`` drives
    the cursor loop; this wrapper just projects out the ``slug`` field
    so the caller never sees the discovery item shape.
    """
    plural = _DISCOVERY_PLURAL.get(kind, f"{kind}s")
    for item in _walk_envelope(get, base=base, kind=plural, counters=counters):
        if not isinstance(item, dict):
            _logger.warning(
                "liquipedia_seed.discovery_drift kind=%s detail=non-dict item",
                kind,
            )
            continue
        slug = item.get("slug")
        if not isinstance(slug, str) or not slug:
            _logger.warning(
                "liquipedia_seed.discovery_drift kind=%s detail=missing or empty slug",
                kind,
            )
            continue
        yield slug


def _walk_envelope(
    get: HttpGet,
    *,
    base: str,
    kind: str,
    counters: _TypeCounters,
) -> Iterable[dict[str, Any]]:
    """Page through ``/{kind}?cursor=`` until ``next_cursor`` is absent.

    Liquipedia list endpoints return one of two envelopes:

    * a bare list ``[...]`` — single-page, no cursor;
    * ``{"items": [...], "next_cursor": "..."}`` — paginated.

    ``_iter_list`` (re-used from the connector) handles the bare list
    case + raises :class:`SchemaDriftError` on anything else, which we
    log and absorb so one drifted page doesn't kill the whole walk.
    """
    cursor: str | None = None
    while True:
        url = f"{base}/{kind}"
        if cursor:
            url = f"{url}?cursor={cursor}"
        envelope = _safe_get(get, url, kind=kind, identifier=cursor or "first")
        if envelope is None:
            counters.transient_errors += 1
            return
        try:
            items = list(_iter_list(envelope))
        except SchemaDriftError as exc:
            _logger.warning(
                "liquipedia_seed.envelope_drift kind=%s detail=%s",
                kind,
                exc,
            )
            counters.schema_drifts += 1
            return
        yield from items
        # Bare-list envelope cannot carry a cursor, so we're done. Same
        # for an ``{"items": [...]}`` envelope without ``next_cursor``.
        if not isinstance(envelope, dict):
            return
        next_cursor = envelope.get("next_cursor")
        if not next_cursor:
            return
        cursor = str(next_cursor)


def _safe_get(
    get: HttpGet,
    url: str,
    *,
    kind: str,
    identifier: str,
) -> Any | None:
    """Run one ``http_get`` and absorb only :class:`TransientFetchError`.

    Mirrors the connector's ``_safe_fetch``: a recoverable miss
    downgrades to ``None`` and the caller skips the record;
    everything else (4xx → ``RuntimeError`` from the default factory,
    a programming bug, etc.) propagates so a misconfigured seed run
    fails loudly rather than producing an empty manifest.
    """
    try:
        return get(url)
    except TransientFetchError as exc:
        _logger.warning(
            "liquipedia_seed.transient_fetch_error url=%s kind=%s id=%s detail=%s",
            url,
            kind,
            identifier,
            exc,
        )
        return None


# --- resolver bridge ------------------------------------------------------


def _resolve_profile(
    session: Session,
    *,
    entity_type: EntityType,
    profile: dict[str, Any],
    counters: _TypeCounters,
    kind: str,
) -> uuid.UUID | None:
    """Single ``resolve_entity`` call per profile, with manifest bookkeeping.

    Returns the canonical_id when the profile resolved to one (MATCHED
    / AUTO_MERGED / CREATED), or ``None`` for the PENDING outcome —
    the seed has nothing to attach to a row that's queued for human
    review.

    ``previous_slug`` is honoured here the same way the steady-state
    connector honours it: feeding the resolver the previous slug as
    ``platform_id`` makes the exact-alias lookup match an existing
    canonical instead of fuzzy-falling-through into a brand-new
    entity. That's how a rebrand survives a seed re-run.

    Rebrand alias extension: after the resolver matches/creates on
    ``previous_slug``, we also register the NEW slug as an alias on
    the same canonical via :func:`handle_rebrand`. Round 2 of Codex
    review caught the gap — without this, a later record carrying
    only the new slug would fuzzy-fall-through into a fork. The
    extension is idempotent so re-running the seed leaves
    ``rebrands_registered`` at zero and increments
    ``rebrands_existing`` instead.
    """
    slug = profile["slug"]
    name = profile["name"]
    previous_slug = profile.get("previous_slug")
    platform_id = previous_slug or slug

    result = resolve_entity(
        session,
        platform=Platform.LIQUIPEDIA,
        platform_id=platform_id,
        platform_name=name,
        entity_type=entity_type,
    )
    session.flush()

    if result.status is ResolutionStatus.CREATED:
        counters.created += 1
    elif result.status is ResolutionStatus.MATCHED:
        counters.matched += 1
    elif result.status is ResolutionStatus.AUTO_MERGED:
        counters.auto_merged += 1
    elif result.status is ResolutionStatus.PENDING:
        counters.review_queued += 1
        _logger.info(
            "liquipedia_seed.review_queued kind=%s slug=%s name=%s confidence=%.4f",
            kind,
            slug,
            name,
            result.confidence,
        )
        return None

    # Rebrand: register the new slug as an alias on the canonical we
    # just resolved. Skip when ``previous_slug`` is absent or equal to
    # the current slug (no rebrand event to record).
    if previous_slug and previous_slug != slug and result.canonical_id is not None:
        _register_rebrand_alias(
            session,
            old_platform_id=previous_slug,
            new_platform_id=slug,
            new_platform_name=name,
            renamed_at=profile.get("renamed_at"),
            counters=counters,
            kind=kind,
        )

    return result.canonical_id


def _register_rebrand_alias(
    session: Session,
    *,
    old_platform_id: str,
    new_platform_id: str,
    new_platform_name: str,
    renamed_at: Any,
    counters: _TypeCounters,
    kind: str,
) -> None:
    """Wire the BUF-12 ``handle_rebrand`` path for a profile carrying ``previous_slug``.

    Resolves the rebrand effective date from the profile's
    ``renamed_at`` field when present; otherwise stamps the alias
    with the current UTC time so a future ``lookup_alias_at`` query
    can still order the chain. Conflicts that point to a *different*
    canonical (the destination handle is already owned elsewhere) are
    logged at WARNING and counted under ``schema_drifts`` rather than
    aborting the seed — the human reviewer is the right authority
    for that ambiguous case.
    """
    effective_date = parse_renamed_at(renamed_at)

    # Pre-check whether the destination alias already exists so we
    # can split the manifest's ``rebrands_registered`` (newly added
    # this run) from ``rebrands_existing`` (idempotent replay). A
    # post-hoc discriminator on ``valid_from`` would mis-count when
    # two seed runs use the same effective date — the pre-check
    # avoids that ambiguity at the cost of one extra SELECT, which
    # is fine because the seed is one-shot.
    pre_existing = session.execute(
        select(EntityAlias).where(
            EntityAlias.platform == Platform.LIQUIPEDIA,
            EntityAlias.platform_id == new_platform_id,
        )
    ).scalar_one_or_none()

    try:
        handle_rebrand(
            session,
            platform=Platform.LIQUIPEDIA,
            old_platform_id=old_platform_id,
            new_platform_id=new_platform_id,
            new_platform_name=new_platform_name,
            effective_date=effective_date,
        )
    except RebrandConflictError as exc:
        _logger.warning(
            "liquipedia_seed.rebrand_conflict kind=%s old=%s new=%s detail=%s",
            kind,
            old_platform_id,
            new_platform_id,
            exc,
        )
        counters.schema_drifts += 1
        return

    session.flush()

    if pre_existing is None:
        counters.rebrands_registered += 1
    else:
        counters.rebrands_existing += 1


# ``_parse_renamed_at`` previously lived here; it's now
# :func:`esports_sim.resolver.parse_renamed_at` so the BUF-12 worker
# extractor (which detects the same Liquipedia rebrand events) can
# project the effective date the same way without duplicating the
# helper.


# --- social aliases -------------------------------------------------------


def _attach_social_aliases(
    session: Session,
    *,
    canonical_id: uuid.UUID,
    profile: dict[str, Any],
    manifest: SeedManifest,
) -> None:
    """Insert Twitter / Twitch aliases at confidence 0.95, idempotently.

    Each handle goes in under a savepoint so a duplicate (returning a
    re-run, or two seed processes racing on the same canonical) catches
    the unique-constraint violation without poisoning the outer
    transaction. The savepoint pattern matches what
    :func:`resolve_entity` does for its own race recoveries — keeping
    the recovery shape consistent across the resolver chokepoint.
    """
    for field_name, platform in _SOCIAL_FIELD_PLATFORMS.items():
        handle = profile.get(field_name)
        if not handle:
            continue
        if not isinstance(handle, str):
            _logger.warning(
                "liquipedia_seed.social_drift canonical_id=%s field=%s detail=non-string",
                canonical_id,
                field_name,
            )
            continue

        bucket = manifest.socials[platform.value]
        try:
            outcome = _insert_social_alias_idempotent(
                session,
                canonical_id=canonical_id,
                platform=platform,
                handle=handle,
            )
        except SocialAliasConflictError as exc:
            # Cross-canonical handle collision — log + count, don't
            # abort. The primary canonical resolution already
            # succeeded; the social alias just doesn't get attached
            # this run. Operator can audit + resolve.
            _logger.warning(
                "liquipedia_seed.social_conflict canonical_id=%s field=%s handle=%s detail=%s",
                canonical_id,
                field_name,
                handle,
                exc,
            )
            bucket.conflicts += 1
            continue
        if outcome == "inserted":
            bucket.inserted += 1
        else:
            bucket.existing += 1


def _insert_social_alias_idempotent(
    session: Session,
    *,
    canonical_id: uuid.UUID,
    platform: Platform,
    handle: str,
) -> str:
    """Insert one social alias; return ``"inserted"`` / ``"existing"``.

    The (platform, platform_id) unique constraint is the source of
    truth for "already there" — we attempt the insert under a
    savepoint and treat a unique-constraint violation as the
    no-op-retry case. Pre-checking with a SELECT first would race
    against a concurrent worker; the catch-the-violation pattern is
    the only one that's correct under concurrency.

    Cross-canonical conflict guard: round 3 of Codex review caught
    that the round-2 code returned ``"existing"`` on ANY collision,
    even when the colliding row belonged to a *different* canonical
    — silently masking a real cross-canonical handle conflict (e.g.
    two different players who both claim ``@TenZ`` on Twitter). On
    a collision we now re-read the winner; if it points at the same
    canonical the call is genuinely idempotent, otherwise it raises
    :class:`SocialAliasConflictError` so an operator can resolve.
    """
    try:
        with session.begin_nested():
            session.add(
                EntityAlias(
                    canonical_id=canonical_id,
                    platform=platform,
                    platform_id=handle,
                    platform_name=handle,
                    confidence=_SOCIAL_ALIAS_CONFIDENCE,
                )
            )
            session.flush()
    except IntegrityError as exc:
        # The only failure mode we know how to recover from is the
        # alias unique-key collision; any other constraint (a different
        # table's unique index, a check constraint we don't know about
        # yet) is a real failure that belongs to the caller.
        if "uq_entity_alias_platform_platform_id" not in str(exc):
            raise
        winner = session.execute(
            select(EntityAlias).where(
                EntityAlias.platform == platform,
                EntityAlias.platform_id == handle,
            )
        ).scalar_one()
        if winner.canonical_id != canonical_id:
            raise SocialAliasConflictError(
                f"({platform.value}, {handle!r}) already maps to canonical "
                f"{winner.canonical_id}; seed expected {canonical_id}"
            ) from exc
        return "existing"
    return "inserted"


class SocialAliasConflictError(RuntimeError):
    """Raised when a social handle is already attached to a different canonical.

    The seed assumes Liquipedia profile pages are the source of truth
    for ``(player_canonical, twitter_handle)`` mappings — but two
    profiles can legitimately both claim the same handle (a typo on
    one editor's page, or a copy-paste mistake on another). Surfacing
    this as a typed exception lets a future ``--allow-conflicts``
    flag downgrade it to a counter without changing the default-safe
    behaviour.
    """


# --- manifest persistence -------------------------------------------------


def _persist_manifest(manifest: SeedManifest, seeds_dir: Path) -> Path:
    """Write the manifest JSON; return the file path.

    The directory is created if missing. Filename is
    ``liquipedia_seed_{YYYY-MM-DD}.json`` so the operator can ``ls
    seeds/`` and read off run history at a glance. Two runs on the
    same date overwrite — that's intentional: the file represents the
    *most recent* attempt, and the structured logs carry the per-run
    detail.
    """
    seeds_dir.mkdir(parents=True, exist_ok=True)
    target = seeds_dir / f"liquipedia_seed_{manifest.seed_date}.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump(manifest.to_json(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return target


# Keep both the lazy-imported HTTP getter type and the manifest
# dataclasses reachable from this module — callers writing
# ``from data_pipeline.seeds.liquipedia import ...`` get a complete
# surface without having to chase down the connector module.
HttpGetter = Callable[[str], Any]


__all__ = [
    "DEFAULT_SEEDS_DIR",
    "HttpGetter",
    "SeedManifest",
    "seed_from_liquipedia",
]
