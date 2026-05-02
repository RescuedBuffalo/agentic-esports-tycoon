"""SQLAlchemy 2.0 models for the BUF-6 schema.

Each table maps 1:1 to the Systems-spec System 01 + 02 design. Downstream code
must import these models (or the Pydantic DTOs in
:mod:`esports_sim.schemas.dtos`) — never reach into raw SQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    event,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, Session, mapped_column, relationship

from esports_sim.db.base import Base
from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus

# Reusable Postgres ENUM types. ``create_type=False`` keeps every model
# definition from trying to recreate the type — the migration owns the
# CREATE TYPE statement, full stop.
_entity_type = PgEnum(
    EntityType,
    name="entity_type",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_platform = PgEnum(
    Platform,
    name="platform",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_staging_status = PgEnum(
    StagingStatus,
    name="staging_status",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_review_status = PgEnum(
    ReviewStatus,
    name="review_status",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)


class Entity(Base):
    """Canonical record. Every other table joins on ``canonical_id``."""

    __tablename__ = "entity"

    canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # ``index=True`` mirrors the ix_entity_entity_type index in the migration.
    # Without this, alembic autogenerate would treat the DB index as drift and
    # drop it in a future revision — costly because lots of downstream queries
    # filter by entity_type.
    entity_type: Mapped[EntityType] = mapped_column(_entity_type, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="true",
    )

    aliases: Mapped[list[EntityAlias]] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class EntityAlias(Base):
    """Many-to-one platform handle.

    The ``(platform, platform_id)`` uniqueness is the schema-level guarantee
    that two scrapers can't accidentally split a player into two canonical
    rows. The resolver (BUF-7) is the runtime guarantee that they don't try.
    """

    __tablename__ = "entity_alias"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platform: Mapped[Platform] = mapped_column(_platform, nullable=False)
    platform_id: Mapped[str] = mapped_column(String(255), nullable=False)
    platform_name: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    entity: Mapped[Entity] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint(
            "platform",
            "platform_id",
            name="uq_entity_alias_platform_platform_id",
        ),
    )


class StagingRecord(Base):
    """Scraped payload waiting for the resolver.

    ``canonical_id`` is nullable to support the documented staging
    lifecycle: ``pending`` rows queued for the resolver, ``review`` rows
    deferred to a human reviewer, and ``blocked`` rows kept only for
    audit may all legitimately carry a null canonical id. The bypass that
    BUF-7 closes is the ``processed`` state — once a row is marked as
    fully processed, the canonical id MUST be set. ``StagingRecord.save()``
    plus the event listener below refuse to flush a ``processed`` row with
    a null canonical id, which is the only way a scraper could "finish"
    a staging row without actually consulting the resolver.
    """

    __tablename__ = "staging_record"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[EntityType] = mapped_column(_entity_type, nullable=False)
    canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[StagingStatus] = mapped_column(
        _staging_status,
        nullable=False,
        # Mirror the server_default at the Python layer so the attribute is
        # populated before ``before_insert`` fires. Without this, a row
        # constructed without an explicit status had ``record.status is None``
        # at validation time — the canonical-id check would raise
        # AttributeError on the error path's ``.value`` deref instead of
        # letting Postgres apply its default. ``server_default`` still owns
        # the on-disk default for hand-written SQL.
        default=StagingStatus.PENDING,
        server_default=StagingStatus.PENDING.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def save(self, session: Session) -> None:
        """Persist this record, enforcing the canonical-id invariant.

        Refuses to add a ``processed`` row whose ``canonical_id`` is still
        null — the only way that combination can arise is a scraper marking
        a staging row as fully processed without actually consulting the
        resolver. ``pending``/``review``/``blocked`` rows may legitimately
        have a null canonical_id and pass through unchanged.

        Direct ``session.add(record)`` is also covered by the event listener
        below; ``save()`` exists so call sites read intentionally and so
        unit tests can assert against a single named code path.
        """
        _check_canonical_invariant(self)
        session.add(self)


class StagingInvariantError(ValueError):
    """Raised when a :class:`StagingRecord` violates BUF-7's resolver rule.

    A ``processed`` staging row whose ``canonical_id`` is null is the
    bypass case: a scraper has declared the row "done" without going
    through :func:`esports_sim.resolver.resolve_entity`. Refuse it loudly
    rather than corrupting the canonical join key.
    """


# Statuses where a null canonical_id is meaningful state, not a bug. ``pending``
# is the pre-resolver queue state; ``review`` and ``blocked`` are terminal
# states the resolver itself produces when it can't (or won't) auto-decide.
# Only ``processed`` requires a non-null canonical_id, because that's the
# state that asserts "this row has a canonical mapping". Kept as a
# module-level frozenset so the listener and the wrapper can't drift apart.
_NULL_CANONICAL_OK: frozenset[StagingStatus] = frozenset(
    {StagingStatus.PENDING, StagingStatus.REVIEW, StagingStatus.BLOCKED}
)


def _check_canonical_invariant(record: StagingRecord) -> None:
    if record.canonical_id is not None:
        return
    # ``status`` may be ``None`` at ``before_insert`` time when a caller
    # constructed the row without setting it and is relying on the column
    # default — the row will land as ``pending``, which is a legal
    # null-canonical state. Treat that case as already-cleared rather than
    # falling through to the error formatter (which would AttributeError on
    # ``None.value``).
    if record.status is None or record.status in _NULL_CANONICAL_OK:
        return
    raise StagingInvariantError(
        f"StagingRecord with status={record.status.value!r} requires a "
        "non-null canonical_id. The resolver (BUF-7) is the only sanctioned "
        "writer of that column; marking a row processed without one means a "
        "scraper bypassed resolve_entity()."
    )


@event.listens_for(StagingRecord, "before_insert")
def _staging_before_insert(
    _mapper: Mapper[StagingRecord],
    _connection: Connection,
    target: StagingRecord,
) -> None:
    # SQLAlchemy event hook. Backstop for ``session.add(StagingRecord(...))``
    # paths that don't go through ``StagingRecord.save``. Note: this does not
    # fire for FK-cascade SET NULL, which happens inside Postgres — that's
    # the audit-trail outcome the schema deliberately allows.
    _check_canonical_invariant(target)


@event.listens_for(StagingRecord, "before_update")
def _staging_before_update(
    _mapper: Mapper[StagingRecord],
    _connection: Connection,
    target: StagingRecord,
) -> None:
    _check_canonical_invariant(target)


class RawRecord(Base):
    """Content-addressed blob store.

    Scrapers hash their payload before insert; ``content_hash`` is unique so
    the same fetch can be replayed without producing duplicates. Useful for
    debugging "why did the resolver decide that?" without re-hitting the
    upstream API.
    """

    __tablename__ = "raw_record"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)


class PatchNote(Base):
    """One game patch's release notes — a document, not a resolver entity (BUF-83).

    Patch notes don't fit the ``(platform, platform_id)`` alias model the
    rest of the schema is built around: there is no fuzzy matching, no
    canonical merging, no per-platform handle. They are versioned
    documents keyed by ``(source, patch_version)``, fed into BUF-24
    patch-intent extraction downstream. Storing them here rather than in
    ``raw_record`` makes ``patch_version`` and ``published_at`` first-class
    queryable columns.

    Source scoping: the uniqueness key is ``(source, patch_version)``,
    not ``patch_version`` alone. The generic ``PatchNoteConnector``
    abstraction lets multiple games' patch-notes connectors live in
    one process; if uniqueness were on ``patch_version`` alone, two
    connectors emitting the same string (e.g. "8.05") would collide
    and silently overwrite each other. Persisting ``source`` (the
    connector's ``source_name``) and using a composite key lets each
    game's history live independently while still UPSERT-deduping
    re-scrapes within a source.

    Idempotency: a re-scrape of the same article updates ``raw_html`` /
    ``body_text`` / ``fetched_at`` in place rather than inserting a
    duplicate row — safe to re-run on the weekly cadence.
    """

    __tablename__ = "patch_note"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # The connector's ``source_name`` (e.g. ``"playvalorant"``).
    # Part of the composite uniqueness key; see class docstring.
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    # 32 chars is more than enough for ``"Major.Minor"`` or ``"Major.Minor.Hotfix"``;
    # narrowing the column makes the unique-index footprint tiny.
    patch_version: Mapped[str] = mapped_column(String(32), nullable=False)
    # Indexed for the freshness check the validator runs against the most
    # recent ``published_at`` per source — see ``nexus validate``.
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    raw_html: Mapped[str] = mapped_column(nullable=False)
    body_text: Mapped[str] = mapped_column(nullable=False)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "source",
            "patch_version",
            name="uq_patch_note_source_patch_version",
        ),
    )


class AliasReviewQueue(Base):
    """Items the fuzzy matcher couldn't auto-decide.

    ``candidates`` is a JSON array of plausible canonical_ids with their
    similarity scores; the human reviewer picks one or marks the row blocked.
    See BUF-16 for the CLI/UI that drains this.

    Uniqueness is enforced via the partial index ``ix_alias_review_queue_pending_unique``
    over ``(platform, platform_id) WHERE status = 'pending'``: at most one
    pending review row per handle. The resolver's pending-enqueue path
    relies on this so two concurrent calls can't both check-then-insert
    and produce duplicate human-review work; the loser catches the unique
    violation and degrades to "already enqueued".
    """

    __tablename__ = "alias_review_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    platform: Mapped[Platform] = mapped_column(_platform, nullable=False)
    platform_id: Mapped[str] = mapped_column(String(255), nullable=False)
    platform_name: Mapped[str] = mapped_column(String(255), nullable=False)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ReviewStatus] = mapped_column(
        _review_status,
        nullable=False,
        server_default=ReviewStatus.PENDING.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Partial unique index: at most one pending row per (platform, platform_id).
        # The literal ``status = 'pending'`` matches Postgres's stored enum
        # value; using the enum object here would round-trip through Python
        # repr and produce the wrong predicate string.
        Index(
            "ix_alias_review_queue_pending_unique",
            "platform",
            "platform_id",
            unique=True,
            postgresql_where=text("status = 'pending'"),
        ),
    )


class PatchEra(Base):
    """One Valorant patch window — the temporal partition every joinable
    record carries (BUF-13, Systems-spec System 04).

    Identity is the ``era_id`` UUID; ``era_slug`` (e.g. ``"e2024_01"``) is
    the human-friendly key the YAML config uses and downstream tooling
    grep-matches against. ``patch_version`` carries the Riot patch label
    that opened the era (e.g. ``"8.0"``); when an era spans multiple
    Riot hotfixes, this is the *first* one — meaningful enough to grep
    out of a log line without claiming the era is bound to a single
    point release.

    Half-open semantics. ``[start_date, end_date)``: a match played at
    exactly ``end_date`` belongs to the *next* era, not this one. The
    closed-then-opened transactional pair :func:`roll_era` exploits this
    so a re-roll from era A to era B can stamp both rows with the same
    timestamp without overlap. The exclusion constraint enforced by the
    migration spells out the same rule at the DB level so two writers
    racing on a roll can't produce overlapping ranges.

    The ``end_date`` column is nullable to mark "the current era". A
    partial unique index (created in the migration) caps the open era
    count at one — the schema-level guarantee that
    :func:`current_era` can't return more than one row.

    ``meta_magnitude`` is a 0..1 hand-tuned magnitude estimate: how
    much the meta shifted at the start of this era. ``is_major_shift``
    is the boolean that gates the BUF-13 ``TEMPORAL_BLEED`` guard;
    aggregations that span across an era marked ``is_major_shift=True``
    raise :class:`TemporalBleedError` so models trained pre-shift never
    silently leak into post-shift evaluation.
    """

    __tablename__ = "patch_era"

    era_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    era_slug: Mapped[str] = mapped_column(String(32), nullable=False, unique=True)
    patch_version: Mapped[str] = mapped_column(String(32), nullable=False)
    start_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    # Nullable: ``end_date IS NULL`` is the open-current-era marker.
    end_date: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    meta_magnitude: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default="0",
    )
    is_major_shift: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="false",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        # Half-open ranges with end_date strictly after start_date. The
        # exclusion constraint in the migration prevents overlap; this
        # CHECK is the guard against a single-row bug (someone setting
        # end_date == start_date or end_date < start_date inside one
        # row). Keeping it CHECK rather than EXCLUDE keeps the per-row
        # validation cost effectively zero.
        CheckConstraint(
            "end_date IS NULL OR end_date > start_date",
            name="ck_patch_era_end_after_start",
        ),
        # ``meta_magnitude`` is a 0..1 estimate. The CHECK keeps a
        # buggy seed loader from inserting nonsense (e.g. -1, 5) that
        # would silently break the magnitude-weighted aggregations.
        CheckConstraint(
            "meta_magnitude >= 0 AND meta_magnitude <= 1",
            name="ck_patch_era_meta_magnitude_range",
        ),
        # At most one open era. The migration installs this as a
        # partial unique index (UNIQUE WHERE end_date IS NULL); the
        # ORM-side declaration mirrors it so alembic autogenerate
        # doesn't see drift in a future revision.
        Index(
            "ix_patch_era_open_unique",
            "end_date",
            unique=True,
            postgresql_where=text("end_date IS NULL"),
        ),
    )


class TemporalBleedError(RuntimeError):
    """Raised when an aggregation would span a major-shift era boundary.

    Systems-spec System 04: every record carries an era context, and no
    cross-era feature aggregation ever happens. The boundaries that
    make the rule load-bearing are the ones marked
    :attr:`PatchEra.is_major_shift` — those are the patches where the
    meta moved enough that a stat aggregated across the boundary is
    measuring two different games. See
    :func:`esports_sim.eras.assert_no_temporal_bleed` for the runtime
    guard.
    """


class Match(Base):
    """One series of maps between two teams (BUF-8 v2, VLR seed).

    Joins three times to :class:`Entity` — home team, away team, and
    the tournament — through nullable FKs. ``ON DELETE SET NULL`` on
    every FK keeps match history intact when an entity-side cleanup
    or merge happens; a row with a dangling participant is still
    valid historical record. Nullability also handles the upstream
    edge cases (TBD opponents represented as id=0, unresolved
    tournaments) that the CSV bootstrap inevitably surfaces.

    The ``vlr_match_id`` uniqueness is the seeder's idempotency
    anchor: a re-run against the same CSV no-ops on existing rows.
    """

    __tablename__ = "match"

    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    vlr_match_id: Mapped[str] = mapped_column(String(64), nullable=False)
    match_date: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    team1_canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    team2_canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    tournament_canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    series_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    team1_map_odds: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    map_results: Mapped[list[MapResult]] = relationship(
        back_populates="match",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint("vlr_match_id", name="uq_match_vlr_match_id"),
    )


class MapResult(Base):
    """One map within a :class:`Match` (BUF-8 v2).

    Per-team rounds (split atk/def), the headline rating, and a JSONB
    blob of the long-tail aggregate stats (ACS/KAST/ADR/HS/FK/FD/
    pistols/ecos/semibuys/fullbuys). The seed module is the single
    documented writer for ``team{1,2}_stats``; readers should treat
    keys outside that documented set as best-effort.

    ``round_breakdown`` is the raw VLR-encoded round-by-round ledger.
    We stash the text and let a downstream feature extractor parse
    it — the encoding has shifted at least once across VLR versions,
    and a parser regression should not be able to break the seed.

    ``vod_url`` lives here, not on ``match``: the CSV ships one VOD
    link per map (different games of the same series often have
    different upload links), so a per-row column is the right shape.
    """

    __tablename__ = "map_result"

    map_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("match.match_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    vlr_game_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # VLR's internal map id (1=Bind, 2=Haven, ..., 11=Abyss in roughly
    # release order). 0 / -1 are upstream sentinels for unplayed
    # / forfeited slots; we keep them verbatim so a future map-name
    # resolver doesn't lose the upstream signal.
    vlr_map_id: Mapped[int] = mapped_column(Integer, nullable=False)
    team1_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    team2_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    team1_atk_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    team1_def_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    team2_atk_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    team2_def_rounds: Mapped[int] = mapped_column(Integer, nullable=False)
    team1_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    team2_rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    team1_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    team2_stats: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    round_breakdown: Mapped[str | None] = mapped_column(Text, nullable=True)
    vod_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    match: Mapped[Match] = relationship(back_populates="map_results")

    __table_args__ = (
        UniqueConstraint("vlr_game_id", name="uq_map_result_vlr_game_id"),
    )


__all__ = [
    "AliasReviewQueue",
    "Entity",
    "EntityAlias",
    "MapResult",
    "Match",
    "PatchEra",
    "PatchNote",
    "RawRecord",
    "StagingInvariantError",
    "StagingRecord",
    "TemporalBleedError",
]
