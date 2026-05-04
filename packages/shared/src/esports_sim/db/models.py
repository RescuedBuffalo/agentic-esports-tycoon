"""SQLAlchemy 2.0 models for the BUF-6 schema.

Each table maps 1:1 to the Systems-spec System 01 + 02 design. Downstream code
must import these models (or the Pydantic DTOs in
:mod:`esports_sim.schemas.dtos`) — never reach into raw SQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
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
from esports_sim.db.enums import (
    EntityType,
    MediaKind,
    Platform,
    RelationshipEdgeType,
    ReviewStatus,
    StagingStatus,
)

# all-MiniLM-L6-v2's output width (BUF-28, ADR-006). Pinned as a
# module-level constant so the model declarations, the migration, and
# the runtime ``Embedder.embed`` call all agree on the width — a
# mismatch would surface as an opaque pgvector cast error at insert.
EMBEDDING_DIM: int = 384

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
_relationship_edge_type = PgEnum(
    RelationshipEdgeType,
    name="relationship_edge_type",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_media_kind = PgEnum(
    MediaKind,
    name="media_kind",
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


class PatchIntent(Base):
    """Structured patch classification produced by the BUF-24 extractor (System 06).

    One row per ``(patch_note_id, prompt_version)`` — re-running the same
    prompt on the same patch UPSERTs in place; bumping the prompt version
    (because the spec or the rubric changed) lands a new row so the older
    classification is auditable.

    ``patch_note_id`` is the FK back to the source document. ``ON DELETE
    CASCADE`` means a patch note rotated out of the corpus takes its
    derived intent with it — there is no useful interpretation of an
    intent record without the underlying notes.

    Float columns (``pro_play_driven_score``,
    ``community_controversy_predicted``, ``confidence``) are 0..1 with
    DB-level CHECK constraints; a buggy model output that emits 1.5 fails
    at insert time rather than silently corrupting downstream weighted
    aggregates. The Pydantic ``PatchIntentResult`` (in
    :mod:`esports_sim.patch_intent.schema`) enforces the same bounds at
    the application boundary so the failure mode is "bad model output"
    not "bad SQL".
    """

    __tablename__ = "patch_intent"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    patch_note_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("patch_note.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    model: Mapped[str] = mapped_column(String(64), nullable=False)
    primary_intent: Mapped[str] = mapped_column(String(64), nullable=False)
    pro_play_driven_score: Mapped[float] = mapped_column(Float, nullable=False)
    agents_affected: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    maps_affected: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    econ_changed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    expected_pickrate_shifts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    community_controversy_predicted: Mapped[float] = mapped_column(Float, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    # Usage attribution — lets a budget retrospective tie an intent row
    # back to its ledger entry without joining on timestamp.
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    usd_cost: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "patch_note_id",
            "prompt_version",
            name="uq_patch_intent_patch_note_id_prompt_version",
        ),
        CheckConstraint(
            "pro_play_driven_score >= 0 AND pro_play_driven_score <= 1",
            name="ck_patch_intent_pro_play_driven_score_range",
        ),
        CheckConstraint(
            "community_controversy_predicted >= 0 AND community_controversy_predicted <= 1",
            name="ck_patch_intent_community_controversy_range",
        ),
        CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_patch_intent_confidence_range",
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

    __table_args__ = (UniqueConstraint("vlr_match_id", name="uq_match_vlr_match_id"),)


class PlayerMatchStat(Base):
    """One per-map player participation row (BUF-85).

    Lands one row per (map, player): which canonical entity played
    which agent on which map, plus the headline stats VLR's per-match
    page exposes (rating, ACS, K/D/A, KAST, ADR, HS%, FK/FD). Source-
    specific long-tail columns live in the ``extra`` JSONB blob so
    the typed schema doesn't have to grow per-source.

    Idempotency: the unique constraint on ``(map_result_id,
    entity_id)`` is the dedup anchor the BUF-85 spec calls for
    (``vlr_game_id`` is 1:1 with ``map_result``, so this is the same
    key without the join). A re-run of the scraper for the same match
    no-ops on conflict; first-writer-wins keeps the schema simple
    until cross-source merge logic is needed.

    Source-agnostic by design. ``source`` + ``source_player_id``
    preserve the upstream identity (VLR numeric id, Riot PUUID) so
    the scraper can audit "who did we attribute this stat to" without
    re-reading the alias table. The Riot connector
    (RescuedBuffalo/agentic-esports-tycoon#2) currently writes only
    to ``staging_record``; once it grows a stat-extraction step it
    can land in this same table without a schema change.
    """

    __tablename__ = "player_match_stat"

    player_match_stat_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    map_result_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("map_result.map_result_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_player_id: Mapped[str] = mapped_column(String(128), nullable=False)
    # ``team1`` / ``team2`` — matches the ``map_result.team{1,2}_*``
    # column naming so a join on side does not need a translation.
    team_side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(32), nullable=True)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    acs: Mapped[float | None] = mapped_column(Float, nullable=True)
    kills: Mapped[int | None] = mapped_column(Integer, nullable=True)
    deaths: Mapped[int | None] = mapped_column(Integer, nullable=True)
    assists: Mapped[int | None] = mapped_column(Integer, nullable=True)
    kast_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    adr: Mapped[float | None] = mapped_column(Float, nullable=True)
    hs_pct: Mapped[float | None] = mapped_column(Float, nullable=True)
    first_kills: Mapped[int | None] = mapped_column(Integer, nullable=True)
    first_deaths: Mapped[int | None] = mapped_column(Integer, nullable=True)
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "map_result_id",
            "entity_id",
            name="uq_player_match_stat_map_result_entity",
        ),
        CheckConstraint(
            "team_side IS NULL OR team_side IN ('team1', 'team2')",
            name="ck_player_match_stat_team_side",
        ),
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

    __table_args__ = (UniqueConstraint("vlr_game_id", name="uq_map_result_vlr_game_id"),)


class PersonalityEmbedding(Base):
    """One embedding per canonical entity (BUF-28, ADR-006).

    The personality extractor (BUF-25) summarises everything we know
    about a player into a paragraph; that paragraph is embedded with
    ``sentence-transformers/all-MiniLM-L6-v2`` and the resulting
    384-dim vector lands here. One row per entity is the right
    cardinality: the personality summary is the entity's current
    state, not a time series. Re-extraction is an UPSERT, anchored on
    ``entity_id`` as the primary key.

    Why a hard FK to ``entity.canonical_id`` with ``ON DELETE
    CASCADE``: the embedding is meaningless without the row that
    produced it. A merged-away duplicate's stale embedding would
    show up in :func:`similar_players` results pointing at a
    canonical id that no longer exists; cascading the delete keeps
    the index honest.

    ``model_version`` records which embedder produced the row (e.g.
    ``"sentence-transformers/all-MiniLM-L6-v2@v1"``). Stored
    per-row so a future model rotation can detect mixed populations
    and re-embed in place without losing the audit trail; the
    HNSW index doesn't care, but recall comparisons across model
    versions are nonsense.
    """

    __tablename__ = "personality_embedding"

    entity_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="CASCADE"),
        primary_key=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class MediaRecord(Base):
    """One audio/video file the local Whisper worker may transcribe (BUF-21).

    The cardinality is "one row per upstream media artifact, regardless
    of whether we've transcribed it yet". The Whisper batch worker
    pulls rows that don't have a corresponding :class:`Transcript`
    child and produces one. Re-runs replace the existing transcript
    in place — the ``transcript`` relationship is one-to-zero-or-one.

    ``source`` + ``source_uri`` is the dedup key — a re-ingest of the
    same Twitch VOD or YouTube upload UPSERTs in place rather than
    minting a duplicate row. The unique constraint at the schema
    layer is the runtime guarantee.

    ``local_path`` is the path on the worker host's filesystem. We
    keep it as a plain string rather than an FK to a blob store
    because the Whisper worker just needs ffmpeg-readable bytes;
    where they live (a local SSD vs. a network mount vs. a copy
    materialised from object storage) is an operational concern,
    not a schema one.

    ``entity_id`` is nullable because not every media artifact maps
    cleanly to one canonical entity — a podcast with three guests,
    a tournament broadcast — and forcing every upstream to pick a
    canonical would be a worse failure mode than leaving the column
    null and letting downstream feature joins handle the multi-
    speaker case explicitly.
    """

    __tablename__ = "media_record"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # Free-form identifier for the upstream that produced this row.
    # ``twitch_vod`` / ``youtube`` / ``podcast`` / ``manual`` are the
    # current populations; new sources just pick a new string. Same
    # convention as :class:`StagingRecord.source` so a single audit
    # query (``SELECT source, count(*) FROM media_record GROUP BY 1``)
    # gives the corpus mix at a glance.
    source: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # The canonical upstream URL or platform identifier. Unique per
    # source so a re-ingest doesn't fork the row. 1024 chars is plenty
    # for any plausible URL — Twitch VOD URLs sit around 80, YouTube
    # ones around 50; tightening this would risk truncating a
    # legitimate redirect chain.
    source_uri: Mapped[str] = mapped_column(String(1024), nullable=False)
    # On-disk location for the worker. Not unique: nothing stops two
    # rows from pointing at the same file (e.g., two upstreams of the
    # same VOD). Keeping it nullable would make every worker check
    # for a NULL before launching ffmpeg — easier to require it at
    # ingest time and have a separate ``download_pending`` workflow
    # if we later want async downloads.
    local_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    media_kind: Mapped[MediaKind] = mapped_column(_media_kind, nullable=False)
    # Probed media duration in seconds. Nullable because a fresh
    # ingest may register the row before the file is ffprobe'd; the
    # worker is allowed to populate it as a side-effect when it opens
    # the file. The transcript row carries its own duration counter
    # for the actually-transcribed span (silence excluded by VAD), so
    # this column is the upstream-reported total.
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    # BCP-47 language tag if known up-front (e.g. ``"en"``, ``"ko"``);
    # null lets Whisper auto-detect on first transcription. The
    # transcript row records the *detected* language separately so a
    # mismatch is auditable.
    language: Mapped[str | None] = mapped_column(String(8), nullable=True)
    # Optional canonical-entity attribution. ``ON DELETE SET NULL``
    # keeps the media row alive if the entity gets merged or deleted
    # — the audio file itself is still useful as raw corpus even
    # after the canonical it pointed at goes away.
    entity_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    # Free-form provenance: episode title, broadcast date, ingest run
    # id. JSONB so a future feature query can pull a structured field
    # out without a follow-up migration.
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    transcript: Mapped[Transcript | None] = relationship(
        back_populates="media",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )

    __table_args__ = (
        UniqueConstraint(
            "source",
            "source_uri",
            name="uq_media_record_source_source_uri",
        ),
    )


class Transcript(Base):
    """Whisper transcription output for one :class:`MediaRecord` (BUF-21).

    One row per ``media_id`` — re-running the worker on the same
    media UPSERTs in place. ``model_version`` is recorded on the row
    rather than the unique key because BUF-21's contract is "we have
    one canonical transcript per file"; if a future workflow wants
    to keep multiple transcripts per media (e.g. for diarization
    A/B), promote ``(media_id, model_version)`` to the unique key
    in that follow-up migration.

    ``text`` is the full concatenated transcript and is the column
    BUF-21's acceptance criterion targets ("transcripts searchable
    via SQL on the transcript column"). A future migration can layer
    a tsvector + GIN index on top for proper full-text search; for
    now plain ``text`` keeps the column queryable with ``ILIKE`` /
    ``%`` matches without forcing a tsvector maintenance cost on
    every insert.

    ``segments`` is the per-segment list Whisper emits, preserved
    as JSONB so a downstream feature extractor (timestamp-aligned
    sentiment, speaker turns) doesn't have to re-run the model.
    """

    __tablename__ = "transcript"

    media_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("media_record.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # BCP-47 tag of what Whisper actually decoded. May differ from
    # the parent ``MediaRecord.language`` (e.g., a "mixed" stream
    # where the upstream-declared tag was wrong); keeping both lets
    # an auditor see the disagreement.
    language: Mapped[str] = mapped_column(String(8), nullable=False)
    # Whisper model identity (e.g. ``"large-v3"``). Per-row so a
    # mixed corpus that's been partly re-transcribed with a newer
    # model is auditable from the table itself.
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # List of objects, one per Whisper segment::
    #   [{"start": 0.0, "end": 4.2, "text": "...", "speaker": null}, ...]
    # ``speaker`` is null for now; pyannote-driven diarization (gated
    # behind a flag in BUF-21's note) will populate it later.
    segments: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=False,
    )
    # The actually-transcribed audio span in seconds. Equal to the
    # parent's ``duration_seconds`` minus VAD-skipped silence. Stored
    # so a throughput query (``SUM(duration_seconds) / SUM(wallclock)``)
    # has the right numerator without re-deriving from segments.
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    # Wallclock the model spent on this file. Lets the dashboard show
    # "audio-seconds per wallclock-second" — BUF-21 acceptance asks
    # for 10h in <30min, i.e., ≥20× realtime, and this is the column
    # that proves it.
    wallclock_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    # Optional path to the per-media ``transcript.json`` sidecar the
    # worker writes. Nullable so a callers that doesn't materialise a
    # sidecar (e.g., a one-off re-transcription called from a unit
    # test) can still land a row.
    transcript_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    transcribed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    media: Mapped[MediaRecord] = relationship(back_populates="transcript")


class TranscriptChunkEmbedding(Base):
    """One embedded chunk of a transcript (BUF-28, ADR-006).

    Whisper (BUF-21) emits transcripts of player and caster media.
    Each transcript gets sliced into ~500-token chunks, embedded
    with the same MiniLM model as :class:`PersonalityEmbedding`,
    and stored here for retrieval-during-inference.

    Why store ``chunk_text`` alongside the vector instead of joining
    back to a ``transcript_chunk`` table on every query: the helper
    that builds an LLM prompt needs the raw text after the kNN
    selects the chunk, and a per-row 500-token blob keeps the join
    out of the hot path. Costs ~2 KB per row in TOAST storage; the
    saving is one less JOIN on every retrieval call.

    ``media_id`` FKs to :class:`MediaRecord` with ``ON DELETE
    CASCADE`` (added in migration 0011 once BUF-21's media table
    landed) — pulling a media row drops its chunks atomically. Until
    that migration ran, the writer was the cleanup authority; this
    docstring is kept for the historical record but the FK is now
    the runtime guarantee.

    Idempotency: ``(media_id, chunk_idx)`` is unique. A re-embedding
    pass for the same media UPSERTs by that key, so a re-run of the
    Whisper pipeline never produces duplicates.
    """

    __tablename__ = "transcript_chunk_embedding"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # FK installed by migration 0011 (BUF-21). Indexed because every
    # read path filters on it (either "all chunks for this media"
    # during cleanup, or "select chunk_text where media_id IN (...)"
    # after the kNN narrows down the candidate set).
    media_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("media_record.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(EMBEDDING_DIM),
        nullable=False,
    )
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    __table_args__ = (
        UniqueConstraint(
            "media_id",
            "chunk_idx",
            name="uq_transcript_chunk_embedding_media_chunk",
        ),
    )


class RelationshipEdge(Base):
    """One pairwise relationship between two canonical entities (BUF-26, System 07).

    Each row encodes a single directed claim of the form "``src_id``
    relates to ``dst_id`` as ``edge_type``" together with two scalar
    attributes:

    * ``strength`` (``[0, 1]``) — how vigorous the relationship is
      *right now*. Decays exponentially with no signal — see
      :func:`ecosystem.relationships.decay_strength`. New
      :class:`RelationshipEvent` rows reset the clock and add to it.
    * ``sentiment`` (``[-1, 1]``) — valence. Does **not** decay; an
      ex-teammate edge keeps its positive sentiment after the strength
      has bled out. Negative sentiment is what turns a TEAMMATE edge
      into a feud rather than a friendship.

    Symmetric edge types (see
    :data:`esports_sim.db.enums.SYMMETRIC_RELATIONSHIP_EDGE_TYPES`)
    must be persisted with ``src_id < dst_id`` so a single canonical
    pair owns at most one row of that type. The migration enforces the
    rule with a CHECK; the bootstrap and the application API
    canonicalise endpoints up front. Asymmetric kinds (mentor,
    manager_of) keep their direction.

    ``last_updated_at`` is the anchor :func:`decay_strength` reads to
    decide how many weeks of decay to apply on the next read or job
    pass. The monthly decay job (see
    :func:`ecosystem.relationships.run_monthly_decay`) is the canonical
    writer; ad-hoc updates set it explicitly.
    """

    __tablename__ = "relationship_edge"

    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    src_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    dst_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    edge_type: Mapped[RelationshipEdgeType] = mapped_column(
        _relationship_edge_type,
        nullable=False,
        index=True,
    )
    strength: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default="0",
    )
    sentiment: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default="0",
    )
    # Anchor for the next ``decay_strength`` call. Initialised to the
    # creation time of the row when the bootstrap or steady-state writer
    # mints it; updated in place by the monthly decay job and by every
    # event that touches strength so the next decay pass measures from
    # the right reference point.
    last_updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    # Free-form provenance. The bootstrap stamps ``shared_maps`` /
    # ``last_shared_match_at`` here so a downstream debug query can
    # explain why a particular edge ended up TEAMMATE vs EX_TEAMMATE.
    extra: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    events: Mapped[list[RelationshipEvent]] = relationship(
        back_populates="edge",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        # One row per (pair, kind). For symmetric kinds the bootstrap
        # canonicalises endpoints to ``src < dst`` so the pair owns at
        # most one row of that type; for asymmetric kinds (mentor,
        # manager_of) the unique constraint distinguishes the two
        # directions.
        UniqueConstraint(
            "src_id",
            "dst_id",
            "edge_type",
            name="uq_relationship_edge_src_dst_type",
        ),
        # A self-edge is meaningless and would always sort to the wrong
        # side of the symmetry CHECK below; pin it out at the schema
        # layer rather than relying on every writer to remember.
        CheckConstraint(
            "src_id <> dst_id",
            name="ck_relationship_edge_no_self_edge",
        ),
        # Symmetric kinds must be stored canonically (src < dst) so a
        # pair owns at most one row of that type. Asymmetric kinds
        # (mentor, manager_of) keep direction. Listing the symmetric
        # kinds inline rather than referencing the Python frozenset
        # keeps the migration self-contained — adding a new symmetric
        # kind is a coordinated migration + enum + frozenset change.
        CheckConstraint(
            "edge_type NOT IN ('teammate', 'ex_teammate', 'rival', 'friend') " "OR src_id < dst_id",
            name="ck_relationship_edge_symmetric_canonical",
        ),
        CheckConstraint(
            "strength >= 0 AND strength <= 1",
            name="ck_relationship_edge_strength_range",
        ),
        CheckConstraint(
            "sentiment >= -1 AND sentiment <= 1",
            name="ck_relationship_edge_sentiment_range",
        ),
    )


class RelationshipEvent(Base):
    """One signal that touched a :class:`RelationshipEdge` (BUF-26, System 07).

    Append-only. Every signal that should affect strength or sentiment
    lands as a row here first; the writer that produced it also folds
    ``delta_strength`` / ``delta_sentiment`` into the parent edge and
    bumps ``last_updated_at`` so the decay job's clock resets.

    Keeping the event log separate from the in-place edge update is the
    BUF-78 event-log pattern: an audit trail that lets a future
    reducer rebuild edge state from genesis if the in-place column
    drifts. The cardinality is several events per edge per season, so
    the table is bounded by roster turnover, not by tick rate.

    ``event_kind`` is a free-form string deliberately — the value
    space is much larger than the small enum on the parent edge
    (``"shared_clutch"``, ``"public_feud"``, ``"social_media_jab"``,
    ``"won_together"``, ``"benched"``, …) and gating it on a
    Postgres ENUM would force an ALTER TYPE migration every time a
    new signal type ships.
    """

    __tablename__ = "relationship_event"

    event_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    edge_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("relationship_edge.edge_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    event_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    # Both deltas are signed and can land outside ``[-1, 1]`` if a
    # particularly weighty event swings the edge past saturation; the
    # parent edge's CHECK constraints clamp the cumulative result.
    delta_strength: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default="0",
    )
    delta_sentiment: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        server_default="0",
    )
    # When the event happened in the simulation/world. Indexed because
    # the event-stream queries (``last 30 days for this edge``,
    # ``replay this edge from genesis``) all filter on it.
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
        default=dict,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    edge: Mapped[RelationshipEdge] = relationship(back_populates="events")


__all__ = [
    "AliasReviewQueue",
    "EMBEDDING_DIM",
    "Entity",
    "EntityAlias",
    "MapResult",
    "Match",
    "MediaRecord",
    "PatchEra",
    "PatchIntent",
    "PatchNote",
    "PersonalityEmbedding",
    "PlayerMatchStat",
    "RawRecord",
    "RelationshipEdge",
    "RelationshipEvent",
    "StagingInvariantError",
    "StagingRecord",
    "TemporalBleedError",
    "Transcript",
    "TranscriptChunkEmbedding",
]
