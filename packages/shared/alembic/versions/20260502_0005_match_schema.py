"""Match-level schema for the VLR CSV bootstrap (BUF-8 v2).

Lands the two tables that hold every map of every VAL match the
seeder ingests from a VLR.gg CSV scrape:

* ``match`` — one row per series (a Bo3/Bo5/Bo1). Joins to ``entity``
  three times: home team, away team, and the tournament. All three FKs
  are ``ON DELETE SET NULL`` so an entity-side cleanup or merge doesn't
  cascade-kill match history; a row with a dangling participant is
  still useful as historical record. Nullability also handles the
  upstream edge cases (TBD opponents represented as id=0, unresolved
  tournaments) that the CSV bootstrap inevitably surfaces.
* ``map_result`` — one row per map within a match. Carries per-team
  rounds (split atk/def), the headline rating, the long-tail per-team
  aggregates as JSONB (so ~25 stats per team don't bloat the column
  count), and the raw round-by-round encoded string from VLR. Cascades
  on match delete because a map without its match is uninterpretable.

Why direct FKs to ``entity.canonical_id`` rather than ``entity_alias``:
the canonical UUID is the join key the resolver guarantees stable
under rebrand. An alias FK would mean a Sentinels rename forces a
match-table backfill; canonical_id stays put.

Per-map player participation deliberately does NOT live in this
schema. The VLR CSV's ``Team1Game1..Team2Game5`` columns are MATCH
ids of each team's five most-recent past matches (a recent-form
ML feature), not player ids — derivable from the ``match`` table
itself by sorting on ``match_date``. Roster-by-map data will land
in a follow-up schema once the per-match VLR scraper extracts it
from ``/match/<id>`` profile pages.

Era integration: matches are timestamped with ``match_date``. The
existing :func:`assign_era` SQL function from migration 0004 resolves
that to a ``patch_era.era_id`` for any analytics view that wants to
filter by era. We deliberately don't materialise an ``era_id`` column
on ``match`` — backfill on era roll would be a write storm; the
function-based lookup keeps it lazy.

Revision ID: 0005
Revises: 0004
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "match",
        sa.Column("match_id", postgresql.UUID(as_uuid=True), primary_key=True),
        # VLR's stable numeric match id, kept as a string so a future
        # source whose id is non-numeric can land in the same column
        # without a schema change. ``unique=True`` is the idempotency
        # anchor for the seeder + the incremental scraper alike.
        sa.Column("vlr_match_id", sa.String(64), nullable=False),
        sa.Column("match_date", sa.DateTime(timezone=True), nullable=False),
        # All three entity FKs are nullable + SET NULL because the
        # CSV occasionally references a TBD opponent (id=0) or an
        # event the resolver hasn't seen; we'd rather keep the row
        # with a dangling participant than reject it.
        sa.Column(
            "team1_canonical_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="SET NULL",
                name="fk_match_team1_canonical_id_entity",
            ),
            nullable=True,
        ),
        sa.Column(
            "team2_canonical_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="SET NULL",
                name="fk_match_team2_canonical_id_entity",
            ),
            nullable=True,
        ),
        sa.Column(
            "tournament_canonical_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="SET NULL",
                name="fk_match_tournament_canonical_id_entity",
            ),
            nullable=True,
        ),
        # Series-level betting odds (decimal). Both columns nullable
        # because VLR omits odds for matches the books didn't list
        # (most regional Challengers fixtures fall in this bucket).
        sa.Column("series_odds", sa.Float(), nullable=True),
        sa.Column("team1_map_odds", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("vlr_match_id", name="uq_match_vlr_match_id"),
    )
    # ``match_date`` is the workhorse filter (date-range queries,
    # era assignment via assign_era). Indexing it once at create time
    # is much cheaper than backfilling later.
    op.create_index("ix_match_match_date", "match", ["match_date"])
    op.create_index("ix_match_team1_canonical_id", "match", ["team1_canonical_id"])
    op.create_index("ix_match_team2_canonical_id", "match", ["team2_canonical_id"])
    op.create_index("ix_match_tournament_canonical_id", "match", ["tournament_canonical_id"])

    op.create_table(
        "map_result",
        sa.Column("map_result_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "match_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "match.match_id",
                ondelete="CASCADE",
                name="fk_map_result_match_id_match",
            ),
            nullable=False,
        ),
        # VLR's stable numeric game/map id. ``unique=True`` is the
        # idempotency anchor for the per-map insert — re-running the
        # seed against the same CSV is a no-op once the row exists.
        sa.Column("vlr_game_id", sa.String(64), nullable=False),
        # VLR's internal map id (1=Bind, 2=Haven, 3=Split, ..., 11=Abyss
        # in roughly release order). ``-1`` and ``0`` show up as
        # sentinel values in the CSV for unplayed/forfeit map slots —
        # we store them verbatim so a future map-name resolver can
        # decide what to do without losing the upstream signal.
        sa.Column("vlr_map_id", sa.Integer(), nullable=False),
        # Round columns. Stored as plain Integer because the values
        # are small (cap of ~25 in overtime) and querying "all maps
        # team won 13-x" is a common analyst question.
        sa.Column("team1_rounds", sa.Integer(), nullable=False),
        sa.Column("team2_rounds", sa.Integer(), nullable=False),
        sa.Column("team1_atk_rounds", sa.Integer(), nullable=False),
        sa.Column("team1_def_rounds", sa.Integer(), nullable=False),
        sa.Column("team2_atk_rounds", sa.Integer(), nullable=False),
        sa.Column("team2_def_rounds", sa.Integer(), nullable=False),
        # Rating is the headline number per VLR's own UI; promoted
        # to a typed column so analysts don't have to dig into the
        # JSONB blob for the most-queried metric. Nullable because
        # forfeits and walkovers ship without ratings.
        sa.Column("team1_rating", sa.Float(), nullable=True),
        sa.Column("team2_rating", sa.Float(), nullable=True),
        # The long tail of per-team aggregates (ACS, kills, deaths,
        # assists, KAST, ADR, HS%, FK/FD, pistols, ecos, semibuys,
        # fullbuys) lives in JSONB. Flattening to ~25 columns per
        # team would double the schema and most downstream queries
        # only ever need a handful at a time. The seed writes a
        # canonical key set documented in the seed module so future
        # readers know what to expect.
        sa.Column("team1_stats", postgresql.JSONB, nullable=False),
        sa.Column("team2_stats", postgresql.JSONB, nullable=False),
        # Encoded round-by-round ledger from VLR (one segment per
        # round, ``state!eco_t1!eco_t2``). We stash the raw text and
        # leave parsing to a downstream feature extractor — the
        # encoding has already mutated once across VLR releases and
        # we don't want a parser regression to break the seed.
        sa.Column("round_breakdown", sa.Text(), nullable=True),
        # 512 chars covers every observed VOD URL plus a comfortable
        # margin for query strings; truncating below would lose
        # YouTube playlist params we want to keep for replay
        # reconstruction. The CSV ships VOD links per map, not per
        # match (different game often = different VOD), so vod_url
        # lives here and not on ``match``.
        sa.Column("vod_url", sa.String(512), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("vlr_game_id", name="uq_map_result_vlr_game_id"),
    )
    op.create_index("ix_map_result_match_id", "map_result", ["match_id"])


def downgrade() -> None:
    op.drop_index("ix_map_result_match_id", table_name="map_result")
    op.drop_table("map_result")
    op.drop_index("ix_match_tournament_canonical_id", table_name="match")
    op.drop_index("ix_match_team2_canonical_id", table_name="match")
    op.drop_index("ix_match_team1_canonical_id", table_name="match")
    op.drop_index("ix_match_match_date", table_name="match")
    op.drop_table("match")
