"""Per-map player participation table for the BUF-85 VLR /match/ scraper.

Lands the ``player_match_stat`` table — one row per (map, player) — so the
follow-up VLR per-match scraper can record who played which agent on
which map with the canonical headline stats. Designed to also accept
Riot API rows once that connector grows a stat-extraction step
(RescuedBuffalo/agentic-esports-tycoon#2 currently writes the same
shape into ``staging_record.payload`` but defers the typed write).

Schema rationale:

* ``map_result_id`` FKs straight to ``map_result.map_result_id`` with
  ``ON DELETE CASCADE`` — a per-map stat without its parent map is
  uninterpretable, and ``map_result`` is the row that owns the
  ``vlr_game_id`` idempotency anchor BUF-8 v2 already maintains.
* ``entity_id`` FKs to ``entity.canonical_id`` with ``ON DELETE
  CASCADE`` — when a canonical entity is removed (a duplicate gets
  merged away) the orphaned stats follow. Other tables (``match``)
  use ``SET NULL`` instead because the row is still useful as a
  "this happened" record without the participant; here, a stat row
  with no player is just noise.
* ``source`` + ``source_player_id`` keep the upstream identity intact
  for audit and let two sources (VLR + Riot) coexist in the same
  table. The dedup unique constraint is on ``(map_result_id,
  entity_id)`` per the BUF-85 spec — first writer wins; a later VLR
  pass over a Riot-seeded match no-ops on conflict.
* Headline stats live in dedicated columns (kills/deaths/assists,
  ACS, ADR, KAST, HS%, FK/FD, rating). Source-specific extras
  (Riot's per-round detail, VLR's pistol/eco breakdown) go into the
  ``extra`` JSONB blob — same pattern as ``map_result.team{1,2}_stats``.
* ``agent`` is a free-form String(32) rather than an enum. Riot adds
  agents per patch and we'd rather not gate the scraper on a
  migration every time a new dueller ships; a downstream feature
  extractor can validate against the canonical agent list in
  ``data/agents/``.
* ``team_side`` is the literal ``"team1"`` / ``"team2"`` matching the
  ``map_result`` column names so a join can resolve which side of
  the map the player was on without an extra mapping table. CHECK
  constraint pins the value space.

Revision ID: 0007
Revises: 0006
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "player_match_stat",
        sa.Column("player_match_stat_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "map_result_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "map_result.map_result_id",
                ondelete="CASCADE",
                name="fk_player_match_stat_map_result_id_map_result",
            ),
            nullable=False,
        ),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="CASCADE",
                name="fk_player_match_stat_entity_id_entity",
            ),
            nullable=False,
        ),
        # ``source`` is the writing connector's ``source_name``
        # (e.g. ``"vlr"`` or ``"riot_api"``). Kept narrow because we
        # only ever store an ingester id here, not a free-form string.
        sa.Column("source", sa.String(32), nullable=False),
        # The upstream identifier we used to mint the canonical alias.
        # Riot's PUUID is 78 chars; VLR's numeric id is short. 128 covers
        # both with margin.
        sa.Column("source_player_id", sa.String(128), nullable=False),
        # ``team1`` / ``team2`` — matches the ``map_result.team{1,2}_*``
        # column naming so a downstream join on side knows which team
        # rounds belong to the player.
        sa.Column("team_side", sa.String(8), nullable=True),
        sa.Column("agent", sa.String(32), nullable=True),
        # Headline stats. Nullable across the board because forfeits,
        # walkovers, and partial scrapes legitimately ship without
        # every column.
        sa.Column("rating", sa.Float(), nullable=True),
        sa.Column("acs", sa.Float(), nullable=True),
        sa.Column("kills", sa.Integer(), nullable=True),
        sa.Column("deaths", sa.Integer(), nullable=True),
        sa.Column("assists", sa.Integer(), nullable=True),
        sa.Column("kast_pct", sa.Float(), nullable=True),
        sa.Column("adr", sa.Float(), nullable=True),
        sa.Column("hs_pct", sa.Float(), nullable=True),
        sa.Column("first_kills", sa.Integer(), nullable=True),
        sa.Column("first_deaths", sa.Integer(), nullable=True),
        # Source-specific long-tail (Riot's per-round breakdown, VLR's
        # 2K/3K/4K/5K columns, etc.) lives here so we don't bloat the
        # column count for fields one source carries.
        sa.Column(
            "extra",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # Idempotency anchor per the BUF-85 spec: re-running the
        # scraper for the same match must no-op. ``map_result_id`` is
        # 1:1 with ``vlr_game_id`` (its UNIQUE column) so this is the
        # same as ``(vlr_game_id, entity_id)`` without the extra join.
        sa.UniqueConstraint(
            "map_result_id",
            "entity_id",
            name="uq_player_match_stat_map_result_entity",
        ),
        # The value space is closed at the application layer; pin it
        # at the DB so a buggy writer can't insert ``"home"`` /
        # ``"away"`` and silently break joins on ``map_result``.
        sa.CheckConstraint(
            "team_side IS NULL OR team_side IN ('team1', 'team2')",
            name="ck_player_match_stat_team_side",
        ),
    )
    # Both FK sides get an index — the typical query pattern is either
    # "all stats for this map" (map_result_id) or "all stats for this
    # player" (entity_id), and we don't want a seq-scan on either side
    # of a 100k+ row table.
    op.create_index(
        "ix_player_match_stat_map_result_id",
        "player_match_stat",
        ["map_result_id"],
    )
    op.create_index(
        "ix_player_match_stat_entity_id",
        "player_match_stat",
        ["entity_id"],
    )
    # Source-id lookup for re-scrape paths that key on the upstream id
    # before the resolver has minted a canonical (e.g. dry-run audits).
    op.create_index(
        "ix_player_match_stat_source_source_player_id",
        "player_match_stat",
        ["source", "source_player_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_player_match_stat_source_source_player_id",
        table_name="player_match_stat",
    )
    op.drop_index("ix_player_match_stat_entity_id", table_name="player_match_stat")
    op.drop_index("ix_player_match_stat_map_result_id", table_name="player_match_stat")
    op.drop_table("player_match_stat")
