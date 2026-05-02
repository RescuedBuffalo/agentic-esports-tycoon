"""Add patch_era table + assign_era SQL function + window view (BUF-13).

Systems-spec System 04: every record carries an era context. The
schema-level pieces this migration installs are the substrate that
makes the rule load-bearing rather than a Python convention:

* The ``patch_era`` table itself, with half-open
  ``[start_date, end_date)`` semantics.
* A CHECK that ``end_date`` is strictly after ``start_date`` (or
  null). Catches single-row bugs (zero-length / negative-length
  ranges) before they can poison any join.
* An ``EXCLUDE USING gist`` on the tstzrange of the era window. Two
  eras whose half-open ranges overlap are rejected at flush, which is
  what makes :func:`roll_era`'s atomic close-then-open safe under
  concurrent writers.
* A partial unique index on ``end_date IS NULL`` so at most one era
  is open. Without this, two operators racing on a roll could both
  succeed at the close step and then both insert an open era,
  duplicating the current-era pointer.
* A SQL function ``assign_era(timestamptz) RETURNS uuid`` so views
  and ad-hoc queries can resolve a timestamp without round-tripping
  to the application layer. The function uses the same predicate as
  :func:`esports_sim.eras.assign_era`, so SQL and Python agree on
  every edge case.
* A ``patch_era_window`` view that resolves the open era's null
  ``end_date`` to ``+infinity``. Downstream views that filter by era
  can join against this without each one re-implementing the
  COALESCE.

Why a Postgres function rather than only a Python helper: the BUF-13
acceptance bullet on SQL views ("filter by era for all common
joins") needs the resolution to live in the database so views like
``v_match_with_era`` (added by future migrations once the matches
table exists) can call ``assign_era(played_at)`` directly.

Revision ID: 0004
Revises: 0003
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Note: this migration deliberately does NOT install ``btree_gist``.
    # Postgres's core GIST opclass already supports range types, so the
    # ``EXCLUDE USING gist (tstzrange(...) WITH &&)`` constraint below
    # works without a contrib extension. Adding ``btree_gist`` would only
    # be required if we wanted to combine a btree-indexable column (e.g.
    # a per-game UUID) into the same EXCLUDE — at which point the
    # migration that introduces that column should install it. Doing it
    # eagerly here would block alembic upgrade in environments where the
    # app role can't install extensions or where ``postgresql-contrib``
    # isn't on the host.
    op.create_table(
        "patch_era",
        sa.Column("era_id", postgresql.UUID(as_uuid=True), primary_key=True),
        # 32 chars covers 'eYYYY_NN' plus a couple of suffix chars; the
        # config file's slug format ('e2024_01') is well within budget.
        sa.Column("era_slug", sa.String(32), nullable=False),
        # 32 chars covers 'Major.Minor[.Hotfix]' Riot patch strings — same
        # width the patch_note table uses for the same field.
        sa.Column("patch_version", sa.String(32), nullable=False),
        sa.Column("start_date", sa.DateTime(timezone=True), nullable=False),
        # Nullable — the open era marker.
        sa.Column("end_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "meta_magnitude",
            sa.Float(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        sa.Column(
            "is_major_shift",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("era_slug", name="uq_patch_era_era_slug"),
        sa.CheckConstraint(
            "end_date IS NULL OR end_date > start_date",
            name="ck_patch_era_end_after_start",
        ),
        sa.CheckConstraint(
            "meta_magnitude >= 0 AND meta_magnitude <= 1",
            name="ck_patch_era_meta_magnitude_range",
        ),
    )
    # ``start_date`` is the predicate column for assign_era's
    # half-open lookup; index it so the lookup is O(log n) rather
    # than a sequential scan once the table is fully seeded.
    op.create_index("ix_patch_era_start_date", "patch_era", ["start_date"])

    # Partial unique index: at most one open era. The literal
    # ``end_date IS NULL`` is what Postgres stores; the ORM mirror in
    # models.py uses the same predicate string so autogenerate doesn't
    # see drift.
    op.create_index(
        "ix_patch_era_open_unique",
        "patch_era",
        ["end_date"],
        unique=True,
        postgresql_where=sa.text("end_date IS NULL"),
    )

    # Exclusion constraint on overlapping ranges. Two eras whose half-
    # open windows overlap are rejected at flush time. The ``[)``
    # bound means closed-then-opened at the same instant DOES NOT
    # overlap (which is what makes ``roll_era`` safe). Using
    # ``COALESCE(end_date, 'infinity')`` so the open era's range
    # extends to +infinity for the overlap check.
    op.execute("""
        ALTER TABLE patch_era
        ADD CONSTRAINT ex_patch_era_no_overlap
        EXCLUDE USING gist (
            tstzrange(start_date, COALESCE(end_date, 'infinity'::timestamptz), '[)')
            WITH &&
        )
        """)

    # ``assign_era`` SQL function. Same half-open semantics as the
    # Python helper. Uses ``ORDER BY start_date DESC LIMIT 1`` so the
    # open era (covering [start_date, +inf)) wins over any older row
    # that might also satisfy the predicate after a buggy seed (which
    # the EXCLUDE constraint should prevent, but defence in depth).
    # ``STABLE`` because it doesn't write and gives the same answer
    # within a transaction; ``RETURNS NULL ON NULL INPUT`` so callers
    # passing a nullable column don't crash on rows we haven't yet
    # backfilled.
    op.execute("""
        CREATE OR REPLACE FUNCTION assign_era(ts timestamptz)
        RETURNS uuid
        LANGUAGE sql
        STABLE
        RETURNS NULL ON NULL INPUT
        AS $$
            SELECT era_id
            FROM patch_era
            WHERE start_date <= ts
              AND (end_date IS NULL OR end_date > ts)
            ORDER BY start_date DESC
            LIMIT 1
        $$
        """)

    # ``patch_era_window`` view: same data with end_date NULL
    # resolved to a far-future sentinel. Future per-table views (e.g.
    # ``v_match_in_era`` once the matches table lands) can join this
    # without each re-implementing the COALESCE.
    #
    # We use ``'9999-12-31'`` rather than ``'infinity'::timestamptz``
    # because psycopg's default datetime adapter cannot decode an
    # infinite timestamp into Python's ``datetime`` (which caps at
    # year 9999). This sentinel is chosen to be later than any
    # plausible Valorant patch and to round-trip cleanly through the
    # Python layer.
    op.execute("""
        CREATE OR REPLACE VIEW patch_era_window AS
        SELECT
            era_id,
            era_slug,
            patch_version,
            start_date,
            COALESCE(end_date, '9999-12-31 23:59:59+00'::timestamptz) AS end_date_resolved,
            end_date IS NULL AS is_current,
            meta_magnitude,
            is_major_shift,
            created_at
        FROM patch_era
        """)


def downgrade() -> None:
    op.execute("DROP VIEW IF EXISTS patch_era_window")
    op.execute("DROP FUNCTION IF EXISTS assign_era(timestamptz)")
    op.execute("ALTER TABLE patch_era DROP CONSTRAINT IF EXISTS ex_patch_era_no_overlap")
    op.drop_index("ix_patch_era_open_unique", table_name="patch_era")
    op.drop_index("ix_patch_era_start_date", table_name="patch_era")
    op.drop_table("patch_era")
