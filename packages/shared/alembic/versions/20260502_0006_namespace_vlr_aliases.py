"""Backfill VLR ``entity_alias.platform_id`` to the namespaced format.

The BUF-8 v2 connector + seed switched from raw VLR numeric ids
(``"9"``, ``"188"``) to entity-type-namespaced ids
(``"player-9"``, ``"team-188"``) so VLR's overlapping per-resource
id spaces don't collide on the
``(platform, platform_id)`` alias unique constraint. Without this
backfill, any database that already had VLR aliases under the raw
format would produce duplicate canonicals on the next scrape: the
exact-alias lookup keys on the new prefixed string and misses the
old row, so the resolver mints a brand-new entity for the same
upstream identity.

The migration is idempotent: the ``platform_id NOT LIKE '%-%'``
filter on upgrade skips rows already in the new format (raw VLR ids
are always digits, so they cannot contain a hyphen). On downgrade
we split on the first hyphen only when the prefix is one of the
entity-type values we own — defensive against any future
non-VLR-alias row that happens to land in the table with a hyphen
in ``platform_id``.

Revision ID: 0006
Revises: 0005
Create Date: 2026-05-02
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # ``entity_type`` lives on the entity row, not on entity_alias —
    # an UPDATE…FROM joins the two and prefixes the alias's
    # ``platform_id`` with the canonical's type. ``::text`` is needed
    # because ``entity_type`` is the Postgres ENUM and ``||`` does not
    # implicitly cast it. The ``NOT LIKE '%-%'`` predicate keeps the
    # migration idempotent: re-running it after a successful upgrade
    # is a no-op because every previously-migrated row contains a
    # hyphen.
    op.execute("""
        UPDATE entity_alias AS ea
        SET platform_id = (e.entity_type::text || '-' || ea.platform_id)
        FROM entity AS e
        WHERE ea.canonical_id = e.canonical_id
          AND ea.platform = 'vlr'
          AND ea.platform_id NOT LIKE '%-%'
    """)


def downgrade() -> None:
    # Strip the entity-type prefix only when the prefix matches one of
    # the values we own. This guards against any future non-VLR
    # alias that happens to have a hyphen — though the WHERE clause
    # already scopes to ``platform = 'vlr'``, the explicit prefix
    # whitelist keeps the migration round-trippable even if some
    # other tooling later inserts hyphenated VLR aliases for an
    # unrelated reason.
    op.execute("""
        UPDATE entity_alias
        SET platform_id = SUBSTRING(platform_id FROM POSITION('-' IN platform_id) + 1)
        WHERE platform = 'vlr'
          AND SPLIT_PART(platform_id, '-', 1) IN ('player', 'team', 'coach', 'tournament')
    """)
