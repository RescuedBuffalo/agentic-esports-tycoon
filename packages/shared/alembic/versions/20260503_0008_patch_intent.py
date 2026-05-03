"""Add patch_intent table for the BUF-24 patch-intent extractor (System 06).

System 06 in the Systems-spec: every patch is classified into a structured
intent record — predicted effects, controversy, agents/maps touched. The
extractor (``esports_sim.patch_intent.extract_patch_intent``) calls Claude
with a cached system prompt and persists the result here so downstream
ecosystem agents can read it without paying for the LLM call.

Schema rationale:

* ``patch_note_id`` FKs straight to ``patch_note.id`` with ``ON DELETE
  CASCADE`` — an intent record without its source patch note is
  uninterpretable; if the patch note is rotated out, the derived
  classification follows. The patch_note row is the durable identity.
* ``prompt_version`` is part of the dedup key. A re-run with the same
  prompt is idempotent (UPSERT path), but bumping the prompt — for a new
  spec or a tighter rubric — produces a new row rather than overwriting
  the previous classification, so we can backtest the rubric change.
* The structured fields mirror the Pydantic ``PatchIntentResult`` shape
  used by the extractor; the JSONB columns (``agents_affected``,
  ``maps_affected``, ``expected_pickrate_shifts``) carry typed Python
  lists/dicts so a downstream feature extractor can join on them
  directly without re-parsing prose.
* ``reasoning`` is unbounded (``Text``) — patch-intent prompts ask the
  model to justify each call so a human spot-check can audit a
  surprising classification.
* ``model`` and ``confidence`` make the row self-describing for
  later trustworthiness analysis (which model produced this, how
  certain was it). ``usd_cost`` and ``input_tokens`` / ``output_tokens``
  let the budget retrospective tie a row back to its ledger entry.

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "patch_intent",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "patch_note_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "patch_note.id",
                ondelete="CASCADE",
                name="fk_patch_intent_patch_note_id_patch_note",
            ),
            nullable=False,
        ),
        # Identifies the prompt rubric used. Bump when the spec changes
        # so a new row lands rather than UPSERTing on top of the older
        # classification.
        sa.Column("prompt_version", sa.String(32), nullable=False),
        # Self-describing row: which Claude model produced this output,
        # for trustworthiness post-mortems and retrospective re-pricing.
        sa.Column("model", sa.String(64), nullable=False),
        # The five required structured fields from the Systems-spec.
        sa.Column("primary_intent", sa.String(64), nullable=False),
        sa.Column("pro_play_driven_score", sa.Float(), nullable=False),
        # JSONB lists keep a downstream feature extractor honest — a
        # GIN index on these columns is cheap to add later if a query
        # pattern needs ``WHERE ? = ANY(agents_affected)``.
        sa.Column("agents_affected", postgresql.JSONB, nullable=False),
        sa.Column("maps_affected", postgresql.JSONB, nullable=False),
        sa.Column("econ_changed", sa.Boolean(), nullable=False),
        # List of objects: ``{"agent": "Chamber", "direction": "down",
        # "magnitude": "large"}``. Stored as JSONB so the schema can
        # evolve (add ``rationale`` per-shift, etc.) without an alembic
        # round-trip.
        sa.Column("expected_pickrate_shifts", postgresql.JSONB, nullable=False),
        sa.Column("community_controversy_predicted", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        # Free-form justification — the model's chain-of-thought
        # rationale for the classification.
        sa.Column("reasoning", sa.Text(), nullable=False),
        # Cost / usage attribution for the budget retrospective.
        sa.Column("input_tokens", sa.Integer(), nullable=False),
        sa.Column("output_tokens", sa.Integer(), nullable=False),
        sa.Column("usd_cost", sa.Float(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # ``(patch_note_id, prompt_version)`` is the dedup key. Same
        # prompt re-run on the same patch is a no-op UPSERT; a bump of
        # ``prompt_version`` produces a new row so the older
        # classification is auditable.
        sa.UniqueConstraint(
            "patch_note_id",
            "prompt_version",
            name="uq_patch_intent_patch_note_id_prompt_version",
        ),
        # Bound the 0..1 floats at the DB layer — a buggy model output
        # (e.g. ``"confidence": 1.5``) should fail the insert, not
        # silently corrupt every downstream weighted aggregate.
        sa.CheckConstraint(
            "pro_play_driven_score >= 0 AND pro_play_driven_score <= 1",
            name="ck_patch_intent_pro_play_driven_score_range",
        ),
        sa.CheckConstraint(
            "community_controversy_predicted >= 0 AND community_controversy_predicted <= 1",
            name="ck_patch_intent_community_controversy_range",
        ),
        sa.CheckConstraint(
            "confidence >= 0 AND confidence <= 1",
            name="ck_patch_intent_confidence_range",
        ),
    )
    # The "find patches without an intent yet" query is what the
    # scheduler hook runs every pass; index the FK so it stays O(log n)
    # against ``patch_note`` once the table grows.
    op.create_index(
        "ix_patch_intent_patch_note_id",
        "patch_intent",
        ["patch_note_id"],
    )


def downgrade() -> None:
    op.drop_index("ix_patch_intent_patch_note_id", table_name="patch_intent")
    op.drop_table("patch_intent")
