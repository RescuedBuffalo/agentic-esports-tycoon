"""kNN-with-cross-entity-filter helpers for the BUF-28 embedding store.

The headline query the BUF-28 acceptance criteria call out is "find
players similar to X *and* on an active T1 roster *and* plays
duelist". That's a JOIN against the personality embedding table plus
a free-form filter on rows the relational schema already exposes —
exactly the shape ADR-006 says lives in Postgres rather than a
separate vector DB.

Public surface is one helper:

* :func:`similar_players` — kNN by cosine distance with an optional
  caller-supplied SQL filter clause. Resolves a string handle (the
  example in the BUF-28 issue is ``similar_players("aspas")``) via
  ``entity_alias`` so call sites don't have to thread canonical UUIDs
  by hand.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Session


class SimilarPlayerNotFoundError(LookupError):
    """The provided handle did not resolve to a personality-embedded entity.

    Three distinct failures share this exception type because they
    are interchangeable from the caller's perspective: the handle
    didn't match any alias, matched multiple, or matched an entity
    that has no personality embedding yet. The error message
    distinguishes them.
    """


@dataclass(frozen=True)
class SimilarPlayer:
    """One row of :func:`similar_players`'s result set.

    ``entity_id`` is the canonical id; ``distance`` is the cosine
    distance the index reported (0 for identical, up to 2 for
    opposite). Sorted ascending in the result list.
    """

    entity_id: uuid.UUID
    distance: float


def _resolve_target(session: Session, target: uuid.UUID | str) -> uuid.UUID:
    """Return the canonical id for ``target``.

    UUIDs are validated to point at a row whose ``entity_type =
    'player'``; strings resolve via ``entity_alias.platform_name``
    (case-insensitive exact match) restricted to ``entity_type =
    'player'``. Raises :class:`SimilarPlayerNotFoundError` on
    no-match, ambiguous-match, or wrong-entity-type because each
    is a programming error the caller should see, not a silent
    zero-result or — worse — a result that ranks players against
    a team-shaped vector.
    """
    if isinstance(target, uuid.UUID):
        # Entity-type guard: similar_players claims to compare
        # players, so a caller who passes a team / coach /
        # tournament UUID with a personality embedding gets a
        # structured error rather than a kNN result that mixes
        # vector spaces. Distinguish "no such row" from "wrong
        # type" so the message points at the actual problem.
        row = session.execute(
            text("SELECT entity_type FROM entity WHERE canonical_id = :id"),
            {"id": target},
        ).first()
        if row is None:
            raise SimilarPlayerNotFoundError(
                f"entity {target} does not exist; pass a canonical id from `entity`"
            )
        if row[0] != "player":
            raise SimilarPlayerNotFoundError(
                f"entity {target} is a {row[0]!r}, not a player. "
                "similar_players is only defined over player-typed entities."
            )
        return target

    rows = session.execute(
        text("""
            SELECT DISTINCT ea.canonical_id
            FROM entity_alias ea
            JOIN entity e ON e.canonical_id = ea.canonical_id
            WHERE LOWER(ea.platform_name) = LOWER(:handle)
              AND e.entity_type = 'player'
            """),
        {"handle": target},
    ).all()
    if not rows:
        raise SimilarPlayerNotFoundError(
            f"no player alias matched {target!r}; check entity_alias.platform_name"
        )
    if len(rows) > 1:
        ids = ", ".join(str(r[0]) for r in rows)
        raise SimilarPlayerNotFoundError(
            f"alias {target!r} resolves to multiple canonical ids: {ids}. "
            "Pass the UUID explicitly to disambiguate."
        )
    # Cast: SQLAlchemy types Row[Any] columns as Any until the bindparam
    # is annotated with a concrete typed column (which would mean
    # importing the EntityAlias model and re-doing the query through
    # the ORM). The runtime value is a UUID because the column is
    # ``UUID(as_uuid=True)``.
    canonical_id: uuid.UUID = rows[0][0]
    return canonical_id


def similar_players(
    session: Session,
    target: uuid.UUID | str,
    *,
    k: int = 10,
    where_sql: str | None = None,
) -> list[SimilarPlayer]:
    """Return the ``k`` players most similar to ``target`` by personality.

    ``target`` is either a canonical UUID or a player alias string
    (``"aspas"``); a string resolves via ``entity_alias.platform_name``
    (see :func:`_resolve_target`). The query plans as an HNSW kNN on
    ``personality_embedding`` joined to ``entity`` so the optional
    ``where_sql`` filter can reference any column on ``entity`` (and
    any future table the caller joins in by extending the predicate
    — until those tables exist, BUF-28's acceptance example
    ``role='duelist' AND active=true`` reads against the columns the
    later issues will land).

    Trust boundary: ``where_sql`` is interpolated literally into the
    final SQL. This helper is developer-facing — the LLM call sites
    that build prompts pass it canned predicates from the codebase,
    not user input. If a caller ever does want to bind end-user
    text, the responsibility to bind-param it is theirs; this helper
    deliberately does not try to whitelist columns because the whole
    point of the cross-entity-filter design is that the filter
    expands as the relational schema grows.

    The target row is excluded from the result (a player is always
    most similar to themselves; including them would crowd out the
    k slots).

    Model-version isolation: cosine distance is only meaningful
    between vectors produced by the same embedder. The query
    constrains neighbors to rows whose ``model_version`` matches
    the target's, so a partial re-embed rollout (some rows on the
    new model, some on the old) returns same-space neighbors only
    rather than silently mixing incompatible vectors. Once a
    rollout completes, every row shares the same model_version and
    the filter becomes a no-op.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    target_id = _resolve_target(session, target)

    # Use an alias for ``entity`` so callers can write predicates like
    # ``e.is_active = true`` without having to know the table's
    # original name. ``pe`` is the personality embedding alias.
    where_clause = ""
    if where_sql is not None:
        where_clause = f"AND ({where_sql})"

    sql = text(f"""
        WITH target AS (
            SELECT embedding, model_version
            FROM personality_embedding
            WHERE entity_id = :target_id
        )
        SELECT pe.entity_id, (pe.embedding <=> target.embedding)::float8 AS distance
        FROM personality_embedding pe
        JOIN entity e ON e.canonical_id = pe.entity_id
        CROSS JOIN target
        WHERE pe.entity_id <> :target_id
          AND e.entity_type = 'player'
          AND pe.model_version = target.model_version
          {where_clause}
        ORDER BY pe.embedding <=> target.embedding
        LIMIT :k
        """).bindparams(
        # ``target_id`` appears twice in the SQL; one bindparam covers both.
        bindparam("target_id", type_=PgUUID(as_uuid=True)),
        bindparam("k"),
    )

    rows = session.execute(sql, {"target_id": target_id, "k": k}).all()
    if not rows:
        # No match could mean: target has no personality embedding,
        # or every other player was filtered out by ``where_sql``.
        # Distinguish the first case so the call site can surface a
        # useful diagnostic instead of a silent empty list.
        has_target = session.execute(
            text("SELECT 1 FROM personality_embedding WHERE entity_id = :id"),
            {"id": target_id},
        ).first()
        if has_target is None:
            raise SimilarPlayerNotFoundError(
                f"entity {target_id} has no personality embedding yet; "
                "BUF-25 must run before similar_players can return results"
            )
    return [SimilarPlayer(entity_id=row[0], distance=float(row[1])) for row in rows]
