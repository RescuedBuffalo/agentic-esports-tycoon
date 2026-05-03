# ADR-006: pgvector instead of Qdrant for embeddings

Status: **Accepted** (BUF-28).
Last touched: 2026-05-03.

## Context

The personality model (BUF-25) and the Whisper transcription pipeline
(BUF-21) both need similarity search. The original Phase 0 stack
(BUF-5) ran Qdrant alongside Postgres for that purpose: a dedicated
vector DB next to the relational store.

Once we mapped out the actual queries we want to run, the picture
changed:

* "Find players similar to *aspas*" is rarely the whole question.
  It is "find players similar to *aspas* **and** currently rostered
  on an active T1 team **and** playing duelist".
* "Find a transcript clip relevant to this prompt" expands to
  "...and from a public English-language interview within the
  last 90 days".

Both shapes are cross-entity filters: a vector kNN combined with
predicates that live on the relational rows (`entity.is_active`,
roster windows, role tags, `media.published_at`, language). Qdrant
supports payload filters, but the predicates we actually want come
from JOINs across `entity`, `entity_alias`, the (planned) roster
table, and `media`. Replicating those columns into Qdrant payloads
keeps two stores in sync forever; running them in Postgres makes
the cross-filter a single SQL statement.

## Decision

Embeddings live in Postgres on the same instance BUF-6 already
provisioned, via the `vector` extension.

* The base image is already `pgvector/pgvector:pg16`. The
  initial-schema migration runs `CREATE EXTENSION IF NOT EXISTS
  vector` so a fresh boot is ready for vector tables without a
  second migration cycle.
* Two tables (BUF-28):
  * `personality_embedding(entity_id PK FK, embedding vector(384), model_version, updated_at)`.
  * `transcript_chunk_embedding(id PK, media_id, chunk_idx, chunk_text, embedding vector(384), model_version, updated_at)`
    with `(media_id, chunk_idx)` unique.
* HNSW indexes on the `embedding` column of each table, built with
  default parameters and `vector_cosine_ops`. Recall + latency get
  measured against a 100k-vector load; we revisit the parameters
  if the 10ms-per-query budget breaks.
* Embedding model is local
  `sentence-transformers/all-MiniLM-L6-v2` (384-dim). No API cost,
  runs on the project's 5090. The model identifier is recorded
  per row in `model_version` so a future rotation can re-embed
  in-place without losing the audit trail.
* Qdrant is removed from `docker-compose.yml` and `.env.example`.
  One fewer service to keep healthy in CI and on dev laptops.

## Consequences

**Good.**

* Cross-entity queries are SQL JOINs, not two-step lookups across
  service boundaries. The `similar_players(player_id, k, where_sql)`
  helper composes one query.
* Migrations are linear: vector tables share Alembic history with
  the rest of the schema, so a fresh `alembic upgrade head`
  produces a working store including the embedding side.
* No dual writes. Personality / transcript rows and their
  embeddings live in the same transactional substrate; an
  upstream rollback can't leave the vector store ahead of the
  source-of-truth row.

**Costs to watch.**

* HNSW index builds run on the Postgres instance — a 100k-vector
  rebuild takes Postgres CPU, not a separate Qdrant process. If
  the embedding tables grow large enough that index maintenance
  starts contending with OLTP traffic, the right move is a read
  replica (still pgvector), not a separate vector DB.
* `vector(384)` columns are not free on disk. At 4 bytes per
  float, 384 dims is ~1.5 KB per row before HNSW overhead. We
  budget for this in the storage plan; revisit if a future
  embedding model pushes us above 1024 dimensions.
* Recall is HNSW-default, which is good but not exact. Workloads
  that need exact kNN (small candidate sets, e.g. per-roster
  search) can fall back to a sequential scan with the same
  `<=>` operator.

## Alternatives considered

* **Keep Qdrant.** Rejected: the cross-entity filter shape we
  actually need is a JOIN, and Qdrant payload-filter parity with
  Postgres state is a maintenance burden out of proportion to
  the upside.
* **Embeddings in object storage + brute-force scan.** Fine for
  ad-hoc analysis; no good for the online "similar to X" path
  the tycoon and the LLM both need.
* **Hosted vector DB (Pinecone, Weaviate Cloud).** Adds an
  external dependency and a recurring bill for a problem
  Postgres already solves.

## Related

* BUF-5: docker-compose + dev stack (Qdrant removal).
* BUF-6: Postgres schema v1 (extension landed in 0001).
* BUF-21: Whisper transcription pipeline (writes `media` rows).
* BUF-25: Personality summary extraction.
* BUF-28: Vector tables, HNSW indexes, `similar_players` helper.
