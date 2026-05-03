# agentic-esports-tycoon

A simulation sandbox for an esports-management tycoon game where the player and
the non-player roles (scouts, coaches, players, sponsors, fans) are driven by
language-model and reinforcement-learning agents.

## Quick start

One command spins up the full local stack:

```bash
make dev
```

That syncs the [`uv`](https://docs.astral.sh/uv/) workspace and brings up
Postgres 16 and Qdrant via docker-compose. Run tests with:

```bash
uv run pytest
```

Copy `.env.example` to `.env` before running anything that needs credentials.

To apply the BUF-6 schema (entity, alias, staging, raw store) to a fresh
Postgres:

```bash
make migrate    # uses $DATABASE_URL; defaults to the dev-compose Postgres
```

Integration tests against Postgres are skipped unless `TEST_DATABASE_URL` is
set:

```bash
TEST_DATABASE_URL=postgresql+psycopg://nexus:nexus@localhost:5432/nexus uv run pytest
```

## Repo layout

```
packages/
  shared/          # Pydantic models, RNG tree, typed clients (esports-sim-shared)
services/
  data_pipeline/   # VLR / Riot / Liquipedia ingest -> data/
  ecosystem/       # tycoon / management sim
  match_wm/        # 2D tactical match engine + neural world model
apps/
  game/            # player-facing client (empty stub)
config/            # data-pipeline configs (BUF-75)
data/              # canonical Valorant content (BUF-76)
docs/              # ARCHITECTURE + design notes
```

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/):
the root `pyproject.toml` declares the members and pins Python 3.12, each
package has its own `pyproject.toml`, and `uv sync` resolves them together.

## Experiment registry (BUF-69)

Every training run, graph snapshot, and world-model pretrain registers
itself in `state/registry.db` and gets a deterministic `run_id`. All
artifacts live under `runs/{run_id}/`. Downstream code resolves paths
via `Registry.get(run_id)` — nothing hardcodes `runs/...`.

```bash
# Register a new run (idempotent: same config + data → same run_id)
uv run nexus run register --kind=graph-snapshot --config=configs/graph/era_7.09.yaml

# List runs (filter by --kind / --status)
uv run nexus run ls --kind=rl-train

# Inspect one
uv run nexus run show <run_id>

# Mark terminal
uv run nexus run finalize <run_id> --status=completed --notes="ok"
```

Public Python API: `from esports_sim.registry import Registry, RunStatus`.

## Patch-era partitioning (BUF-13)

Every record that carries a timestamp also carries an era context.
`patch_era` is the temporal-partition table; `assign_era(ts)` (Python
helper or the matching Postgres SQL function) maps a timestamp onto
its era. Aggregations across an era marked `is_major_shift=True`
raise `TemporalBleedError` — the runtime guard for the System 04
"no cross-era feature aggregation" rule.

Bootstrap the historical 2020→present timeline once per environment:

```bash
DATABASE_URL=postgresql+psycopg://nexus:nexus@localhost:5432/nexus \
    uv run python -m data_pipeline.seeds patch-eras
```

Steady-state era transitions (called from BUF-24's patch-intent
extractor when a new patch ships):

```python
from esports_sim.eras import roll_era

closed, opened = roll_era(
    session,
    new_slug="e2026_02",
    new_patch_version="11.05",
    boundary_at=patch_release_ts,
    is_major_shift=True,
    meta_magnitude=0.85,
)
```

The close + open pair is atomic (single savepoint, half-open ranges,
EXCLUDE constraint at the DB layer) — no gap, no overlap.

## Patch-intent extraction (BUF-24)

Every patch ingested by `data_pipeline.connectors.playvalorant` is
classified by `esports_sim.patch_intent.extract_patch_intent` — primary
intent, agents/maps affected, expected pickrate shifts, predicted
community controversy. Results land in the `patch_intent` table keyed on
`(patch_note_id, prompt_version)`; bumping the prompt produces a new
row instead of overwriting the older classification.

The Phase 0 scheduler hook (`extract_intent_for_pending`) enumerates
patch notes that don't yet have an intent for the current prompt and
runs the extractor against each. Idempotent — a re-run after every
patch is classified is a no-op:

```python
from esports_sim.budget import Governor, default_caps
from esports_sim.patch_intent import extract_intent_for_pending

# ``source`` defaults to "playvalorant" — the prompt rubric is
# Valorant-specific so the hook scopes to that connector's rows.
stats = extract_intent_for_pending(session, governor=Governor(caps=default_caps()))
# stats.inserted, stats.updated, stats.skipped_existing, stats.budget_exhausted
```

System prompt is wrapped in `cache_control` (1 h TTL) so a corpus-wide
re-classification amortises the rubric across patches. One patch costs
well under the BUF-22 `patch_intent` per-purpose soft cap ($3/wk).

## Claude API budget governor (BUF-22)

Every Claude call goes through `esports_sim.budget.claude_call`. The governor
enforces a $30/week hard cap (with $10 buffer under the $40 weekly budget) and
per-purpose soft caps, logs every call to a SQLite ledger, and prices usage
against a versioned per-model table. Inspect spend with:

```bash
uv run nexus budget report
```

See `packages/shared/src/esports_sim/budget/__init__.py` for the public API.

## Useful entry points

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system overview, three-layer breakdown.
- [`docs/design/observation-action-space.md`](docs/design/observation-action-space.md) — match-engine obs/action contract (BUF-79).
- [`docs/design/world-model-state.md`](docs/design/world-model-state.md) — neural world-model latent shape (BUF-81).
- [`docs/design/tycoon-obs-action-space.md`](docs/design/tycoon-obs-action-space.md) — management-layer obs/action.
- [`docs/design/tycoon-state-factorisation.md`](docs/design/tycoon-state-factorisation.md) — management-layer state.
- [`docs/setup/api-and-hardware-checklist.md`](docs/setup/api-and-hardware-checklist.md) — what we need to procure before training (BUF-80).

## Common tasks

| Command           | What it does                                            |
| ----------------- | ------------------------------------------------------- |
| `make dev`        | Sync workspace + bring up Postgres & Qdrant.            |
| `make up`         | Bring up the data plane only.                           |
| `make down`       | Stop the data plane (volumes preserved).                |
| `make sync`       | Resolve and install the uv workspace.                   |
| `make migrate`    | Apply Alembic migrations against `$DATABASE_URL`.       |
| `make test`       | Run pytest across all workspace members.                |
| `make lint`       | Run ruff + black --check.                               |
| `make typecheck`  | Run mypy.                                               |
| `make ci`         | Lint + typecheck + test (mirrors CI).                   |
| `make precommit`  | Run all configured pre-commit hooks on every file.      |
| `make clean`      | Tear down volumes and caches.                           |

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/) 0.11+
- Docker / Docker Compose v2
- GNU `make`
