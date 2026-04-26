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
