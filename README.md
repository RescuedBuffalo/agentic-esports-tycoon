# agentic-esports-tycoon

A simulation sandbox for an esports-management tycoon game where the player and the
non-player roles (scouts, coaches, players, sponsors, fans) are driven by language-model
and reinforcement-learning agents.

This repository is in the **planning phase**. Useful entry points:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system overview.
- [`docs/design/observation-action-space.md`](docs/design/observation-action-space.md) — agent I/O contract.
- [`docs/design/world-model-state.md`](docs/design/world-model-state.md) — provisional world-model state.
- [`docs/setup/api-and-hardware-checklist.md`](docs/setup/api-and-hardware-checklist.md) — what we need to procure before training.
- [`schemas/`](schemas/) — YAML schemas for configs and game data.
- [`src/aet/`](src/aet/) — Python package (currently: RNG tree + event schemas).

## Quick start (developers)

```bash
python -m pip install -e '.[dev]'
pytest
```
