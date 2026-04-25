# agentic-esports-tycoon

A simulation sandbox for an esports-management tycoon game where the player and the
non-player roles (scouts, coaches, players, sponsors, fans) are driven by language-model
and reinforcement-learning agents.

This repository is in the **planning phase**. Useful entry points:

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system overview, three-layer breakdown.
- [`docs/design/observation-action-space.md`](docs/design/observation-action-space.md) — match-engine obs/action contract (BUF-79).
- [`docs/design/world-model-state.md`](docs/design/world-model-state.md) — neural world-model latent shape (BUF-81).
- [`docs/design/tycoon-obs-action-space.md`](docs/design/tycoon-obs-action-space.md) — management-layer obs/action (companion).
- [`docs/design/tycoon-state-factorisation.md`](docs/design/tycoon-state-factorisation.md) — management-layer state (companion).
- [`docs/setup/api-and-hardware-checklist.md`](docs/setup/api-and-hardware-checklist.md) — what we need to procure before training (BUF-80).
- [`config/`](config/) — data-pipeline configs (BUF-75).
- [`data/`](data/) — canonical Valorant content the sim consumes (BUF-76).
- [`src/esports_sim/`](src/esports_sim/) — Python package (currently: RNG tree + event schemas).

## Quick start (developers)

```bash
python -m pip install -e '.[dev]'
pytest
```
