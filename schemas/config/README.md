# Config schemas

JSON-Schema-flavoured YAML for everything that controls **how** the simulation runs
(as opposed to game data, which controls **what** lives inside the simulation — see
`../game/`).

| File | Loaded as | Purpose |
| --- | --- | --- |
| `run.schema.yaml` | `RunConfig` | Top-level run: seed, ticks, snapshotting, IO paths. |
| `sim.schema.yaml` | `SimConfig` | Sim-loop knobs: tick length, scheduler caps, economy. |
| `agent.schema.yaml` | `AgentConfig` | Per-role agent backend bindings + budgets. |

A run merges these three files; later files override earlier keys. See BUF-75 for the
issue this closes.
