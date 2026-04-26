# Architecture

Status: **Planning / draft**. Owners: core team. Last touched: BUF-74.

This document describes the intended shape of the system at the end of the
planning phase. Anything marked _provisional_ is a working assumption that we
expect to revise before the first vertical slice ships.

## 1. Goals

1. Run a deterministic, replayable simulation of a Valorant esports
   organisation across multiple seasons.
2. Inside that simulation, simulate individual matches at tile resolution so
   on-server agents can be trained with RL.
3. Let language-model and RL agents control roles at both layers — the GM /
   scout / coach / sponsor / fan layer above, and the per-agent in-match
   layer below.
4. Keep configs and game data in human-editable YAML so designers can
   iterate without touching code.
5. Make every run reproducible from a single root seed, even when many
   agents and pipeline stages pull randomness concurrently.

## 2. Non-goals (for now)

- Real-time graphical client. The simulation is headless; visualisation is a
  separate surface.
- Multiplayer / networked play.
- Modelling other esports titles. The match engine is a 2D Valorant tactical
  simulator; see §3 for the generic-vs-Valorant scoping note.

## 3. Scoping: generic engine vs Valorant-native

It would be cheap to claim the system is title-agnostic. It isn't. To avoid
ambiguity:

- The **data pipeline** (BUF-75 configs, BUF-76 data) is **Valorant-native**:
  VLR scrapers, Riot patch eras, agents / weapons / maps are all Valorant
  concepts. Replacing the title would mean replacing this layer wholesale.
- The **match engine** (BUF-30) is **Valorant-shaped**: 5v5, ability slots
  q/e/c/x with charges, credit economy, spike plant + defuse, attacker /
  defender side swap. It does not contain Valorant-specific magic numbers in
  code — those come from `data/` — but the rule schema is Valorant.
- The **tycoon / management layer** (contracts, sponsors, tournaments) is
  **mostly title-agnostic** in shape, though current data is Valorant.
  Repurposing it for a different title is plausible; doing so for the match
  engine is not.

The earlier draft of this document said "a generic team-vs-team title
configurable through game data." That overstated the engine layer's
flexibility. Treat the system as **Valorant-native by design at the data and
match layers, title-agnostic in shape only at the tycoon layer**.

## 4. Top-level layers

```
+----------------------------------------------------------+
|  Data pipeline (BUF-75, BUF-76)                          |
|  VLR / Riot / Liquipedia -> entity graph -> data/        |
+--------------------------+-------------------------------+
                           |  reads at boot
                           v
+----------------------------------------------------------+
|  Tycoon / management sim                                 |
|  Tick = 1 in-game day. Tycoon obs/action: see            |
|  docs/design/tycoon-obs-action-space.md                  |
+--------------------------+-------------------------------+
                           |  invokes per scheduled match
                           v
+----------------------------------------------------------+
|  2D tactical match engine (BUF-30)                       |
|  Tick = ~100 ms. PettingZoo wrapper (BUF-34) exposes     |
|  the per-agent obs/action contract from BUF-79.          |
|  World model (BUF-81) consumes (obs, action, latent)     |
|  trajectories produced here.                             |
+----------------------------------------------------------+
```

`RngTree` (BUF-77) and the event log are cross-cutting: every layer pulls
named subtrees and emits typed events (BUF-78).

## 5. Components

### 5.1 Data pipeline (`config/`, `data/`)

- DAG of stages defined in `config/scheduler.yaml`. Each stage reads exactly
  one config file (entity resolution, derivation, relationships, validation,
  graph export). See `config/README.md`.
- Output is the canonical `data/` tree the sim loads: `attributes.yaml`,
  `agents/`, `weapons/`, `maps/`. Manual corrections live under
  `data/_overrides/` and survive re-export.
- Patch-era boundaries (`config/patch_eras.yaml`) are hand-curated and
  trigger downstream re-aggregation when a new era ships.

### 5.2 Tycoon sim (`src/esports_sim/sim/`, planned)

- Tick-based, single-threaded. One in-game day per tick by default.
- Pure functions over an immutable `WorldState` snapshot plus an `RngTree`
  cursor. Event log is the source of truth; world state is a fold over it.
- Tycoon obs/action contract is defined in
  `docs/design/tycoon-obs-action-space.md`.

### 5.3 2D tactical match engine (`src/esports_sim/match/`, BUF-30, planned)

- Tile-grid sim, ~100 ms per step. Simulates one Valorant round at a time;
  rounds compose into maps and maps into matches.
- Reads agents / weapons / maps from `data/`. Magic numbers live there, not
  in code.
- PettingZoo `parallel_env` wrapper (BUF-34) exposes the per-agent
  observation and action space from BUF-79
  (`docs/design/observation-action-space.md`).

### 5.4 World model (`src/esports_sim/wm/`, BUF-81, planned)

- Hybrid RSSM latent: deterministic GRU state + categorical stochastic
  state. See `docs/design/world-model-state.md` for the shape and the
  implications for BUF-38 replay storage.
- Trained offline from match-engine trajectories; consumed online by
  policies that want imagination rollouts.

### 5.5 Agent host (`src/esports_sim/agents/`, planned)

- Subscribes to obs views and writes back actions through typed APIs at both
  layers.
- Backends: scripted heuristics, RL policies (PPO baseline at the match
  layer; TBD at the tycoon layer), and LLM-driven agents at the tycoon
  layer.
- Each agent gets its own `RngTree` child so swapping a backend does not
  perturb unrelated agents' randomness.

### 5.6 Event log (BUF-78)

- Append-only. Each event carries `(tick, source, kind, payload, rng_path)`
  — see `src/esports_sim/schemas/events.py`.
- Snapshots are taken every N ticks (configurable) so long replays don't
  re-fold from genesis.

### 5.7 RNG tree (BUF-77)

- Hierarchical splittable RNG. `src/esports_sim/rng/tree.py`.
- Each subsystem and each agent gets a deterministic, named child stream so
  parallel reads do not race for entropy.

## 6. Determinism contract

A run is reproducible when **all** of the following hold:

- The root seed is fixed.
- The `data/` tree is byte-identical (we hash it into the run header) and
  was produced by a known-good pipeline run.
- Every agent's backend is either deterministic or seeded from the
  `RngTree`.
- Wall-clock time is never read by sim code (use `tick` instead).

Any deviation must be flagged in the run header so replays can refuse to run
silently.

## 7. Module map (target)

```
src/esports_sim/
    sim/         # tycoon tick loop, reducers, scheduler         (planned)
    match/       # 2D tactical match engine                      (BUF-30)
    agents/      # agent host + backends                         (planned)
    wm/          # world-model train + serve                     (BUF-81)
    obs/         # observation builders for both layers          (planned)
    actions/     # action validators and appliers                (planned)
    rng/         # RngTree                                       (BUF-77, this PR)
    schemas/
        events.py    # Pydantic event union                      (BUF-78, this PR)
        config.py    # loaders for config/*.yaml                 (planned)
        gamedata.py  # loaders for data/                         (planned)
config/      # data-pipeline configs                             (BUF-75, this PR)
data/        # canonical Valorant content                        (BUF-76, this PR)
docs/
    ARCHITECTURE.md
    design/      # design notes for non-trivial subsystems
    setup/       # operator-facing checklists
tests/
```

## 8. Open questions tracked elsewhere

- Match-engine obs / action shape — see BUF-79.
- World-model latent shape and replay-buffer implications — see BUF-81 and
  BUF-38.
- Tycoon obs / action shape — see `docs/design/tycoon-obs-action-space.md`.
- Hardware and API budget — see BUF-80.
