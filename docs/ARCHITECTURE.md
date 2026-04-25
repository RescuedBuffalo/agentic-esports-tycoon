# Architecture

Status: **Planning / draft**. Owners: core team. Last touched: BUF-74.

This document describes the intended shape of the system at the end of the planning
phase. Anything marked _provisional_ is a working assumption that we expect to revise
before the first vertical slice ships.

## 1. Goals

1. Run a deterministic, replayable simulation of an esports organisation, season after
   season, season after season.
2. Let language-model and RL agents control roles inside that simulation (the player /
   GM, scouts, coaches, players, sponsors, fans, casters).
3. Keep configs and game data in human-editable YAML so designers can iterate without
   touching code.
4. Make every run reproducible from a single root seed, even when multiple agents pull
   randomness concurrently.

## 2. Non-goals (for now)

- Real-time graphical client. The simulation is headless; visualisation is a separate
  surface.
- Multiplayer / networked play.
- Modelling specific real-world esports titles. We model a generic team-vs-team title
  (`Title`) configurable through game data.

## 3. Top-level components

```
                    +--------------------------+
                    |   Config (YAML, BUF-75)  |
                    +-----------+--------------+
                                |
                                v
+----------------+      +-------+--------+      +---------------------+
|  Game data     +----->+   Simulation   +<-----+  RngTree (BUF-77)   |
|  (YAML, BUF-76)|      |     core       |      +---------------------+
+----------------+      +---+--------+---+
                            |        |
              events (BUF-78)|        | observations / actions (BUF-79)
                            v        v
                  +---------+--+  +--+-----------+
                  |  Event log |  |  Agent host  |
                  | (immutable)|  | (LLM + RL)   |
                  +------------+  +------+-------+
                                         |
                                         v
                                +--------+--------+
                                |  World model    |
                                |  state (BUF-81) |
                                +-----------------+
```

### 3.1 Simulation core (`src/aet/sim/`, not yet implemented)

- Tick-based, single-threaded loop. One in-game day per tick by default; configurable.
- Pure functions over an immutable `WorldState` snapshot plus an `RngTree` cursor.
- Emits a stream of `Event` objects (BUF-78). Events are the canonical source of truth;
  the world state is a fold over them.

### 3.2 Agent host (`src/aet/agents/`, not yet implemented)

- Subscribes to observation views (BUF-79) and writes back actions through a typed
  action API.
- Supports multiple backends behind a common interface: scripted heuristics, RL policies
  (PPO baseline), and LLM-driven agents (Anthropic / OpenAI / local).
- Each agent gets its own `RngTree` child so that swapping a backend does not perturb
  unrelated agents' randomness.

### 3.3 World model (BUF-81)

- A learned, compact state representation used by planning agents.
- Trained offline from event logs; consumed online through the same observation API as
  the rest of the agents. The provisional schema lives in
  [`docs/design/world-model-state.md`](design/world-model-state.md).

### 3.4 Event log

- Append-only. Each event carries `(tick, source, kind, payload, rng_path)`.
- Snapshots are taken every N ticks (configurable) so that long replays don't have to
  re-fold from genesis.

### 3.5 RNG tree (BUF-77)

- Hierarchical splittable RNG (NumPy `SeedSequence` under the hood).
- Each subsystem and each agent gets a deterministic, named child stream so that
  parallel reads do not race for entropy.

## 4. Data flow per tick

1. Sim core advances its clock by one tick.
2. For each scheduled job (matches, training, contract negotiations, fan reactions):
   1. Pull a child RNG from the tree using a stable path.
   2. Build the relevant observation view.
   3. Ask the bound agent for an action (may be a no-op).
   4. Validate and apply the action; emit one or more events.
3. The reducer folds events into the new world state.
4. Snapshot if `tick % snapshot_every == 0`.

## 5. Determinism contract

A run is reproducible when **all** of the following hold:

- The root seed is fixed.
- The game data and config files are byte-identical (we hash them into the run header).
- Every agent's backend is either deterministic or seeded from the RNG tree.
- Wall-clock time is never read by sim code (use `tick` instead).

Any deviation must be flagged in the run header so replays can refuse to run silently.

## 6. Module map (target)

```
src/aet/
    sim/         # tick loop, reducers, scheduler                (planned)
    agents/      # agent host + backends                         (planned)
    rng/         # RngTree                                       (BUF-77, this PR)
    schemas/
        events.py    # Pydantic event union                      (BUF-78, this PR)
        config.py    # loaders for schemas/config/*.yaml         (planned)
        gamedata.py  # loaders for schemas/game/*.yaml           (planned)
    obs/         # observation builders                          (planned)
    actions/     # action validators and appliers                (planned)
    wm/          # world model train + serve                     (planned)
schemas/
    config/      # config schemas + example configs              (BUF-75, this PR)
    game/        # game data schemas + seed data                 (BUF-76, this PR)
docs/
    ARCHITECTURE.md
    design/      # design notes for non-trivial subsystems
    setup/       # operator-facing checklists
tests/
```

## 7. Open questions tracked elsewhere

- Observation / action space shape — see BUF-79.
- World-model state shape — see BUF-81.
- Hardware and API budget — see BUF-80.
