# Observation / action space — 2D tactical match engine

Status: **Planning / draft**. Closes BUF-79. Unblocks BUF-30 (2D tactical
match engine) and BUF-34 (PettingZoo wrapper). Complements
[`tycoon-obs-action-space.md`](tycoon-obs-action-space.md), which covers the
management layer.

This document fixes the per-agent observation and action vocabulary for the
match engine — the layer that simulates a single Valorant round at tile
resolution. Anything flagged _provisional_ here will need a follow-up before
BUF-34 ships.

## 1. Scope

- One match = one map = up to 24 rounds (best-of-13 or first-to-13 with overtime).
- Each round simulates 10 agents (5 attackers + 5 defenders).
- Tile-grid world; each tile is 64 × 64 game units; map is up to 80 × 80 tiles.
- One sim step ≈ 100 ms (configurable). RL agents act every step.

PettingZoo `parallel_env` API: every step the env returns a dict
`{agent_id: obs}` and accepts a dict `{agent_id: action}`. Dead agents continue
to receive a zero observation and a single legal `no_op` action so the env
shape stays fixed across the round.

## 2. Per-agent observation

The view is **first-person and incomplete**: each agent sees only what their
ray-cast field of view reveals plus information their team has shared
(spotted enemies, ability cues). The observation is a `gymnasium.spaces.Dict`
with the following keys.

### 2.1 Local grid patch

A square crop centred on the agent's tile.

| Field | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `grid_terrain` | `(P, P)` | `uint8` | Tile class: empty, wall, half-cover, smoke, plant-zone, teleporter. |
| `grid_visible` | `(P, P)` | `bool` | True where the agent currently has line of sight. |
| `grid_team` | `(P, P)` | `int8` | Per-tile occupancy: -1 self, 0 empty, 1..4 teammates, 5..9 enemies (only on visible tiles). |
| `grid_audio` | `(P, P)` | `float32` | Decaying audio cue map (footsteps, gunshots, ability casts) clamped to `[0, 1]`. |

`P` is the patch radius in tiles; provisional default `P = 21` (10 tiles in
each direction). Pad with zeros at the map edge.

### 2.2 LOS raycast

A polar sample of the visibility from the agent's eye. Decouples vision from
the grid so the policy can learn long-range engagement geometry without
needing a wide patch.

| Field | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `los_distance` | `(R,)` | `float32` | Hit distance per ray, normalised by max sight range. |
| `los_hit_kind` | `(R,)` | `int8` | 0 wall, 1 smoke, 2 enemy, 3 teammate, 4 spike, 5 nothing. |
| `los_hit_health` | `(R,)` | `float32` | Hit entity HP fraction; 0 if not an entity. |

Provisional `R = 64` rays evenly spaced across a 103° horizontal FOV plus 8
peripheral rays for hearing-direction cues.

### 2.3 Self state

| Field | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `hp` | `(1,)` | `float32` | `[0, 1]`. |
| `armour` | `(1,)` | `float32` | `[0, 1]`. |
| `creds` | `(1,)` | `float32` | Round-start credits, normalised by the cap from `data/attributes.yaml`. |
| `weapon_id` | `(1,)` | `int32` | Index into `data/weapons/`. |
| `ammo_mag` | `(1,)` | `float32` | Fraction. |
| `ammo_reserve` | `(1,)` | `float32` | Fraction. |
| `ability_charges` | `(4,)` | `int8` | Remaining charges per slot (q, e, c, x). |
| `ability_cooldowns` | `(4,)` | `float32` | Seconds remaining, clamped to `[0, 30]` then normalised. |
| `ult_progress` | `(1,)` | `float32` | `points / required`. |
| `agent_id_onehot` | `(A,)` | `bool` | `A` from `data/agents/`. Provisional `A = 32`. |
| `velocity` | `(2,)` | `float32` | Self velocity, normalised. |
| `facing_sin_cos` | `(2,)` | `float32` | `[sin(yaw), cos(yaw)]`. |

### 2.4 Round / team state

| Field | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `round_phase_onehot` | `(5,)` | `bool` | buy, freeze, action, post-plant, end. |
| `round_clock` | `(1,)` | `float32` | Seconds remaining in current phase, normalised. |
| `score_onehot` | `(2, 13)` | `bool` | Self / opponent score, capped at 12. |
| `side` | `(1,)` | `bool` | True = attacker, False = defender. |
| `team_alive_mask` | `(5,)` | `bool` | Includes self at index 0. |
| `team_hp` | `(5,)` | `float32` | `[0, 1]`. |
| `team_last_seen` | `(5, 2)` | `float32` | Last known teammate position, agent-relative; zero if unknown. |
| `enemy_spotted_mask` | `(5,)` | `bool` | Enemies the team has currently spotted. |
| `spike_state_onehot` | `(4,)` | `bool` | not-on-field, carried, planted, defused. |
| `spike_position` | `(2,)` | `float32` | If planted, agent-relative tile coords; zero otherwise. |
| `spike_timer` | `(1,)` | `float32` | Seconds remaining if planted, normalised. |

### 2.5 Action mask

The env publishes a mask alongside the observation; samplers use it to skip
illegal actions. Masking is required (not optional): unmasked sampling lets
agents try to fire while reloading, plant when not carrying, etc., and wastes
training steps.

| Field | Shape | dtype | Notes |
| --- | --- | --- | --- |
| `action_kind_mask` | `(K,)` | `bool` | One bit per action kind in §3. |
| `ability_slot_mask` | `(4,)` | `bool` | Which of q / e / c / x is castable now. |

## 3. Action space

A `gymnasium.spaces.Dict` with one discrete head for the action kind plus
per-kind continuous heads. Illegal heads are ignored using the mask.

### 3.1 Action kinds

| Kind | Index | Notes |
| --- | --- | --- |
| `no_op` | 0 | Default; only legal action when dead. |
| `move` | 1 | Continuous direction + crouch + walk flag. |
| `aim` | 2 | Continuous yaw delta. |
| `fire` | 3 | Single shot or burst-tick depending on weapon mode. |
| `reload` | 4 | |
| `switch_weapon` | 5 | Discrete slot. |
| `cast_ability` | 6 | One of `q / e / c / x` — slot picked via `ability_slot_mask`. |
| `interact` | 7 | Pick up spike, drop spike, open door, board teleporter. |
| `plant_spike` | 8 | Legal only as attacker on a plant zone with the spike. |
| `defuse_spike` | 9 | Legal only as defender on the spike. |
| `comm_ping` | 10 | Sends a comm cue visible to teammates next step. |

Provisional total `K = 11`.

### 3.2 Per-kind payloads

| Kind | Payload |
| --- | --- |
| `move` | `direction: Box(2, [-1, 1])`, `crouch: Discrete(2)`, `walk: Discrete(2)`. Direction is unit-clamped at apply time. |
| `aim` | `delta_yaw_rad: Box(1, [-pi/4, pi/4])`. Hard clamp avoids unrealistic snaps in one step. |
| `fire` | `tap: Discrete(2)`. `tap=1` releases on the same step (semi / sniper); `tap=0` is hold (full-auto). |
| `switch_weapon` | `slot: Discrete(3)` — primary / secondary / sidearm. |
| `cast_ability` | `slot: Discrete(4)`, `aim_pitch: Box(1, [-pi/6, pi/6])` for arc abilities. |
| `interact` | `target_kind: Discrete(5)` — spike, door, teleporter, ult-orb, dropped weapon. |
| `comm_ping` | `kind: Discrete(6)` — danger, attention, on-my-way, requesting, defending, going-a/b/c. |

## 4. Reward shaping

Out of strict scope for BUF-79 (BUF-34 owns the wrapper) but the obs/action
contract assumes the per-step reward components are computable from the
observation alone: damage delta, kill flag, death flag, plant flag, defuse
flag, round-win flag. No private opponent state is required.

## 5. Determinism

Each env episode pulls its own RNG subtree from the run's `RngTree` under
`match/<match_id>/round/<round_idx>`; per-agent stochastic policies pull
under `agent/<player_id>`. Sibling agents do not perturb each other's
randomness (BUF-77).

## 6. Versioning

`obs_schema_version` is part of the env metadata and bumped whenever any
field is added, removed, or its dtype changes. The PettingZoo wrapper logs
this version with every replay so BUF-38 storage knows which decoder to
load.

## 7. Open questions

- Patch size `P=21` and ray count `R=64` are rough first guesses. We need a
  sweep before the BUF-30 milestone closes.
- Continuous-vs-discrete action heads: sticking with the hybrid above
  because it's friendly to PPO and SAC; revisit if we adopt MuZero-style
  planners.
- Communication channel breadth (`comm_ping` kinds): six is a pragmatic
  starter; learned-comm extensions get a follow-up issue.
- Whether the audio cue map should be a separate channel or fused into
  `grid_audio` — current sketch keeps it unified for tractability.

## 8. Acceptance for closing BUF-79

- [x] Per-agent obs structure defined including grid patch, LOS raycast,
      ability charges, and spike state.
- [x] Action vocabulary defined with payload shapes and a mask contract.
- [x] PettingZoo `parallel_env` shape (dict in / dict out) is fixed.
- [x] Determinism hook through `RngTree` is documented.
- [x] Versioning hook for replay storage (BUF-38) is documented.
