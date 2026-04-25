# Provisional world-model state representation

Status: **Planning / draft**. Closes BUF-81. Pairs with
[`observation-action-space.md`](observation-action-space.md).

The world model is a learned compression of the full simulation state, used by
planning agents to roll forward "what if" trajectories without re-running the
full simulator. This document fixes a **provisional** representation so that
the rest of the planning artifacts (data layout, encoder skeleton, evaluation
plan) can lock in. Expect the shapes here to change after the first training
run.

## 1. Why a learned state instead of the raw state

The raw `WorldState` is large, sparse, and contains identifiers that are not
useful as model inputs (UUIDs, free-text scout notes). A learned state lets us:

1. Keep a fixed-dimensional representation regardless of league size.
2. Drop fields the model has learned to be irrelevant.
3. Share weights between the dynamics head and the policy / value heads.

## 2. State factorisation

The world model state `z_t` is the concatenation of three sub-states. Each
sub-state is produced by its own encoder; the dynamics module operates on the
concatenation.

```
z_t = [ z_org_t  ‖  z_league_t  ‖  z_economy_t ]
       (D_org=64) (D_league=128) (D_econ=32)
```

Total provisional dimension: **`D = 224`**. (Subject to change after the first
representation-learning sweep.)

### 2.1 `z_org` — owned organisation (D=64)

Encodes everything about the player's own org: roster, finances, morale,
upcoming commitments. Inputs come from the `gm` observation view.

- Roster encoder: per-player MLP → mean-pool over starters, separate mean-pool
  over subs, concat. Player input features: 5 skill ratings, role one-hot,
  age, contract length remaining, morale.
- Finance encoder: small MLP over `[bank_cents, payroll_run_rate,
  sponsor_run_rate, debt]`, all log-scaled.

### 2.2 `z_league` — competitive landscape (D=128)

Encodes other orgs and tournaments. The space is variable-sized (number of
orgs and tournaments changes over time), so we use set encoders + cross-attn
to a fixed bank of learned slot tokens.

- Per-org token: `[public_reputation, public_form_last_8, region_one_hot,
  ...]`.
- Per-tournament token: `[tier, days_until, prize_pool_log, region_one_hot]`.
- Aggregator: cross-attention from a fixed pool of 16 slot queries (chosen so
  the model can specialise slots to "rivals", "cupcakes", "next opponent",
  etc.). Output flattened to D=128.

### 2.3 `z_economy` — macro (D=32)

A small MLP over `[inflation_pct, salary_index, sponsor_demand_index,
fan_growth_rate]`. Mostly a stability signal; rarely changes between adjacent
ticks.

## 3. Dynamics

We learn a recurrent dynamics module:

```
z_{t+1} = f_theta( z_t, a_t, e_t )
```

- `a_t` — discrete action kind one-hot (≤32 dims) + small continuous payload
  (≤16 dims), padded.
- `e_t` — exogenous tick info: in-game date features, scheduled-match flag,
  is-end-of-season flag.
- `f_theta` — GRU cell of width 256, plus a residual MLP.

Heads attached to `z_t`:
- Reward head (scalar): `r_t = MLP(z_t)`.
- Value head (scalar): `V(z_t) = MLP(z_t)`.
- Policy head (per role): `pi(a | z_t, role) = MLP(z_t, role_embed)`.
- Reconstruction heads: bank balance bucket, league-table position; trained
  as auxiliary losses to keep `z_t` informative.

## 4. Training signal

Phase 0 (BUF-81 — this issue): no training, just the schema and encoder
skeleton. We need the shapes nailed down so that BUF-79's tensor view can
align with the encoder inputs.

Phase 1 (post-planning): supervised pre-training on logged event streams.
Targets: next-tick reconstruction of (a) bank balance, (b) league-table
position, (c) outcome of the next match. This gives us a useful `z_t` before
any RL touches it.

Phase 2: RL fine-tuning of the policy heads with the world model frozen, then
joint training. PPO baseline; Dreamer-style imagination rollouts are out of
scope until the supervised phase is healthy.

## 5. Reproducibility

The world-model trainer pulls its own RNG subtree from the run's `RngTree`
under `wm/<seed_id>` so that swapping a model version does not perturb sim
randomness for unrelated subsystems (BUF-77).

## 6. What this document does **not** lock in

- Exact encoder hyperparameters — keep notebook tuning rights.
- Dataset partitioning across runs — needs an offline ETL plan.
- Choice of optimiser — default to AdamW; revisit once we have a loss curve.
- Which agents actually consume `z_t` first. Likely `gm` only, then `coach`,
  then `scout`.

## 7. Acceptance for closing BUF-81

- [x] Sub-state factorisation (org / league / economy) is named and
      dimensioned.
- [x] Dynamics shape and heads are sketched.
- [x] Training plan is split into phases with explicit phase-0 deliverables.
- [x] Reproducibility hook into `RngTree` is documented.
