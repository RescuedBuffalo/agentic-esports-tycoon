# World-model state representation (provisional)

Status: **Planning / draft**. Closes BUF-81. Companions:
[`observation-action-space.md`](observation-action-space.md) (BUF-79) and
[`tycoon-state-factorisation.md`](tycoon-state-factorisation.md) (the
management-layer state). Feeds BUF-38 (replay buffer / experience storage
format).

This document picks the **provisional** neural world-model state shape for
the match-engine policy and the dynamics learner. The decision matters for
BUF-38 because the buffer needs to know whether it's storing discrete tokens,
continuous vectors, or both — and how big each step is on disk.

## 1. Why we need this now

Every other piece of the RL pipeline depends on a state encoding being
nameable:

- BUF-30 (match engine) needs to know what its `info` payload should expose
  to make the world model trainable.
- BUF-34 (PettingZoo wrapper) needs to know which fields land in obs vs.
  which are deferred until the dynamics learner asks.
- BUF-38 (replay buffer) needs a stable on-disk schema for the latent. A
  256-dim continuous vector and a 32-token-of-categorical-32 sequence have
  very different storage and lookup costs.
- The world model itself needs an encoder target shape before any
  representation-learning sweep can start.

## 2. Provisional choice: hybrid RSSM-style latent

We commit to **a Recurrent State-Space Model (RSSM) latent with a hybrid
deterministic/stochastic split**, following the Dreamer family:

```
state_t = (h_t, z_t)
  h_t ∈ R^H        # deterministic recurrent state
  z_t ∈ {one_hot(C)}^N  # stochastic state, N categorical groups of C classes
```

Provisional dimensions:

| Symbol | Meaning | Default | Rationale |
| --- | --- | --- | --- |
| `H` | GRU hidden size | 512 | Matches Dreamer-V3 baseline; fits comfortably on a 24 GB GPU. |
| `N` | Number of categorical groups | 32 | Enough capacity for a 5v5 map state without exploding sample size. |
| `C` | Classes per group | 32 | Powers-of-two friendly; maps to 5-bit symbols on disk. |

### 2.1 Why discrete latents (z_t)

- Dreamer-V3 evidence: discrete categoricals stabilise training across
  domains without per-task tuning.
- Compact on disk: `N · ceil(log2 C) = 32 · 5 = 160 bits = 20 bytes` per
  step's stochastic state. The deterministic `h_t` is what dominates BUF-38
  storage.
- Plays well with cross-entropy losses on the dynamics head, which avoids
  the KL-collapse pathology continuous latents are prone to.

### 2.2 Why also keep h_t deterministic

- Long-horizon credit assignment in match rounds (10–145 s, 100–1450 sim
  steps) benefits from a recurrent backbone.
- A single discrete sample per step would be a high-variance signal for the
  policy head; mixing it with `h_t` gives a stable substrate.

### 2.3 What we explicitly defer

- **JEPA-style joint-embedding predictive architectures**: attractive as a
  pre-training signal because they sidestep reconstruction, but the
  literature on multi-agent partially-observed environments is thinner. We
  treat JEPA as an alternative encoder that can plug in behind the same
  `(h_t, z_t)` interface; revisit after a Dreamer baseline lands.
- **Pure continuous Gaussian latents** (Dreamer-V1/V2 style): worse
  empirical stability in our setting; rejected.
- **Token-only models** (no `h_t`): rejected for the round-length reasons
  above.

## 3. Encoder + decoder shapes

Inputs are the per-agent observation from BUF-79. The encoder is per-agent
(weight-shared) and emits one `(h_t, z_t)` per agent; the dynamics module
optionally fuses across teammates via attention.

```
encoder: BUF-79 obs -> z_t ~ Categorical(N, C)
prior:   h_t -> z_hat_t ~ Categorical(N, C)        # KL target
recurrent: h_{t+1} = GRU(h_t, embed(z_t), embed(a_t))
heads:
  reward      r_t = MLP(h_t, z_t)
  termination d_t = MLP(h_t, z_t)
  obs decoder o_t = MLP(h_t, z_t)        # auxiliary; small reconstruction
```

We do **not** decode the full observation pixel-perfectly — the auxiliary
decoder is a lightweight projection over a few salient channels (own HP, own
position, spike state, observed enemies). Full reconstruction is too large
to be worth the gradient.

## 4. Implications for BUF-38 (replay buffer storage)

Per env step, per agent, we store:

| Field | Size | Notes |
| --- | --- | --- |
| Observation packed bytes | ~6–8 kB | Per BUF-79 layout; dominated by grid patch and rays. |
| Action packed bytes | ~16 B | Discrete head + small payload. |
| `h_t` snapshot | 512 × 4 B = 2 kB | float32; switching to bf16 halves this. |
| `z_t` snapshot | 20 B | 32×32 categorical packed to 5 bits each. |
| Reward / done | 8 B | |
| Step metadata | ~32 B | tick, agent_id, rng_path, schema_version. |

Provisional total: **~10 kB / agent / step**. A best-of-13 match capped at
13 rounds × ~1450 steps × 10 agents ≈ 1.9M steps ≈ **~19 GB / match raw**.
That number sets a hard ceiling on how many matches we can replay-buffer per
host without compression; BUF-38 will own that decision.

Storage hooks the buffer must support:

1. **Decouple obs and latent**: the buffer must be able to drop `(h_t, z_t)`
   if the model version changes (re-encoded from obs on demand) without
   rewriting the obs payload.
2. **Versioned latents**: every saved `(h_t, z_t)` carries
   `(model_version, obs_schema_version)`. Mismatched reads either re-encode
   (if obs schema matches) or are skipped.
3. **Trajectory-level access**: the dynamics trainer needs contiguous
   round-length sequences, so the buffer indexes by `(match_id, round_idx,
   agent_id)`.

## 5. Determinism

The world-model trainer pulls its own RNG subtree from the run's `RngTree`
under `wm/<seed_id>` (BUF-77). Encoder dropout, KL temperature schedules,
and replay-buffer shuffling all draw from this subtree so swapping a model
version does not perturb sim randomness for unrelated subsystems.

## 6. Open questions

- `(N, C) = (32, 32)` is a starting point; an ablation between `(32, 16)`,
  `(32, 32)`, and `(48, 32)` should run before any policy training that
  relies on the latent.
- Whether the cross-agent attention fuse lives inside the encoder or the
  dynamics module. Lean toward the dynamics module so single-agent encoders
  remain reusable for evaluation.
- KL balancing: Dreamer-V3 uses fixed coefficients; we may need a schedule
  for highly-asymmetric round openings (eco rounds vs. full-buy).
- Whether to expose `z_t` to the policy head as samples or as the full
  categorical distribution. Current plan: samples during rollout, mean
  during evaluation.

## 7. Acceptance for closing BUF-81

- [x] Latent shape committed: hybrid RSSM `(h_t ∈ R^512, z_t ∈ Categorical^{32×32})`.
- [x] Discrete-vs-continuous decision is recorded with rationale.
- [x] JEPA, pure-continuous, and token-only alternatives explicitly weighed
      and either deferred or rejected.
- [x] On-disk implications are itemised for BUF-38 with a per-step byte
      budget.
- [x] Determinism + versioning hooks documented.
