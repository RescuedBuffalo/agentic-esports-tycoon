# API access + hardware checklist

Status: **Planning**. Closes BUF-80. This is a procurement / readiness checklist
used to gate the start of the first training run. Tick everything before
kicking off Phase 1.

## 1. API access

| Provider | Why we need it | Account / key location | Owner | Status |
| --- | --- | --- | --- | --- |
| Anthropic API | LLM-driven `gm` agent (default backend). | `ANTHROPIC_API_KEY` in shared 1Password vault `aet/secrets`. | Platform | ☐ |
| OpenAI API | LLM agent A/B comparisons; embeddings for scout reports. | `OPENAI_API_KEY` in same vault. | Platform | ☐ |
| Local model gateway | Offline inference for budget runs (vLLM on internal cluster). | Internal endpoint URL in `.env.example`. | Infra | ☐ |
| Object storage | Event-log + snapshot uploads. | S3-compatible bucket `aet-runs-prod`. | Infra | ☐ |
| Telemetry sink | Run metrics. | Prometheus push-gateway or hosted equivalent. | Infra | ☐ |

### Per-key requirements

- **Rate limits**: at least 1M input + 200k output tokens per hour during
  training to absorb burst spikes.
- **Spend caps**: hard budget alarm at 80% of `max_usd` from `agent.schema.yaml`.
- **Key rotation**: every 90 days; rotation runbook lives in the platform repo.
- **Provider redundancy**: every agent role with `backend: llm` must work
  against at least two of {`anthropic`, `openai`, `local`} so a single
  provider outage does not stall a run.

## 2. Hardware

Two distinct profiles. The simulation core is CPU-bound; only the
representation-learning trainer needs a GPU.

### 2.1 Sim host (no GPU)

| Resource | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| CPU | 8 modern cores | 16 cores, AVX2 | Sim is single-threaded today; spare cores run agents. |
| RAM | 16 GB | 32 GB | Snapshots are mmap'd; bigger seasons need more. |
| Disk | 50 GB SSD | 200 GB NVMe | Event logs grow ~1 GB / season-month. |
| Network | 100 Mbit | 1 Gbit | LLM calls dominate latency; bandwidth is fine. |

### 2.2 Trainer host (GPU)

For the world-model + RL training in Phase 1+. Single-host is enough for the
first vertical slice.

| Resource | Minimum | Recommended | Notes |
| --- | --- | --- | --- |
| GPU | 1 × 24 GB (e.g. L4, A10) | 1 × 80 GB (H100, A100) | Mixed precision; bf16 preferred. |
| CPU | 8 cores | 32 cores | Sim rollouts on CPU feed the GPU trainer. |
| RAM | 64 GB | 128 GB | Replay buffer lives on host RAM. |
| Disk | 500 GB NVMe | 2 TB NVMe | Checkpoints + replay buffer + dataset cache. |
| CUDA | 12.x | 12.x | Match what the project's `torch` pin requires. |

### 2.3 Cluster (later)

Out of scope for the first training run. Re-evaluate after we have a stable
single-host loop and a measured tokens-per-USD figure.

## 3. Operational gates

Tick all of these before kicking off the first run:

- [ ] Two LLM providers wired up and budget-alarmed.
- [ ] Sim host meets at least the minimum spec.
- [ ] Trainer host provisioned **or** explicit decision to defer training.
- [ ] Object storage bucket exists, versioning + lifecycle policies configured.
- [ ] Run header fields (build hash, data hash, root seed, key fingerprints)
      logged on every run (see determinism contract in `ARCHITECTURE.md` §5).
- [ ] On-call rotation defined for the first month after launch.
- [ ] Cost dashboard shared with the team.

## 4. Open items

- Pick the local-inference cluster (vLLM vs. SGLang). Defer until we know the
  quantised model we want to serve.
- Decide whether scout/coach RL agents share GPU time with the world model or
  get a dedicated trainer host.
