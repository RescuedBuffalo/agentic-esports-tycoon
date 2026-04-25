# Data pipeline configuration

These files configure the **data pipeline** that ingests Valorant esports data
(VLR scrapers, Riot API, patch notes) and turns it into the entity graph the
sim consumes from `data/`. Closes BUF-75.

The pipeline is laid out as a DAG of stages. Each stage reads its config from
exactly one of these files; the scheduler stitches them together.

| File | Stage | Reads | Writes |
| --- | --- | --- | --- |
| `entity_resolution.yaml` | Match raw records to canonical entities (players, teams, tournaments). | Raw scraper / API rows. | Canonical entity table. |
| `derivation.yaml` | Derive features from raw stats (KAST, ADR, agent-specific metrics). | Canonical match logs. | Derived feature tables. |
| `relationships.yaml` | Extract relationships (player-on-team-during, team-in-tournament, agent-played-on-map). | Canonical entities + matches. | Relationship edge tables. |
| `scheduler.yaml` | DAG topology, schedules, retries, alerting. | All other config files. | Run history. |
| `patch_eras.yaml` | Define Valorant patch boundaries used to slice data by meta. | None — hand-curated. | Era labels for downstream joins. |
| `graph_export.yaml` | Export the entity + relationship graph to the format `data/` expects. | Canonical entities, derived features, relationships. | YAML files under `data/`. |
| `validation.yaml` | Validation rules + thresholds; the gate before `graph_export` runs. | All upstream tables. | Validation report; pass/fail. |

All files share these conventions:

- Paths are relative to the repository root unless prefixed with `s3://`.
- Time windows are ISO-8601 (`2025-01-15T00:00:00Z`).
- Entity ids match `^[a-z0-9][a-z0-9_-]{2,63}$` (same convention as `data/`).
- Every config supports a top-level `enabled: true|false` so a stage can be
  short-circuited without removing it from the DAG.
