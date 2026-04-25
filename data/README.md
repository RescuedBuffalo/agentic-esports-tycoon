# Game data

Canonical Valorant data the simulator loads at boot. Closes BUF-76. The
content here is **generated** by the data pipeline (see `config/graph_export.yaml`)
from the entity graph; manual edits should go through `data/_overrides/` and
will be re-applied on every export.

| Path | Purpose |
| --- | --- |
| `attributes.yaml` | Canonical attribute list — every numeric field the rest of the data tree may reference. |
| `agents/<id>.yaml` | One file per Valorant agent. |
| `weapons/<id>.yaml` | One file per weapon. |
| `maps/<id>.yaml` | One file per map. |
| `_overrides/` | Manual corrections applied by the data pipeline; not consumed directly. |

## Conventions

- `id` matches `^[a-z0-9][a-z0-9_-]{2,63}$` and is stable across patches.
- Numeric ranges follow `attributes.yaml`. Anything outside the declared
  range is rejected by `config/validation.yaml`.
- `released_in_patch` and `retired_in_patch` are Riot patch strings (e.g.
  `"8.08"`); cross-reference `config/patch_eras.yaml` to map them to eras.
- Currency is integer Valorant credits in-match; out-of-match prize money
  uses cents with currency declared at the run level.

## Coverage in this drop

The first drop seeds the canonical list with enough entries to exercise the
match engine and the data pipeline's validation rules. Full coverage of the
agent / weapon / map roster will come from the pipeline once it lands.

- Agents: jett, sage, sova, cypher, omen — covers all five role classes.
- Weapons: classic, sheriff, vandal, phantom, operator — covers SMG/rifle
  pricing tiers minus shotguns.
- Maps: bind, haven, ascent — covers 2-site, 3-site, and a current-pool map.

The pipeline owns expanding this; do not hand-edit beyond `_overrides/`.
