# Game data schemas

Schemas for the **content** that lives inside the simulation: players, teams,
tournaments, matches, sponsors. Anything that a designer might want to author or tune
without rebuilding the simulator. See BUF-76 for the issue this closes.

| File | Loaded as | Notes |
| --- | --- | --- |
| `player.schema.yaml` | `Player` | Individual competitor. |
| `team.schema.yaml` | `Team` | Roster + org metadata. References players by id. |
| `tournament.schema.yaml` | `Tournament` | Calendar, format, prize pool. |
| `match.schema.yaml` | `Match` | A scheduled or simulated match between two teams. |
| `sponsor.schema.yaml` | `Sponsor` | Sponsor offers and contract terms. |

## Conventions

- All ids are `^[a-z0-9][a-z0-9_-]{2,63}$`. They are stable and never reused.
- Money is stored as integer cents in the `currency` declared by `RunConfig.economy`.
- Skills and ratings are `0..100` integers unless noted otherwise.
- Times are ISO-8601 strings (`YYYY-MM-DD` or full timestamps); the loader converts to
  in-game ticks at boot.
