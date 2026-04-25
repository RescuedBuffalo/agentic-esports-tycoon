# Observation / action space (provisional)

Status: **Planning / draft**. Closes BUF-79. Pairs with
[`world-model-state.md`](world-model-state.md).

We need a single typed contract that every agent backend (scripted, RL, LLM) can
satisfy. This document fixes that contract for the first vertical slice. Anything
flagged _provisional_ here will need a follow-up before training launches.

## 1. Roles

The first slice exposes three roles. Each role gets its own observation view and
its own action vocabulary; the keys below are stable role identifiers used in
`AgentConfig.bindings` (BUF-75).

| Role | What it controls |
| --- | --- |
| `gm` | Strategic decisions: contracts, sponsors, tournament entries, budget. |
| `scout` | Player evaluation: who to watch, who to recommend. |
| `coach` | Match-prep / training-week decisions for an owned team. |

We deliberately omit `player` agents in the first slice — match outcomes are
simulated mechanically.

## 2. Observation view

Observations are read-only Pydantic models built per-tick from the current
world state. They expose a **filtered, role-appropriate** projection — not the
full state — so that agents cannot peek at hidden information (e.g. another
team's balance sheet).

Common envelope:

```python
class ObservationEnvelope(BaseModel):
    tick: int
    role: Literal["gm", "scout", "coach"]
    run_id: str
    schema_version: int  # bumped when the view shape changes
    body: GMObservation | ScoutObservation | CoachObservation
```

### 2.1 `gm` view (provisional)

| Field | Type | Notes |
| --- | --- | --- |
| `org` | `OrgSnapshot` | Owned org's bank, reputation, fan base, payroll burn. |
| `roster` | `list[PlayerCard]` | Public ratings, contract length, morale, role. |
| `pipeline` | `list[PlayerCard]` | Scouted prospects, capped at config limit. |
| `calendar` | `list[CalendarEntry]` | Upcoming matches & tournaments, 8 weeks ahead. |
| `offers` | `list[SponsorOffer]` | Open sponsor offers. |
| `market` | `MarketSummary` | Aggregated salary trends + inflation. |
| `recent_events` | `list[EventSummary]` | Last 32 events relevant to this org. |

### 2.2 `scout` view (provisional)

| Field | Type | Notes |
| --- | --- | --- |
| `assignment` | `ScoutingAssignment` | Region + budget + deadline. |
| `candidates` | `list[PlayerCard]` | Players currently visible to this scout. |
| `prior_reports` | `list[ScoutReport]` | This scout's last reports, capped. |

### 2.3 `coach` view (provisional)

| Field | Type | Notes |
| --- | --- | --- |
| `team` | `TeamSnapshot` | Same shape as `gm.org` but match-prep oriented. |
| `next_match` | `MatchCard \| None` | Opponent profile, scheduled tick, format. |
| `practice_log` | `list[PracticeBlock]` | Last 14 days of training mix. |

## 3. Action space

Actions are **typed, bounded, and side-effect-free at construction time**. The
action validator owns side effects: it accepts an action, checks invariants
against the world state, and (on success) emits domain events (BUF-78).

### 3.1 `gm` actions

| Action | Payload | Notes |
| --- | --- | --- |
| `gm.no_op` | `{}` | Default fallback. |
| `gm.sign_player` | `{ player_id, salary_cents_per_year, term_years }` | Fails if cap exceeded. |
| `gm.release_player` | `{ player_id, reason }` | Triggers buy-out cost. |
| `gm.accept_sponsor` | `{ offer_id }` | Locks that sponsor's prestige & terms. |
| `gm.enter_tournament` | `{ tournament_id }` | Subject to qualifier results. |
| `gm.adjust_budget` | `{ category, new_amount_cents }` | Categories: salaries, marketing, ops. |

### 3.2 `scout` actions

| Action | Payload | Notes |
| --- | --- | --- |
| `scout.no_op` | `{}` | |
| `scout.report` | `{ player_id, recommendation, confidence, free_text }` | `recommendation in {sign, watch, pass}`; `confidence in [0, 1]`. |
| `scout.travel` | `{ region, days }` | Costs from scouting budget. |

### 3.3 `coach` actions

| Action | Payload | Notes |
| --- | --- | --- |
| `coach.no_op` | `{}` | |
| `coach.set_practice_mix` | `{ aim_pct, strats_pct, scrims_pct, rest_pct }` | Must sum to 100. |
| `coach.start_player` | `{ player_id }` | Subject to roster legality. |

All actions inherit a common envelope:

```python
class ActionEnvelope(BaseModel):
    role: Literal["gm", "scout", "coach"]
    tick: int
    body: GMAction | ScoutAction | CoachAction
```

## 4. RL-friendly tensor view (provisional)

For RL agents (PPO baseline) we additionally publish a flat numeric encoding:

- Observations: `np.ndarray` of fixed dimension per role; categorical fields
  are one-hot, ordinal fields min-max-scaled, list fields padded with a mask.
- Actions: a `gymnasium.spaces.Dict` with one `Discrete` head for the action
  kind plus per-kind continuous heads. Illegal actions are masked at sample
  time using a boolean mask shipped alongside the observation.

The mapping between Pydantic views and tensor views lives in
`src/aet/obs/encode.py` (planned). LLM agents skip the tensor view entirely
and consume the Pydantic models serialised to JSON.

## 5. Open questions

- Hidden information policy: how much should `gm` see about rival orgs? Today
  we expose only public ratings and league-table results; revisit before we
  add a fan-sentiment subsystem.
- Action latency: do we need an asynchronous action API for slow LLM calls, or
  is a per-tick deadline enough? The current plan is hard deadline + fall back
  to `no_op`.
- Curriculum: should the RL coach see the full match card or a redacted one in
  early training? Park until we have a baseline policy.
