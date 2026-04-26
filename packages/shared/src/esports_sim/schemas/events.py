"""Event schemas (BUF-78).

Events are the canonical record of everything that happens in a run. The world
state is a fold over the event log; replay loads events in tick order.

Design rules
------------
- Every event is a Pydantic v2 model with ``model_config = ConfigDict(frozen=True,
  extra="forbid")`` so payloads are immutable and unknown fields fail loudly.
- Every event carries the same envelope fields (``tick``, ``source``,
  ``rng_path``, ``id``). Concrete events extend ``Event`` and add a
  literal ``kind`` for dispatch.
- ``AnyEvent`` is a discriminated union on ``kind``. Loaders should validate
  against this union so that unknown kinds raise instead of silently turning
  into ``Event``.
- Versioning: when a payload shape changes, bump ``schema_version`` on the
  envelope and write a migrator. We do not rename ``kind``s.

This module intentionally only defines the **first wave** of events needed to
exercise the simulation core (clock, matches, contracts, finance). The full
event vocabulary will grow as subsystems land.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

EVENT_SCHEMA_VERSION = 1


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Event(_Frozen):
    """Envelope shared by every event."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    tick: int = Field(ge=0, description="Sim tick at which this event was emitted.")
    source: str = Field(
        description=(
            "Producer of the event. 'sim' for the core; an agent role "
            "name (gm, scout, ...) for agent-driven events."
        ),
    )
    rng_path: str | None = Field(
        default=None,
        description="RngTree path the producer pulled from; null for deterministic events.",
    )
    schema_version: int = Field(default=EVENT_SCHEMA_VERSION, ge=1)


# --- Clock --------------------------------------------------------------------


class TickAdvanced(Event):
    kind: Literal["tick.advanced"] = "tick.advanced"
    new_tick: int = Field(ge=0)


# --- Match lifecycle ----------------------------------------------------------


class MatchScheduled(Event):
    kind: Literal["match.scheduled"] = "match.scheduled"
    match_id: str
    tournament_id: str
    home_team_id: str
    away_team_id: str
    scheduled_for_tick: int = Field(ge=0)
    best_of: Literal[1, 3, 5, 7]


class MatchStarted(Event):
    kind: Literal["match.started"] = "match.started"
    match_id: str


class MatchCompleted(Event):
    kind: Literal["match.completed"] = "match.completed"
    match_id: str
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)


# --- Contracts ----------------------------------------------------------------


class ContractSigned(Event):
    kind: Literal["contract.signed"] = "contract.signed"
    player_id: str
    team_id: str
    salary_cents_per_year: int = Field(ge=0)
    ends_on_tick: int = Field(ge=0)


class ContractTerminated(Event):
    kind: Literal["contract.terminated"] = "contract.terminated"
    player_id: str
    team_id: str
    reason: Literal["expired", "released", "retired", "transferred"]


# --- Finance ------------------------------------------------------------------


class CashTransferred(Event):
    kind: Literal["finance.cash_transferred"] = "finance.cash_transferred"
    from_account: str
    to_account: str
    amount_cents: int
    memo: str | None = None


# --- Agent decisions ----------------------------------------------------------


class AgentDecisionMade(Event):
    """An agent took an action. The action payload itself is opaque here; the
    action validator emits more specific follow-on events (contract.signed,
    cash_transferred, ...). Useful for auditing prompts and policies."""

    kind: Literal["agent.decision_made"] = "agent.decision_made"
    role: str
    backend: Literal["scripted", "rl", "llm"]
    action_kind: str
    latency_ms: int = Field(ge=0)
    tokens_in: int | None = Field(default=None, ge=0)
    tokens_out: int | None = Field(default=None, ge=0)


# --- Discriminated union ------------------------------------------------------


AnyEvent = Annotated[
    TickAdvanced
    | MatchScheduled
    | MatchStarted
    | MatchCompleted
    | ContractSigned
    | ContractTerminated
    | CashTransferred
    | AgentDecisionMade,
    Field(discriminator="kind"),
]


__all__ = [
    "EVENT_SCHEMA_VERSION",
    "Event",
    "TickAdvanced",
    "MatchScheduled",
    "MatchStarted",
    "MatchCompleted",
    "ContractSigned",
    "ContractTerminated",
    "CashTransferred",
    "AgentDecisionMade",
    "AnyEvent",
]
