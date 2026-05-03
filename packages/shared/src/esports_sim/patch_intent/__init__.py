"""Patch-intent extraction (BUF-24, Systems-spec System 06).

For every new patch, classify its intent — what Riot was trying to
accomplish, which agents/maps shifted, expected pickrate moves, and how
controversial the community will find it. The extractor calls Claude
through the budget governor with a cached system prompt; the result
lands in the ``patch_intent`` table so downstream agents can read it
without paying for the LLM call.

Public surface::

    PatchIntentResult              # Pydantic shape Claude must return
    ExpectedPickrateShift          # one entry inside expected_pickrate_shifts
    PROMPT_VERSION                 # bumps when the prompt is reworked
    extract_patch_intent           # patch_note_text + dev_blog -> result
    upsert_patch_intent            # persist a result against a PatchNote
    extract_intent_for_pending     # scheduler hook: every patch_note without
                                   #   an intent gets one (idempotent)
"""

from esports_sim.patch_intent.extractor import (
    PROMPT_VERSION,
    extract_patch_intent,
)
from esports_sim.patch_intent.persistence import (
    extract_intent_for_pending,
    upsert_patch_intent,
)
from esports_sim.patch_intent.schema import (
    ExpectedPickrateShift,
    PatchIntentResult,
)

__all__ = [
    "ExpectedPickrateShift",
    "PROMPT_VERSION",
    "PatchIntentResult",
    "extract_intent_for_pending",
    "extract_patch_intent",
    "upsert_patch_intent",
]
