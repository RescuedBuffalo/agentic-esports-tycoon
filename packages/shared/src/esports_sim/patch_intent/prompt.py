"""Prompt template for the BUF-24 patch-intent extractor.

Lives in its own module so the system prompt — the *cacheable* portion
of every patch-intent call — is grep-able and version-controlled
independently of the extractor wiring. Bumping
``PROMPT_VERSION`` (in :mod:`esports_sim.patch_intent.extractor`) is
mandatory whenever this file changes in a way that should produce a
fresh classification rather than UPSERT on top of the existing one.

Design notes:

* The system prompt embeds the JSON schema *literally* in the prompt
  (rather than relying on tool-use). The output shape is small enough
  to bind by example, and the extractor parses + validates with the
  Pydantic model in :mod:`schema` — the prompt is one of two pieces of
  the contract, the schema is the other.
* The Systems-spec rubric is in the system prompt so it's cached
  across patches (each patch only pays the small per-patch user
  message). We rely on Anthropic's prompt caching to make the
  per-patch call cheap; the wrapper sets ``cache_control`` on the
  system block (1h TTL).
* Output is asked to be raw JSON, no surrounding prose. The extractor
  strips a leading ```` ```json `` fence as a safety net but expects
  the model to honour the instruction.
"""

from __future__ import annotations

# The cacheable rubric. Versioned as a constant so a future spec change
# is a single-edit + version-bump.
SYSTEM_PROMPT = """\
You are the patch-intent classifier for a Valorant esports simulator.

For each patch you receive, return a JSON object that classifies the patch
along the following axes. Output MUST be valid JSON, with no surrounding
prose, code fences, or commentary.

Rubric:

- primary_intent: a short hyphenated label describing what Riot was
  trying to accomplish with this patch. Examples:
  "nerf-meta-outlier", "buff-underused-agent", "map-rotation-tweak",
  "econ-rebalance", "qol-bugfix", "new-agent-release".
- pro_play_driven_score: float in [0, 1]. 1.0 means "this patch was
  driven entirely by pro-play feedback" (e.g. nerfing an agent that
  dominated VCT but is mid-tier in ranked). 0.0 means "purely ranked-
  ladder data drove this".
- agents_affected: list of agent display names that the patch directly
  touches (e.g. ["Chamber", "Jett"]). Use the canonical Valorant
  display names. Empty list if no agents changed.
- maps_affected: list of map display names directly changed (geometry,
  callouts, rotation in/out). Empty list if no maps changed.
- econ_changed: true if any credit value moved — kill rewards, ability
  costs, weapon prices, plant/defuse rewards. False otherwise.
- expected_pickrate_shifts: list of objects, one per agent or map you
  predict will shift in pickrate as a result of this patch. Each
  object: {"subject": <name>, "direction": "up" | "down" | "flat",
  "magnitude": "small" | "medium" | "large", "rationale": <one-line
  why, optional>}. Use "flat" to record an explicit no-change call
  on a subject the patch touched but you expect inertia on. Omit
  subjects you have no view on.
- community_controversy_predicted: float in [0, 1]. 1.0 means "this
  patch will dominate Reddit / Twitter for the next week with
  near-universal negative reaction". 0.0 means "no one will care".
- confidence: float in [0, 1]. Your overall confidence in the
  classification. Lower for short patches with little context, higher
  for patches with detailed dev blogs.
- reasoning: a short paragraph (2-5 sentences) justifying the call.
  Reference specific changes from the patch notes. This is what a
  human spot-checker reads first when auditing your output.

Constraints:

- Be conservative on magnitude. "large" is reserved for
  game-warping shifts (Chamber 5.12, Yoru rework, agent removal).
  When in doubt, choose "medium" or "small".
- Be honest about confidence. If the patch is a hotfix with two bullet
  points, output confidence around 0.4, not 0.9.
- Output ONLY the JSON object. Do not wrap it in code fences. Do not
  prepend "Here is the classification:". Do not append a sign-off.
"""


def build_user_message(patch_notes_text: str, dev_blog_text: str | None) -> str:
    """Compose the per-patch user message.

    The patch-notes text (and optional dev blog) is the *uncached*
    portion of the prompt — every patch's call pays for it once. The
    cacheable system prompt above is what amortises across the corpus.
    """
    parts: list[str] = []
    parts.append("Classify the following patch.")
    parts.append("")
    parts.append("=== PATCH NOTES ===")
    parts.append(patch_notes_text.strip())
    if dev_blog_text and dev_blog_text.strip():
        parts.append("")
        parts.append("=== DEV BLOG ===")
        parts.append(dev_blog_text.strip())
    parts.append("")
    parts.append("Return the JSON classification now.")
    return "\n".join(parts)


__all__ = ["SYSTEM_PROMPT", "build_user_message"]
