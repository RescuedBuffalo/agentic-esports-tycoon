"""``extract_patch_intent`` — call Claude, parse, validate.

The single sanctioned entry point for BUF-24 patch-intent extraction.
Composes:

* The cached system prompt from :mod:`esports_sim.patch_intent.prompt`
  — wrapped in ``cache_control`` so a corpus-wide re-classification
  pays the system-prompt cost once per hour, not once per patch.
* The per-patch user message (notes + optional dev blog).
* The budget governor (BUF-22) — every call is pre-flighted against
  the weekly hard cap and the per-purpose ``patch_intent`` soft cap
  ($3/wk by default).
* The ``PatchIntentResult`` schema — the LLM's JSON output is parsed
  and validated; a malformed response raises
  ``pydantic.ValidationError`` rather than crashing downstream
  consumers with a ``KeyError``.

Output: ``ExtractionOutcome`` carrying the validated result plus the
usage / cost / model triple the persistence layer needs to populate the
``patch_intent`` table.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from esports_sim.budget import claude_call
from esports_sim.budget.pricing import DEFAULT_MODEL, cost_from_usage_obj
from esports_sim.patch_intent.prompt import SYSTEM_PROMPT, build_user_message
from esports_sim.patch_intent.schema import PatchIntentResult

if TYPE_CHECKING:
    import anthropic

    from esports_sim.budget import Governor


_logger = logging.getLogger("esports_sim.patch_intent")

# Bumped whenever the prompt rubric in :mod:`prompt` changes in a way
# that should produce a fresh classification. The ``patch_intent``
# table dedupes on ``(patch_note_id, prompt_version)`` so a bump means
# a new row lands rather than UPSERTing the existing classification.
PROMPT_VERSION = "v1"

# Issue spec: temperature 0.1 — low enough that two runs agree on the
# headline classification, high enough that the model still picks a
# reasoned choice on borderline patches rather than collapsing to a
# generic answer.
DEFAULT_TEMPERATURE = 0.1

# Output is small (one JSON object, ~1k tokens worst-case). Cap at 2k
# to leave headroom for a long ``reasoning`` paragraph without inviting
# unbounded spend on a runaway response.
DEFAULT_MAX_TOKENS = 2048

# 1-hour cache TTL on the system prompt. The corpus-wide re-classification
# path (re-running every patch in one batch) wants the system prompt cached
# across all patches — Anthropic's 5m default would expire mid-batch on a
# slow run. The 2x write multiplier is fine: the system prompt is small.
SYSTEM_CACHE_TTL = "1h"

# Strip a leading ```` ```json `` (or bare ```` ``` ``) fence and the
# trailing fence the model occasionally emits despite the instructions.
# ``re.DOTALL`` lets the body span newlines.
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


@dataclass(frozen=True)
class ExtractionOutcome:
    """The validated result plus the metadata persistence needs.

    Carries the model name + usage + cost so the persistence layer can
    write the ``patch_intent`` row without re-deriving them. Frozen so a
    leaked instance can't be mutated by a downstream caller.
    """

    result: PatchIntentResult
    model: str
    prompt_version: str
    input_tokens: int
    output_tokens: int
    usd_cost: float


def extract_patch_intent(
    *,
    governor: Governor,
    patch_notes_text: str,
    dev_blog_text: str | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    client: anthropic.Anthropic | None = None,
    notes: str | None = None,
) -> ExtractionOutcome:
    """Run one patch-intent extraction call.

    Parameters
    ----------
    governor:
        BUF-22 budget governor — every call is pre-flighted against the
        weekly hard cap and the ``patch_intent`` per-purpose soft cap.
    patch_notes_text:
        The cleaned ``body_text`` from the ``patch_note`` row. Required.
    dev_blog_text:
        Optional companion dev-blog text. Riot occasionally publishes a
        long-form piece alongside a numbered patch; passing it gives the
        classifier richer context for ``primary_intent`` and
        ``confidence``.
    model:
        Claude model. Defaults to Opus 4.7 — the BUF-24 cost target
        ($0.50/patch) is well under what Opus 4.7 charges for a
        ~10k-token input + ~1k-token output, especially with the system
        prompt cached.
    max_tokens / temperature:
        Forwarded to the SDK call. ``temperature=0.1`` is the BUF-24
        spec value.
    client:
        Anthropic SDK client. ``None`` means construct one from the
        process environment (ANTHROPIC_API_KEY); tests inject a mock.
    notes:
        Free-form annotation that lands on the ledger row.

    Returns
    -------
    ExtractionOutcome
        The validated result plus usage / cost metadata for the
        persistence layer.

    Raises
    ------
    pydantic.ValidationError
        Model returned a JSON shape that doesn't match
        :class:`PatchIntentResult` (missing field, out-of-range score,
        extra key, etc.). A regression in the model's output shape
        surfaces here, not as a downstream KeyError.
    json.JSONDecodeError
        Model returned non-JSON. Usually a hint that the system prompt
        needs tightening — the wrapper does best-effort fence stripping
        but does not try to repair invalid JSON.
    BudgetExhausted
        The weekly hard cap or the ``patch_intent`` soft cap would be
        exceeded by this call. The SDK is never contacted.
    """
    if not patch_notes_text or not patch_notes_text.strip():
        # The ``patch_note.body_text`` column is NOT NULL with min_length=1
        # at the connector boundary, so an empty string here is a caller
        # bug (or a cleaning regression); raise clearly rather than
        # paying for a Claude call that can't possibly be useful.
        raise ValueError("patch_notes_text is empty; nothing to classify")

    # System prompt is wrapped as a single cacheable text block. The
    # ``cache_control`` marker tells Anthropic to cache the prefix; the
    # extractor's wrapper (``claude_call``) detects the marker and
    # prices the pre-flight estimate at the cache-write rate so an
    # over-cap call gets blocked before the network.
    system_blocks: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral", "ttl": SYSTEM_CACHE_TTL},
        }
    ]
    user_text = build_user_message(patch_notes_text, dev_blog_text)

    response = claude_call(
        governor=governor,
        purpose="patch_intent",
        model=model,
        max_tokens=max_tokens,
        system=system_blocks,
        messages=[{"role": "user", "content": user_text}],
        temperature=temperature,
        cache_write_ttl=SYSTEM_CACHE_TTL,
        client=client,
        notes=notes,
    )

    raw_text = _extract_text(response)
    payload = _parse_json(raw_text)
    result = PatchIntentResult.model_validate(payload)

    usage = response.usage
    return ExtractionOutcome(
        result=result,
        model=model,
        prompt_version=PROMPT_VERSION,
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
        usd_cost=cost_from_usage_obj(model, usage, cache_write_ttl=SYSTEM_CACHE_TTL),
    )


# --- private helpers -------------------------------------------------------


def _extract_text(response: Any) -> str:
    """Concatenate the text blocks from a ``Message`` response.

    The SDK returns ``content`` as a list of typed blocks. We only
    expect one ``text`` block from this prompt (output is JSON, no
    tool use, no thinking), but we concatenate any text blocks
    defensively so a future ``thinking`` mode toggle doesn't silently
    drop the actual answer.
    """
    blocks = getattr(response, "content", None) or []
    pieces: list[str] = []
    for block in blocks:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text = getattr(block, "text", "")
            if text:
                pieces.append(text)
    return "".join(pieces).strip()


def _parse_json(raw: str) -> dict[str, Any]:
    """Strip a stray ``json`` code fence and parse.

    The system prompt forbids fences but defensive stripping keeps a
    one-time prompt drift from tipping the whole pipeline over. Anything
    other than a single fence-wrapped object falls through to
    ``json.loads`` directly — invalid JSON is the caller's signal to
    investigate.
    """
    stripped = raw
    fence_match = _FENCE_RE.match(raw)
    if fence_match is not None:
        stripped = fence_match.group(1)
        _logger.warning(
            "patch_intent.fence_stripped",
            extra={"length": len(raw)},
        )
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        # ``json.loads`` happily returns a list / string / number for
        # the right input; the schema demands an object. Normalise the
        # error so the failure is "shape mismatch", not "ValidationError
        # on an unexpected scalar".
        raise ValueError(
            f"patch_intent response is not a JSON object (got {type(parsed).__name__})"
        )
    return parsed


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TEMPERATURE",
    "ExtractionOutcome",
    "PROMPT_VERSION",
    "SYSTEM_CACHE_TTL",
    "extract_patch_intent",
]
