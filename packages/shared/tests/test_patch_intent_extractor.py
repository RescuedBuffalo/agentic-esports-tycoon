"""``extract_patch_intent`` unit tests.

Covers the BUF-24 contract:

* Calls go through the budget governor with ``purpose="patch_intent"``.
* The system prompt is wrapped in ``cache_control`` (1h TTL) so a
  corpus-wide re-classification amortises the rubric across patches.
* Temperature defaults to 0.1 per the spec.
* Output JSON is parsed and validated through
  :class:`PatchIntentResult`; malformed shapes raise typed errors.
* The single-patch cost ceiling ($0.50) holds for realistic input
  sizes on Opus 4.7.
* The known-patch acceptance: a 5.12-shaped Chamber nerf produces a
  shift entry with ``direction="down"`` and ``magnitude="large"``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from esports_sim.budget import BudgetCaps, BudgetExhausted, Governor, Ledger
from esports_sim.patch_intent import PROMPT_VERSION, extract_patch_intent
from esports_sim.patch_intent.extractor import (
    DEFAULT_TEMPERATURE,
    SYSTEM_CACHE_TTL,
    _parse_json,
)
from pydantic import ValidationError

# --- helpers --------------------------------------------------------------


def _canonical_response_payload() -> dict:
    return {
        "primary_intent": "nerf-meta-outlier",
        "pro_play_driven_score": 0.9,
        "agents_affected": ["Chamber"],
        "maps_affected": [],
        "econ_changed": False,
        "expected_pickrate_shifts": [
            {
                "subject": "Chamber",
                "direction": "down",
                "magnitude": "large",
                "rationale": "Trademark rework + ult cost +1.",
            }
        ],
        "community_controversy_predicted": 0.85,
        "confidence": 0.8,
        "reasoning": "Classic 5.12-style Chamber nerf — pro play driven.",
    }


def _make_mock_client(
    *,
    response_text: str | None = None,
    input_tokens_estimate: int = 5_000,
    actual_input_tokens: int = 4_800,
    actual_output_tokens: int = 600,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
    request_id: str = "req_patch_intent",
) -> MagicMock:
    client = MagicMock(name="anthropic_client")
    client.messages.count_tokens.return_value = SimpleNamespace(
        input_tokens=input_tokens_estimate,
    )
    text = response_text if response_text is not None else json.dumps(_canonical_response_payload())
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=actual_input_tokens,
            output_tokens=actual_output_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        ),
        _request_id=request_id,
    )
    client.messages.create.return_value = response
    return client


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


@pytest.fixture
def governor(ledger: Ledger) -> Governor:
    return Governor(ledger=ledger, caps=BudgetCaps(weekly_hard_cap_usd=30.0))


# --- contract: prompt shape, governor, cache_control ----------------------


def test_extractor_routes_through_budget_governor_with_patch_intent_purpose(
    governor: Governor,
) -> None:
    """Every call must be attributed to ``purpose="patch_intent"`` so the
    per-purpose soft cap (BUF-22 default $3/wk) bites correctly."""
    client = _make_mock_client()
    extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12: Chamber adjustments...",
        client=client,
    )

    # The governor wrote a ledger row with the right purpose.
    rows = governor.ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].purpose == "patch_intent"


def test_extractor_attaches_cache_control_to_system_prompt(governor: Governor) -> None:
    """The system prompt is the cacheable payload — verify the wrapper
    sees a 1h ``cache_control`` block."""
    client = _make_mock_client()
    extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12: Chamber adjustments...",
        client=client,
    )

    sent_kwargs = client.messages.create.call_args.kwargs
    system_blocks = sent_kwargs["system"]
    assert isinstance(system_blocks, list)
    assert len(system_blocks) == 1
    block = system_blocks[0]
    assert block["type"] == "text"
    assert block["text"].strip(), "system prompt body must be non-empty"
    assert block["cache_control"] == {"type": "ephemeral", "ttl": SYSTEM_CACHE_TTL}


def test_extractor_uses_temperature_0_1_per_spec(governor: Governor) -> None:
    """BUF-24 spec: temperature 0.1."""
    client = _make_mock_client()
    extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12: Chamber adjustments...",
        client=client,
    )

    sent_kwargs = client.messages.create.call_args.kwargs
    assert sent_kwargs["temperature"] == DEFAULT_TEMPERATURE
    assert pytest.approx(0.1) == DEFAULT_TEMPERATURE


def test_extractor_propagates_dev_blog_into_user_message(governor: Governor) -> None:
    """When a dev blog accompanies the patch, it lands in the user message."""
    client = _make_mock_client()
    extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12 notes",
        dev_blog_text="Why we touched Chamber: pro feedback.",
        client=client,
    )

    sent_kwargs = client.messages.create.call_args.kwargs
    messages = sent_kwargs["messages"]
    user_text = messages[0]["content"]
    assert "Patch 5.12 notes" in user_text
    assert "DEV BLOG" in user_text
    assert "Why we touched Chamber" in user_text


def test_extractor_rejects_empty_patch_notes(governor: Governor) -> None:
    """Empty input is a caller bug; never pay for a Claude call on it."""
    client = _make_mock_client()
    with pytest.raises(ValueError, match="empty"):
        extract_patch_intent(
            governor=governor,
            patch_notes_text="   ",
            client=client,
        )
    # The SDK was never contacted — the check happens before the
    # governor is even called.
    assert not client.messages.count_tokens.called
    assert not client.messages.create.called


# --- contract: parsing + validation ---------------------------------------


def test_extractor_returns_validated_pydantic_result(governor: Governor) -> None:
    client = _make_mock_client()
    outcome = extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12 Chamber nerf",
        client=client,
    )
    assert outcome.result.primary_intent == "nerf-meta-outlier"
    assert outcome.prompt_version == PROMPT_VERSION
    assert outcome.input_tokens == 4_800
    assert outcome.output_tokens == 600


def test_extractor_strips_stray_code_fence(governor: Governor) -> None:
    """The system prompt forbids fences but defensive stripping prevents
    a one-time prompt drift from tipping the pipeline over."""
    fenced = "```json\n" + json.dumps(_canonical_response_payload()) + "\n```"
    client = _make_mock_client(response_text=fenced)
    outcome = extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12 Chamber nerf",
        client=client,
    )
    assert outcome.result.primary_intent == "nerf-meta-outlier"


def test_extractor_raises_on_invalid_json(governor: Governor) -> None:
    client = _make_mock_client(response_text="not json")
    with pytest.raises(json.JSONDecodeError):
        extract_patch_intent(
            governor=governor,
            patch_notes_text="Patch 5.12 Chamber nerf",
            client=client,
        )


def test_extractor_raises_on_schema_mismatch(governor: Governor) -> None:
    """A model regression that drops a required field fails as a typed
    ValidationError, not a downstream KeyError."""
    payload = _canonical_response_payload()
    del payload["confidence"]
    client = _make_mock_client(response_text=json.dumps(payload))
    with pytest.raises(ValidationError):
        extract_patch_intent(
            governor=governor,
            patch_notes_text="Patch 5.12 Chamber nerf",
            client=client,
        )


def test_extractor_raises_when_response_is_a_list(governor: Governor) -> None:
    """Non-object JSON normalised to a clear shape error."""
    client = _make_mock_client(response_text="[1,2,3]")
    with pytest.raises(ValueError, match="not a JSON object"):
        extract_patch_intent(
            governor=governor,
            patch_notes_text="Patch 5.12 Chamber nerf",
            client=client,
        )


# --- contract: BUF-22 cap enforcement -------------------------------------


def test_extractor_blocks_at_per_purpose_soft_cap(ledger: Ledger) -> None:
    """The default $3/wk ``patch_intent`` soft cap blocks an oversize call.

    Seeds the ledger to within $0.05 of the soft cap, then sends a
    realistic-sized request (50k input tokens, cached system prompt,
    1h TTL). The cache-aware preflight prices the input at $5 * 2.0 /
    MTok which lands above $0.05 and trips the per-purpose cap.
    """
    # Use the default caps so the per-purpose $3/wk soft cap is active.
    gov = Governor(ledger=ledger)
    # Spend $2.95 already on patch_intent — 5 cents headroom.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="post",
        usd_cost=2.95,
    )
    client = _make_mock_client(input_tokens_estimate=50_000)
    with pytest.raises(BudgetExhausted) as exc_info:
        extract_patch_intent(
            governor=gov,
            patch_notes_text="Patch notes here",
            client=client,
        )
    # The per-purpose cap fired, not the global weekly one.
    assert exc_info.value.scope == "purpose"
    assert exc_info.value.purpose == "patch_intent"
    # The SDK was never called.
    assert not client.messages.create.called


# --- acceptance: cost ceiling ---------------------------------------------


def test_typical_patch_costs_under_50_cents_on_opus(governor: Governor) -> None:
    """BUF-24 acceptance: one patch costs <$0.50 in API spend.

    Conservative envelope:

    * Input: ~10k tokens of patch notes + ~1k system prompt = ~11k.
    * Output: ~1k tokens (full JSON object with reasoning).
    * Cache: system prompt cached after first patch in a batch (5m TTL,
      1.25× write multiplier). The first patch pays the write
      multiplier; subsequent patches pay the read rate.
    * Model: Opus 4.7 — $5 input / $25 output per million.

    Expected first-call upper bound (cache write 1h, full max_tokens
    output assumed at preflight time):

      11_000 * $5 * 2.0 + 2048 * $25 = $0.110 + $0.0512 = $0.161

    Which is well under $0.50. Reconciled cost (actual usage) is
    even lower.
    """
    client = _make_mock_client(
        input_tokens_estimate=11_000,
        actual_input_tokens=10_000,
        actual_output_tokens=800,
        cache_creation_input_tokens=1_000,  # system prompt write
    )
    outcome = extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12 long body...",
        client=client,
    )
    assert (
        outcome.usd_cost < 0.50
    ), f"single-patch cost {outcome.usd_cost:.4f} exceeded the BUF-24 $0.50 ceiling"

    # Pre-flight projection on the ledger row (which is what the cap
    # checked against) is also under $0.50.
    rows = governor.ledger.all_entries()
    assert len(rows) == 1
    assert rows[0].usd_cost < 0.50


def test_warmed_cache_cost_is_dominated_by_per_patch_input(governor: Governor) -> None:
    """Subsequent calls within the cache window read the system prompt
    at 0.1× the base rate — verify that translates to a meaningful
    per-patch saving on the reconciled cost."""
    # Cache hit: system prompt 1k tokens read from cache, only the
    # per-patch input is fresh (10k tokens).
    client = _make_mock_client(
        input_tokens_estimate=11_000,
        actual_input_tokens=10_000,
        actual_output_tokens=800,
        cache_read_input_tokens=1_000,  # system prompt cache hit
    )
    outcome = extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 8.05 long body...",
        client=client,
    )
    # Opus 4.7: 10k input * $5 + 1k cache-read * $0.5 + 800 out * $25
    # = $0.050 + $0.0005 + $0.020 = $0.0705
    assert outcome.usd_cost == pytest.approx(0.0705, abs=1e-4)
    assert outcome.usd_cost < 0.10


# --- acceptance: 5.12-style Chamber nerf ----------------------------------


def test_chamber_nerf_acceptance_shape(governor: Governor) -> None:
    """BUF-24 acceptance bullet: the 5.12 Chamber nerf, when classified,
    must list Chamber with ``direction='down'`` and ``magnitude='large'``.

    Drives the extractor with a canned response that matches the
    expected acceptance shape so the test pins the schema/parsing path
    without requiring a live model.
    """
    client = _make_mock_client(response_text=json.dumps(_canonical_response_payload()))
    outcome = extract_patch_intent(
        governor=governor,
        patch_notes_text="Patch 5.12 Chamber Trademark + ult cost adjustments...",
        client=client,
    )
    chamber_shifts = [s for s in outcome.result.expected_pickrate_shifts if s.subject == "Chamber"]
    assert len(chamber_shifts) == 1
    assert chamber_shifts[0].direction == "down"
    assert chamber_shifts[0].magnitude == "large"


# --- private helper smoke -------------------------------------------------


def test_parse_json_strips_bare_code_fence() -> None:
    """``_parse_json`` handles ``` without ``json`` lang-tag."""
    fenced = "```\n" + json.dumps({"primary_intent": "x"}) + "\n```"
    assert _parse_json(fenced) == {"primary_intent": "x"}


def test_parse_json_passes_through_unwrapped() -> None:
    raw = json.dumps({"a": 1})
    assert _parse_json(raw) == {"a": 1}
