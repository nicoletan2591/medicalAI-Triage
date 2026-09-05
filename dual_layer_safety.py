"""
dual_layer_safety.py

Combines layer (a) keyword_redflag_layer and layer (b)
semantic_intent_classifier into the dual-layer safety system.

CHANGE FROM PRIOR VERSION: safety_check() now returns a three-way
tier (routine / urgent / emergency) instead of a binary escalate
flag. The design principle is unchanged and, if anything, matters
MORE now: a positive signal from either layer overrides generation
with a fixed, deterministic response -- for BOTH the urgent and
emergency tiers, not just emergency. Only 'routine' falls through to
RAG + generation. This keeps the safety-critical path auditable and
independent of generative model behavior, exactly as before.

safety_check() (the old binary-returning function) is kept as a thin
backward-compatible wrapper. Prefer safety_check_tier() in new code.
"""

from keyword_redflag_layer import check_keyword_tier
from semantic_intent_classifier import classify_tier, TIERS

# Fixed, deterministic responses -- plain text, not model output.
# What a person sees for a given tier does not vary based on model
# sampling, prompt sensitivity, or any other generative unpredictability.
FIXED_EMERGENCY_RESPONSE = (
    "This sounds like it could be a medical emergency. Please call your local "
    "emergency number or go to the nearest emergency department right away. "
    "Do not wait to see if symptoms improve on their own."
)
FIXED_URGENT_RESPONSE = (
    "This should be seen soon -- please contact your doctor or an urgent care "
    "clinic rather than waiting for a routine appointment. If it gets worse, "
    "seek emergency care."
)

_TIER_RANK = {t: i for i, t in enumerate(TIERS)}  # routine=0, urgent=1, emergency=2


def _higher_tier(a, b):
    """Returns whichever of two tiers is higher-acuity. None is treated
    as lower than any real tier (i.e. a skipped layer never overrides
    a triggered one)."""
    if a is None:
        return b
    if b is None:
        return a
    return a if _TIER_RANK[a] >= _TIER_RANK[b] else b


def safety_check_tier(message, use_semantic_layer=True):
    """Runs the dual-layer safety check on a user message and returns
    a three-way tier.

    Returns a dict:
        tier: str -- 'routine', 'urgent', or 'emergency' (combined result)
        response: str or None -- fixed response for 'urgent'/'emergency', else None
        keyword_tier: str or None -- tier from the keyword layer alone
        keyword_matches: list[str]
        semantic_tier: str or None (None if semantic layer skipped)
        semantic_reasoning: str or None

    The two layers are combined by taking the HIGHER-acuity tier of
    the two -- neither layer is allowed to downgrade what the other
    flagged. This preserves the original design principle (a positive
    signal from either layer cannot be silently overridden) in the
    three-tier setting.

    The keyword layer always runs (cheap, deterministic, no model
    call). The semantic layer can be skipped (use_semantic_layer=False)
    ONLY when the keyword layer already returned 'emergency' -- nothing
    the semantic layer says can lower that result, so the extra model
    call is pure latency cost in that specific case. For 'urgent' or
    no keyword signal, the semantic layer still needs to run, since it
    could raise the tier further (e.g. keyword says nothing, semantic
    layer catches a paraphrased emergency).
    """
    keyword_tier, keyword_matches = check_keyword_tier(message)

    semantic_tier = None
    semantic_reasoning = None

    # Short-circuit ONLY when the keyword layer already hit the
    # ceiling (emergency) -- the semantic layer cannot raise the tier
    # any further in that case. Any other keyword result (urgent or
    # None) still needs the semantic layer to run, since it could
    # raise -- not lower -- the combined tier.
    skip_semantic = (keyword_tier == "emergency") or not use_semantic_layer
    if not skip_semantic:
        semantic_tier, semantic_reasoning = classify_tier(message)

    combined_tier = _higher_tier(keyword_tier, semantic_tier)
    if combined_tier is None:
        combined_tier = "routine"

    if combined_tier == "emergency":
        response = FIXED_EMERGENCY_RESPONSE
    elif combined_tier == "urgent":
        response = FIXED_URGENT_RESPONSE
    else:
        response = None  # routine falls through to RAG + generation

    return {
        "tier": combined_tier,
        "response": response,
        "keyword_tier": keyword_tier,
        "keyword_matches": keyword_matches,
        "semantic_tier": semantic_tier,
        "semantic_reasoning": semantic_reasoning,
    }


def safety_check(message, use_semantic_layer=True):
    """Backward-compatible binary wrapper around safety_check_tier(),
    for any other code (e.g. an un-upgraded triage_agent.py) still
    calling the old interface. Prefer safety_check_tier() in new code
    -- this collapses urgent and emergency into a single
    escalate=True, which is exactly the granularity the three-tier
    upgrade exists to add back.
    """
    result = safety_check_tier(message, use_semantic_layer)
    escalate = result["tier"] in ("urgent", "emergency")
    return {
        "escalate": escalate,
        "response": result["response"] if escalate else None,
        "keyword_triggered": result["keyword_tier"] is not None,
        "keyword_matches": result["keyword_matches"],
        "semantic_triggered": (
            None if result["semantic_tier"] is None
            else result["semantic_tier"] in ("urgent", "emergency")
        ),
        "semantic_reasoning": result["semantic_reasoning"],
    }
