"""
semantic_intent_classifier.py

Layer (b) of the dual-layer safety system: a prompted local LLM used
as a semantic classifier to catch paraphrased or implicit acuity
signals that the deterministic keyword layer (keyword_redflag_layer.py)
would miss -- e.g., "I feel like something is really wrong with my
heart and I can't catch my breath" rather than the literal phrase
"chest pain."

CHANGE FROM PRIOR VERSION: this now classifies into three tiers
(routine / urgent / emergency) instead of a binary emergency yes/no.
The binary version conflated "needs prompt care soon" with "needs
immediate care right now," which Phase 4's false-positive analysis
identified as a major source of over-escalation. classify_tier()
replaces classify_intent() as the primary entry point;
classify_intent() is kept as a thin backward-compatible wrapper in
case other code still calls it directly.

This is a PROMPTED classifier, not a fine-tuned one -- same scope
decision as before (see Phase I hardware constraint documentation).
Per Phase I's finding that a 3B LLM judge was unreliable for retrieval
relevance judgment (55% agreement with human review), this
classifier's output should NOT be trusted blindly for a safety-critical
task -- it runs alongside, never instead of, the deterministic keyword
layer, and its reliability should be validated with
validate_intent_classifier() (now updated for three-way agreement)
before being relied upon.

REPRODUCIBILITY FIX: prior versions of this file called ollama.chat()
with no temperature or seed specified, meaning each call sampled from
Ollama's default (non-zero-temperature) settings. This caused
significant run-to-run variance -- the SAME unchanged code and prompt
produced accuracy figures 9+ points apart across two otherwise
identical 210-case evaluation runs (78.1% vs 69.5%). For a
safety-critical classifier, that variance is itself a finding that
needs to be measured and reported, not something to average away
silently. GENERATION_OPTIONS below fixes temperature to a low, non-zero
value (0.1 -- fully deterministic temperature=0 can behave oddly with
some Ollama model backends, so a low-but-nonzero value is the safer
choice) and sets a fixed seed, so repeated runs on the same input
should now produce the same tier the large majority of the time.
Re-run validate_intent_classifier() or the full 210-case evaluation
at least twice after this change to confirm variance has actually
dropped before trusting any single accuracy number, including this
version's.
"""

import re
import ollama

CLASSIFIER_MODEL = "qwen2.5"  # matches the model already validated in Phase I

# Low temperature (not zero -- some Ollama backends handle temperature=0
# inconsistently) plus a fixed seed, specifically to make repeated runs
# on the same input reproducible. If you still see meaningful variance
# after this change, that's worth reporting as a real property of the
# system, not a bug to keep chasing.
GENERATION_OPTIONS = {
    "temperature": 0.1,
    "seed": 42,
}

TIERS = ["routine", "urgent", "emergency"]  # fixed ordinal order


def classify_tier(message, model=CLASSIFIER_MODEL):
    """Asks the local LLM to classify a message into one of three
    acuity tiers, even if it doesn't use direct red-flag language.

    Returns (tier: str, reasoning: str) where tier is one of
    'routine', 'urgent', 'emergency'.

    Uses chain-of-thought (reasoning before answering) since this was
    shown in Phase I to improve small-model judgment reliability versus
    a direct answer-only prompt.
    """
    prompt = (
        f"A person said the following in a medical triage conversation:\n\n"
        f"\"{message}\"\n\n"
        f"Classify the acuity of this statement into exactly ONE of these three "
        f"tiers, considering paraphrased, vague, or implicit descriptions of "
        f"symptoms as well as explicit ones:\n\n"
        f"EMERGENCY - an immediate, potentially life-threatening situation that "
        f"cannot wait (e.g. chest pain suggestive of a heart attack, stroke signs, "
        f"severe breathing difficulty, active suicidal intent, severe uncontrolled "
        f"bleeding, anaphylaxis).\n"
        f"URGENT - concerning and needs medical attention soon (within hours to a "
        f"day), but is not an immediate life threat (e.g. a worsening infection, "
        f"new but non-severe neurological symptoms, significant but stable pain).\n"
        f"ROUTINE - mild, stable, or chronic, and can reasonably wait for a "
        f"regular scheduled appointment.\n\n"
        f"If you are uncertain between two tiers, choose the HIGHER-acuity one -- "
        f"erring toward more urgent is the safer default here.\n\n"
        f"Think briefly about your reasoning, then on a new line write exactly: "
        f"Final answer: EMERGENCY or Final answer: URGENT or Final answer: ROUTINE"
    )
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options=GENERATION_OPTIONS,
    )
    reply = response["message"]["content"]

    cleaned = reply.strip().upper()
    match = re.search(r"(?:FINAL ANSWER|ANSWER)[:\s]*\(?(EMERGENCY|URGENT|ROUTINE)\)?", cleaned)
    if match:
        tier = match.group(1).lower()
    else:
        # fallback: last standalone tier word mentioned, consistent with the
        # last-mention approach validated in Phase I's
        # rag_test_batch_llm_judge_v2.py, extended from YES/NO to three terms.
        positions = {t: cleaned.rfind(t.upper()) for t in TIERS}
        tier = max(positions, key=positions.get)
        if positions[tier] == -1:
            # Model produced no recognizable tier at all -- do not silently
            # default to 'routine'. Fail safe toward 'urgent' and flag it,
            # since silently under-triaging an unparseable response is the
            # one failure mode this whole layer exists to avoid.
            tier = "urgent"
            reply += "\n\n[PARSE_FAILURE: no tier keyword found in model output, defaulted to 'urgent']"

    return tier, reply


def classify_intent(message, model=CLASSIFIER_MODEL):
    """Backward-compatible binary wrapper around classify_tier(), for any
    other code still calling the old interface directly. Prefer calling
    classify_tier() in new code -- this collapses urgent and emergency
    into a single True, which is exactly the granularity loss the
    three-tier upgrade is meant to fix.
    """
    tier, reasoning = classify_tier(message, model)
    return tier == "emergency", reasoning


def validate_intent_classifier(test_cases, model=CLASSIFIER_MODEL):
    """Runs the classifier against a labeled test set and reports
    accuracy -- the same reliability-checking discipline established
    in Phase I (check_judge_agreement.py), extended to three-way
    agreement instead of binary.

    test_cases: list of (message, expected_tier) tuples, where
    expected_tier is one of 'routine', 'urgent', 'emergency'.
    """
    correct = 0
    results = []
    for message, expected in test_cases:
        if expected not in TIERS:
            raise ValueError(f"expected_tier must be one of {TIERS}, got {expected!r}")
        predicted, reasoning = classify_tier(message, model)
        is_correct = predicted == expected
        correct += int(is_correct)
        results.append({
            "message": message,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
        })
        print(f"{'OK' if is_correct else 'MISS'}: expected={expected}, got={predicted} - {message[:60]}")

    accuracy = round((correct / len(test_cases)) * 100, 1)
    print(f"\nClassifier accuracy on {len(test_cases)} labeled test cases: {accuracy}%")
    print("If this is not high (e.g. below ~90%), do not rely on this layer alone --")
    print("this is exactly the kind of number that should be reported alongside any")
    print("safety-layer accuracy claim, per the judge-reliability lesson from Phase I.")
    print("NOTE: for the full metric picture (which errors are dangerous vs merely")
    print("costly), run this same test set through triage_metrics.evaluate() rather")
    print("than relying on plain accuracy alone.")
    return accuracy, results
