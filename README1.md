# Three-Tier Safety Layer Upgrade

Upgrades `dual_layer_safety.py` and `semantic_intent_classifier.py` from
binary escalate/don't-escalate to three-way routine/urgent/emergency
output, per Phase 4's false-positive analysis recommendation.

## Files

| File | Status |
|---|---|
| `semantic_intent_classifier.py` | Rewritten. `classify_tier()` replaces `classify_intent()` as the primary function (old one kept as a wrapper). |
| `dual_layer_safety.py` | Rewritten. `safety_check_tier()` replaces `safety_check()` (old one kept as a wrapper). |
| `keyword_redflag_layer.py` | **Template only — not your real file.** You didn't share the original, so this has placeholder terms in the right structure. **Replace `EMERGENCY_KEYWORDS` and `URGENT_KEYWORDS` with your actual curated list before using this for real.** |

Tested (with a mocked semantic layer, since this sandbox has no local
Ollama/model to call): short-circuit behavior, tier combination
(higher tier always wins, never gets silently downgraded), and both
backward-compatible wrappers. All passed.

## What changed and why

**Combining the two layers:** the old version did
`escalate = keyword_triggered or semantic_triggered`. The new version
takes the **higher** of the two layers' tiers — same principle
(neither layer can silently override a positive signal from the
other), extended from a boolean OR to an ordinal max.

**Short-circuit logic changed subtly.** The old version skipped the
semantic layer whenever the keyword layer triggered at all (since
either triggering meant `escalate=True` regardless). The new version
can only skip the semantic layer when the keyword layer hits
**emergency** specifically — because now, if the keyword layer says
"urgent," the semantic layer could still raise the combined result to
"emergency." Skipping it in that case would silently cap the result at
"urgent" even when the true tier is higher. This is a genuine
behavior change from the binary version, not just a rename — worth
noting explicitly in your methodology as a deliberate fix, since it
affects correctness, not just output granularity.

**Semantic layer prompt** changed from "is this an emergency, yes/no"
to "classify into routine/urgent/emergency, erring toward the higher
tier when uncertain" — same chain-of-thought structure Phase I found
improved small-model reliability, just extended to three options.

**Fail-safe parsing:** if the model's response doesn't contain a
recognizable tier word at all, the old code's fallback logic
(`last_yes > last_no`) would have silently defaulted toward NO in a
tie. The new version defaults an unparseable response to **'urgent'**
(not 'routine') and flags it in the returned reasoning text —
deliberately avoiding a silent under-triage on parser failure.

## Required next steps (in order)

1. **Get your real `keyword_redflag_layer.py` term list into the
   template.** Split your existing flat list into `EMERGENCY_KEYWORDS`
   and `URGENT_KEYWORDS`. Per your own corpus's stated principle
   (erring toward higher acuity when uncertain), anything you're not
   sure how to tier goes in the higher bucket.

2. **Specifically re-examine "sunken fontanelle."** Phase 4 already
   flagged this exact term as a case where keyword presence
   (suggesting emergency) conflicted with the intended, lower true
   acuity. This is your first real test case for whether the two-tier
   keyword split actually fixes that conflict or just relocates it —
   worth checking by hand before trusting the rest of the list.

3. **Update `triage_agent.py`** (not shared, so not rewritten here) to
   call `safety_check_tier()` instead of `safety_check()`, and branch
   three ways: `emergency` → fixed response (as before), `urgent` →
   the new fixed `FIXED_URGENT_RESPONSE`, `routine` → continue to the
   existing RAG + clarifying-question flow. If other code still calls
   the old binary functions, it keeps working via the compatibility
   wrappers, but should be migrated to the tier-based interface — the
   wrappers exist for a smooth transition, not as a long-term crutch.

4. **Re-run all 210 vignettes** through the upgraded pipeline and save
   the three-way output as `predicted_tier` in your evaluation CSV.

5. **Run the real evaluation:**
   ```bash
   python3 run_eval.py triage_relabeled_draft.csv --true-col true_tier --pred-col predicted_tier --category-col system_mode
   ```
