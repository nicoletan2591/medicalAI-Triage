"""
rerun_testbed.py

Re-runs all 210 vignettes from phase3_agent_progress.json through the
upgraded three-tier triage_agent.py, and writes out predicted_tier
for each one -- ready to merge into triage_relabeled_draft.csv for
the real run_eval.py evaluation.

REQUIRES (all must be true on your machine, not this sandbox):
  - Ollama running locally with the qwen2.5 model pulled
  - chroma_db already built in this folder (from Phase I's build_index.py)
  - prompt_templates.py present in this folder
  - The four upgraded safety files already copied in:
      dual_layer_safety.py, semantic_intent_classifier.py,
      keyword_redflag_layer.py, triage_agent.py

Run from the same folder as triage_agent.py:
    python3 rerun_testbed.py

This will take a while -- 210 calls to a local model, one per
vignette. If a vignette hits 'urgent' or 'emergency', that's a single
fast call (safety layer short-circuits before generation). 'routine'
vignettes are slower (RAG retrieval + generation).

Output: predicted_tiers.csv with columns vignette_id, predicted_tier,
response, mode (if applicable). Also predicted_tiers_raw.json with the
full result dict per vignette, in case you want the safety_detail /
retrieved_context for your own inspection.
"""

import json
import csv
import sys

try:
    import triage_agent as ta
except ImportError as e:
    sys.exit(
        f"Could not import triage_agent: {e}\n"
        f"Make sure this script is in the same folder as triage_agent.py, "
        f"dual_layer_safety.py, semantic_intent_classifier.py, "
        f"keyword_redflag_layer.py, and prompt_templates.py."
    )

SOURCE_JSON = "phase3_agent_progress.json"  # your original 210-vignette file
OUTPUT_CSV = "predicted_tiers.csv"
OUTPUT_JSON = "predicted_tiers_raw.json"


def main():
    with open(SOURCE_JSON) as f:
        original = json.load(f)

    print(f"Loaded {len(original)} vignettes from {SOURCE_JSON}")

    results = []
    for i, rec in enumerate(original):
        statement = rec.get("_patient_statement", "").strip()
        vignette_id = f"V{i:03d}"

        if not statement:
            print(f"[{vignette_id}] SKIPPED - empty patient statement")
            continue

        # Fresh dialogue manager per vignette -- these are independent
        # single-turn test cases, not a continuous conversation.
        dm = ta.DialogueManager()
        try:
            result = ta.handle_message(statement, dm)
        except Exception as e:
            print(f"[{vignette_id}] ERROR: {e}")
            results.append({
                "vignette_id": vignette_id,
                "patient_statement": statement,
                "predicted_tier": None,
                "response": None,
                "mode": None,
                "error": str(e),
            })
            continue

        tier = result.get("tier")
        print(f"[{vignette_id}] {tier}  -  {statement[:70]}")

        results.append({
            "vignette_id": vignette_id,
            "patient_statement": statement,
            "predicted_tier": tier,
            "response": result.get("response"),
            "mode": result.get("mode"),
            "raw_result": result,
        })

    # Write CSV (for merging into triage_relabeled_draft.csv)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vignette_id", "predicted_tier", "response", "mode"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "vignette_id": r["vignette_id"],
                "predicted_tier": r["predicted_tier"],
                "response": r["response"],
                "mode": r["mode"],
            })

    # Write full raw JSON (includes safety_detail, retrieved_context, etc.)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    n_ok = sum(1 for r in results if r["predicted_tier"] is not None)
    n_err = len(results) - n_ok
    print(f"\nDone. {n_ok} succeeded, {n_err} errored.")
    print(f"Wrote {OUTPUT_CSV} and {OUTPUT_JSON}")
    if n_err:
        print("Check predicted_tiers_raw.json for the 'error' field on failed cases.")


if __name__ == "__main__":
    main()
