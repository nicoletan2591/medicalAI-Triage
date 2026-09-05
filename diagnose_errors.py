"""
diagnose_errors.py

Breaks down every misclassification by WHICH LAYER caused it --
keyword layer, semantic layer, or the combination -- so you know
whether to fix regex patterns (cheap, deterministic) or the semantic
prompt (harder, less predictable) before spending time on either.

Requires predicted_tiers_raw.json (from rerun_testbed.py) and
triage_final_for_eval.csv (from merge_predictions.py) in this folder.

Run from your phase 5_core folder:
    python3 diagnose_errors.py
"""

import json
import pandas as pd
from collections import Counter

RAW_JSON = "predicted_tiers_raw.json"
EVAL_CSV = "triage_final_for_eval.csv"


def main():
    with open(RAW_JSON) as f:
        raw = json.load(f)
    truth_df = pd.read_csv(EVAL_CSV)
    true_tier_by_id = dict(zip(truth_df["vignette_id"], truth_df["true_tier"]))

    rows = []
    for r in raw:
        vid = r["vignette_id"]
        true_tier = true_tier_by_id.get(vid)
        pred_tier = r.get("predicted_tier")
        if true_tier is None or pred_tier is None:
            continue

        safety_detail = (r.get("raw_result") or {}).get("safety_detail", {}) or {}
        keyword_tier = safety_detail.get("keyword_tier")
        semantic_tier = safety_detail.get("semantic_tier")
        keyword_matches = safety_detail.get("keyword_matches", [])

        rows.append({
            "vignette_id": vid,
            "patient_statement": r.get("patient_statement", ""),
            "true_tier": true_tier,
            "predicted_tier": pred_tier,
            "keyword_tier": keyword_tier,
            "semantic_tier": semantic_tier,
            "keyword_matches": keyword_matches,
            "correct": true_tier == pred_tier,
        })

    df = pd.DataFrame(rows)
    errors = df[~df["correct"]].copy()

    print(f"Total: {len(df)}  |  Errors: {len(errors)}  |  Accuracy: {df['correct'].mean():.3f}\n")

    # ---- classify each error by which layer is responsible ----
    def attribute_error(row):
        kw, sem = row["keyword_tier"], row["semantic_tier"]
        kw_present = pd.notna(kw)
        sem_present = pd.notna(sem)
        if kw_present and not sem_present:
            return "keyword_only (semantic was skipped/not needed)"
        if not kw_present and sem_present:
            return "semantic_only (no keyword match at all -- keyword layer had nothing to work with)"
        if kw_present and sem_present:
            return "both_ran (combined via max -- check which one was wrong)"
        return "neither_layer_fired (result came from elsewhere -- check manually)"

    errors["attribution"] = errors.apply(attribute_error, axis=1)

    print("=== Error count by which layer(s) were involved ===")
    print(errors["attribution"].value_counts().to_string())
    print()

    # ---- specifically the emergency -> urgent misses (the critical safety gap) ----
    em_to_urgent = errors[(errors["true_tier"] == "emergency") & (errors["predicted_tier"] == "urgent")]
    print(f"=== Emergency->Urgent misses: {len(em_to_urgent)} ===")
    print(em_to_urgent["attribution"].value_counts().to_string())
    print()
    print("Detail (vignette_id | keyword_tier | semantic_tier | statement):")
    for _, row in em_to_urgent.iterrows():
        print(f"  {row['vignette_id']} | kw={row['keyword_tier']} | sem={row['semantic_tier']} | {row['patient_statement'][:80]}")
    print()

    # ---- specifically the routine -> urgent misses (lower stakes but still worth checking) ----
    routine_to_urgent = errors[(errors["true_tier"] == "routine") & (errors["predicted_tier"] == "urgent")]
    print(f"=== Routine->Urgent misses: {len(routine_to_urgent)} ===")
    print(routine_to_urgent["attribution"].value_counts().to_string())
    print()
    # specifically flag ones caused by the keyword layer (deterministic, fixable by editing regex)
    kw_caused = routine_to_urgent[routine_to_urgent["keyword_tier"] == "urgent"]
    print(f"Of these, {len(kw_caused)} were triggered by the KEYWORD layer specifically (fixable by editing patterns):")
    for _, row in kw_caused.iterrows():
        print(f"  {row['vignette_id']} | matched: {row['keyword_matches']} | {row['patient_statement'][:80]}")
    print()

    # ---- save full error detail to CSV for further inspection ----
    errors_out = errors[[
        "vignette_id", "true_tier", "predicted_tier", "keyword_tier",
        "semantic_tier", "attribution", "keyword_matches", "patient_statement",
    ]]
    errors_out.to_csv("error_diagnosis.csv", index=False)
    print(f"Wrote error_diagnosis.csv with all {len(errors)} error rows for manual review.")


if __name__ == "__main__":
    main()
