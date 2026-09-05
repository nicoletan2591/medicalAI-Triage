"""
merge_predictions.py

Merges predicted_tiers.csv (output of rerun_testbed.py) into
triage_relabeled_draft.csv, joining on vignette_id, producing a final
CSV ready for run_eval.py.

Run from the same folder as both CSVs:
    python3 merge_predictions.py
"""

import pandas as pd
import sys

TRUTH_CSV = "triage_relabeled_draft.csv"
PRED_CSV = "predicted_tiers.csv"
OUTPUT_CSV = "triage_final_for_eval.csv"


def main():
    try:
        truth = pd.read_csv(TRUTH_CSV)
    except FileNotFoundError:
        sys.exit(f"Could not find {TRUTH_CSV} in this folder.")
    try:
        pred = pd.read_csv(PRED_CSV)
    except FileNotFoundError:
        sys.exit(f"Could not find {PRED_CSV} -- run rerun_testbed.py first.")

    if "true_tier" not in truth.columns:
        if "draft_true_tier" in truth.columns:
            sys.exit(
                f"{TRUTH_CSV} has 'draft_true_tier' but not 'true_tier'. "
                f"Rename the column once you've reviewed the draft labels:\n"
                f'  python3 -c "import pandas as pd; df = pd.read_csv(\'{TRUTH_CSV}\'); '
                f"df.rename(columns={{'draft_true_tier':'true_tier'}}).to_csv('{TRUTH_CSV}', index=False)\""
            )
        sys.exit(f"{TRUTH_CSV} has no 'true_tier' or 'draft_true_tier' column -- check the file.")

    merged = truth.merge(
        pred[["vignette_id", "predicted_tier"]],
        on="vignette_id",
        how="left",
    )

    missing = merged["predicted_tier"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no matching prediction (vignette_id mismatch or error during rerun).")
        print("These rows will be dropped before evaluation -- check predicted_tiers_raw.json for errors.")
        merged = merged.dropna(subset=["predicted_tier"])

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV} with {len(merged)} rows ready for evaluation.")
    print(f"\nNext: python3 run_eval.py {OUTPUT_CSV} --true-col true_tier --pred-col predicted_tier --category-col system_mode")


if __name__ == "__main__":
    main()
