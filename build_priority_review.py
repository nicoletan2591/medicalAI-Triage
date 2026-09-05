"""
build_priority_review.py

Rebuilds priority_review.csv from triage_final_for_eval.csv.

Flags a case for review if either:
  - draft_confidence == 'L' (low-confidence draft label), or
  - true_tier != 'emergency' but the ORIGINAL binary system escalated
    it anyway (system_escalated_binary == True) -- i.e. the AI draft
    and the existing binary system disagreed on whether this was an
    emergency.

Run:
    python3 build_priority_review.py
"""

import pandas as pd

SOURCE_CSV = "triage_final_for_eval.csv"
OUTPUT_CSV = "priority_review.csv"


def main():
    df = pd.read_csv(SOURCE_CSV)

    # normalize the binary column: it may load as bool, string "True"/"False", or NaN
    escalated = df["system_escalated_binary"].astype(str).str.strip().str.lower() == "true"

    low_confidence = df["draft_confidence"] == "L"
    system_disagreed = (df["true_tier"] != "emergency") & escalated

    def build_reason(row_low, row_disagree):
        reasons = []
        if row_low:
            reasons.append("low_confidence_draft")
        if row_disagree:
            reasons.append("system_over_escalated")
        return "; ".join(reasons)

    df["review_reason"] = [
        build_reason(lc, sd) for lc, sd in zip(low_confidence, system_disagreed)
    ]

    flagged = df[low_confidence | system_disagreed].copy()

    # sort: low-confidence + disagreement first, then disagreement-only, then low-confidence-only
    def sort_key(reason):
        has_low = "low_confidence_draft" in reason
        has_dis = "system_over_escalated" in reason
        if has_low and has_dis:
            return 0
        if has_low:
            return 1
        return 2
    flagged["_sort"] = flagged["review_reason"].apply(sort_key)
    flagged = flagged.sort_values("_sort").drop(columns="_sort")

    flagged.to_csv(OUTPUT_CSV, index=False)

    n_low = int(low_confidence.sum())
    n_dis = int(system_disagreed.sum())
    n_both = int((low_confidence & system_disagreed).sum())
    n_total = len(flagged)

    print(f"Low-confidence drafts: {n_low}")
    print(f"System-disagreement cases: {n_dis}")
    print(f"Overlap (both): {n_both}")
    print(f"Total unique flagged: {n_total}  (check: {n_low} + {n_dis} - {n_both} = {n_low + n_dis - n_both})")
    print(f"\nWrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
