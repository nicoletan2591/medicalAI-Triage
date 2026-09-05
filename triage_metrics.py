"""
triage_metrics.py

Evaluation metrics for the three-tier (routine / urgent / emergency)
triage classification output, extending the binary confusion-matrix
work in phase_4_core (sensitivity/specificity/ROC on emergency vs.
safe) to the three-way output the Phase 4 false-positive analysis
recommended.

Design rationale (see README.md for full detail):
  - Tiers are ORDINAL: routine < urgent < emergency.
  - Errors are ASYMMETRIC in cost: under-triage (true emergency
    predicted as routine) is far worse than over-triage (true routine
    predicted as emergency).
  - Therefore plain accuracy / macro-F1 alone is not sufficient; this
    module reports those PLUS an ordinal agreement measure (quadratic
    weighted kappa), a configurable cost-weighted score, and a
    standalone missed-emergency rate that is never averaged away.

No dependency on any project-internal module — this only needs a
CSV/DataFrame with true and predicted tier columns. Wire it to your
real testbed once relabeling is done (see relabel_template.csv).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
)
from scipy.stats import chi2
from dataclasses import dataclass, field
from typing import Optional

TIERS = ["routine", "urgent", "emergency"]  # fixed ordinal order, do not reorder
TIER_INDEX = {t: i for i, t in enumerate(TIERS)}

# ---------------------------------------------------------------
# Cost matrix: rows = TRUE tier, cols = PREDICTED tier.
# Default reflects "under-triage costs much more than over-triage."
# Edit these numbers to match your thesis's stated risk framing —
# treat this as a design decision you defend in the methodology,
# not a fixed constant.
# ---------------------------------------------------------------
DEFAULT_COST_MATRIX = pd.DataFrame(
    [
        # pred:   routine  urgent  emergency
        [0,        1,      2],   # true: routine
        [3,        0,      1],   # true: urgent
        [10,       5,      0],   # true: emergency
    ],
    index=[f"true_{t}" for t in TIERS],
    columns=[f"pred_{t}" for t in TIERS],
)


@dataclass
class TierEvalResult:
    n: int
    confusion: pd.DataFrame
    per_class: pd.DataFrame
    accuracy: float
    macro_f1: float
    weighted_f1: float
    quadratic_weighted_kappa: float
    missed_emergency_rate: float
    missed_emergency_count: int
    total_true_emergencies: int
    mean_cost: float
    total_cost: float
    cost_matrix: pd.DataFrame
    by_category: Optional[pd.DataFrame] = None  # breakdown by standard/adversarial etc.


def _validate_tiers(series: pd.Series, name: str) -> pd.Series:
    s = series.str.lower().str.strip()
    bad = set(s.unique()) - set(TIERS)
    if bad:
        raise ValueError(
            f"Column '{name}' has values outside {TIERS}: {bad}. "
            f"Fix labels before scoring — do not silently coerce them."
        )
    return s


def evaluate(
    df: pd.DataFrame,
    true_col: str = "true_tier",
    pred_col: str = "predicted_tier",
    category_col: Optional[str] = "case_category",
    cost_matrix: pd.DataFrame = DEFAULT_COST_MATRIX,
) -> TierEvalResult:
    """
    df must contain at minimum `true_col` and `pred_col` with values in
    {'routine','urgent','emergency'} (case-insensitive). If
    `category_col` is present (e.g. 'standard'/'adversarial', matching
    phase_4_core's breakdown), a per-category missed-emergency
    breakdown is also produced — phase_4 found standard (non-adversarial)
    vignettes were the single biggest source of false positives, so
    that split is worth preserving here too.
    """
    y_true = _validate_tiers(df[true_col], true_col)
    y_pred = _validate_tiers(df[pred_col], pred_col)

    n = len(df)

    # ---- confusion matrix, fixed ordinal ordering ----
    cm = confusion_matrix(y_true, y_pred, labels=TIERS)
    cm_df = pd.DataFrame(
        cm, index=[f"true_{t}" for t in TIERS], columns=[f"pred_{t}" for t in TIERS]
    )

    # ---- per-class precision/recall/F1 ----
    report = classification_report(
        y_true, y_pred, labels=TIERS, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame(report).T.loc[TIERS + ["macro avg", "weighted avg"]]

    accuracy = report["accuracy"]
    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]

    # ---- ordinal agreement: quadratic weighted kappa ----
    y_true_ord = y_true.map(TIER_INDEX)
    y_pred_ord = y_pred.map(TIER_INDEX)
    qwk = cohen_kappa_score(y_true_ord, y_pred_ord, weights="quadratic")

    # ---- standalone safety number: missed-emergency rate ----
    # "Missed" = true emergency predicted as urgent OR routine (any
    # downgrade at all). This is reported on its own, never folded
    # into an averaged score, per the design principle in Phase II /
    # phase_4_core that a missed emergency costs far more than a
    # false alarm.
    true_emerg_mask = y_true == "emergency"
    total_true_emergencies = int(true_emerg_mask.sum())
    missed_mask = true_emerg_mask & (y_pred != "emergency")
    missed_count = int(missed_mask.sum())
    missed_rate = missed_count / total_true_emergencies if total_true_emergencies else float("nan")

    # ---- cost-weighted score ----
    costs = [
        cost_matrix.loc[f"true_{t}", f"pred_{p}"]
        for t, p in zip(y_true, y_pred)
    ]
    total_cost = float(np.sum(costs))
    mean_cost = float(np.mean(costs))

    # ---- optional per-category breakdown ----
    by_category = None
    if category_col and category_col in df.columns:
        rows = []
        for cat, sub in df.groupby(category_col):
            sub_true = _validate_tiers(sub[true_col], true_col)
            sub_pred = _validate_tiers(sub[pred_col], pred_col)
            sub_emerg_mask = sub_true == "emergency"
            sub_total_emerg = int(sub_emerg_mask.sum())
            sub_missed = int((sub_emerg_mask & (sub_pred != "emergency")).sum())
            sub_acc = float((sub_true == sub_pred).mean())
            rows.append({
                "category": cat,
                "n": len(sub),
                "accuracy": sub_acc,
                "true_emergencies": sub_total_emerg,
                "missed_emergencies": sub_missed,
                "missed_emergency_rate": (
                    sub_missed / sub_total_emerg if sub_total_emerg else float("nan")
                ),
            })
        by_category = pd.DataFrame(rows).set_index("category")

    return TierEvalResult(
        n=n,
        confusion=cm_df,
        per_class=per_class,
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        quadratic_weighted_kappa=qwk,
        missed_emergency_rate=missed_rate,
        missed_emergency_count=missed_count,
        total_true_emergencies=total_true_emergencies,
        mean_cost=mean_cost,
        total_cost=total_cost,
        cost_matrix=cost_matrix,
        by_category=by_category,
    )


def stuart_maxwell_test(pred_a: pd.Series, pred_b: pd.Series) -> dict:
    """
    Stuart-Maxwell test of marginal homogeneity between two PAIRED sets
    of predictions on the SAME cases (e.g. variant A's tier assignment
    vs. variant B's tier assignment on the identical 210-case testbed).

    This is the multi-class generalization of the McNemar's test your
    Phase III methodology already specifies for paired binary escalation
    outcomes — use this instead of McNemar's now that the output is
    three-way, not two-way.

    Use this to compare architectural variants (e.g. guardrail-on vs
    guardrail-off, or two quantization levels), NOT to compare a
    prediction against ground truth — that's what `evaluate()` above
    is for. Both series must be the same length, same case order,
    same label set.
    """
    a = _validate_tiers(pd.Series(pred_a), "pred_a").map(TIER_INDEX).to_numpy()
    b = _validate_tiers(pd.Series(pred_b), "pred_b").map(TIER_INDEX).to_numpy()
    if len(a) != len(b):
        raise ValueError("pred_a and pred_b must be the same length (paired cases).")

    k = len(TIERS)
    n_table = np.zeros((k, k), dtype=int)
    for ai, bi in zip(a, b):
        n_table[ai, bi] += 1

    row_sums = n_table.sum(axis=1)  # variant A's marginal tier counts
    col_sums = n_table.sum(axis=0)  # variant B's marginal tier counts
    d = (row_sums - col_sums)[:-1].astype(float)  # drop last (linearly dependent)

    # Covariance matrix (Stuart-Maxwell / Bhapkar formulation)
    S = np.zeros((k - 1, k - 1))
    for i in range(k - 1):
        S[i, i] = row_sums[i] + col_sums[i] - 2 * n_table[i, i]
        for j in range(k - 1):
            if i != j:
                S[i, j] = -(n_table[i, j] + n_table[j, i])

    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        return {
            "statistic": float("nan"),
            "df": k - 1,
            "p_value": float("nan"),
            "note": "Covariance matrix singular (often means A and B never disagreed "
                    "on some tier pair) — inspect the raw cross-tab instead.",
        }

    statistic = float(d @ S_inv @ d)
    df = k - 1
    p_value = float(1 - chi2.cdf(statistic, df))

    return {
        "statistic": statistic,
        "df": df,
        "p_value": p_value,
        "cross_tab": pd.DataFrame(
            n_table,
            index=[f"A_{t}" for t in TIERS],
            columns=[f"B_{t}" for t in TIERS],
        ),
        "interpretation": (
            "Reject H0 (marginal homogeneity) at alpha=0.05 -> the two variants' "
            "tier-assignment distributions differ significantly."
            if p_value < 0.05 else
            "Fail to reject H0 at alpha=0.05 -> no significant difference detected "
            "in tier-assignment distribution between the two variants."
        ),
    }


def print_report(result: TierEvalResult) -> None:
    print(f"\n{'='*60}\nTHREE-TIER TRIAGE EVALUATION  (n={result.n})\n{'='*60}")
    print("\n--- Confusion Matrix (rows=true, cols=predicted) ---")
    print(result.confusion.to_string())
    print("\n--- Per-Class Precision / Recall / F1 ---")
    print(result.per_class.round(3).to_string())
    print(f"\nOverall accuracy:            {result.accuracy:.3f}")
    print(f"Macro F1:                    {result.macro_f1:.3f}")
    print(f"Weighted F1:                 {result.weighted_f1:.3f}")
    print(f"Quadratic Weighted Kappa:    {result.quadratic_weighted_kappa:.3f}")
    print(f"\n*** Missed-Emergency Rate:   {result.missed_emergency_rate:.3f} "
          f"({result.missed_emergency_count}/{result.total_true_emergencies}) ***")
    print(f"Mean cost per case:          {result.mean_cost:.3f}")
    print(f"Total cost:                  {result.total_cost:.1f}")
    if result.by_category is not None:
        print("\n--- Breakdown by Case Category ---")
        print(result.by_category.round(3).to_string())
    print()
