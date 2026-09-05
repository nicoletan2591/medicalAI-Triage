"""
run_eval.py

CLI entry point: score a three-tier triage testbed CSV against the
metrics in triage_metrics.py, save a text report + confusion matrix
heatmap + JSON summary.

Usage:
    python run_eval.py testbed_scored.csv
    python run_eval.py testbed_scored.csv --true-col true_tier --pred-col predicted_tier
    python run_eval.py --compare variant_a.csv variant_b.csv --pred-col predicted_tier

See relabel_template.csv for the expected input shape.
"""

import argparse
import json
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from triage_metrics import evaluate, stuart_maxwell_test, print_report, TIERS


def plot_confusion(cm_df: pd.DataFrame, cost_matrix: pd.DataFrame, out_path: str):
    """
    Colors cells by RISK (the cost matrix), not raw count. This matters:
    coloring by count alone makes the diagonal (correct calls) show up
    as the most intense color purely because it has the most cases,
    which reads backwards for a safety chart. Here, color always means
    "how dangerous is this cell" regardless of how often it occurs;
    the actual counts are printed as text on top.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cost_matrix.values, cmap="RdYlGn_r", vmin=0)
    ax.set_xticks(range(len(TIERS)))
    ax.set_yticks(range(len(TIERS)))
    ax.set_xticklabels(TIERS)
    ax.set_yticklabels(TIERS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Three-Tier Triage Confusion Matrix\n(color = risk cost, number = case count)")
    for i in range(len(TIERS)):
        for j in range(len(TIERS)):
            count = cm_df.values[i, j]
            cost = cost_matrix.values[i, j]
            # pick readable text color against the risk-colored background
            text_color = "white" if cost >= 5 else "black"
            ax.text(j, i - 0.08, str(count), ha="center", va="center",
                     color=text_color, fontsize=15, fontweight="bold")
            ax.text(j, i + 0.28, f"cost {cost}", ha="center", va="center",
                     color=text_color, fontsize=7.5, alpha=0.85)
    fig.colorbar(im, ax=ax, label="cost weight (risk)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Evaluate three-tier triage classification.")
    ap.add_argument("csv_path", nargs="?", help="Path to scored testbed CSV.")
    ap.add_argument("--true-col", default="true_tier")
    ap.add_argument("--pred-col", default="predicted_tier")
    ap.add_argument("--category-col", default="case_category")
    ap.add_argument("--out-prefix", default="eval_report")
    ap.add_argument("--compare", nargs=2, metavar=("CSV_A", "CSV_B"),
                     help="Run Stuart-Maxwell paired comparison between two variants' "
                          "predictions instead of a single-file evaluation.")
    args = ap.parse_args()

    if args.compare:
        df_a = pd.read_csv(args.compare[0])
        df_b = pd.read_csv(args.compare[1])
        if len(df_a) != len(df_b):
            sys.exit("--compare files must have the same number of rows (paired cases).")
        result = stuart_maxwell_test(df_a[args.pred_col], df_b[args.pred_col])
        print("\n=== Stuart-Maxwell Paired Comparison ===")
        print(result["cross_tab"].to_string())
        print(f"\nstatistic = {result['statistic']:.4f}, df = {result['df']}, "
              f"p = {result['p_value']:.4f}")
        print(result["interpretation"])
        return

    if not args.csv_path:
        sys.exit("Provide a CSV path, or use --compare A.csv B.csv.")

    df = pd.read_csv(args.csv_path)
    result = evaluate(df, true_col=args.true_col, pred_col=args.pred_col,
                       category_col=args.category_col)
    print_report(result)

    plot_confusion(result.confusion, result.cost_matrix, f"{args.out_prefix}_confusion.png")

    summary = {
        "n": result.n,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "weighted_f1": result.weighted_f1,
        "quadratic_weighted_kappa": result.quadratic_weighted_kappa,
        "missed_emergency_rate": result.missed_emergency_rate,
        "missed_emergency_count": result.missed_emergency_count,
        "total_true_emergencies": result.total_true_emergencies,
        "mean_cost": result.mean_cost,
        "total_cost": result.total_cost,
    }
    with open(f"{args.out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    result.confusion.to_csv(f"{args.out_prefix}_confusion.csv")
    result.per_class.to_csv(f"{args.out_prefix}_per_class.csv")
    if result.by_category is not None:
        result.by_category.to_csv(f"{args.out_prefix}_by_category.csv")

    print(f"Saved: {args.out_prefix}_confusion.png, _summary.json, _confusion.csv, "
          f"_per_class.csv" + (", _by_category.csv" if result.by_category is not None else ""))


if __name__ == "__main__":
    main()
