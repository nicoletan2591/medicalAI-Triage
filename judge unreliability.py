"""
make_judge_agreement_breakdown_chart.py

Generates a horizontal stacked bar showing the raw agree/disagree
counts from check_judge_agreement.py's 20-query sample -- the actual
numbers behind the 55% agreement figure.

Install matplotlib first if you don't have it:
    pip install matplotlib --break-system-packages

Run it:
    python3 make_judge_agreement_breakdown_chart.py

Saves judge_agreement_breakdown.png in the same folder.
"""

import matplotlib
matplotlib.use("Agg")  # renders to a file instead of trying to open a window
import matplotlib.pyplot as plt

# ---- EDIT THESE if you re-run check_judge_agreement.py and get different counts ----
agreed = 11
disagreed = 9
total = agreed + disagreed
# --------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 2.8))

ax.barh(["20-query sample"], [agreed], color="#1BAF7A", edgecolor="#333", linewidth=1.0, label=f"Agreed ({agreed})")
ax.barh(["20-query sample"], [disagreed], left=[agreed], color="#E34948", edgecolor="#333", linewidth=1.0, label=f"Disagreed ({disagreed})")

# labels centered inside each segment
ax.text(agreed / 2, 0, str(agreed), ha="center", va="center", fontsize=13, fontweight="bold", color="white")
ax.text(agreed + disagreed / 2, 0, str(disagreed), ha="center", va="center", fontsize=13, fontweight="bold", color="white")

ax.set_xlim(0, total)
ax.set_xlabel(f"Out of {total} sampled queries", fontsize=11, labelpad=10)
ax.set_title("Judge vs. Human Agreement Breakdown", fontsize=13, fontweight="bold", pad=14)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=2, frameon=False, fontsize=10)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.set_yticks([])

plt.tight_layout()
plt.subplots_adjust(bottom=0.32)
plt.savefig("judge_agreement_breakdown.png", dpi=200, bbox_inches="tight")
print("Saved judge_agreement_breakdown.png")
