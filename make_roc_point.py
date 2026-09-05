"""
make_roc_point.py

Plots the classifier's single operating point in ROC space (false
positive rate vs true positive rate), against the diagonal reference
line representing random guessing.

IMPORTANT LIMITATION, honestly stated: this is NOT a true ROC curve.
A real ROC curve requires a continuous confidence score that can be
swept across multiple thresholds, producing many points. The current
semantic_intent_classifier.py only outputs a flat YES/NO with no
confidence score, so there is exactly one point to plot. Calling this
a "ROC curve" in a report would overstate what was actually measured --
label it as an "ROC space operating point" instead.

Install matplotlib first if you don't have it:
    pip install matplotlib --break-system-packages

Run it:
    python3 make_roc_point.py

Saves roc_point.png in the same folder.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- EDIT THESE if you re-run test_semantic_classifier.py and get different counts ----
true_positives = 15
false_negatives = 0
false_positives = 6
true_negatives = 17
# -----------------------------------------------------------------------------------

tpr = true_positives / (true_positives + false_negatives)  # sensitivity
fpr = false_positives / (false_positives + true_negatives)  # 1 - specificity

fig, ax = plt.subplots(figsize=(6, 6))

# diagonal reference line (random guessing)
ax.plot([0, 1], [0, 1], linestyle="--", color="#B4B2A9", linewidth=1.5, label="Random guessing")

# the single real operating point
ax.scatter([fpr], [tpr], s=140, color="#2A78D6", edgecolor="white", linewidth=2, zorder=5, label="Classifier operating point")
ax.annotate(f"({fpr:.2f}, {tpr:.2f})", (fpr, tpr), textcoords="offset points", xytext=(12, -6), fontsize=10, color="#2A78D6")

ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.set_xlabel("False positive rate", fontsize=11)
ax.set_ylabel("True positive rate", fontsize=11)
ax.set_title("Classifier Position in ROC Space\n(single point, not a full curve -- see script docstring)", fontsize=12, fontweight="bold")
ax.legend(loc="lower right", frameon=False, fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig("roc_point.png", dpi=200, bbox_inches="tight")
print("Saved roc_point.png")
print(f"\nTrue positive rate (sensitivity): {tpr:.3f}")
print(f"False positive rate (1 - specificity): {fpr:.3f}")
print("\nReminder: this is one operating point, not a full ROC curve. A real curve")
print("needs a confidence score to sweep across thresholds -- the current classifier")
print("only outputs a flat YES/NO.")
