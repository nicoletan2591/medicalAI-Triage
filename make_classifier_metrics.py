"""
make_classifier_metrics.py

Generates a bar chart of sensitivity, specificity, precision, and F1
score for the semantic intent classifier, computed from the confusion
matrix in make_confusion_matrix.py (15 TP, 0 FN, 6 FP, 17 TN).

Install matplotlib first if you don't have it:
    pip install matplotlib --break-system-packages

Run it:
    python3 make_classifier_metrics.py

Saves classifier_metrics.png in the same folder.
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

sensitivity = true_positives / (true_positives + false_negatives) * 100  # aka recall
specificity = true_negatives / (true_negatives + false_positives) * 100
precision = true_positives / (true_positives + false_positives) * 100
recall = sensitivity  # same value, used in F1 calc for clarity
f1 = 2 * (precision * recall) / (precision + recall)

labels = ["Sensitivity\n(recall)", "Specificity", "Precision", "F1 score"]
values = [sensitivity, specificity, precision, f1]
colors = ["#1BAF7A", "#EB6834", "#EB6834", "#2A78D6"]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
bars = ax.bar(labels, values, color=colors, alpha=0.9, edgecolor="#333", linewidth=1.0, width=0.55)

for bar, value in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{value:.1f}%",
            ha="center", va="bottom", fontsize=13, fontweight="bold")

ax.set_ylim(0, 110)
ax.set_ylabel("Percent", fontsize=11)
ax.set_title("Semantic Classifier Performance Metrics", fontsize=13, fontweight="bold", pad=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.savefig("classifier_metrics.png", dpi=200, bbox_inches="tight")
print("Saved classifier_metrics.png")
print(f"\nSensitivity: {sensitivity:.1f}%")
print(f"Specificity: {specificity:.1f}%")
print(f"Precision: {precision:.1f}%")
print(f"F1 score: {f1:.1f}%")
