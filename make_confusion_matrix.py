"""
make_confusion_matrix.py

Generates a confusion matrix for the semantic intent classifier,
using the real counts derived from test_semantic_classifier.py's
output: 15 true positives, 0 false negatives, 17 true negatives,
6 false positives (38 test cases total).

Install matplotlib first if you don't have it:
    pip install matplotlib --break-system-packages

Run it:
    python3 make_confusion_matrix.py

Saves confusion_matrix.png in the same folder.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- EDIT THESE if you re-run test_semantic_classifier.py and get different counts ----
true_positives = 15
false_negatives = 0
false_positives = 6
true_negatives = 17
# -----------------------------------------------------------------------------------

# matrix layout: rows = actual, columns = predicted
# row 0 = actual emergency, row 1 = actual safe
# col 0 = predicted emergency, col 1 = predicted safe
matrix = np.array([
    [true_positives, false_negatives],
    [false_positives, true_negatives],
])

labels = [["True Positive", "False Negative"], ["False Positive", "True Negative"]]
colors = [["#E1F5EE", "#FCEBEB"], ["#FAEEDA", "#E1F5EE"]]
text_colors = [["#04342C", "#501313"], ["#412402", "#04342C"]]

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.set_xlim(-0.05, 2.05)
ax.set_ylim(0, 2)
ax.invert_yaxis()
ax.axis("off")

for i in range(2):
    for j in range(2):
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=colors[i][j], edgecolor="#999", linewidth=1))
        ax.text(j + 0.5, i + 0.35, str(matrix[i][j]), ha="center", va="center", fontsize=26, fontweight="bold", color=text_colors[i][j])
        ax.text(j + 0.5, i + 0.68, labels[i][j], ha="center", va="center", fontsize=11, color=text_colors[i][j])

ax.text(1, -0.15, "Predicted: emergency", ha="center", va="center", fontsize=11, color="#555")
ax.text(2, -0.15, "Predicted: safe", ha="center", va="center", fontsize=11, color="#555")
ax.text(-0.15, 0.5, "Actual:\nemergency", ha="center", va="center", fontsize=11, color="#555")
ax.text(-0.15, 1.5, "Actual:\nsafe", ha="center", va="center", fontsize=11, color="#555")

ax.set_title("Semantic Classifier Confusion Matrix (n=38)", fontsize=13, fontweight="bold", pad=30)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200, bbox_inches="tight")
print("Saved confusion_matrix.png")
