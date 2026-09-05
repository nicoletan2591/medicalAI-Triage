"""
make_testbed_composition.py

Generates the bar chart showing your vignette testbed's composition:
6 vignette types (3 standard + 3 adversarial), 2 per type per topic,
across 35 corpus topics = ~420 vignettes total.

Install matplotlib first if you don't have it:
    pip install matplotlib --break-system-packages

Run it:
    python3 make_testbed_composition.py

Saves testbed_composition.png in the same folder.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---- EDIT THESE if your testbed settings change ----
categories = ["Standard:\nEmergency", "Standard:\nUrgent", "Standard:\nRoutine",
              "Adversarial:\nParaphrased\nemergency", "Adversarial:\nAmbiguous\nborderline", "Adversarial:\nFalse-alarm\ntrigger"]
per_topic_count = [2, 2, 2, 2, 2, 2]  # vignettes per type, per topic (matches standard_per_category / adversarial_per_type)
num_topics = 35
colors = ["#2A9D8F", "#2A9D8F", "#2A9D8F", "#E76F51", "#E76F51", "#E76F51"]  # teal = standard, coral = adversarial
# -----------------------------------------------------

total_vignettes = sum(per_topic_count) * num_topics

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(categories, per_topic_count, color=colors, alpha=0.85, edgecolor="#333", linewidth=1.1, width=0.6)
for bar, count in zip(bars, per_topic_count):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(count),
            ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_ylim(0, max(per_topic_count) + 1)
ax.set_ylabel("Vignettes per topic", fontsize=11)
ax.set_title(f"Vignette Testbed Composition (per topic, x{num_topics} topics = ~{total_vignettes} total)",
             fontsize=12.5, fontweight="bold", pad=14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

legend_elements = [
    Patch(facecolor="#2A9D8F", alpha=0.85, edgecolor="#333", label="Standard (tests acuity categorization)"),
    Patch(facecolor="#E76F51", alpha=0.85, edgecolor="#333", label="Adversarial (tests guardrail robustness)"),
]
ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=9.5)

plt.tight_layout()
plt.savefig("testbed_composition.png", dpi=200, bbox_inches="tight")
print("Saved testbed_composition.png")
print(f"Total vignettes: {total_vignettes}")
