"""
analyze_false_positives.py

Investigates whether the 75 false positives from your Step 3 rubric
scoring run (54.9% accuracy, 4 false negatives, 75 false positives)
cluster in specific vignette types -- particularly whether they're
concentrated in the false_alarm_trigger adversarial type (which was
specifically designed to probe over-triggering) versus spread across
standard "routine" vignettes too.

This distinction matters: if false positives cluster almost entirely
in false_alarm_trigger cases, that's the guardrail behaving as
designed on a deliberately hard adversarial case -- less alarming.
If they're spread significantly into ordinary "routine" vignettes too,
that's a broader over-triggering problem affecting normal, everyday
interactions, not just adversarial edge cases -- more concerning.

Run this from your project folder, after phase3_full_results.csv
exists (produced by phase3_evaluation_runner.py):

    python3 analyze_false_positives.py
"""

import csv
from collections import Counter

RESULTS_FILE = "phase3_full_results.csv"


def load_results(filepath=RESULTS_FILE):
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def find_false_positives(rows):
    """A false positive: rubric marked 'incorrect', ground truth said
    NO escalation was expected, but the agent escalated anyway."""
    false_positives = []
    for row in rows:
        if row.get("rubric_verdict") != "incorrect":
            continue
        expected_trigger = row.get("expected_guardrail_trigger", "")
        actual_escalated = row.get("actual_escalated", "")
        # CSV values are strings -- normalize before comparing
        expected_false = expected_trigger in ("False", "false", "")
        actual_true = actual_escalated in ("True", "true")
        if expected_false and actual_true:
            false_positives.append(row)
    return false_positives


def analyze(filepath=RESULTS_FILE):
    rows = load_results(filepath)
    false_positives = find_false_positives(rows)

    print(f"Total rows: {len(rows)}")
    print(f"False positives found: {len(false_positives)}")
    print()

    if not false_positives:
        print("No false positives found in this file -- if you expected 75,")
        print("check that phase3_full_results.csv is from the completed run,")
        print("not a partial/interrupted one.")
        return

    # Breakdown by vignette_type (standard vs adversarial)
    by_vignette_type = Counter(fp.get("vignette_type", "unknown") for fp in false_positives)
    print("Breakdown by vignette_type:")
    for vtype, count in by_vignette_type.most_common():
        pct = round(count / len(false_positives) * 100, 1)
        print(f"  {vtype}: {count} ({pct}%)")
    print()

    # Breakdown by specific category / adversarial_type (finer-grained)
    by_category = Counter(fp.get("category", "unknown") for fp in false_positives)
    print("Breakdown by category/topic label:")
    for cat, count in by_category.most_common():
        pct = round(count / len(false_positives) * 100, 1)
        print(f"  {cat}: {count} ({pct}%)")
    print()

    # The key question: how concentrated is this?
    adversarial_count = by_vignette_type.get("adversarial", 0)
    standard_count = by_vignette_type.get("standard", 0)

    print("=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if standard_count == 0:
        print(f"All {len(false_positives)} false positives are in ADVERSARIAL vignettes.")
        print("This means the guardrail did NOT over-trigger on any ordinary,")
        print("everyday case in this run -- every false positive came from a")
        print("case specifically designed to be a hard, adversarial test.")
        print("This is a meaningfully less concerning finding than if standard")
        print("vignettes were also affected -- worth stating explicitly in your report.")
    elif standard_count / len(false_positives) < 0.15:
        print(f"{standard_count} of {len(false_positives)} false positives ({round(standard_count/len(false_positives)*100,1)}%)")
        print("are in standard (non-adversarial) vignettes -- a small minority.")
        print("Most over-triggering is concentrated in the adversarial cases")
        print("designed to probe exactly this weakness, which is a less")
        print("concerning pattern than broad over-triggering on ordinary input.")
    else:
        print(f"{standard_count} of {len(false_positives)} false positives ({round(standard_count/len(false_positives)*100,1)}%)")
        print("are in STANDARD (non-adversarial, ordinary) vignettes.")
        print("This is a meaningful fraction -- the over-triggering problem is")
        print("not confined to deliberately hard adversarial cases, it's also")
        print("affecting ordinary interactions. This is worth treating as a real")
        print("limitation in your report, not just an adversarial-case quirk.")

    print()
    print("Sample false-positive patient statements (first 5), for manual review:")
    for fp in false_positives[:5]:
        print(f"\n  [{fp.get('vignette_type')}] {fp.get('patient_statement', '')[:100]}")


if __name__ == "__main__":
    analyze()
