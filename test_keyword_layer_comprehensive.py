"""
test_keyword_layer_comprehensive.py

Comprehensive positive/negative test suite for keyword_redflag_layer.py,
covering every major red-flag category the pattern list is meant to
catch, plus deliberately tricky negative cases (messages that mention
health topics or even similar-sounding words without actually
describing a red flag) to check for false positives, not just false
negatives.

This is a fully deterministic test (no model calls, no randomness) --
running it twice on the same code always gives the same result, so a
single comprehensive run is sufficient evidence, unlike the semantic
classifier test (see test_semantic_classifier.py), which needs to be
re-validated any time the model or prompt changes.

Run it:
    python3 test_keyword_layer_comprehensive.py
"""

from keyword_redflag_layer import check_keyword_redflags

# (message, should_trigger, category) -- category is just for organizing
# the printed report, not used in the pass/fail logic itself.
TEST_CASES = [
    # --- Cardiac / circulatory: positives ---
    ("I have severe chest pain that won't go away", True, "cardiac"),
    ("The chest pressure is radiating to my arm", True, "cardiac"),
    ("I fainted while walking to the kitchen", True, "cardiac"),
    ("My heart is beating irregularly and I feel dizzy", True, "cardiac"),
    ("I have sudden severe shortness of breath at rest", True, "cardiac"),
    # --- Cardiac / circulatory: negatives ---
    ("I get a bit out of breath climbing stairs, is that normal?", False, "cardiac"),
    ("My resting heart rate has been a bit higher lately", False, "cardiac"),
    ("I have mild chest tightness after a big meal sometimes", False, "cardiac"),

    # --- Neurological / stroke: positives ---
    ("My face is drooping on one side and my speech is slurred", True, "neuro"),
    ("I have sudden weakness in my left arm", True, "neuro"),
    ("This is the worst headache of my life", True, "neuro"),
    ("I just had a first-time seizure", True, "neuro"),
    ("My seizure has been going on for several minutes and won't stop", True, "neuro"),
    ("I can't speak properly all of a sudden", True, "neuro"),
    # --- Neurological / stroke: negatives ---
    ("I get headaches sometimes when I'm stressed", False, "neuro"),
    ("I have a known seizure disorder and this matches my usual pattern", False, "neuro"),
    ("My speech therapist gave me some new exercises to try", False, "neuro"),

    # --- Respiratory: positives ---
    ("I can't breathe and I'm gasping for air", True, "respiratory"),
    ("My lips have turned blue", True, "respiratory"),
    ("I'm choking and can't speak or cough", True, "respiratory"),
    # --- Respiratory: negatives ---
    ("I have a mild cough from a cold", False, "respiratory"),
    ("My allergies are making me sneeze a lot today", False, "respiratory"),

    # --- Abdominal / GI: positives ---
    ("I have severe sudden abdominal pain", True, "abdominal"),
    ("I'm vomiting blood", True, "abdominal"),
    ("My stool is black and tarry", True, "abdominal"),
    ("My abdomen feels rigid and very tender", True, "abdominal"),
    # --- Abdominal / GI: negatives ---
    ("I have mild indigestion after eating spicy food", False, "abdominal"),
    ("I've had constipation for a couple of days", False, "abdominal"),

    # --- Mental health: positives ---
    ("Someone told me they intend to harm themselves", True, "mental_health"),
    ("I've been feeling suicidal lately", True, "mental_health"),
    ("I want to kill myself", True, "mental_health"),
    # --- Mental health: negatives ---
    ("I've been feeling a bit down lately, nothing serious", False, "mental_health"),
    ("I want general information about therapy options", False, "mental_health"),

    # --- Allergic / anaphylaxis: positives ---
    ("My throat feels tight and I'm having trouble breathing after a bee sting", True, "allergy"),
    ("I'm having a severe allergic reaction with facial swelling", True, "allergy"),
    # --- Allergic / anaphylaxis: negatives ---
    ("I have mild seasonal allergies with itchy eyes", False, "allergy"),
    ("I get a small rash from certain fabrics sometimes", False, "allergy"),

    # --- Bleeding / shock: positives ---
    ("I have uncontrolled bleeding from a wound", True, "bleeding"),
    ("The bleeding won't stop even with pressure", True, "bleeding"),
    # --- Bleeding / shock: negatives ---
    ("I have a small paper cut that already stopped bleeding", False, "bleeding"),

    # --- Infection / sepsis: positives ---
    ("I have a high fever and I'm confused", True, "infection"),
    ("I have a rash that doesn't fade when I press on it", True, "infection"),
    # --- Infection / sepsis: negatives ---
    ("I have a low-grade fever and feel a bit tired", False, "infection"),

    # --- Pediatric-specific: positives ---
    ("My infant has a fever and is only 6 weeks old", True, "pediatric"),
    ("My child is unusually drowsy and hard to wake with a fever", True, "pediatric"),
    # --- Pediatric-specific: negatives ---
    ("My child has a mild cold and a runny nose", False, "pediatric"),
    ("My toddler is being a bit fussy today", False, "pediatric"),

    # --- Pregnancy-specific: positives ---
    ("I'm pregnant and having vaginal bleeding", True, "pregnancy"),
    ("I'm pregnant with a severe headache and vision changes", True, "pregnancy"),
    # --- Pregnancy-specific: negatives ---
    ("I'm pregnant and have mild morning sickness", False, "pregnancy"),

    # --- Dental: positives ---
    ("I have a knocked-out tooth from a fall", True, "dental"),
    # --- Dental: negatives ---
    ("I have mild tooth sensitivity to cold drinks", False, "dental"),

    # --- Tricky negatives: mentions similar words without describing a red flag ---
    ("I'm learning to speak Spanish and my accent sounds a bit slurred to natives", False, "tricky"),
    ("I watched a movie about someone having a heart attack last night", False, "tricky"),
    ("My friend is training to be a paramedic for chest trauma cases", False, "tricky"),
    ("I read an article about seizure disorders for a school project", False, "tricky"),
    ("The blood donation clinic is looking for volunteers this week", False, "tricky"),
]


def run_suite():
    total = len(TEST_CASES)
    passed = 0
    failures = []

    by_category = {}

    for message, should_trigger, category in TEST_CASES:
        triggered, matches = check_keyword_redflags(message)
        is_correct = triggered == should_trigger

        by_category.setdefault(category, {"total": 0, "passed": 0})
        by_category[category]["total"] += 1

        if is_correct:
            passed += 1
            by_category[category]["passed"] += 1
        else:
            failures.append((message, should_trigger, triggered, category))

    print(f"{'='*70}")
    print(f"KEYWORD LAYER TEST SUITE: {passed}/{total} passed ({round(passed/total*100, 1)}%)")
    print(f"{'='*70}\n")

    print("By category:")
    for cat, stats in sorted(by_category.items()):
        print(f"  {cat:20s}: {stats['passed']}/{stats['total']}")

    if failures:
        print(f"\n{'='*70}")
        print(f"FAILURES ({len(failures)}):")
        print(f"{'='*70}")
        for message, expected, got, category in failures:
            kind = "FALSE NEGATIVE (missed a real red flag)" if expected else "FALSE POSITIVE (triggered on safe content)"
            print(f"\n  [{category}] {kind}")
            print(f"  Message: {message}")
            print(f"  Expected trigger={expected}, got trigger={got}")
    else:
        print("\nNo failures.")

    print(f"\n{'='*70}")
    if failures:
        false_negatives = sum(1 for _, exp, _, _ in failures if exp)
        false_positives = sum(1 for _, exp, _, _ in failures if not exp)
        print(f"False negatives (missed real red flags): {false_negatives} -- these are the more serious failure type")
        print(f"False positives (triggered on safe content): {false_positives} -- less serious, but still worth fixing")
    print(f"{'='*70}")

    return passed, total, failures


if __name__ == "__main__":
    run_suite()
