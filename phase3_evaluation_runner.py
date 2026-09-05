"""
phase3_evaluation_runner.py

Ties together components (2) and (3) of the Phase III protocol: runs
the vignette testbed through the Phase II triage_agent, scores every
response with BOTH the rubric scorer and the independent LLM judge,
and reports agreement between the two scoring methods as an explicit
validity check on the protocol itself -- per the requirement that
"agreement between the two scoring methods [be] reported explicitly
as a validity check on the protocol itself."

This is the same reliability-checking discipline used throughout this
project: Phase I validated its LLM judge against human review before
trusting it; Phase II validated its semantic classifier the same way;
this script validates whether two independent, automated scoring
methods (one deterministic, one model-based) actually agree with each
other, since agreement between two flawed methods is not the same
thing as either being correct, but persistent disagreement would be a
clear signal that at least one of them has a problem worth
investigating -- the same posture taken toward the Phase I judge.

Run this from your project folder (with chroma_db, triage_agent.py,
and all Phase II files already in place):

    python3 phase3_evaluation_runner.py
"""

import json
import csv

from triage_vignette_generator import build_testbed
from triage_agent import DialogueManager, handle_message
from rubric_scorer import score_testbed_rubric
from llm_judge_scorer import score_testbed_llm_judge

# A small starter topic set -- expand this to cover more of your 35
# corpus topics for a more thorough testbed. Kept small here so a
# first run completes in a reasonable time; this is a starting
# configuration, not a fixed final testbed size.
DEFAULT_TOPICS = [
    "red flag symptoms",
    "triage acuity levels",
    "symptom mapping questions",
    "stroke seizure minor injuries",
    "chronic conditions",
    "heart disease chest pain",
    "diabetes emergency",
    "stroke symptoms",
    "cancer symptoms",
    "high cholesterol blood pressure",
    "asthma COPD breathing difficulty",
    "mental health crisis",
    "pregnancy complications",
    "pediatric child symptoms",
    "infection sepsis fever",
    "kidney disease",
    "liver disease",
    "autoimmune condition flare",
    "medication side effects",
    "allergic reaction",
    "bowel disease abdominal pain",
    "blood disorder bleeding",
    "brain neurological disorder",
    "eye disease vision problem",
    "skin condition rash",
    "older adult symptoms",
    "nose bleeding sinus",
    "lung disease breathing",
    "men's health testicular pain",
    "women's health pelvic pain",
    "osteoporosis frailty",
    "autism triage considerations",
    "cerebral palsy symptoms",
    "general symptoms fatigue",
    "tooth dental pain",
]


def run_agent_on_testbed(vignettes, output_file="phase3_agent_progress.json"):
    """Runs every vignette's patient statement through the Phase II
    triage_agent, using a fresh DialogueManager per vignette (each
    vignette is treated as the start of a new, independent conversation,
    not a continuation of a prior one).

    Saves progress incrementally and resumes from output_file if it
    already has partial results, matching the same crash-safety
    pattern used in triage_vignette_generator.py (Step 1) and
    llm_judge_scorer.py (Step 4) -- so a full restart of the pipeline
    never re-does work that already completed in ANY stage, not just
    generation and judging."""
    agent_results = []
    already_processed_statements = set()

    try:
        with open(output_file) as f:
            saved = json.load(f)
        agent_results = saved
        already_processed_statements = {r["_patient_statement"] for r in agent_results}
        print(f"Resuming: found {len(agent_results)} already-processed agent results in {output_file}")
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Results must stay in the same order as `vignettes` for later
    # zip()-based scoring to line up correctly, so we rebuild the list
    # in vignette order rather than just appending resumed + new.
    results_by_statement = {r["_patient_statement"]: r for r in agent_results}

    for i, vignette in enumerate(vignettes):
        statement = vignette["patient_statement"]
        if statement in already_processed_statements:
            continue

        print(f"Running agent on vignette {i+1}/{len(vignettes)}: {statement[:60]}...")
        dm = DialogueManager()
        result = handle_message(statement, dm)
        result["_patient_statement"] = statement  # tag for resume matching
        results_by_statement[statement] = result
        already_processed_statements.add(statement)

        with open(output_file, "w") as f:
            json.dump(list(results_by_statement.values()), f, indent=2)

    # Return in the same order as the input vignettes list
    ordered_results = [results_by_statement[v["patient_statement"]] for v in vignettes]
    print(f"\nAgent run complete: {len(ordered_results)} vignettes processed, saved to {output_file}")
    return ordered_results


def compute_scoring_agreement(rubric_scored, judge_scored):
    """Compares the rubric verdict against the judge verdict for every
    STRICTLY-SCORED vignette (excludes ambiguous cases the rubric
    deliberately doesn't score, since there's no rubric verdict to
    compare against there).

    Note: the rubric outputs 'correct'/'incorrect' relative to ground
    truth, while the judge outputs 'appropriate'/'inappropriate'
    relative to general clinical judgment with no ground truth shown
    to it. These map as: rubric 'correct' should correspond to judge
    'appropriate', and rubric 'incorrect' should correspond to judge
    'inappropriate', IF the ground truth and the judge's independent
    clinical assessment are both valid. Disagreement between them
    doesn't automatically mean the rubric is wrong -- it's a signal
    worth investigating, the same way Phase I's judge-vs-human
    disagreements were investigated rather than assumed to indict one
    side automatically."""
    comparable = [
        (r, j) for r, j in zip(rubric_scored, judge_scored)
        if r["rubric_verdict"] != "not_scored"
    ]

    agree = 0
    disagree = 0
    disagreements = []

    for rubric_row, judge_row in comparable:
        rubric_says_correct = rubric_row["rubric_verdict"] == "correct"
        judge_says_appropriate = judge_row["judge_verdict"] == "appropriate"

        if rubric_says_correct == judge_says_appropriate:
            agree += 1
        else:
            disagree += 1
            disagreements.append({
                "patient_statement": rubric_row["patient_statement"],
                "rubric_verdict": rubric_row["rubric_verdict"],
                "judge_verdict": judge_row["judge_verdict"],
                "expected_guardrail_trigger": rubric_row["expected_guardrail_trigger"],
                "actual_escalated": rubric_row["actual_escalated"],
            })

    total = agree + disagree
    agreement_pct = round((agree / total) * 100, 1) if total else None

    return {
        "total_compared": total,
        "agree": agree,
        "disagree": disagree,
        "agreement_pct": agreement_pct,
    }, disagreements


def run_full_evaluation(topics=None, standard_per_category=2, adversarial_per_type=2):
    topics = topics or DEFAULT_TOPICS

    print("=" * 70)
    print("STEP 1: Generating vignette testbed")
    print("=" * 70)
    vignettes = build_testbed(topics, standard_per_category, adversarial_per_type, output_file="phase3_vignettes.json")

    print("\n" + "=" * 70)
    print("STEP 2: Running Phase II agent on every vignette")
    print("=" * 70)
    agent_results = run_agent_on_testbed(vignettes)

    print("\n" + "=" * 70)
    print("STEP 3: Rubric-based scoring (deterministic, ground-truth-based)")
    print("=" * 70)
    rubric_summary, rubric_scored = score_testbed_rubric(vignettes, agent_results)
    print(json.dumps(rubric_summary, indent=2))

    print("\n" + "=" * 70)
    print("STEP 4: Independent LLM-judge scoring (no ground truth shown)")
    print("=" * 70)
    judge_scored = score_testbed_llm_judge(vignettes, agent_results)

    print("\n" + "=" * 70)
    print("STEP 5: Agreement between the two scoring methods (validity check)")
    print("=" * 70)
    agreement_summary, disagreements = compute_scoring_agreement(rubric_scored, judge_scored)
    print(json.dumps(agreement_summary, indent=2))

    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s) between rubric and judge:")
        for d in disagreements:
            print(f"\n  Statement: {d['patient_statement'][:80]}")
            print(f"  Rubric: {d['rubric_verdict']} (expected trigger={d['expected_guardrail_trigger']}, actual escalated={d['actual_escalated']})")
            print(f"  Judge: {d['judge_verdict']}")

    # Save full results for later analysis / report writing
    with open("phase3_full_results.csv", "w", newline="") as f:
        fieldnames = ["patient_statement", "vignette_type", "category", "expected_category",
                      "expected_guardrail_trigger", "actual_escalated", "rubric_verdict",
                      "judge_verdict"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r, j in zip(rubric_scored, judge_scored):
            writer.writerow({**r, "judge_verdict": j["judge_verdict"]})

    print(f"\nFull results saved to phase3_full_results.csv")
    print(f"\nSUMMARY:")
    print(f"  Rubric accuracy: {rubric_summary['accuracy_pct']}% ({rubric_summary['correct']}/{rubric_summary['strictly_scored']})")
    print(f"  False negatives: {rubric_summary['false_negatives']} (missed expected escalations -- the more serious type)")
    print(f"  False positives: {rubric_summary['false_positives']}")
    print(f"  Rubric-judge agreement: {agreement_summary['agreement_pct']}% -- report this alongside")
    print(f"  the accuracy figure, per the validity-check requirement in the Phase III protocol.")

    return {
        "vignettes": vignettes,
        "agent_results": agent_results,
        "rubric_summary": rubric_summary,
        "agreement_summary": agreement_summary,
        "disagreements": disagreements,
    }


if __name__ == "__main__":
    run_full_evaluation()
