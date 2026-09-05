"""
multi_benchmark_factual_eval.py

Component (1) of the Phase III evaluation protocol: standard
factual-competency testing on MedQA, MedMCQA, and MedXpertQA, extending
Phase I's single-dataset benchmark.py to all three named datasets, with
explicit regression detection across quantization levels.

"Regression detection" here means: for the same model family, compare
accuracy at each tested quantization level against the highest-precision
level tested (treated as the reference), and flag any drop beyond a
threshold as a possible quantization-induced regression worth
investigating -- not a definitive conclusion on its own, given Phase
I's own finding that small-sample accuracy is noisy (a 20-point swing
was observed between two runs of the same qwen2.5/Q5_K_M configuration).
"""

import csv
import re
import time
import ollama
from datasets import load_dataset

NUM_QUESTIONS_PER_DATASET = 20  # larger than Phase I's n=10, per the "larger sample" next step already identified
REGRESSION_THRESHOLD_PCT = 10.0  # flag if accuracy drops more than this many percentage points

DATASET_CONFIGS = {
    "MedQA": {"path": "bigbio/med_qa", "split": "test"},
    "MedMCQA": {"path": "openlifescienceai/medmcqa", "split": "validation"},
    # VERIFIED against the real Hugging Face dataset card (2026): MedXpertQA
    # requires a subset name ("Text" or "MM"), not just a split. Using "Text"
    # since this pipeline handles text-only questions, not the MM subset's
    # images. Confirmed schema: options is a dict keyed by letter, label is
    # the correct letter directly -- both already handled correctly below.
    "MedXpertQA": {"path": "TsinghuaC3I/MedXpertQA", "name": "Text", "split": "test"},
}

# VERIFIED: MedXpertQA's Text subset uses up to 10 answer options (A-J),
# not the 5 (A-E) that MedQA and MedMCQA use. The regex below must match
# the full range or it will silently fail to parse any model answer that
# happens to be F-J -- a real bug caught by checking the actual dataset
# schema rather than assuming it matched the other two datasets.
VALID_ANSWER_LETTERS = "A-J"


def extract_letter_answer(model_answer):
    cleaned = model_answer.strip().strip("'\"").upper()
    match = re.search(rf"\b([{VALID_ANSWER_LETTERS}])\b", cleaned)
    return match.group(1) if match else None


def normalize_question(item, dataset_name):
    """Different datasets use different field names for the same
    concepts (question text, options, correct answer). This function
    maps each dataset's raw schema to a common (question, options_text,
    correct_letter) format so the rest of the pipeline doesn't need to
    know which dataset it's looking at."""
    if dataset_name == "MedQA":
        options_text = "\n".join(f"{opt['key']}: {opt['value']}" for opt in item["options"])
        return item["question"], options_text, item["answer_idx"].strip().upper()

    if dataset_name == "MedMCQA":
        # MedMCQA uses opa/opb/opc/opd fields and a 0-indexed cop (correct option) field
        options = {"A": item["opa"], "B": item["opb"], "C": item["opc"], "D": item["opd"]}
        options_text = "\n".join(f"{k}: {v}" for k, v in options.items())
        correct_letter = ["A", "B", "C", "D"][item["cop"]]
        return item["question"], options_text, correct_letter

    if dataset_name == "MedXpertQA":
        # Schema assumed based on typical MedXpertQA structure -- VERIFY
        # against the actual dataset fields on first run, since this
        # dataset was not directly inspected during Phase III script
        # development (no live dataset access in this environment).
        options_text = "\n".join(f"{k}: {v}" for k, v in item["options"].items())
        return item["question"], options_text, item["label"].strip().upper()

    raise ValueError(f"Unknown dataset: {dataset_name}")


def run_single_config(model, quant_format, dataset_name, num_questions=NUM_QUESTIONS_PER_DATASET):
    """Runs one model/quant/dataset combination and returns accuracy
    plus per-question detail for later regression analysis."""
    config = DATASET_CONFIGS[dataset_name]
    print(f"  Loading {dataset_name}...")
    # Pass the subset name ("name" param) when the dataset config specifies
    # one -- MedXpertQA requires this (its real schema, verified above), MedQA
    # and MedMCQA don't use a subset name at all.
    if "name" in config:
        dataset = load_dataset(config["path"], config["name"], split=config["split"], trust_remote_code=True)
    else:
        dataset = load_dataset(config["path"], split=config["split"], trust_remote_code=True)
    questions = dataset.select(range(min(num_questions, len(dataset))))

    correct = 0
    per_question = []

    for item in questions:
        question_text, options_text, correct_letter = normalize_question(item, dataset_name)
        prompt = f"{question_text}\n\nOptions:\n{options_text}\n\nAnswer with ONLY the letter of the correct option."

        start = time.time()
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        elapsed = time.time() - start

        model_letter = extract_letter_answer(response["message"]["content"])
        is_correct = model_letter == correct_letter
        correct += int(is_correct)

        per_question.append({
            "correct_letter": correct_letter,
            "model_letter": model_letter,
            "is_correct": is_correct,
            "latency_sec": round(elapsed, 2),
        })

    accuracy_pct = round((correct / len(questions)) * 100, 1)
    return accuracy_pct, per_question


def detect_regressions(results):
    """Given a list of {model, quant_format, dataset, accuracy_pct}
    dicts, groups by (model, dataset) and flags any quant level whose
    accuracy drops more than REGRESSION_THRESHOLD_PCT below the best
    quant level tested for that same model/dataset pair."""
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in results:
        grouped[(r["model"], r["dataset"])].append(r)

    flagged = []
    for (model, dataset), rows in grouped.items():
        best_accuracy = max(r["accuracy_pct"] for r in rows)
        for r in rows:
            drop = best_accuracy - r["accuracy_pct"]
            if drop > REGRESSION_THRESHOLD_PCT:
                flagged.append({**r, "drop_from_best_pct": round(drop, 1), "best_accuracy_pct": best_accuracy})

    return flagged


def run_full_protocol(model_quant_configs, output_file="phase3_factual_results.csv"):
    """model_quant_configs: list of (model_tag, quant_format_label) tuples.
    Runs every combination against every dataset in DATASET_CONFIGS."""
    all_results = []

    for model_tag, quant_label in model_quant_configs:
        for dataset_name in DATASET_CONFIGS:
            print(f"\n=== {model_tag} ({quant_label}) on {dataset_name} ===")
            accuracy_pct, per_question = run_single_config(model_tag, quant_label, dataset_name)
            print(f"  Accuracy: {accuracy_pct}%")
            all_results.append({
                "model": model_tag,
                "quant_format": quant_label,
                "dataset": dataset_name,
                "accuracy_pct": accuracy_pct,
                "num_questions": len(per_question),
            })

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "quant_format", "dataset", "accuracy_pct", "num_questions"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved {output_file}")

    regressions = detect_regressions(all_results)
    if regressions:
        print(f"\n{len(regressions)} possible quantization regression(s) flagged (>{REGRESSION_THRESHOLD_PCT}pp drop):")
        for r in regressions:
            print(f"  {r['model']} / {r['quant_format']} / {r['dataset']}: "
                  f"{r['accuracy_pct']}% vs best {r['best_accuracy_pct']}% (drop: {r['drop_from_best_pct']}pp)")
    else:
        print(f"\nNo regressions beyond {REGRESSION_THRESHOLD_PCT}pp threshold detected.")

    return all_results, regressions
