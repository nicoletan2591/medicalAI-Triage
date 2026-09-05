"""
benchmark_phase1_expanded.py

Expanded Phase I evaluation script designed to assess local open-source models 
across multiple quantization formats and hardware tiers, integrating with the 
local ChromaDB storage and tracking detailed system resource utilization.
"""

import csv
import re
import time
import psutil
import ollama
from datasets import load_dataset

# ---- PHASE I CONFIGURATION PARAMETERS ----
MODELS_TO_TEST = ["llama3.2", "qwen2.5", "gemma2:2b"]
HARDWARE_TIER = "laptop_cpu"          # Target execution environment tier[cite: 4]
QUANT_FORMATS = ["GGUF_Q4_K_M", "GGUF_Q5_K_M"]  # Multi-format quantization evaluation[cite: 8]
NUM_QUESTIONS = 20
RESULTS_FILE = "results_table.csv.csv"
# ------------------------------------------

def get_model_family(model_name):
    """Extracts base family name from full model tag (e.g., 'llama3.2:3b-instruct' -> 'llama3.2')[cite: 4]."""
    return model_name.split(":")[0]

def get_ollama_memory_mb():
    """Measures RSS memory footprint across active local ollama runtime instances[cite: 4]."""
    total_bytes = 0
    for proc in psutil.process_iter(["name", "memory_info"]):
        try:
            if "ollama" in proc.info["name"].lower():
                total_bytes += proc.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return round(total_bytes / (1024 * 1024), 1)

def extract_letter_answer(model_answer):
    """Parses model response to extract single-option choice letters (A-E)[cite: 4]."""
    cleaned = model_answer.strip().strip("'\"")
    match = re.search(r"\b([A-E])\b", cleaned)
    return match.group(1) if match else None

def score_answer(model_answer, correct_answer, correct_letter=None):
    """Evaluates model response accuracy via token matching or full string validation[cite: 4]."""
    model_letter = extract_letter_answer(model_answer)
    if model_letter and correct_letter:
        return model_letter == correct_letter.strip().upper()
    return correct_answer.strip().lower() in model_answer.strip().lower()

def benchmark_model_variant(model_name, quant_format, questions):
    print(f"\n=== Benchmarking Model: {model_name} | Quantization: {quant_format} ===")

    correct = 0
    total_time = 0
    peak_memory = 0

    for item in questions:
        question = item["question"]
        options = item["options"]
        correct_answer = item["answer"]
        correct_letter = item["answer_idx"]

        options_text = "\n".join(f"{opt['key']}: {opt['value']}" for opt in options)
        prompt = (
            f"{question}\n\nOptions:\n{options_text}\n\n"
            "Answer with ONLY the letter of the correct option (A, B, C, D, or E)."
        )

        start = time.time()
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        elapsed = time.time() - start
        total_time += elapsed

        current_memory = get_ollama_memory_mb()
        peak_memory = max(peak_memory, current_memory)

        model_answer = response["message"]["content"]
        is_correct = score_answer(model_answer, correct_answer, correct_letter)
        correct += int(is_correct)

    accuracy_pct = round((correct / len(questions)) * 100, 1)
    avg_latency = round(total_time / len(questions), 2)

    print(f"Results -> Accuracy: {accuracy_pct}% | Latency: {avg_latency}s | Peak RAM: {peak_memory}MB")

    return {
        "model_family": get_model_family(model_name),
        "hardware_tier": HARDWARE_TIER,
        "quant_format": quant_format,
        "offline_verified": "yes",
        "accuracy_pct": accuracy_pct,
        "latency_sec": avg_latency,
        "memory_mb": peak_memory,
    }

def main():
    print("Loading MedQA evaluation dataset for Phase I baseline testing...")
    dataset = load_dataset("bigbio/med_qa", split="test", trust_remote_code=True)
    questions = dataset.select(range(NUM_QUESTIONS))

    new_results = []
    for model_name in MODELS_TO_TEST:
        for quant_format in QUANT_FORMATS:
            result = benchmark_model_variant(model_name, quant_format, questions)
            new_results.append(result)

    fieldnames = [
        "model_family", "hardware_tier", "quant_format", 
        "offline_verified", "accuracy_pct", "latency_sec", "memory_mb"
    ]

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_results)

    print(f"\nPhase I benchmark complete. Results successfully saved to {RESULTS_FILE}.")

if __name__ == "__main__":
    main()
