"""
tune_rag_params.py

Systematically tests different RAG chunk sizes and top_k retrieval parameters
against the gold-standard evaluation set (gold_set.py) to find the optimal
configuration for your local clinical triage agent's vector database.
"""

import csv
from gold_set import GOLD_SET
from build_index_variant import build_variant_index

# --- PHASE I RAG TUNING CONFIGURATION ---
CHUNK_SIZES_TO_TEST = [300, 600, 900]
TOP_K_VALUES_TO_TEST = [1, 3, 5]
OUTPUT_CSV = "gold_standard_results.csv"
DB_PATH_PREFIX = "./chroma_variant_db_"
# ----------------------------------------

def evaluate_variant(collection, embedder, chunk_size, top_k):
    """Evaluates hit rate across all gold-standard queries for a given index variant."""
    hits = 0
    total = len(GOLD_SET)

    for query, expected_files in GOLD_SET.items():
        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=top_k)
        
        # Extract source filenames from retrieved chunk IDs (format: filename-chunk_index)
        retrieved_ids = results["ids"][0] if results["ids"] else []
        retrieved_files = set(id_str.rsplit("-", 1)[0] for id_str in retrieved_ids)

        # Check if any expected source file matches the retrieved files
        match_found = any(f in retrieved_files for f in expected_files)
        if match_found:
            hits += 1

    hit_rate_pct = round((hits / total) * 100, 1)
    return hit_rate_pct

def main():
    print("Starting automated RAG parameter tuning for Phase I objectives...")
    results_rows = []

    for chunk_size in CHUNK_SIZES_TO_TEST:
        db_path = f"{DB_PATH_PREFIX}{chunk_size}"
        print(f"\nBuilding variant index with target chunk size: {chunk_size}...")
        
        collection, embedder = build_variant_index(chunk_size, db_path)

        for top_k in TOP_K_VALUES_TO_TEST:
            print(f"  Evaluating with top_k = {top_k}...")
            hit_rate_pct = evaluate_variant(collection, embedder, chunk_size, top_k)
            print(f"  -> Hit Rate: {hit_rate_pct}%")

            results_rows.append({
                "chunk_size": str(chunk_size),
                "top_k": str(top_k),
                "hit_rate_pct": hit_rate_pct
            })

    # Save optimization results to CSV for check_phase1_complete.py verification
    fieldnames = ["chunk_size", "top_k", "hit_rate_pct"]
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_rows)

    print(f"\nOptimization complete! Results saved to {OUTPUT_CSV}.")

if __name__ == "__main__":
    main()
