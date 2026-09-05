"""
check_judge_agreement.py

Since a larger judge model isn't reliably runnable on this hardware
(memory errors when combined with the embedding pipeline), this script
instead quantifies how much the existing 3B judge can be trusted, by
sampling some already-judged queries and asking you to review them
by hand against the actual retrieved chunks.

This produces a real number (agreement rate) that tells you how
reliable the automated judge is -- which is more useful than either
blindly trusting it or blindly distrusting it.
"""

import csv
import random
import chromadb
from sentence_transformers import SentenceTransformer

RESULTS_FILE = "retrieval_review.csv"
SAMPLE_SIZE = 20  # how many rows to manually re-check


def load_existing():
    with open(RESULTS_FILE, newline="") as f:
        return list(csv.DictReader(f))


def main():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection("triage_docs")

    rows = load_existing()
    if len(rows) < SAMPLE_SIZE:
        print(f"Only {len(rows)} rows logged, using all of them.")
        sample = rows
    else:
        sample = random.sample(rows, SAMPLE_SIZE)

    agree = 0
    disagree = 0

    for row in sample:
        query = row["test_query"]
        judge_said = row["relevant_chunk_retrieved"]

        query_embedding = embedder.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=3)
        chunks = results["documents"][0] if results["documents"][0] else []

        print("\n" + "=" * 70)
        print(f"Query: {query}")
        print(f"Judge previously said: {judge_said}")
        print("-" * 70)
        for i, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {i}:\n{chunk[:300]}")

        your_answer = input("\nDo YOU think any chunk is relevant? (y/n): ").strip().lower()
        your_verdict = "yes" if your_answer == "y" else "no"

        if your_verdict == judge_said:
            agree += 1
            print("-> Agreement")
        else:
            disagree += 1
            print(f"-> DISAGREEMENT (you said {your_verdict}, judge said {judge_said})")

    total = agree + disagree
    agreement_pct = round((agree / total) * 100, 1) if total else 0

    print("\n" + "=" * 70)
    print(f"RESULTS: {agree}/{total} agreement ({agreement_pct}%)")
    print("=" * 70)
    print("\nThis agreement rate is your real validity number for the automated")
    print("judge. Report it alongside your retrieval scores -- e.g. 'the 3B")
    print("LLM judge showed X% agreement with human review on a 20-query sample,'")
    print("which tells the reader how much to trust the raw retrieval percentage.")


if __name__ == "__main__":
    main()
