"""
debug_suspicious_misses.py

For a handful of queries that SHOULD obviously match a document
(near-exact matches to a Q1 or early line in a specific document),
but were judged "not relevant" by the LLM judge -- print the actual
retrieved chunks in full, so we can see whether:
  (a) retrieval genuinely failed to find the matching document (a
      real indexing/embedding problem), or
  (b) retrieval found it fine, but the LLM judge incorrectly said "no"
      (a judge reliability problem, not a retrieval problem)

These are two very different bugs with two very different fixes, so
this distinction matters before doing anything else.
"""

import chromadb
from sentence_transformers import SentenceTransformer

SUSPICIOUS_QUERIES = [
    "My rescue inhaler isn't helping, what should I do?",
    "Someone told me they intend to harm themselves, what should happen?",
    "I have a chemical splash in my eye, what should I do?",
    "My face feels droopy and my speech is slurred, what should I do?",
    "I have a knocked-out permanent tooth, is this urgent?",
]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("triage_docs")

print(f"Total chunks in index: {collection.count()}\n")

for query in SUSPICIOUS_QUERIES:
    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    ids = results["ids"][0] if results["ids"] else []
    docs = results["documents"][0] if results["documents"][0] else []
    distances = results["distances"][0] if results.get("distances") else [None] * len(docs)

    if not docs:
        print("  NOTHING RETRIEVED AT ALL")
        continue

    for i, (chunk_id, doc, dist) in enumerate(zip(ids, docs, distances), start=1):
        print(f"\n  --- Chunk {i} (id: {chunk_id}, distance: {dist}) ---")
        print(f"  {doc[:400]}")
    print()

print("\nLook at the chunk IDs above -- do they come from the document you'd expect")
print("(e.g. respiratory_qa_50.txt for the inhaler query)? If yes, retrieval is")
print("working and the judge model is the problem. If the right document never")
print("appears in the top 3 at all, retrieval itself is failing.")
