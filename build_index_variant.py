"""
build_index_variant.py

Same chunking logic as build_index.py, but parameterized so you can
build multiple index variants (different chunk sizes) into separate
Chroma paths, without overwriting your real production index at
./chroma_db.

Used by tune_rag_params.py to test several chunk sizes automatically.
Not meant to be run directly in normal use -- use build_index.py for that.
"""

import chromadb
from sentence_transformers import SentenceTransformer
import os


def paragraph_chunks(text, target_size):
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for para in paragraphs:
        if current and len(current) + len(para) + 2 > target_size:
            chunks.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        chunks.append(current)
    return chunks


def build_variant_index(target_chunk_size, db_path, docs_dir="docs"):
    """Builds a Chroma index at db_path using the given chunk target
    size. Returns the collection for immediate querying."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=db_path)

    try:
        client.delete_collection("triage_docs")
    except Exception:
        pass
    collection = client.get_or_create_collection("triage_docs")

    for filename in os.listdir(docs_dir):
        if not filename.endswith(".txt"):
            continue
        with open(f"{docs_dir}/{filename}", encoding="utf-8") as f:
            text = f.read()
        chunks = paragraph_chunks(text, target_chunk_size)
        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{filename}-{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)

    return collection, embedder
