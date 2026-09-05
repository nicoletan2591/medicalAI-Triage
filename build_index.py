import chromadb
from sentence_transformers import SentenceTransformer
import os

TARGET_CHUNK_SIZE = 600  # aim for roughly this many characters per chunk


def paragraph_chunks(text, target_size=TARGET_CHUNK_SIZE):
    """Split text into chunks along paragraph boundaries (blank lines),
    packing consecutive paragraphs together up to roughly target_size
    characters. This avoids cutting words, bullet points, or sentences
    in half, unlike naive fixed-length slicing."""
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


embedder = SentenceTransformer("all-MiniLM-L6-v2")  # small, runs locally, no internet needed after first download
client = chromadb.PersistentClient(path="./chroma_db")

# Delete and recreate the collection so old fixed-length chunks don't
# linger alongside the new paragraph-based ones.
try:
    client.delete_collection("triage_docs")
except Exception:
    pass
collection = client.get_or_create_collection("triage_docs")

for filename in os.listdir("docs"):
    if not filename.endswith(".txt"):
        # skips hidden system files like .DS_Store, and anything that
        # isn't actually one of our documents
        continue
    with open(f"docs/{filename}", encoding="utf-8") as f:
        text = f.read()

    chunks = paragraph_chunks(text)
    embeddings = embedder.encode(chunks).tolist()
    ids = [f"{filename}-{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)

print(f"Indexed {collection.count()} chunks.")
