"""search.py - find the document chunks that best match a question."""
import sys

import chromadb

# Same paths + same embedding model as ingest.py (must match!)
from app.rag.ingest import COLLECTION_NAME, DB_DIR, get_embedding_function

_collection = None  # cached so the model loads once


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_DIR))
        _collection = client.get_collection(
            name=COLLECTION_NAME, embedding_function=get_embedding_function()
        )
    return _collection


def search(question: str, k: int = 3) -> list[dict]:
    """Return the k closest chunks: [{text, source, score}], best first. score 0-1."""
    results = get_collection().query(query_texts=[question], n_results=k)
    hits = []
    for text, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": text, "source": meta["source"], "score": round(1 - distance, 3)})
    return hits


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "When is the surgery open?"
    print(f"Q: {question}\n")
    for hit in search(question):
        first_lines = " | ".join(hit["text"].splitlines()[:2])
        print(f"{hit['score']:.3f}  {hit['source']:40}  {first_lines}")
