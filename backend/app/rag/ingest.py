"""
ingest.py - prepares the surgery documents so they can be searched by meaning.

Run once (and again whenever the documents change):
    cd backend
    python -m app.rag.ingest

Steps:
    1. LOAD   - read every .md file in domains/gp/docs
    2. CHUNK  - split each file into sections at every "## " heading
    3. EMBED  - turn each chunk into a vector (numbers that capture meaning)
    4. STORE  - save chunks + vectors + source filename in ChromaDB
"""
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

# ---------- Paths ----------
# This file is guardrail/backend/app/rag/ingest.py
# parents[0]=rag, [1]=app, [2]=backend, [3]=guardrail
ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "domains" / "gp" / "docs"
DB_DIR = ROOT / "chroma_db"

COLLECTION_NAME = "gp_documents"

# Multilingual model: an Urdu or Polish question can match an English document.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def get_embedding_function():
    """The model that turns text into vectors. Shared by ingest and search."""
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def load_docs() -> list[tuple[str, str]]:
    """Return [(filename, text), ...] for every .md file in DOCS_DIR."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append((path.name, path.read_text(encoding="utf-8")))

    # Fail loudly: a silent "0 documents" would make the bot say "I don't know" to everything.
    if not docs:
        raise FileNotFoundError(f"No .md files found in {DOCS_DIR}")
    return docs


def chunk(text: str) -> list[str]:
    """
    Split a document into one chunk per '## ' section.
    The '# Title' line is added to the start of every chunk so each chunk
    still makes sense on its own (e.g. "Repeat Prescriptions / How long it takes").
    """
    lines = text.splitlines()
    title = lines[0] if lines else ""

    chunks: list[str] = []
    current: list[str] = []

    for line in lines:
        # A new "## " heading means the previous section is finished.
        if line.startswith("## ") and current:
            chunks.append("\n".join(current).strip())
            current = [title]  # start the next chunk with the document title
        current.append(line)

    # Don't forget the last section after the loop ends.
    if current:
        chunks.append("\n".join(current).strip())

    # Keep only real sections. This drops the intro (just the title + SYNTHETIC banner).
    return [c for c in chunks if "\n## " in c]


def build_index() -> None:
    """Rebuild the ChromaDB collection from scratch."""
    docs = load_docs()

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Delete the old collection so edited/removed docs never leave stale chunks behind.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # first run: nothing to delete

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=get_embedding_function(),
        metadata={"hnsw:space": "cosine"},  # measure closeness by meaning (cosine distance)
    )

    ids, texts, metadatas = [], [], []
    for filename, text in docs:
        for i, piece in enumerate(chunk(text)):
            ids.append(f"{filename}-{i}")         # unique id, e.g. "04_repeat_prescriptions.md-2"
            texts.append(piece)
            metadatas.append({"source": filename})  # so answers can show where they came from

    # Chroma embeds the texts for us using the embedding function.
    collection.add(ids=ids, documents=texts, metadatas=metadatas)

    print(f"Root:   {ROOT}")
    print(f"Stored {len(ids)} chunks from {len(docs)} documents in {DB_DIR}")


if __name__ == "__main__":
    build_index()