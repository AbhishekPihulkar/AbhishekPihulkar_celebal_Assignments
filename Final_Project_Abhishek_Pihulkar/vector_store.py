"""
vector_store.py
----------------
Embeds chunks (from data/chunks.json) with a local sentence-transformer
model and stores them in a persistent ChromaDB collection, with full
metadata attached to every vector for filtering at query time.

Run once after ingest.py:  python vector_store.py
"""

import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")
CHROMA_DIR = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "drivewise_brochures"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2" 


def get_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


def get_client():
    return chromadb.PersistentClient(path=CHROMA_DIR)


def build_index():
    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError("Run ingest.py first to produce data/chunks.json")

    with open(CHUNKS_FILE) as fh:
        chunks = json.load(fh)

    embedder = get_embedder()
    client = get_client()

   
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]

    print(f"Embedding {len(texts)} chunks with {EMBED_MODEL_NAME} ...")
    embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    
    BATCH = 500
    for i in range(0, len(texts), BATCH):
        collection.add(
            ids=ids[i:i+BATCH],
            embeddings=embeddings[i:i+BATCH],
            documents=texts[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {CHROMA_DIR}")


def get_collection():
    client = get_client()
    return client.get_collection(COLLECTION_NAME)


if __name__ == "__main__":
    build_index()
