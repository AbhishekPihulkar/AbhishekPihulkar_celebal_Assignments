"""
retriever.py
------------
Implements the retrieval layer of DriveWise:
  1. Metadata filtering (brand + model selected by the user)
  2. HYBRID retrieval: dense vector search (semantic) + BM25 (exact lexical
     match), combined via Reciprocal Rank Fusion (RRF)
  3. Re-ranking of fused candidates with a cross-encoder
  4. Confidence gate (drop anything below MIN_RERANK_SCORE)
  5. Context window control (top-N chunks, char budget) before generation

"""

import json
import os
import time
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from vector_store import get_embedder, get_collection

RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

DENSE_TOP_K = 15            
BM25_TOP_K = 15              
RRF_K = 60                   
FUSION_TOP_K = 20            
FINAL_TOP_N = 4              
MAX_CONTEXT_CHARS = 3000    

MIN_RERANK_SCORE = -9.0

CHUNKS_FILE = os.path.join("data", "chunks.json")

_embedder = None
_reranker = None
_all_chunks = None          
_bm25_cache = {}             


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def _load_all_chunks():
  
    global _all_chunks
    if _all_chunks is None:
        with open(CHUNKS_FILE) as fh:
            raw = json.load(fh)
        _all_chunks = [
            {"chunk_id": f"chunk_{i}", "text": c["text"], "metadata": c["metadata"]}
            for i, c in enumerate(raw)
        ]
    return _all_chunks


def _get_bm25_for(brand: str, model: str):
 
    key = (brand, model)
    if key not in _bm25_cache:
        chunks = [
            c for c in _load_all_chunks()
            if c["metadata"]["brand"] == brand and c["metadata"]["model"] == model
        ]
        tokenized = [c["text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized) if tokenized else None
        _bm25_cache[key] = (bm25, chunks)
    return _bm25_cache[key]


def _bm25_search(query: str, brand: str, model: str, top_k: int = BM25_TOP_K):
   
    bm25, chunks = _get_bm25_for(brand, model)
    if bm25 is None or not chunks:
        return []
    scores = bm25.get_scores(query.lower().split())
    ranked_idx = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)
    return [chunks[i]["chunk_id"] for i in ranked_idx[:top_k]]


def _reciprocal_rank_fusion(*ranked_id_lists, k: int = RRF_K):

    fused_scores = defaultdict(float)
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            fused_scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


def retrieve(query: str, brand: str, model: str):
    """
    Returns (selected, diagnostics):
      selected    -> list of dicts {chunk_id, text, metadata, rerank_score},
                     trimmed to fit MAX_CONTEXT_CHARS, ordered by relevance.
      diagnostics -> dict with per-stage chunk ids/scores and timing.
    """
    collection = get_collection()
    embedder = _get_embedder()
    all_chunks_by_id = {c["chunk_id"]: c for c in _load_all_chunks()}

    stage_start = time.time()

    # --- Dense retrieval ---
    query_embedding = embedder.encode([query]).tolist()
    dense_results = collection.query(
        query_embeddings=query_embedding,
        n_results=DENSE_TOP_K,
        where={"$and": [{"brand": {"$eq": brand}}, {"model": {"$eq": model}}]},
    )
    dense_ids = dense_results.get("ids", [[]])[0]
    dense_distances = dense_results.get("distances", [[]])[0]

    # --- BM25 retrieval ---
    bm25_ids = _bm25_search(query, brand, model)

    retrieval_time_ms = (time.time() - stage_start) * 1000

    diagnostics = {
        "dense_candidate_ids": dense_ids,
        "dense_candidate_distances": [round(float(d), 4) for d in dense_distances],
        "bm25_candidate_ids": bm25_ids,
        "num_candidates": len(set(dense_ids) | set(bm25_ids)),
        "reranked_ids": [],
        "reranked_scores": [],
        "num_below_threshold": 0,
        "num_final_chunks": 0,
        "retrieval_time_ms": round(retrieval_time_ms, 1),
        "reranking_time_ms": 0.0,
    }

    if not dense_ids and not bm25_ids:
        return [], diagnostics

    # --- Fuse dense + BM25 rankings ---
    fused = _reciprocal_rank_fusion(dense_ids, bm25_ids)
    fused_top = fused[:FUSION_TOP_K]

    candidates = [
        all_chunks_by_id[chunk_id] for chunk_id, _ in fused_top
        if chunk_id in all_chunks_by_id
    ]
    if not candidates:
        return [], diagnostics

    # --- Re-rank fused candidates against the query ---
    rerank_start = time.time()
    reranker = _get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    diagnostics["reranking_time_ms"] = round((time.time() - rerank_start) * 1000, 1)

    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

    # --- Confidence gate ---
    confident = [(c, s) for c, s in ranked if s >= MIN_RERANK_SCORE]
    diagnostics["num_below_threshold"] = len(ranked) - len(confident)

    top_ranked = confident[:FINAL_TOP_N]
    diagnostics["reranked_ids"] = [c["chunk_id"] for c, s in top_ranked]
    diagnostics["reranked_scores"] = [round(float(s), 4) for c, s in top_ranked]


    selected = []
    total_chars = 0
    for chunk, score in top_ranked:
        doc = chunk["text"]
        if total_chars + len(doc) > MAX_CONTEXT_CHARS:
            continue
        selected.append({
            "chunk_id": chunk["chunk_id"],
            "text": doc,
            "metadata": chunk["metadata"],
            "rerank_score": float(score),
        })
        total_chars += len(doc)

    diagnostics["num_final_chunks"] = len(selected)

    return selected, diagnostics


def list_available_brands_models():
    """Utility for populating the UI dropdowns from whatever was indexed."""
    collection = get_collection()
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    pairs = sorted({(m["brand"], m["model"]) for m in all_meta})
    return pairs