import csv
import json
import os
import time
from datetime import datetime

LOG_FILE = os.path.join("logs", "query_log.csv")

FIELDS = [
    "timestamp", "brand", "model", "query", "status",
    "dense_candidate_ids", "dense_candidate_distances",
    "bm25_candidate_ids", "num_candidates",
    "reranked_ids", "reranked_scores", "num_below_threshold", "num_final_chunks",
    "retrieval_time_ms", "reranking_time_ms", "generation_time_ms", "total_time_ms",
    "error",
]


def _ensure_log_file():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()


def log_query(brand, model, query, status, diagnostics=None,
              generation_time_ms=0.0, total_time_ms=0.0, error=""):
    """
    diagnostics: the dict returned by retriever.retrieve() -- dense_candidate_ids,
    dense_candidate_distances, bm25_candidate_ids, num_candidates, reranked_ids,
    reranked_scores, num_below_threshold, num_final_chunks, retrieval_time_ms,
    reranking_time_ms. Pass None for a failure that happened before retrieval ran.
    """
    _ensure_log_file()
    diagnostics = diagnostics or {}

    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "brand": brand,
        "model": model,
        "query": query,
        "status": status,  # "success" or "failed"
        "dense_candidate_ids": json.dumps(diagnostics.get("dense_candidate_ids", [])),
        "dense_candidate_distances": json.dumps(diagnostics.get("dense_candidate_distances", [])),
        "bm25_candidate_ids": json.dumps(diagnostics.get("bm25_candidate_ids", [])),
        "num_candidates": diagnostics.get("num_candidates", 0),
        "reranked_ids": json.dumps(diagnostics.get("reranked_ids", [])),
        "reranked_scores": json.dumps(diagnostics.get("reranked_scores", [])),
        "num_below_threshold": diagnostics.get("num_below_threshold", 0),
        "num_final_chunks": diagnostics.get("num_final_chunks", 0),
        "retrieval_time_ms": diagnostics.get("retrieval_time_ms", 0.0),
        "reranking_time_ms": diagnostics.get("reranking_time_ms", 0.0),
        "generation_time_ms": round(generation_time_ms, 1),
        "total_time_ms": round(total_time_ms, 1),
        "error": error,
    }

    with open(LOG_FILE, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writerow(row)


class Timer:
    """Context manager for timing a single stage. .elapsed is in seconds."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start