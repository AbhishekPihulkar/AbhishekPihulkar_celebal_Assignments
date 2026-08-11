# DriveWise

Metadata-aware RAG assistant that answers questions about cars using their official brochures as the only source of truth.

## Overview

DriveWise lets a user pick a brand and model, then ask natural-language questions about that vehicle. Every answer is grounded in the selected brochure — retrieval is scoped by metadata (brand/model), and the system explicitly declines to answer rather than guess when the brochure doesn't cover something.

## Architecture

```
PDF brochures  →  ingest.py  →  chunks.json  →  vector_store.py  →  ChromaDB
                                                                        │
User query ──▶ retriever.py ◀── metadata filter (brand + model) ──────┘
                  │
                  ├─ dense vector search + BM25 lexical search
                  ├─ Reciprocal Rank Fusion (RRF)
                  ├─ cross-encoder re-ranking
                  ├─ confidence gate (MIN_RERANK_SCORE)
                  └─ context-window control
                  │
                  ▼
            generator.py  →  Groq LLM  →  grounded answer + sources
                  │
                  ▼
             logger.py  →  logs/query_log.csv
```

## Features

- **Metadata filtering** — retrieval scoped to the selected brand + model
- **Structured chunking** — brochure text split by detected section (engine, safety, dimensions, interior, infotainment, mileage)
- **Hybrid retrieval** — dense embeddings (semantic) + BM25 (lexical), fused via RRF, to handle both natural-language and short/keyword-style queries
- **Re-ranking** — cross-encoder re-scores fused candidates before selection
- **Source attribution** — every answer cites brand, model, section, and page number
- **Faithful generation** — answers only from retrieved context; explicitly says so when the brochure doesn't cover the question
- **Logging** — every query, its retrieval trace, timing, and outcome logged to CSV
- **Evaluation** — context relevance, faithfulness, and correctness scored per test case via LLM-as-judge

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF extraction | pypdf |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Lexical search | rank_bm25 |
| Re-ranking | cross-encoder (`ms-marco-MiniLM-L-6-v2`) |
| Generation | Groq (`llama-3.3-70b-versatile`) |
| Logging | CSV (pandas-readable) |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your GROQ_API_KEY (free at console.groq.com)
```

Add brochure PDFs to `data/`, named `<brand>_<model>_<version>.pdf`:

```
data/hyundai_creta_2026.pdf
data/hyundai_alcazar_2026.pdf
```

Ingest and index:

```bash
python ingest.py
python vector_store.py
```

Run:

```bash
streamlit run app.py
```

## Evaluation

```bash
python eval.py
```

Runs the test cases in `evaluation/test_cases.json` against the live pipeline and scores each answer on context relevance, faithfulness, and correctness. Results are saved to `evaluation/eval_results.json`.

## Project structure

```
app.py              Streamlit UI
ingest.py            PDF → structured chunks
vector_store.py       Chunk embedding + ChromaDB indexing
retriever.py          Hybrid retrieval, fusion, re-ranking
generator.py           Grounded answer generation
logger.py               Query/retrieval logging
eval.py                  Evaluation harness
preview_chunks.py         Inspect indexed chunks (for writing eval questions)
debug_scores.py            Inspect retrieval/rerank scores for a given query
evaluation/test_cases.json  Evaluation questions + reference answers
```

## Known limitations

- Section detection is keyword-based, not layout-aware — works well on brochures with clean paragraph structure, less reliably on dense tables.
- Brand/model/version are parsed from the filename; multi-word model names need a filename convention adjustment.
