"""
generator.py
------------
Takes the retrieved, re-ranked chunks and generates a grounded answer
via Groq's free/fast LLM API. Falls back to a clear "not found" response
if the context doesn't actually contain the answer, instead of letting
the model guess (this is what makes the system "faithful").
"""

import os
from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """You are DriveWise, a car brochure assistant. You answer ONLY using the
provided brochure context. Rules:
1. If the answer is present in the context, answer clearly and concisely.
2. If the context does NOT contain the answer, say so explicitly — do not guess or use
   outside knowledge about cars.
3. Never invent specifications, numbers, or features not present in the context.
4. Keep answers short and direct (2-4 sentences) unless the user asks for detail.
"""


def _client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com "
            "and set it as an environment variable or in a .env file."
        )
    return Groq(api_key=api_key)


def build_context_block(chunks):
    lines = []
    for i, c in enumerate(chunks, start=1):
        meta = c["metadata"]
        tag = f"[Source {i}: {meta['brand']} {meta['model']}, {meta['section']}, p.{meta['page_number']}]"
        lines.append(f"{tag}\n{c['text']}")
    return "\n\n".join(lines)


def generate_answer(query: str, chunks: list):
    if not chunks:
        return {
            "answer": "I couldn't find relevant information for this car in the brochure. "
                      "Try rephrasing, or check that the right brand/model is selected.",
            "sources": [],
        }

    context_block = build_context_block(chunks)

    user_prompt = f"""Brochure context:
{context_block}

User question: {query}

Answer using only the context above."""

    client = _client()
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,   # low temperature: we want faithful, not creative
        max_tokens=400,
    )

    answer_text = response.choices[0].message.content

    sources = [
        {
            "brand": c["metadata"]["brand"],
            "model": c["metadata"]["model"],
            "section": c["metadata"]["section"],
            "page": c["metadata"]["page_number"],
            "relevance_score": round(c["rerank_score"], 3),
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources}
