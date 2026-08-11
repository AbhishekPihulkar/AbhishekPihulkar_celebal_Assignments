"""
eval.py
-------
Lightweight evaluation harness covering the three metrics the spec asks for:
  - Context Relevance: mean re-rank score of retrieved chunks
  - Faithfulness: does the answer avoid asserting things absent from context
    (approximated via an LLM-as-judge call, kept cheap/fast)
  - Answer Correctness: compares the model's answer to a reference answer
    you supply, via LLM-as-judge (1-5 scale)

Run:  python eval.py
Test cases live in evaluation/test_cases.json (brand/model/question/reference_answer).
"""

import os
import json
import math
from dotenv import load_dotenv
from groq import Groq
from retriever import retrieve
from generator import generate_answer

load_dotenv()


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

JUDGE_MODEL = "llama-3.3-70b-versatile"
TEST_CASES_FILE = os.path.join("evaluation", "test_cases.json")
RESULTS_FILE = os.path.join("evaluation", "eval_results.json")


def load_test_cases():
    with open(TEST_CASES_FILE) as fh:
        return json.load(fh)


def judge(question, context, answer, reference_answer):
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    prompt = f"""You are grading a RAG system's answer.

Question: {question}
Retrieved context: {context}
Model's answer: {answer}
Reference answer: {reference_answer}

Score two things from 1 (bad) to 5 (excellent), as JSON only:
{{"faithfulness": <int, does the answer stick strictly to the context with no invented facts>,
  "correctness": <int, does the answer match the reference answer>}}
"""
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=100,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"faithfulness": None, "correctness": None, "raw": text}


def run_eval():
    test_cases = load_test_cases()
    results = []

    for case in test_cases:
        # retrieve() returns (selected_chunks, diagnostics) -- unpack both,
        # diagnostics isn't needed here but must still be unpacked.
        chunks, _diagnostics = retrieve(case["question"], case["brand"], case["model"])

        context_relevance = (
            round(sum(sigmoid(c["rerank_score"]) for c in chunks) / len(chunks), 3)
            if chunks else 0.0
        )
        gen = generate_answer(case["question"], chunks)
        context_text = "\n".join(c["text"] for c in chunks)
        scores = judge(case["question"], context_text, gen["answer"], case["reference_answer"])

        results.append({
            "brand": case["brand"],
            "model": case["model"],
            "question": case["question"],
            "answer": gen["answer"],
            "context_relevance": context_relevance,
            **scores,
        })

    for r in results:
        print(json.dumps(r, indent=2))

    os.makedirs("evaluation", exist_ok=True)
    with open(RESULTS_FILE, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nSaved {len(results)} results to {RESULTS_FILE}")

    return results


if __name__ == "__main__":
    run_eval()