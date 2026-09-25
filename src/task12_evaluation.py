"""
Task 12 — Evaluation runner và đo lường A/B metrics.

Chạy đánh giá trên golden dataset cho hai cấu hình:
  - Config A: Dense-only (semantic search)
  - Config B: Hybrid + RRF (dense + BM25 + Reciprocal Rank Fusion)

Đo 4 metrics chuẩn RAG:
  1. Faithfulness: Mức độ trung thực của câu trả lời với context được truy xuất.
  2. Answer Relevance: Độ phù hợp và chính xác của câu trả lời so với câu hỏi / expected answer.
  3. Context Recall: Tỉ lệ thông tin cần thiết từ ground truth context xuất hiện trong retrieved chunks.
  4. Context Precision: Thứ hạng của các chunk chứa thông tin liên quan (Mean Average Precision - MAP).
"""

import json
import math
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

from .task4_chunking_indexing import embed_texts
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search
from .task10_generation import (
    call_llm,
    format_context,
    reorder_for_llm,
    SAFE_REFUSAL,
    SYSTEM_PROMPT,
    TOP_K,
)

load_dotenv()

ROOT = Path(__file__).parent.parent
EVALUATION_DIR = ROOT / "group_project" / "evaluation"
GOLDEN_DATASET_PATH = EVALUATION_DIR / "golden_dataset.json"
CACHE_PATH = EVALUATION_DIR / ".eval_embeddings_cache.json"

SCORE_THRESHOLD = 0.66


def load_query_embeddings(queries: list[str]) -> dict[str, list[float]]:
    """Load or precompute query embeddings in batch requests."""
    from google import genai
    from google.genai import types

    cache = {}
    if CACHE_PATH.is_file():
        try:
            cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    missing = [q for q in queries if q not in cache]
    if missing:
        print(f"Pre-embedding {len(missing)} queries in batch...")
        load_dotenv(override=True)
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        for i in range(0, len(missing), 20):
            batch = missing[i : i + 20]
            resp = client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=768,
                ),
            )
            for q, emb in zip(batch, resp.embeddings):
                raw_vec = emb.values
                norm = math.sqrt(sum(x * x for x in raw_vec))
                cache[q] = [x / norm for x in raw_vec]

        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        print(f"Successfully cached {len(cache)} query embeddings.")
    return cache


def retrieve_with_cache(
    query: str,
    query_vector: list[float],
    top_k: int = 5,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieve chunks for a query using cached embedding."""
    from .task4_chunking_indexing import get_collection
    from .contracts import validate_search_results

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    # Semantic search with precomputed vector
    n_results = min(top_k * 2, count)
    response = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    dense = [
        {
            "id": item_id,
            "content": content,
            "score": max(0.0, min(1.0, 1.0 - float(distance))),
            "metadata": metadata,
            "retrieval_method": "dense",
        }
        for item_id, content, metadata, distance in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        )
    ]
    dense.sort(key=lambda item: item["score"], reverse=True)
    validate_search_results(dense, top_k=top_k * 2, expected_method="dense")

    sparse = lexical_search(query, top_k=top_k * 2)

    results = (
        rerank_rrf([dense, sparse], top_k=top_k)
        if use_reranking
        else dense[:top_k]
    )

    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < SCORE_THRESHOLD:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            pass

    return results


def generate_answer(query: str, chunks: list[dict]) -> dict:
    """Generate answer with citation using standard prompt and model, with retries."""
    valid_ids = {chunk["id"] for chunk in chunks}
    answer = ""
    if chunks:
        context = format_context(reorder_for_llm(chunks))
        for attempt in range(4):
            try:
                answer = call_llm(
                    SYSTEM_PROMPT,
                    f"Context:\n{context}\n\nQuestion: {query}",
                ).strip()
                break
            except Exception as exc:
                if attempt < 3:
                    time.sleep(3.0 * (attempt + 1))
                else:
                    print(f"LLM call failed after 4 attempts: {exc}")
                    answer = ""

    cited_ids = set(re.findall(r"\[([^\[\]]+)\]", answer))
    if not answer or answer == SAFE_REFUSAL or not cited_ids or not cited_ids <= valid_ids:
        return {"answer": SAFE_REFUSAL, "sources": [], "retrieval_source": "none"}
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0]["retrieval_method"],
    }


def _extract_facts(text: str) -> list[str]:
    """Extract key factual statements, numbers, and clauses for recall/precision."""
    cleaned = re.sub(r"[#*>`_-]", " ", text)
    # Split by newlines, periods, semicolons, commas
    parts = re.split(r"[\n.;]+", cleaned)
    facts = []
    for p in parts:
        p = p.strip()
        if len(p) >= 4 and not p.startswith("http") and "Source:" not in p and "URL:" not in p:
            facts.append(p)
    return facts


def compute_context_recall(expected_context: str, retrieved_chunks: list[dict], is_ood: bool) -> float:
    """Measure fraction of expected facts present in retrieved chunks."""
    if is_ood:
        return 1.0

    facts = _extract_facts(expected_context)
    if not facts:
        return 1.0

    combined_text = " ".join(chunk["content"].lower() for chunk in retrieved_chunks)
    matched = 0
    for fact in facts:
        # Check sub-phrases / numbers in fact
        words = [w.lower() for w in re.findall(r"[\w%+/]+", fact) if len(w) > 1]
        if not words:
            continue
        found_count = sum(1 for w in words if w in combined_text)
        if found_count / len(words) >= 0.6:
            matched += 1

    return round(min(1.0, matched / len(facts)), 4)


def compute_context_precision(expected_context: str, retrieved_chunks: list[dict], is_ood: bool) -> float:
    """Mean Average Precision (MAP) of retrieved chunks."""
    if is_ood:
        return 1.0

    if not retrieved_chunks:
        return 0.0

    facts = _extract_facts(expected_context)
    key_terms = [w.lower() for w in re.findall(r"[\w%+/]+", expected_context) if len(w) > 2]

    # Evaluate relevance of each chunk
    relevance = []
    for chunk in retrieved_chunks:
        content = chunk["content"].lower()
        # chunk-0 is metadata-only header: if it doesn't contain actual payload facts, mark not fully relevant
        if chunk["id"].endswith("chunk-0") and ("entity type:" in content or "source: riot" in content):
            # Check if it has actual gameplay text
            has_payload = any(k in content for k in ["hồi chiêu", "sát thương", "tốc độ", "vàng", "nội tại", "giá"])
            relevance.append(1 if has_payload else 0)
            continue

        # For other chunks, check keyword coverage
        term_matches = sum(1 for term in key_terms if term in content)
        rel = 1 if (term_matches >= 3 or (key_terms and term_matches / len(key_terms) >= 0.25)) else 0
        relevance.append(rel)

    num_relevant = sum(relevance)
    if num_relevant == 0:
        return 0.0

    precisions = []
    running_rel = 0
    for rank, rel in enumerate(relevance, start=1):
        if rel == 1:
            running_rel += 1
            precisions.append(running_rel / rank)

    return round(sum(precisions) / num_relevant, 4)


def compute_faithfulness(answer: str, retrieved_chunks: list[dict], is_ood: bool) -> float:
    """Verify that every statement/number in answer is supported by retrieved context."""
    if is_ood:
        return 1.0 if answer == SAFE_REFUSAL else 0.0

    if answer == SAFE_REFUSAL:
        return 1.0

    combined_text = " ".join(c["content"].lower() for c in retrieved_chunks)
    # Check numbers and key words in answer
    answer_tokens = re.findall(r"[\w%+/]+", answer.lower())
    if not answer_tokens:
        return 0.0

    # Numbers in answer must exist in context
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", answer.lower())
    supported_numbers = sum(1 for num in numbers if num in combined_text)
    number_score = (supported_numbers / len(numbers)) if numbers else 1.0

    # General token support
    supported_tokens = sum(1 for t in answer_tokens if t in combined_text)
    token_score = supported_tokens / len(answer_tokens)

    return round(0.5 * number_score + 0.5 * token_score, 4)


def compute_answer_relevance(
    answer: str,
    expected_answer: str,
    is_ood: bool,
) -> float:
    """Measure alignment between answer and ground truth."""
    if is_ood:
        return 1.0 if answer == SAFE_REFUSAL else 0.0

    if answer == SAFE_REFUSAL:
        return 0.0

    exp_words = set(re.findall(r"[\w%+/]+", expected_answer.lower()))
    ans_words = set(re.findall(r"[\w%+/]+", answer.lower()))

    if not exp_words:
        return 1.0

    overlap = exp_words & ans_words
    precision = len(overlap) / len(ans_words) if ans_words else 0.0
    recall = len(overlap) / len(exp_words) if exp_words else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return round(f1, 4)


def evaluate_run() -> dict:
    """Execute full evaluation across both configurations."""
    cases = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    queries = [c["question"] for c in cases]
    q_map = load_query_embeddings(queries)

    results_dense = []
    results_hybrid = []

    print(f"Starting evaluation of {len(cases)} cases across 2 configurations...")

    for i, case in enumerate(cases):
        q = case["question"]
        expected_ans = case["expected_answer"]
        expected_ctx = case["expected_context"]
        is_ood = (expected_ans == SAFE_REFUSAL)
        q_vec = q_map[q]

        print(f"\n--- Case {i+1}/{len(cases)}: {q[:60]}... ---")

        # Config A: Dense-only
        dense_chunks = retrieve_with_cache(q, q_vec, top_k=TOP_K, use_reranking=False)
        dense_gen = generate_answer(q, dense_chunks)
        dense_eval = {
            "case_id": i + 1,
            "question": q,
            "config": "dense-only",
            "is_ood": is_ood,
            "retrieved_chunk_ids": [c["id"] for c in dense_chunks],
            "answer": dense_gen["answer"],
            "faithfulness": compute_faithfulness(dense_gen["answer"], dense_chunks, is_ood),
            "answer_relevance": compute_answer_relevance(dense_gen["answer"], expected_ans, is_ood),
            "context_recall": compute_context_recall(expected_ctx, dense_chunks, is_ood),
            "context_precision": compute_context_precision(expected_ctx, dense_chunks, is_ood),
        }
        dense_eval["average"] = round(
            (dense_eval["faithfulness"] + dense_eval["answer_relevance"] +
             dense_eval["context_recall"] + dense_eval["context_precision"]) / 4.0, 4
        )
        results_dense.append(dense_eval)

        # Config B: Hybrid + RRF
        hybrid_chunks = retrieve_with_cache(q, q_vec, top_k=TOP_K, use_reranking=True)
        hybrid_gen = generate_answer(q, hybrid_chunks)
        hybrid_eval = {
            "case_id": i + 1,
            "question": q,
            "config": "hybrid+rrf",
            "is_ood": is_ood,
            "retrieved_chunk_ids": [c["id"] for c in hybrid_chunks],
            "answer": hybrid_gen["answer"],
            "faithfulness": compute_faithfulness(hybrid_gen["answer"], hybrid_chunks, is_ood),
            "answer_relevance": compute_answer_relevance(hybrid_gen["answer"], expected_ans, is_ood),
            "context_recall": compute_context_recall(expected_ctx, hybrid_chunks, is_ood),
            "context_precision": compute_context_precision(expected_ctx, hybrid_chunks, is_ood),
        }
        hybrid_eval["average"] = round(
            (hybrid_eval["faithfulness"] + hybrid_eval["answer_relevance"] +
             hybrid_eval["context_recall"] + hybrid_eval["context_precision"]) / 4.0, 4
        )
        results_hybrid.append(hybrid_eval)

        print(f"  Config A (Dense):  Avg: {dense_eval['average']:.4f} | Rec: {dense_eval['context_recall']:.4f} | Prec: {dense_eval['context_precision']:.4f}")
        print(f"  Config B (Hybrid): Avg: {hybrid_eval['average']:.4f} | Rec: {hybrid_eval['context_recall']:.4f} | Prec: {hybrid_eval['context_precision']:.4f}")

    def avg_metric(res_list, key):
        return round(sum(r[key] for r in res_list) / len(res_list), 4)

    summary = {
        "dataset_size": len(cases),
        "overall_scores": {
            "faithfulness": {
                "dense": avg_metric(results_dense, "faithfulness"),
                "hybrid": avg_metric(results_hybrid, "faithfulness"),
                "delta": round(avg_metric(results_hybrid, "faithfulness") - avg_metric(results_dense, "faithfulness"), 4),
            },
            "answer_relevance": {
                "dense": avg_metric(results_dense, "answer_relevance"),
                "hybrid": avg_metric(results_hybrid, "answer_relevance"),
                "delta": round(avg_metric(results_hybrid, "answer_relevance") - avg_metric(results_dense, "answer_relevance"), 4),
            },
            "context_recall": {
                "dense": avg_metric(results_dense, "context_recall"),
                "hybrid": avg_metric(results_hybrid, "context_recall"),
                "delta": round(avg_metric(results_hybrid, "context_recall") - avg_metric(results_dense, "context_recall"), 4),
            },
            "context_precision": {
                "dense": avg_metric(results_dense, "context_precision"),
                "hybrid": avg_metric(results_hybrid, "context_precision"),
                "delta": round(avg_metric(results_hybrid, "context_precision") - avg_metric(results_dense, "context_precision"), 4),
            },
            "average": {
                "dense": avg_metric(results_dense, "average"),
                "hybrid": avg_metric(results_hybrid, "average"),
                "delta": round(avg_metric(results_hybrid, "average") - avg_metric(results_dense, "average"), 4),
            },
        },
    }

    # Save raw outputs
    (EVALUATION_DIR / "eval_results_dense.json").write_text(
        json.dumps(results_dense, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVALUATION_DIR / "eval_results_hybrid.json").write_text(
        json.dumps(results_hybrid, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (EVALUATION_DIR / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n================== SUMMARY ==================")
    print(f"Metrics (Dense vs Hybrid + RRF):")
    for m, vals in summary["overall_scores"].items():
        print(f"  {m:18s}: Dense={vals['dense']:.4f} | Hybrid={vals['hybrid']:.4f} | Delta={vals['delta']:+.4f}")

    return summary


if __name__ == "__main__":
    evaluate_run()
