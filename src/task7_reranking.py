"""Task 7 — Reciprocal Rank Fusion for dense and BM25 results."""

from .contracts import validate_search_results


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse ranked lists by ID using ``sum(1 / (k + rank))``."""
    if top_k <= 0:
        return []
    if k < 0:
        raise ValueError("RRF k must be non-negative")

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        seen = set()
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items.setdefault(item_id, item)

    results = [
        {
            **items[item_id],
            "score": scores[item_id],
            "retrieval_method": "hybrid",
        }
        for item_id in sorted(scores, key=scores.get, reverse=True)[:top_k]
    ]
    validate_search_results(results, top_k=top_k, expected_method="hybrid")
    return results


if __name__ == "__main__":
    print("Use rerank_rrf([dense_results, bm25_results]).")
