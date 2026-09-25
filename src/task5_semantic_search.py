"""Task 5 — semantic search over the Task 4 Chroma collection."""

from .contracts import validate_search_results
from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    response = collection.query(
        query_embeddings=[embed_texts([query], task_type="RETRIEVAL_QUERY")[0]],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    results = [
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
    results.sort(key=lambda item: item["score"], reverse=True)
    validate_search_results(results, top_k=top_k, expected_method="dense")
    return results


if __name__ == "__main__":
    for result in semantic_search("Ahri có những kỹ năng gì?", top_k=3):
        print(result["score"], result["metadata"]["title"])
