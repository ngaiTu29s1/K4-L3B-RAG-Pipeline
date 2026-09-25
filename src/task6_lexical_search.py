"""Task 6 — BM25 lexical search over the Task 4 chunks."""

import re

from .contracts import validate_search_results
from .task4_chunking_indexing import get_collection


CORPUS: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _get_corpus() -> list[dict]:
    global CORPUS
    if CORPUS is None:
        stored = get_collection().get(include=["documents", "metadatas"])
        CORPUS = [
            {"id": item_id, "content": content, "metadata": metadata}
            for item_id, content, metadata in zip(
                stored["ids"], stored["documents"] or [], stored["metadatas"] or []
            )
        ]
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Create a BM25 index from Task 4 chunks."""
    from rank_bm25 import BM25Okapi

    if not corpus:
        raise ValueError("Cannot build BM25 index from an empty corpus")
    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Return BM25 SearchResult items sorted by descending score."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []

    corpus = _get_corpus()
    if not corpus:
        return []
    scores = build_bm25_index(corpus).get_scores(_tokenize(query))
    indices = sorted(range(len(corpus)), key=scores.__getitem__, reverse=True)
    results = [
        {
            "id": corpus[index]["id"],
            "content": corpus[index]["content"],
            "score": float(scores[index]),
            "metadata": corpus[index]["metadata"],
            "retrieval_method": "bm25",
        }
        for index in indices[:top_k]
        if scores[index] > 0
    ]
    validate_search_results(results, top_k=top_k, expected_method="bm25")
    return results


if __name__ == "__main__":
    for result in lexical_search("Volibear patch 26.19", top_k=3):
        print(result["score"], result["metadata"]["title"])
