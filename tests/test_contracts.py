import asyncio
import inspect
import json

import pytest

from src.contracts import (
    validate_document,
    validate_generation_result,
    validate_search_results,
)


def test_data_dragon_download_pins_version_and_writes_catalogs(tmp_path, monkeypatch):
    import src.task1_collect_legal_docs as task1

    calls = []

    def fake_fetch(url):
        calls.append(url)
        return ["26.19.1"] if url == task1.VERSIONS_URL else {"data": {"fixture": {}}}

    monkeypatch.setattr(task1, "DATA_DIR", tmp_path)
    monkeypatch.setattr(task1, "_fetch_json", fake_fetch)

    task1.download_documents()
    assert json.loads((tmp_path / "ddragon_version.json").read_text())["version"] == "26.19.1"
    assert set(task1.CATALOGS) <= {path.name for path in tmp_path.iterdir()}

    calls.clear()
    task1.download_documents()
    assert task1.VERSIONS_URL not in calls


def test_patch_crawler_returns_grounded_metadata():
    from src.task2_crawl_news import crawl_article

    class Result:
        success = True
        markdown = "# League of Legends Patch 26.19 Notes\n\n## Champions\n\n### Aatrox"
        metadata = {"title": "League of Legends Patch 26.19 Notes"}
        error_message = ""

    class Crawler:
        async def arun(self, *, url):
            return Result()

    url = (
        "https://www.leagueoflegends.com/en-us/news/game-updates/"
        "league-of-legends-patch-26-19-notes/"
    )
    article = asyncio.run(crawl_article(Crawler(), "26.19", url))
    assert article["url"] == url
    assert article["patch"] == "26.19"
    assert article["source_tier"] == "official"
    assert article["content_markdown"].startswith("# League of Legends")


def test_patch_parser_separates_summoners_rift_entries():
    from src.task3_convert_markdown import PATCH_SECTIONS, _entries, _top_sections

    markdown = """## Champions
### [Aatrox](https://example.com/aatrox)
SR change
#### Q - The Darkin Blade
More SR detail
## Classic
### Aatrox
Classic-only change
## Arena
### Aatrox
Arena-only change
"""
    sections = _top_sections(markdown)
    titles = [title for title, _ in _entries(sections["Champions"], "Champions")]

    assert titles == ["[Aatrox](https://example.com/aatrox)"]
    assert {name for name in sections if name in PATCH_SECTIONS} == {"Champions"}


def metadata(source: str = "tuition.md", chunk_index: int = 0) -> dict:
    return {
        "source": source,
        "title": "Tuition policy",
        "doc_type": "legal",
        "url": "https://university.example/tuition",
        "chunk_index": chunk_index,
    }


def result(
    item_id: str,
    score: float,
    method: str = "dense",
    content: str = "Tuition is paid per semester.",
) -> dict:
    return {
        "id": item_id,
        "content": content,
        "score": score,
        "metadata": metadata(chunk_index=int(item_id.rsplit("-", 1)[-1])),
        "retrieval_method": method,
    }


def test_public_function_signatures_are_stable():
    from src.task4_chunking_indexing import chunk_documents, load_documents
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    assert list(inspect.signature(load_documents).parameters) == []
    assert list(inspect.signature(chunk_documents).parameters) == ["documents"]
    assert list(inspect.signature(semantic_search).parameters) == ["query", "top_k"]
    assert list(inspect.signature(lexical_search).parameters) == ["query", "top_k"]
    assert list(inspect.signature(rerank_rrf).parameters) == ["ranked_lists", "top_k", "k"]
    assert list(inspect.signature(pageindex_search).parameters) == ["query", "top_k"]
    assert list(inspect.signature(retrieve).parameters) == [
        "query", "top_k", "score_threshold", "use_reranking"
    ]
    assert list(inspect.signature(generate_with_citation).parameters) == ["query", "top_k"]


def test_document_validator_accepts_contract():
    validate_document(
        {
            "id": "tuition",
            "content": "Tuition policy content",
            "metadata": {key: value for key, value in metadata().items() if key != "chunk_index"},
        }
    )


@pytest.mark.parametrize("missing", ["id", "content", "metadata"])
def test_document_validator_rejects_missing_fields(missing):
    document = {
        "id": "tuition",
        "content": "Tuition policy content",
        "metadata": {key: value for key, value in metadata().items() if key != "chunk_index"},
    }
    document.pop(missing)
    with pytest.raises(ValueError):
        validate_document(document)


def test_search_result_validator_checks_order_method_and_uniqueness():
    valid = [result("chunk-0", 0.9), result("chunk-1", 0.7)]
    validate_search_results(valid, top_k=2, expected_method="dense")

    with pytest.raises(ValueError, match="sorted"):
        validate_search_results(list(reversed(valid)))
    with pytest.raises(ValueError, match="unique"):
        validate_search_results([valid[0], valid[0]])
    with pytest.raises(ValueError, match="expected"):
        validate_search_results(valid, expected_method="bm25")


def test_chunk_documents_preserves_identity_and_metadata():
    from src.task4_chunking_indexing import CHUNK_SIZE, chunk_documents

    document = {
        "id": "tuition",
        "content": "Tuition policy. " * 100,
        "metadata": {
            "source": "tuition.md",
            "title": "Tuition policy",
            "doc_type": "legal",
            "url": None,
        },
    }
    chunks = chunk_documents([document])

    assert chunks
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for index, chunk in enumerate(chunks):
        validate_document(chunk, require_chunk=True)
        assert chunk["metadata"]["source"] == "tuition.md"
        assert chunk["metadata"]["chunk_index"] == index
        assert len(chunk["content"]) <= int(CHUNK_SIZE * 1.1)


def test_semantic_search_uses_shared_embedding_and_contract(monkeypatch):
    import src.task5_semantic_search as semantic

    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            assert kwargs["query_embeddings"] == [[0.1, 0.2]]
            return {
                "ids": [["chunk-0", "chunk-1"]],
                "documents": [["Relevant tuition text", "Less relevant text"]],
                "metadatas": [[metadata(chunk_index=0), metadata(chunk_index=1)]],
                "distances": [[0.1, 0.4]],
            }

    def fake_embed(texts, **kwargs):
        assert kwargs == {"task_type": "RETRIEVAL_QUERY"}
        return [[0.1, 0.2]]

    monkeypatch.setattr(semantic, "embed_texts", fake_embed)
    monkeypatch.setattr(semantic, "get_collection", lambda: FakeCollection())
    output = semantic.semantic_search("tuition", top_k=2)
    validate_search_results(output, top_k=2, expected_method="dense")


def test_lexical_search_returns_bm25_contract(monkeypatch):
    import src.task6_lexical_search as lexical

    corpus = [
        {
            "id": "chunk-0",
            "content": "tuition fee payment policy",
            "metadata": metadata(chunk_index=0),
        },
        {
            "id": "chunk-1",
            "content": "library opening hours",
            "metadata": metadata(source="library.md", chunk_index=1),
        },
        {
            "id": "chunk-2",
            "content": "student housing application",
            "metadata": metadata(source="housing.md", chunk_index=2),
        },
    ]
    monkeypatch.setattr(lexical, "CORPUS", corpus)
    output = lexical.lexical_search("tuition fee", top_k=2)
    validate_search_results(output, top_k=2, expected_method="bm25")
    assert output[0]["id"] == "chunk-0"


def test_rrf_uses_rank_deduplicates_and_marks_hybrid():
    from src.task7_reranking import rerank_rrf

    dense = [result("chunk-0", 0.9), result("chunk-1", 0.8)]
    bm25 = [
        result("chunk-1", 7.0, "bm25"),
        result("chunk-2", 5.0, "bm25"),
    ]
    fused = rerank_rrf([dense, bm25], top_k=3, k=60)

    validate_search_results(fused, top_k=3, expected_method="hybrid")
    assert [item["id"] for item in fused][0] == "chunk-1"
    expected = 1 / 62 + 1 / 61
    assert fused[0]["score"] == pytest.approx(expected)


def test_pageindex_nodes_are_mapped_to_search_results():
    from src.task8_pageindex_vectorless import _parse_nodes

    nodes = [
        {
            "node_id": "42",
            "title": "Volibear Patch 26.19",
            "relevant_contents": [
                [
                    {
                        "relevant_content": "news/patch__26_19__champion__volibear.md\nQ damage increased"
                    }
                ]
            ],
        }
    ]
    output = _parse_nodes("doc-1", nodes, top_k=1)

    validate_search_results(output, top_k=1, expected_method="pageindex")
    assert output[0]["metadata"]["source"].startswith("news/")
    assert output[0]["content"].endswith("Q damage increased")


def test_reorder_is_non_mutating_and_context_contains_source():
    from src.task10_generation import format_context, reorder_for_llm

    chunks = [result(f"chunk-{index}", 1 - index / 10, "hybrid") for index in range(5)]
    original_ids = [item["id"] for item in chunks]
    reordered = reorder_for_llm(chunks)

    assert [item["id"] for item in chunks] == original_ids
    assert sorted(item["id"] for item in reordered) == sorted(original_ids)
    assert reordered[0]["id"] == "chunk-0"
    context = format_context(reordered)
    assert "tuition.md" in context
    assert "Tuition policy" in context


def test_retrieve_uses_dense_score_for_fallback(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [result("chunk-0", 0.2, "dense")]
    sparse = [result("chunk-1", 4.0, "bm25")]
    fallback = [result("chunk-2", 1.0, "pageindex")]

    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: sparse)
    monkeypatch.setattr(pipeline, "rerank_rrf", lambda lists, top_k: [])
    monkeypatch.setattr(pipeline, "pageindex_search", lambda query, top_k: fallback)

    output = pipeline.retrieve("tuition", top_k=2, score_threshold=0.5)
    assert output == fallback
    validate_search_results(output, top_k=2, expected_method="pageindex")


def test_retrieve_fuses_once_when_dense_is_confident(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [result("chunk-0", 0.9, "dense")]
    sparse = [result("chunk-1", 4.0, "bm25")]
    fused = [result("chunk-0", 0.03, "hybrid")]
    calls = {"rrf": 0, "fallback": 0}

    def fake_rrf(lists, top_k):
        calls["rrf"] += 1
        assert lists == [dense, sparse]
        return fused

    def fake_fallback(query, top_k):
        calls["fallback"] += 1
        return []

    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: sparse)
    monkeypatch.setattr(pipeline, "rerank_rrf", fake_rrf)
    monkeypatch.setattr(pipeline, "pageindex_search", fake_fallback)

    output = pipeline.retrieve("tuition", top_k=2, score_threshold=0.5)
    assert output == fused
    assert calls == {"rrf": 1, "fallback": 0}


def test_retrieve_survives_fallback_provider_error(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [result("chunk-0", 0.2, "dense")]
    hybrid = [result("chunk-0", 0.02, "hybrid")]
    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: [])
    monkeypatch.setattr(pipeline, "rerank_rrf", lambda lists, top_k: hybrid)

    def unavailable(query, top_k):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(pipeline, "pageindex_search", unavailable)
    output = pipeline.retrieve("tuition", top_k=2, score_threshold=0.5)
    assert output == hybrid


def test_generation_result_validator_accepts_safe_refusal():
    validate_generation_result(
        {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }
    )


def test_generation_checks_citations_and_survives_provider_error(monkeypatch):
    import src.task10_generation as generation

    chunks = [result("chunk-0", 0.03, "hybrid")]
    monkeypatch.setattr(generation, "retrieve", lambda query, top_k: chunks)
    monkeypatch.setattr(
        generation,
        "call_llm",
        lambda system_prompt, user_message: "Học phí theo học kỳ [chunk-0].",
    )
    output = generation.generate_with_citation("Học phí thế nào?", top_k=1)
    validate_generation_result(output)
    assert output["sources"] == chunks
    assert output["retrieval_source"] == "hybrid"

    monkeypatch.setattr(
        generation,
        "call_llm",
        lambda system_prompt, user_message: "Thông tin không có nguồn [chunk-999].",
    )
    assert generation.generate_with_citation("Học phí thế nào?", top_k=1)[
        "retrieval_source"
    ] == "none"

    def unavailable(system_prompt, user_message):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(generation, "call_llm", unavailable)
    assert generation.generate_with_citation("Học phí thế nào?", top_k=1)[
        "answer"
    ] == generation.SAFE_REFUSAL

    monkeypatch.setattr(generation, "retrieve", lambda query, top_k: unavailable(None, None))
    assert generation.generate_with_citation("Học phí thế nào?", top_k=1)[
        "retrieval_source"
    ] == "none"
