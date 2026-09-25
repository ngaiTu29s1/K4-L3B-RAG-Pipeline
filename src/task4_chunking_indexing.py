"""Task 4 — load Markdown, chunk, embed with Gemini, and index in Chroma."""

import hashlib
import math
import os
import re
import time
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from .contracts import validate_document


ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
CHROMA_DIR = ROOT / "chroma_db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768
EMBEDDING_BATCH_SIZE = 20
MIN_REQUEST_INTERVAL = 10.0
MAX_RETRIES = 5

COLLECTION_NAME = "rag_documents"
_LAST_REQUEST_AT = 0.0


@lru_cache(maxsize=1)
def _client():
    from google import genai

    load_dotenv(ROOT / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing from .env")
    return genai.Client(api_key=api_key)


def _throttle() -> None:
    global _LAST_REQUEST_AT
    wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _LAST_REQUEST_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_REQUEST_AT = time.monotonic()


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if not norm:
        raise ValueError("Gemini returned a zero embedding")
    return [value / norm for value in vector]


def _embed_batch(texts: list[str], task_type: str) -> list[list[float]]:
    from google.genai import errors, types

    for attempt in range(MAX_RETRIES):
        _throttle()
        try:
            response = _client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=EMBEDDING_DIM,
                ),
            )
            vectors = [_normalize(item.values or []) for item in response.embeddings or []]
            if len(vectors) != len(texts):
                raise RuntimeError("Gemini returned an unexpected embedding count")
            return vectors
        except errors.APIError as exc:
            if exc.code == 429 and "PerDay" in str(exc.details):
                raise RuntimeError(
                    "Gemini daily embedding quota is exhausted; rerun after its reset"
                ) from exc
            if exc.code not in {429, 500, 503} or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(min(60, 10 * 2**attempt))
    raise RuntimeError("Embedding retries exhausted")


def embed_texts(
    texts: list[str], *, task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    """Embed texts in quota-friendly batches using one Gemini project."""
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding texts must be non-empty strings")
    return [
        vector
        for start in range(0, len(texts), EMBEDDING_BATCH_SIZE)
        for vector in _embed_batch(texts[start : start + EMBEDDING_BATCH_SIZE], task_type)
    ]


def get_collection():
    """Open the persistent Chroma collection with cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR)).get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _field(markdown: str, name: str) -> str | None:
    match = re.search(rf"^\*\*{re.escape(name)}:\*\*\s*(.+)$", markdown, re.MULTILINE)
    return match.group(1).strip() if match else None


def load_documents() -> list[dict]:
    """Read standardized Markdown and preserve its source metadata."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        relative = path.relative_to(STANDARDIZED_DIR)
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        document = {
            "id": relative.as_posix(),
            "content": content,
            "metadata": {
                "source": _field(content, "Source") or path.name,
                "title": title_match.group(1).strip() if title_match else path.stem,
                "doc_type": "legal" if "legal" in relative.parts else "news",
                "url": _field(content, "URL"),
            },
        }
        validate_document(document)
        documents.append(document)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split documents into stable, overlapping recursive chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for document in documents:
        validate_document(document)
        for index, content in enumerate(splitter.split_text(document["content"])):
            chunk = {
                "id": f"{document['id']}::chunk-{index}",
                "content": content,
                "metadata": {**document["metadata"], "chunk_index": index},
            }
            validate_document(chunk, require_chunk=True)
            chunks.append(chunk)
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return chunks with document embeddings attached."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _pending_chunks(chunks: list[dict]) -> list[dict]:
    collection = get_collection()
    stored = collection.get(ids=[chunk["id"] for chunk in chunks], include=["metadatas"])
    hashes = {
        item_id: (metadata or {}).get("_content_sha256")
        for item_id, metadata in zip(stored["ids"], stored["metadatas"] or [])
    }
    return [
        chunk
        for chunk in chunks
        if hashes.get(chunk["id"]) != _content_hash(chunk["content"])
    ]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert embedded chunks into ChromaDB."""
    if not chunks:
        return
    get_collection().upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[
            {
                **{
                    key: "" if value is None else value
                    for key, value in chunk["metadata"].items()
                },
                "_content_sha256": _content_hash(chunk["content"]),
            }
            for chunk in chunks
        ],
    )


def run_pipeline() -> None:
    """Load, chunk, resume unchanged content, embed, and index."""
    chunks = chunk_documents(load_documents())
    pending = _pending_chunks(chunks)
    for start in range(0, len(pending), EMBEDDING_BATCH_SIZE):
        index_to_vectorstore(embed_chunks(pending[start : start + EMBEDDING_BATCH_SIZE]))
        print(f"Indexed {min(start + EMBEDDING_BATCH_SIZE, len(pending))}/{len(pending)}")
    print(f"Ready: {len(chunks)} chunks ({len(pending)} new or changed)")


if __name__ == "__main__":
    run_pipeline()
