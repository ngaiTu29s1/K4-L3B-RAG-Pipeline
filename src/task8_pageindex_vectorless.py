"""Task 8 — PageIndex vectorless fallback with one cached corpus PDF."""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from pageindex import PageIndexAPIError

from .contracts import validate_search_results


ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = ROOT / "data" / "standardized"
PDF_PATH = ROOT / "pageindex_pdfs" / "patchlens_corpus.pdf"
CACHE_PATH = ROOT / "pageindex_doc_ids.json"
BASE_URL = "https://api.pageindex.ai"
REQUEST_TIMEOUT = 60
JOB_TIMEOUT = 30 * 60
POLL_INTERVAL = 5
MAX_FREE_PAGES = 200


def _api_key() -> str:
    load_dotenv(ROOT / ".env")
    key = os.getenv("PAGEINDEX_API_KEY")
    if not key:
        raise RuntimeError("PAGEINDEX_API_KEY is missing from .env")
    return key


def _request(method: str, path: str, **kwargs) -> dict:
    response = requests.request(
        method,
        f"{BASE_URL}{path}",
        headers={"api_key": _api_key()},
        timeout=REQUEST_TIMEOUT,
        **kwargs,
    )
    if response.status_code != 200:
        raise PageIndexAPIError(
            f"PageIndex {method} {path} failed ({response.status_code}): "
            f"{response.text[:500]}"
        )
    return response.json()


def _corpus_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        digest.update(path.relative_to(STANDARDIZED_DIR).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def _build_pdf() -> int:
    from fpdf import FPDF

    regular = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    pdf = FPDF(format="A4")
    pdf.set_margins(4, 4, 4)
    pdf.set_auto_page_break(True, 4)
    pdf.add_font("DejaVu", "", fname=regular)
    pdf.add_font("DejaVu", "B", fname=bold)
    pdf.add_page()
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        pdf.set_font("DejaVu", "B", 6)
        pdf.multi_cell(
            0,
            2.5,
            text=path.relative_to(STANDARDIZED_DIR).as_posix(),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.set_font("DejaVu", "", 5)
        pdf.multi_cell(
            0,
            2.2,
            text=path.read_text(encoding="utf-8"),
            new_x="LMARGIN",
            new_y="NEXT",
        )
        pdf.ln(0.6)

    pages = len(pdf.pages)
    if pages > MAX_FREE_PAGES:
        raise RuntimeError(
            f"PageIndex corpus has {pages} pages; free limit is {MAX_FREE_PAGES}"
        )
    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(PDF_PATH)
    return pages


def _wait_document(doc_id: str) -> None:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        status = _request("GET", f"/doc/{doc_id}/metadata/").get("status", "")
        if status == "completed":
            return
        if status == "failed":
            raise RuntimeError(f"PageIndex failed to process document {doc_id}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"PageIndex document {doc_id} was not ready within {JOB_TIMEOUT}s")


def upload_documents() -> None:
    """Build and upload one PDF, caching its PageIndex document ID."""
    corpus_hash = _corpus_hash()
    cache = _load_cache()
    if cache.get("corpus_hash") == corpus_hash and cache.get("doc_id"):
        _wait_document(cache["doc_id"])
        return

    pages = _build_pdf()
    with PDF_PATH.open("rb") as file:
        result = _request(
            "POST",
            "/doc/",
            files={"file": file},
            data={"if_retrieval": True},
        )
    doc_id = result["doc_id"]
    CACHE_PATH.write_text(
        json.dumps(
            {"doc_id": doc_id, "corpus_hash": corpus_hash, "pages": pages},
            indent=2,
        )
    )
    print(f"Uploaded {pages} pages as {doc_id}; waiting for processing")
    _wait_document(doc_id)


def _wait_retrieval(retrieval_id: str) -> list[dict]:
    deadline = time.monotonic() + JOB_TIMEOUT
    while time.monotonic() < deadline:
        result = _request("GET", f"/retrieval/{retrieval_id}/")
        status = result.get("status")
        if status == "completed":
            return result.get("retrieved_nodes", [])
        if status == "failed":
            raise RuntimeError(f"PageIndex retrieval {retrieval_id} failed")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"PageIndex retrieval {retrieval_id} timed out")


def _relevant_text(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [text for item in value for text in _relevant_text(item)]
    if isinstance(value, dict):
        return _relevant_text(value.get("relevant_content", ""))
    return []


def _parse_nodes(doc_id: str, nodes: list[dict], top_k: int) -> list[dict]:
    results = []
    for rank, node in enumerate(nodes[:top_k], 1):
        parts = _relevant_text(node.get("relevant_contents", []))
        content = "\n".join(part for part in parts if part).strip()
        content = content or node.get("text") or node.get("summary") or node.get("title")
        source_match = re.search(
            r"((?:legal|news)/\S+\.md)", f"{node.get('title', '')}\n{content}"
        )
        source = source_match.group(1) if source_match else PDF_PATH.name
        results.append(
            {
                "id": f"pageindex:{doc_id}:{node.get('node_id', rank)}",
                "content": content,
                "score": 1.0 / rank,
                "metadata": {
                    "source": source,
                    "title": node.get("title") or source,
                    "doc_type": "news" if source.startswith("news/") else "legal",
                    "url": None,
                    "chunk_index": rank - 1,
                },
                "retrieval_method": "pageindex",
            }
        )
    validate_search_results(results, top_k=top_k, expected_method="pageindex")
    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Return PageIndex nodes as ranked SearchResult items."""
    if not isinstance(query, str) or not query.strip() or top_k <= 0:
        return []
    cache = _load_cache()
    if cache.get("corpus_hash") != _corpus_hash() or not cache.get("doc_id"):
        raise RuntimeError("Run upload_documents() for the current corpus first")
    doc_id = cache["doc_id"]
    _wait_document(doc_id)
    job = _request(
        "POST",
        "/retrieval/",
        json={"doc_id": doc_id, "query": query, "thinking": False},
    )
    return _parse_nodes(doc_id, _wait_retrieval(job["retrieval_id"]), top_k)


if __name__ == "__main__":
    upload_documents()
