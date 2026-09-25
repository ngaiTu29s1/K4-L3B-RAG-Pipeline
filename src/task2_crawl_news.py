"""Crawl the five official Riot patch notes used by PatchLens."""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
CACHE_DIR = Path(__file__).parent.parent / ".cache"

PATCH_URLS = {
    patch: (
        "https://www.leagueoflegends.com/en-us/news/game-updates/"
        f"league-of-legends-patch-{patch.replace('.', '-')}-notes/"
    )
    for patch in ("26.15", "26.16", "26.17", "26.18", "26.19")
}


async def crawl_article(crawler, patch: str, url: str) -> dict:
    result = await crawler.arun(url=url)
    markdown = str(result.markdown or "").strip()
    if not result.success or not markdown:
        raise RuntimeError(result.error_message or f"Empty response from {url}")
    return {
        "url": url,
        "title": (result.metadata or {}).get("title", f"Patch {patch} Notes"),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "patch": patch,
        "source_tier": "official",
        "content_markdown": markdown,
    }


async def crawl_all() -> None:
    """Crawl all pinned patch notes with one browser session."""
    os.environ.setdefault("CRAWL4_AI_BASE_DIRECTORY", str(CACHE_DIR))
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_crawler_strategy import AsyncHTTPCrawlerStrategy

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    failures = []

    async with AsyncWebCrawler(
        crawler_strategy=AsyncHTTPCrawlerStrategy(),
        base_directory=str(CACHE_DIR),
    ) as crawler:
        for patch, url in PATCH_URLS.items():
            try:
                article = await crawl_article(crawler, patch, url)
                output = DATA_DIR / f"patch_{patch.replace('.', '_')}.json"
                output.write_text(
                    json.dumps(article, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"Saved: {output}")
            except Exception as error:
                failures.append(patch)
                print(f"Failed: {url} — {error}")

    if failures:
        raise RuntimeError(f"Failed to crawl patches: {', '.join(failures)}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
