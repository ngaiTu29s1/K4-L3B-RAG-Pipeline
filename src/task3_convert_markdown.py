"""Normalize Data Dragon and Riot patch notes into entity-level Markdown."""

import json
import re
import unicodedata
from html import unescape
from pathlib import Path


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
TARGET_PATCH = "26.19"
PATCH_SECTIONS = {
    "Champions": "champion",
    "Items": "item",
    "Runes": "rune",
    "Systems": "system",
    "Ranked Season 3 Start": "system",
    "Aegis of Valor": "system",
    "Apex Duo Restrictions": "system",
}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _plain(value: str) -> str:
    return " ".join(unescape(re.sub(r"<[^>]+>", " ", value or "")).split())


def _slug(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", ascii_value.lower()).strip("-") or "entry"


def _header(
    title: str,
    source: str,
    url: str,
    entity_type: str,
    entity: str,
    patch: str = TARGET_PATCH,
) -> str:
    return (
        f"# {title}\n\n"
        f"**Source:** {source}\n\n"
        f"**URL:** {url}\n\n"
        f"**Patch:** {patch}\n\n"
        f"**Entity type:** {entity_type}\n\n"
        f"**Entity name:** {entity}\n"
    )


def _write(directory: Path, filename: str, content: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(content.strip() + "\n", encoding="utf-8")


def convert_legal_docs() -> None:
    source_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.md"):
        path.unlink()

    version = _load(source_dir / "ddragon_version.json")["version"]
    base_url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/vi_VN"
    champions = _load(source_dir / "champion_details_vi_VN.json")["data"]
    items = _load(source_dir / "items_vi_VN.json")["data"]
    spells = _load(source_dir / "summoner_spells_vi_VN.json")["data"]

    for champion in champions.values():
        champion_id = champion["id"]
        lines = [
            _header(
                f"{champion['name']} — {champion['title']}",
                "Riot Data Dragon",
                f"{base_url}/champion/{champion_id}.json",
                "champion",
                champion["name"],
            ),
            f"**Data Dragon version:** {version}",
            "\n## Tổng quan",
            _plain(champion.get("lore") or champion.get("blurb", "")),
            f"\n**Vai trò:** {', '.join(champion.get('tags', []))}",
            f"\n**Tài nguyên:** {champion.get('partype', '')}",
            "\n## Chỉ số cơ bản",
            *[f"- {name}: {value}" for name, value in champion.get("stats", {}).items()],
            f"\n## Nội tại — {champion['passive']['name']}",
            _plain(champion["passive"].get("description", "")),
        ]
        for slot, spell in zip("QWER", champion.get("spells", [])):
            lines.extend(
                [
                    f"\n## {slot} — {spell['name']}",
                    _plain(spell.get("description", "")),
                    f"\n**Hồi chiêu:** {spell.get('cooldownBurn', '')}",
                    f"\n**Chi phí:** {spell.get('costBurn', '')} {spell.get('costType', '')}",
                    f"\n**Tầm:** {spell.get('rangeBurn', '')}",
                ]
            )
        _write(output_dir, f"champion__{champion_id.lower()}.md", "\n".join(lines))

    for item_id, item in items.items():
        if (
            not item.get("maps", {}).get("11")
            or not item.get("gold", {}).get("purchasable")
            or item.get("inStore") is False
        ):
            continue
        lines = [
            _header(
                item["name"],
                "Riot Data Dragon",
                f"{base_url}/item.json",
                "item",
                item["name"],
            ),
            f"**Data Dragon version:** {version}",
            "\n## Mô tả",
            _plain(item.get("description") or item.get("plaintext", "")),
            "\n## Giá",
            f"- Mua: {item['gold'].get('total', 0)} vàng",
            f"- Bán: {item['gold'].get('sell', 0)} vàng",
            "\n## Chỉ số",
            *[f"- {name}: {value}" for name, value in item.get("stats", {}).items()],
        ]
        _write(output_dir, f"item__{item_id}.md", "\n".join(lines))

    for spell_id, spell in spells.items():
        if "CLASSIC" not in spell.get("modes", []):
            continue
        content = "\n".join(
            [
                _header(
                    spell["name"],
                    "Riot Data Dragon",
                    f"{base_url}/summoner.json",
                    "summoner_spell",
                    spell["name"],
                ),
                f"**Data Dragon version:** {version}",
                "\n## Mô tả",
                _plain(spell.get("description", "")),
                f"\n**Hồi chiêu:** {spell.get('cooldownBurn', '')}",
                f"\n**Tầm:** {spell.get('rangeBurn', '')}",
            ]
        )
        _write(output_dir, f"spell__{spell_id.lower()}.md", content)


def _top_sections(markdown: str) -> dict[str, str]:
    sections = {}
    title = None
    lines = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if title:
                sections[title] = "\n".join(lines)
            title, lines = line[3:].strip(), []
        elif title:
            lines.append(line)
    if title:
        sections[title] = "\n".join(lines)
    return sections


def _entries(section: str, fallback_title: str) -> list[tuple[str, str]]:
    headings = [
        (len(match.group(1)), match.group(2))
        for line in section.splitlines()
        if (match := re.match(r"^(#{3,4})\s+(.+)", line))
    ]
    if not headings:
        return [(fallback_title, section)]
    level = min(level for level, _ in headings)
    pattern = re.compile(rf"^#{{{level}}}\s+(.+)")
    entries = []
    title = None
    lines = []
    for line in section.splitlines():
        match = pattern.match(line)
        if match:
            if title:
                entries.append((title, "\n".join(lines)))
            title, lines = match.group(1), []
        elif title:
            lines.append(line)
    if title:
        entries.append((title, "\n".join(lines)))
    return entries


def _clean_patch_text(value: str) -> str:
    value = re.sub(r"!\[[^]]*]\([^)]+\)", "", value)
    value = re.sub(r"^\[\]\([^)]+\)\s*$", "", value, flags=re.MULTILINE)
    value = re.sub(r"^\s*\*\s+\*\s+\*\s*$", "", value, flags=re.MULTILINE)
    return "\n".join(line.rstrip() for line in value.splitlines()).strip()


def convert_news_articles() -> None:
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.glob("*.md"):
        path.unlink()

    for path in sorted((LANDING_DIR / "news").glob("patch_*.json")):
        article = _load(path)
        patch = article["patch"]
        for section, entity_type in PATCH_SECTIONS.items():
            text = _top_sections(article["content_markdown"]).get(section)
            if not text:
                continue
            for title, body in _entries(text, section):
                entity = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", title).strip()
                body = _clean_patch_text(body)
                if not body:
                    continue
                content = "\n".join(
                    [
                        _header(
                            f"{entity} — Patch {patch}",
                            "Riot Games Patch Notes",
                            article["url"],
                            entity_type,
                            entity,
                            patch,
                        ),
                        f"**Crawled:** {article['date_crawled']}",
                        "\n## Thay đổi",
                        body,
                    ]
                )
                filename = (
                    f"patch__{patch.replace('.', '_')}__{entity_type}__{_slug(entity)}.md"
                )
                _write(output_dir, filename, content)


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
