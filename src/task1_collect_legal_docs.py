"""Download version-pinned Vietnamese League of Legends reference data."""

import json
from pathlib import Path
from urllib.request import urlopen


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CATALOGS = {
    "champions_vi_VN.json": "champion.json",
    "champion_details_vi_VN.json": "championFull.json",
    "items_vi_VN.json": "item.json",
    "summoner_spells_vi_VN.json": "summoner.json",
}


def _fetch_json(url: str) -> object:
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def download_documents() -> None:
    """Download Data Dragon catalogs, reusing the first downloaded version."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    version_path = DATA_DIR / "ddragon_version.json"

    if version_path.exists():
        version = json.loads(version_path.read_text(encoding="utf-8")).get("version")
    else:
        versions = _fetch_json(VERSIONS_URL)
        version = versions[0] if isinstance(versions, list) and versions else None
        if not isinstance(version, str) or not version:
            raise ValueError("Data Dragon returned no usable version")
        version_path.write_text(
            json.dumps({"version": version, "source": VERSIONS_URL}, indent=2),
            encoding="utf-8",
        )

    if not isinstance(version, str) or not version:
        raise ValueError(f"Invalid pinned version in {version_path}")

    base_url = f"https://ddragon.leagueoflegends.com/cdn/{version}/data/vi_VN"
    for filename, endpoint in CATALOGS.items():
        payload = _fetch_json(f"{base_url}/{endpoint}")
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise ValueError(f"Invalid Data Dragon catalog: {endpoint}")
        output = DATA_DIR / filename
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Saved: {output}")


if __name__ == "__main__":
    download_documents()
