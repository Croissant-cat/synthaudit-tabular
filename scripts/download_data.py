"""Download the public datasets used in the paper (UCI ML Repository).

Usage
-----
python scripts/download_data.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import RAW_DIR, get_logger  # noqa: E402

log = get_logger()

URLS = {
    "adult.zip": "https://archive.ics.uci.edu/static/public/2/adult.zip",
    "bank.zip": "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip",
}

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) synthaudit/0.1"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        log.info("exists: %s", dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading %s", url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f)
    except Exception as exc:  # fall back to curl (useful behind TLS inspection)
        log.warning("urllib failed (%s); trying curl", exc)
        subprocess.run(["curl", "-L", "-k", "--ssl-no-revoke", "-o", str(dest), url], check=True)
    log.info("saved %s (%.1f KB)", dest, dest.stat().st_size / 1024)


def main() -> None:
    for name, url in URLS.items():
        download(url, RAW_DIR / name)
    log.info("raw data ready in %s", RAW_DIR)


if __name__ == "__main__":
    main()
