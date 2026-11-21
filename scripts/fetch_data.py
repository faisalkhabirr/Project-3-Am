#!/usr/bin/env python3
"""Download the official hackathon dataset into data/raw/.

The Google Sheets "export" endpoint requires authentication, but the public
``gviz`` CSV endpoint serves the same sheet. The resulting CSV is bundled in
the repository so the dashboard runs fully offline. Run this script only when
you want to refresh the data.

Usage:
    python scripts/fetch_data.py [--force]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "dengue_weather.csv"
URL = (
    "https://docs.google.com/spreadsheets/d/"
    "1QFI0ACJRvJQkiQT3LfGHklM3g4PgaJWUt8xSVU07NIE/gviz/tq?tqx=out:csv"
)


def download(url: str = URL, out: Path = OUT, force: bool = False) -> Path:
    """Download the sheet CSV to ``out`` unless it already exists."""
    if out.exists() and not force:
        size = out.stat().st_size / 1_000_000
        print(f"Dataset already present: {out} ({size:.2f} MB)")
        return out

    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    out.write_bytes(payload)
    size = out.stat().st_size / 1_000_000
    print(f"Saved {out} ({size:.2f} MB, {len(payload.splitlines()) - 1:,} rows)")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite the cached CSV")
    args = parser.parse_args()
    try:
        download(force=args.force)
    except Exception as exc:  # noqa: BLE001 - CLI should print a clean message
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
