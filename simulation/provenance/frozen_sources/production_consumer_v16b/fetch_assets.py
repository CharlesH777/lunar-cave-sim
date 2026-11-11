#!/usr/bin/env python3
"""Fetch pinned CC0 photographic PBR maps, or verify the bundled local copies.

Examples:
  python fetch_assets.py --verify
  python fetch_assets.py

All download URLs, author credits, license pages, image sizes, and SHA-256
digests are recorded in assets/ASSET_SOURCES.json. No provider API needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def valid(path: Path, entry: dict) -> bool:
    return (path.is_file() and path.stat().st_size == entry["bytes"]
            and digest(path) == entry["sha256"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true",
                        help="Check all local bytes without using the network")
    args = parser.parse_args()
    manifest = json.loads((ASSETS / "ASSET_SOURCES.json").read_text())
    failures = []
    count = 0
    total_bytes = 0
    for asset in manifest["assets"]:
        for entry in asset["files"]:
            target = ASSETS / entry["path"]
            if not valid(target, entry):
                if args.verify:
                    failures.append(entry["path"])
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                partial = target.with_suffix(target.suffix + ".download")
                request = urllib.request.Request(entry["url"], headers={
                    "User-Agent": "IsaacCaveDemo/1.0 (CC0 asset retrieval)"})
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        with partial.open("wb") as output:
                            while block := response.read(1024 * 1024):
                                output.write(block)
                    if not valid(partial, entry):
                        raise RuntimeError(f"Checksum or length mismatch: {entry['url']}")
                    partial.replace(target)
                finally:
                    partial.unlink(missing_ok=True)
            print(f"OK {entry['path']}")
            count += 1
            total_bytes += entry["bytes"]
    if failures:
        raise SystemExit("Missing or modified assets: " + ", ".join(failures))
    print(f"Verified {count} pinned CC0 maps ({total_bytes / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
