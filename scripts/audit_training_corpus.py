#!/usr/bin/env python3
"""Inventory candidate training sources without training or copying any image.

This is intentionally provenance-first: a source whose path or supplied name
resembles an organizer benchmark is rejected.  The final training launcher must
still run the content-hash isolation gate before it reads a manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
FORBIDDEN = ("coco", "wildfake", "val2017", "held_out", "benchmark")


def bucket(width: int, height: int) -> str:
    edge = min(width, height)
    if edge <= 256: return "<=256"
    if edge <= 512: return "257-512"
    if edge <= 1024: return "513-1024"
    if edge <= 2048: return "1025-2048"
    return ">2048"


def source(value: str) -> tuple[str, Path]:
    name, sep, raw_path = value.partition("=")
    if not sep or not name or not raw_path:
        raise argparse.ArgumentTypeError("sources must be written NAME=/absolute/path")
    if any(token in f"{name} {raw_path}".lower() for token in FORBIDDEN):
        raise argparse.ArgumentTypeError("organizer/benchmark-like sources are forbidden from candidate training inventory")
    path = Path(raw_path)
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"source directory does not exist: {path}")
    return name, path


def inspect(name: str, root: Path) -> dict:
    counts = {key: 0 for key in ("<=256", "257-512", "513-1024", "1025-2048", ">2048")}
    unreadable = 0
    total = 0
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                counts[bucket(*image.size)] += 1
                total += 1
        except (OSError, ValueError):
            unreadable += 1
    return {"name": name, "provenance_root": str(root), "image_count": total, "resolution_policy": "shorter image edge", "resolution_buckets": counts, "unreadable_files": unreadable, "training_approved": False, "approval_reason": "Inventory only. Content-hash isolation and source-license review are still required."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=source, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    records = [inspect(name, root) for name, root in args.source]
    payload = {"schema_version": 1, "sources": records, "total_images": sum(item["image_count"] for item in records), "status": "INVENTORIED_NOT_APPROVED"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"sources": len(records), "total_images": payload["total_images"], "status": payload["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
