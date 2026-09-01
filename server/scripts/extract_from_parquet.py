#!/usr/bin/env python3
"""Direct high-speed Parquet extractor for Community Forensics and SID_Set.
Reads downloaded parquet shards and saves balanced authentic & synthetic image files.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
import pandas as pd
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet", required=True, help="Path to downloaded parquet file")
    p.add_argument("--out", required=True, help="Destination directory")
    p.add_argument("--limit", type=int, default=10000, help="Maximum images to extract")
    args = p.parse_args()

    out = Path(args.out)
    real_dir = out / "real"
    fake_dir = out / "synthetic"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.parquet}...")
    df = pd.read_parquet(args.parquet)
    print(f"Loaded DataFrame with {len(df)} rows. Columns: {list(df.columns)}")

    manifest = []
    n_real = 0
    n_fake = 0
    total = 0
    per_class_limit = args.limit // 2

    for idx, row in df.iterrows():
        if total >= args.limit:
            break

        # Extract label
        label = row.get("label")
        label_text = str(label).lower()
        is_fake = label_text in {"1", "fake", "synthetic", "generated", "1.0", "true"}

        if is_fake and n_fake >= per_class_limit:
            continue
        if not is_fake and n_real >= per_class_limit:
            continue

        raw = row.get("image_data") or row.get("image")
        if raw is None:
            continue
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]

        try:
            if isinstance(raw, (bytes, bytearray)):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            elif hasattr(raw, "convert"):
                img = raw.convert("RGB")
            else:
                continue
        except Exception:
            continue

        cls = "synthetic" if is_fake else "real"
        count = n_fake if is_fake else n_real
        filename = f"cf_{count:07d}.jpg"
        img.save(out / cls / filename, quality=95)

        manifest.append({
            "image_path": f"{cls}/{filename}",
            "label": int(is_fake),
            "source": "OwensLab/CommunityForensics-Small",
            "model_name": row.get("model_name", "unknown"),
        })

        if is_fake:
            n_fake += 1
        else:
            n_real += 1
        total += 1

        if total % 1000 == 0:
            print(f"Extracted {total}/{args.limit} ({n_real} real, {n_fake} synthetic)...")

    manifest_path = out / "manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nCOMPLETED: Extracted {total} images ({n_real} real, {n_fake} synthetic) to {out}")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
