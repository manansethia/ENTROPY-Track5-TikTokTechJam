#!/usr/bin/env python3
"""Materialize a bounded Community Forensics-Small slice without filling the disk.

Example:
  python 05_stream_community_forensics.py --limit 10000 --out /mnt/ai-storage/aigc_data/datasets/cf_slice

Streams directly from Hugging Face with authenticated token.
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from PIL import Image
from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = Path(args.out)
    real = out / "real"
    fake = out / "synthetic"
    real.mkdir(parents=True, exist_ok=True)
    fake.mkdir(parents=True, exist_ok=True)

    # Load token
    token = os.environ.get("HF_TOKEN")
    if not token:
        env_file = Path(__file__).resolve().parents[2] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break

    print(f"Streaming OwensLab/CommunityForensics-Small (Token: {'YES' if token else 'NO'}, Limit: {args.limit})...")

    ds = load_dataset(
        "OwensLab/CommunityForensics-Small",
        split="train",
        streaming=True,
        token=token,
    ).shuffle(seed=args.seed, buffer_size=1000)

    manifest = []
    n_real = 0
    n_fake = 0
    total = 0
    per_class_limit = args.limit // 2

    for row in ds:
        if total >= args.limit:
            break

        label = row.get("label")
        label_text = str(label).lower()
        is_fake = label_text in {"1", "fake", "synthetic", "generated", "1.0"}

        if is_fake and n_fake >= per_class_limit:
            continue
        if not is_fake and n_real >= per_class_limit:
            continue

        raw = row.get("image_data") or row.get("image")
        if raw is None:
            continue
        if isinstance(raw, dict) and "bytes" in raw:
            raw = raw["bytes"]
        elif isinstance(raw, str):
            raw = raw.encode()

        try:
            if isinstance(raw, bytes):
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            elif hasattr(raw, "convert"):
                img = raw.convert("RGB")
            else:
                continue
        except Exception:
            continue

        cls = "synthetic" if is_fake else "real"
        idx = n_fake if is_fake else n_real
        path = Path(cls) / f"cf_{idx:07d}.jpg"
        img.save(out / path, quality=95)

        manifest.append({
            "image_path": str(path),
            "label": int(is_fake),
            "source": "OwensLab/CommunityForensics-Small",
            "model_name": row.get("model_name"),
        })

        if is_fake:
            n_fake += 1
        else:
            n_real += 1
        total += 1

        if total % 200 == 0:
            print(f"Streamed {total}/{args.limit} images ({n_real} real, {n_fake} synthetic)...")

    with open(out / "manifest.jsonl", "w", encoding="utf-8") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"FINISHED: Wrote {total} examples ({n_real} real, {n_fake} synthetic) to {out}")


if __name__ == "__main__":
    main()
