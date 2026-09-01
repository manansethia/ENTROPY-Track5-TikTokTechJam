#!/usr/bin/env python3
"""Memory-efficient streaming Parquet extractor for massive multi-generator datasets.
Uses PyArrow batch streaming (near-zero RAM footprint) and multi-core process pool.
"""

import argparse
import gc
import glob
import hashlib
import io
import os

import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm


def process_single_cf_shard(pfile, output_fake_dir, max_per_shard=600):
    shard_name = Path(pfile).stem
    extracted = 0
    try:
        parquet_file = pq.ParquetFile(pfile)
        # Stream in 128-row record batches to avoid memory spikes
        for batch in parquet_file.iter_batches(batch_size=128):
            if extracted >= max_per_shard:
                break
            batch_pydict = batch.to_pydict()
            images_list = batch_pydict.get("image_data", [])

            for row_img in images_list:
                if extracted >= max_per_shard:
                    break
                if row_img is None:
                    continue

                raw_data = row_img.get("bytes", None) if isinstance(row_img, dict) else row_img
                if not isinstance(raw_data, bytes):
                    continue

                try:
                    img = Image.open(io.BytesIO(raw_data)).convert("RGB")
                    img_hash = hashlib.sha256(raw_data).hexdigest()[:16]
                    out_path = os.path.join(output_fake_dir, f"img_{img_hash}.jpg")
                    img.save(out_path, format="JPEG", quality=95)
                    extracted += 1
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {shard_name}: {e}")
    finally:
        gc.collect()
    return shard_name, extracted


def process_single_sid_shard(pfile, output_real_dir, output_fake_dir, max_per_shard=1000):
    shard_name = Path(pfile).stem
    real_cnt = 0
    fake_cnt = 0
    try:
        parquet_file = pq.ParquetFile(pfile)
        for batch in parquet_file.iter_batches(batch_size=128):
            if (real_cnt + fake_cnt) >= max_per_shard:
                break
            batch_pydict = batch.to_pydict()
            labels_list = batch_pydict.get("label", [])
            images_list = batch_pydict.get("image", [])

            for lbl, row_img in zip(labels_list, images_list):
                if (real_cnt + fake_cnt) >= max_per_shard:
                    break
                if row_img is None or lbl is None:
                    continue

                raw_data = row_img.get("bytes", None) if isinstance(row_img, dict) else row_img
                if not isinstance(raw_data, bytes):
                    continue

                try:
                    img = Image.open(io.BytesIO(raw_data)).convert("RGB")
                    img_hash = hashlib.sha256(raw_data).hexdigest()[:16]
                    if int(lbl) == 0:
                        out_path = os.path.join(output_real_dir, f"img_{img_hash}.jpg")
                        img.save(out_path, format="JPEG", quality=95)
                        real_cnt += 1
                    elif int(lbl) in (1, 2):
                        out_path = os.path.join(output_fake_dir, f"img_{img_hash}.jpg")
                        img.save(out_path, format="JPEG", quality=95)
                        fake_cnt += 1
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading {shard_name}: {e}")
    finally:
        gc.collect()
    return shard_name, real_cnt, fake_cnt


def main():
    p = argparse.ArgumentParser(description="Streaming Multi-Shard Parquet Extractor")
    p.add_argument("--cf_parquet_dir", default="/mnt/ai-storage/aigc_data/datasets/parquet")
    p.add_argument("--sid_parquet_dir", default="/mnt/ai-storage/aigc_data/datasets/sid_parquet")
    p.add_argument("--cf_real_dir", default="/mnt/ai-storage/aigc_data/datasets/cf_slice/real")
    p.add_argument("--output_dir", default="/mnt/ai-storage/aigc_data/datasets/scaled_massive")
    p.add_argument("--max_per_cf_shard", type=int, default=300)
    p.add_argument("--max_workers", type=int, default=8)
    p.add_argument("--balance_quota", type=int, default=20000, help="Equal quota for real and synthetic images")
    args = p.parse_args()

    real_out = os.path.join(args.output_dir, "real")
    fake_out = os.path.join(args.output_dir, "synthetic")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)

    # 1. Copy base real photos
    if os.path.isdir(args.cf_real_dir):
        print(f"Copying authentic photos from {args.cf_real_dir}...")
        for img_p in glob.glob(os.path.join(args.cf_real_dir, "*.*")):
            shutil.copy(img_p, os.path.join(real_out, f"coco_{os.path.basename(img_p)}"))
        print(f"Current base real count: {len(os.listdir(real_out))}")

    # 2. Parallel Extraction for Community Forensics
    cf_files = sorted(glob.glob(os.path.join(args.cf_parquet_dir, "HFCF_small_*.parquet")))
    print(f"\n[Parallel Stream] Extracting {len(cf_files)} Community Forensics shards (Workers: {args.max_workers})...")
    total_cf = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_cf_shard, pf, fake_out, args.max_per_cf_shard) for pf in cf_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="CF Shards Streamed"):
            name, cnt = f.result()
            total_cf += cnt

    # 3. Parallel Extraction for SID_Set (Real + T2I + Deepfakes/Edits)
    sid_files = sorted(glob.glob(os.path.join(args.sid_parquet_dir, "train-*-of-00249.parquet")))
    print(f"\n[Parallel Stream] Extracting {len(sid_files)} SID_Set shards (Workers: {args.max_workers})...")
    total_sid_real = 0
    total_sid_fake = 0
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_single_sid_shard, pf, real_out, fake_out) for pf in sid_files]
        for f in tqdm(as_completed(futures), total=len(futures), desc="SID Shards Streamed"):
            name, r_cnt, f_cnt = f.result()
            total_sid_real += r_cnt
            total_sid_fake += f_cnt

    # 4. Strict 1:1 Class Balancing Enforcement
    all_reals = sorted(glob.glob(os.path.join(real_out, "*.*")))
    all_fakes = sorted(glob.glob(os.path.join(fake_out, "*.*")))
    quota = min(len(all_reals), len(all_fakes), args.balance_quota)

    # Prune surplus to ensure exact 1:1 ratio
    if len(all_reals) > quota:
        for p in all_reals[quota:]:
            os.remove(p)
    if len(all_fakes) > quota:
        for p in all_fakes[quota:]:
            os.remove(p)

    final_real = len(os.listdir(real_out))
    final_fake = len(os.listdir(fake_out))
    print(f"\n========================================================")
    print(f"STRICTLY BALANCED 1:1 DATASET READY in {args.output_dir}:")
    print(f"  Authentic Real Photos:    {final_real:,}")
    print(f"  Synthetic / AIGC / Edits: {final_fake:,}")
    print(f"  Total Training Samples:   {final_real + final_fake:,} (1:1 Ratio)")
    print(f"========================================================")


if __name__ == "__main__":
    main()

