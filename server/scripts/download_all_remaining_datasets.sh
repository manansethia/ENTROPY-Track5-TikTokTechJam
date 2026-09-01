#!/usr/bin/env bash
set -euo pipefail

CF_DIR="/mnt/ai-storage/aigc_data/datasets/parquet"
SID_DIR="/mnt/ai-storage/aigc_data/datasets/sid_parquet"
GENIMAGE_DIR="/mnt/ai-storage/aigc_data/datasets/genimage_parquet"
mkdir -p "$CF_DIR" "$SID_DIR" "$GENIMAGE_DIR"

echo "=== Starting Master Continuous Dataset Archive Ingestion ==="

# 1. Download Community Forensics Small shards (6 to 50)
for i in $(seq 6 50); do
  p="HFCF_small_${i}.parquet"
  if [[ ! -f "$CF_DIR/$p" ]]; then
    echo "Downloading CF Shard $p..."
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true \
      -d "$CF_DIR" -o "$p" \
      "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data/$p" || true
  fi
done

# 2. Download SID_Set shards (5 to 50)
for i in $(seq -w 5 50); do
  p="train-000${i}-of-00249.parquet"
  if [[ ! -f "$SID_DIR/$p" ]]; then
    echo "Downloading SID_Set Shard $p..."
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true \
      -d "$SID_DIR" -o "$p" \
      "https://huggingface.co/datasets/saberzl/SID_Set/resolve/main/data/$p" || true
  fi
done

# 3. Download GenImage & Multi-Generator Benchmark splits
for split in biggan vqdm sdv5 wukong adm gligen midjourney; do
  p="${split}.parquet"
  if [[ ! -f "$GENIMAGE_DIR/$p" ]]; then
    echo "Downloading GenImage Split $p..."
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true \
      -d "$GENIMAGE_DIR" -o "$p" \
      "https://huggingface.co/datasets/GenImage/GenImage_subset/resolve/main/data/$p" || true
  fi
done

echo "=== Master Continuous Dataset Archive Ingestion Complete! ==="
