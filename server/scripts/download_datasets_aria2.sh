#!/usr/bin/env bash
set -euo pipefail

DEST_DIR="/mnt/ai-storage/aigc_data/datasets/parquet"
SID_DIR="/mnt/ai-storage/aigc_data/datasets/sid_parquet"
mkdir -p "$DEST_DIR" "$SID_DIR"

echo "=== Downloading Community Forensics Small shards via aria2c ==="
for i in 1 2 3 4 5; do
  p="HFCF_small_${i}.parquet"
  if [[ ! -f "$DEST_DIR/$p" ]]; then
    echo "Downloading $p..."
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true \
      -d "$DEST_DIR" -o "$p" \
      "https://huggingface.co/datasets/OwensLab/CommunityForensics-Small/resolve/main/data/$p"
  fi
done

echo "=== Downloading SID_Set shards via aria2c ==="
for i in $(seq -w 0 5); do
  p="train-000${i}-of-00249.parquet"
  if [[ ! -f "$SID_DIR/$p" ]]; then
    echo "Downloading $p..."
    aria2c -x 16 -s 16 -k 1M --allow-overwrite=true \
      -d "$SID_DIR" -o "$p" \
      "https://huggingface.co/datasets/saberzl/SID_Set/resolve/main/data/$p"
  fi
done

echo "=== All Parquet Downloads Finished Successfully ==="
