#!/bin/bash
set -e

source $HOME/.venvs/aigc-detector/bin/activate
cd $HOME/aigc_robust_detection

echo "=========================================================="
echo "[Scale Pipeline] Step 1: Ingesting All Available Shards..."
echo "=========================================================="
python scripts/extract_multishard_dataset.py \
  --cf_parquet_dir /mnt/ai-storage/aigc_data/datasets/parquet \
  --sid_parquet_dir /mnt/ai-storage/aigc_data/datasets/sid_parquet \
  --cf_real_dir /mnt/ai-storage/aigc_data/datasets/cf_slice/real \
  --output_dir /mnt/ai-storage/aigc_data/datasets/scaled_massive \
  --max_per_cf_shard 600 \
  --max_workers 12

echo "=========================================================="
echo "[Scale Pipeline] Step 2: High-Throughput 3-Stream GPU Feature Caching..."
echo "=========================================================="
python scripts/cache_backbone_features.py \
  --data_dir /mnt/ai-storage/aigc_data/datasets/scaled_massive \
  --output_h5 /mnt/ai-storage/aigc_data/cache/scaled_massive_3stream.h5 \
  --siglip_dir /mnt/ai-storage/aigc_data/models/siglip_base_224 \
  --clip_dir /mnt/ai-storage/aigc_data/models/clip_vitl14 \
  --dinov2_dir /mnt/ai-storage/aigc_data/models/dinov2_large \
  --include_dinov2 \
  --batch_size 128 \
  --num_workers 8 \
  --device cuda

echo "=========================================================="
echo "[Scale Pipeline] Step 3: Training 3-Stream Dynamic Gating Fusion Network..."
echo "=========================================================="
mkdir -p checkpoints/tri_hybrid_massive_v4
python scripts/train_tri_hybrid_gating.py \
  --cache_h5 /mnt/ai-storage/aigc_data/cache/scaled_massive_3stream.h5 \
  --output_dir checkpoints/tri_hybrid_massive_v4 \
  --epochs 20 \
  --batch_size 128 \
  --lr 3e-4 \
  --device cuda

echo "=========================================================="
echo "[Scale Pipeline] Step 4: 15-Condition Matrix Evaluation..."
echo "=========================================================="
python scripts/evaluate_tri_hybrid_matrix.py \
  --checkpoint checkpoints/tri_hybrid_massive_v4/best_model.pt \
  --coco_dir /mnt/ai-storage/aigc_data/validation_LOCKED/val2017 \
  --fake_dir /mnt/ai-storage/aigc_data/datasets/cf_slice/synthetic \
  --max_images 100 \
  --output_csv reports/tri_hybrid_massive_robustness_results.csv \
  --device cuda

echo "=========================================================="
echo "[Scale Pipeline] Step 5: Hard-Negative / CGI Guardrail Audit..."
echo "=========================================================="
python scripts/benchmark_hard_negatives.py \
  --checkpoint checkpoints/tri_hybrid_massive_v4/best_model.pt \
  --hard_neg_dir /mnt/ai-storage/aigc_data/validation_LOCKED/val2017 \
  --max_images 250 \
  --output_json reports/hard_negative_massive_audit.json \
  --device cuda

echo "=========================================================="
echo "[Scale Pipeline] COMPLETED SUCCESSFULLY!"
echo "=========================================================="
