#!/usr/bin/env bash
set -e

echo "================================================================="
echo "   AETHERFORENSICS 50K MASSIVE QUAD-HYBRID TRAINING PIPELINE     "
echo "================================================================="

source $HOME/.venvs/aigc-detector/bin/activate
cd $HOME/aigc_robust_detection

# 1. Feature Extraction across all 4 Foundation Backbones (12 CPU Workers, FP16 GPU)
echo ""
echo "[Step 1/3] Caching 4-Stream Foundation Features into HDF5..."
python scripts/cache_backbone_features.py \
  --data_dir /mnt/ai-storage/aigc_data/datasets/massive_balanced_50k \
  --output_h5 /mnt/ai-storage/aigc_data/cache/quad_features_50k.h5 \
  --siglip_dir /mnt/ai-storage/aigc_data/models/siglip_base_224 \
  --clip_dir /mnt/ai-storage/aigc_data/models/clip_vitl14 \
  --dinov2_dir /mnt/ai-storage/aigc_data/models/dinov2_large \
  --convnext_dir /mnt/ai-storage/aigc_data/models/convnextv2_tiny \
  --batch_size 128 \
  --num_workers 12 \
  --device cuda

# 2. Train Quad-Hybrid Dynamic Softmax Fusion Head
echo ""
echo "[Step 2/3] Training 4-Stream Dynamic Softmax Router with Hard-Negative Mining..."
python scripts/train_quad_hybrid_gating.py \
  --cache_h5 /mnt/ai-storage/aigc_data/cache/quad_features_50k.h5 \
  --output_dir checkpoints/quad_hybrid_50k_v1 \
  --epochs 25 \
  --batch_size 128 \
  --lr 3e-4 \
  --device cuda

# 3. Comprehensive 15-Condition Robustness Benchmark Matrix
echo ""
echo "[Step 3/3] Evaluating 15-Condition Perturbation Matrix & Hard-Negative Audit..."
python scripts/evaluate_quad_hybrid_matrix.py \
  --checkpoint checkpoints/quad_hybrid_50k_v1/best_model.pt \
  --coco_dir /mnt/ai-storage/aigc_data/validation_LOCKED/val2017 \
  --fake_dir /mnt/ai-storage/aigc_data/datasets/cf_slice/synthetic \
  --siglip_dir /mnt/ai-storage/aigc_data/models/siglip_base_224 \
  --clip_dir /mnt/ai-storage/aigc_data/models/clip_vitl14 \
  --dinov2_dir /mnt/ai-storage/aigc_data/models/dinov2_large \
  --convnext_dir /mnt/ai-storage/aigc_data/models/convnextv2_tiny \
  --output_csv reports/quad_hybrid_50k_robustness_results.csv \
  --device cuda

echo "================================================================="
echo "   50K MASSIVE PIPELINE COMPLETED SUCCESSFULLY!                  "
echo "================================================================="
