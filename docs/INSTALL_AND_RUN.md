# Installation and Runbook

## A. Server prerequisites

Fedora Linux, NVIDIA driver, `nvidia-smi`, Python 3.10+ recommended, Git, Git LFS, compiler toolchain, ~32 GB RAM, and enough free storage for the selected datasets.

Run:

```bash
bash server/scripts/00_hardware_audit.sh
bash server/scripts/13_system_dependencies_fedora.sh
bash server/scripts/01_prepare_server.sh
bash server/scripts/02_install_ml_stack.sh
bash server/scripts/14_preflight_all.sh
bash server/scripts/08_gpu_smoke_test.sh
```

## B. Optional research stack

Only after the core stack works:

```bash
bash server/scripts/12_install_optional_stack.sh
```

This installs Accelerate, Lightning, DeepSpeed, and OverflowML. Unsloth is intentionally not installed by default.

## C. Model download

```bash
bash server/scripts/03_download_model_pool.sh
```

Before downloading all models, inspect storage:

```bash
bash server/scripts/10_storage_guard.sh
```

## D. Dataset download

Download in stages. Large datasets should be streamed/sharded where possible.

```bash
bash server/scripts/04_download_datasets_full.sh --community-small
bash server/scripts/04_download_datasets_full.sh --sid
bash server/scripts/04_download_datasets_full.sh --wildfake
```

CIFAKE requires Kaggle credentials; the script intentionally does not silently fetch credentials.

## E. Validation lock

```bash
bash server/scripts/05_lock_validation.sh
```

Only place the exact challenge-provided validation files in `validation_LOCKED/`.

## F. Memory diagnostics

Before/after large expert jobs:

```bash
bash server/scripts/11_memory_diagnostics.sh
```

For the strongest reset between large models, run each expert in a subprocess and let that process exit after writing its features.

## G. Inference contract

```bash
python scripts/run_inference.py \
  --image_dir ./test_images \
  --checkpoint ./checkpoints/final.pth \
  --output ./results.json \
  --device cuda
```

## H. Robustness evaluation

```bash
python scripts/evaluate_robustness.py \
  --checkpoint ./checkpoints/final.pth \
  --coco_dir /mnt/ai-storage/aigc_data/validation_LOCKED/coco_val2017 \
  --dalle_dir /mnt/ai-storage/aigc_data/validation_LOCKED/dalle_advanced \
  --output_csv ./robustness_results.csv \
  --device cuda
```

Never use the validation paths for training.
