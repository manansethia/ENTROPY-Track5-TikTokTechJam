# Fedora GPU Server Setup

## Hardware assumed

- Fedora Linux
- NVIDIA GPU with `nvidia-smi`
- 32 GB RAM
- RTX 3050 6 GB VRAM
- `/mnt/ai-storage` mounted on the large HDD
- ~475 GB NVMe root filesystem

## Storage policy

Datasets and Hugging Face/ModelScope caches live under:

`/mnt/ai-storage/aigc_data`

Do not download the large datasets to the Mac first.

## First run

```bash
cd /path/to/aigc_robust_detection
bash server/scripts/00_hardware_audit.sh
bash server/scripts/01_prepare_server.sh
source ~/.venvs/aigc-detector/bin/activate
bash server/scripts/02_install_ml_stack.sh
bash server/scripts/08_gpu_smoke_test.sh
```

Optional Fedora packages:

```bash
bash server/scripts/09_install_fedora_optional.sh
```

## Download the model pool

```bash
bash server/scripts/03_download_model_pool.sh
```

This intentionally downloads multiple candidate models before selection. We will benchmark them rather than assuming the biggest encoder is best.

## Download training data

```bash
bash server/scripts/04_download_datasets_full.sh --community-small
bash server/scripts/04_download_datasets_full.sh --sid
bash server/scripts/04_download_datasets_full.sh --wildfake
```

CIFAKE is optional and can be obtained through Kaggle credentials.

## Lock validation

```bash
bash server/scripts/05_lock_validation.sh
```

The challenge validation data must be kept outside the training tree. The supplied challenge specification requires COCO val2017 (4,998) and WildFake DALL-E Advanced (8,843) to remain unused during training.

## Verify

```bash
bash server/scripts/06_verify_assets.sh
bash server/scripts/10_storage_guard.sh
```

## Important

The RTX 3050 has 6 GB VRAM. Large encoders should be loaded sequentially for feature extraction. Use cached embeddings, mixed precision, gradient accumulation, and parameter-efficient tuning rather than naïve multi-model end-to-end training.
