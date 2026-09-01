# Dataset Guide

## Training

The project supports a directory-based binary classification layout:

```text
data/train/
├── real/
└── synthetic/
```

Images can be nested under either directory.

Candidate sources named in the supplied materials include:

- WildFake training partitions
- SID_Set
- balanced GenImage subsets
- CIFAKE subsets

Use only data that is public or properly licensed for the intended use.

## Demonstration benchmark isolation

The hackathon brief reserves:

- **COCO val2017:** 4,998 authentic images
- **WildFake DALL-E Advanced:** 8,843 synthetic images

These are demonstration/validation data and must not be used in training, model selection, hyperparameter tuning, or any other optimization step.

Recommended physical separation:

```text
data/
├── train/
│   ├── real/
│   └── synthetic/
└── val_demo/
    ├── coco_val2017/
    └── dalle_advanced/
```

Do not create symlinks from `data/train` to the validation benchmark.

## Dataset leakage checklist

Before final training:

1. Search filenames and metadata for overlap between train and benchmark.
2. Remove exact duplicates and obvious near-duplicates.
3. Keep generator-specific folders from the benchmark out of training.
4. Record dataset versions and download dates.
5. Freeze the benchmark before reporting results.

## Labels

The repository uses:

```text
0 = authentic / real
1 = AIGC / synthetic
```

The inference output `pred` is therefore the estimated probability of class 1.
