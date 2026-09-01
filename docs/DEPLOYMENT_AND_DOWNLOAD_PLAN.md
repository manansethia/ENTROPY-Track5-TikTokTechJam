# AIGC Robust Detection — Server-First Download, Model and Training Plan

**Target machine:** `buildabot`  
**GPU:** NVIDIA RTX 3050 6 GB  
**RAM:** 32 GB  
**CPU:** Intel i5 12th gen  
**NVMe:** ~475.4 GB mounted at `/`  
**HDD:** ~931.5 GB mounted at `/mnt/ai-storage`  
**Network:** Tailscale

---

## 1. The important change: download directly on the GPU server

You do **not** need to download the 100–500+ GB datasets onto your Mac.

The correct workflow is:

```text
Mac
  │
  │ Tailscale / SSH
  ▼
buildabot
  │
  ├── /                     NVMe
  │     ├── OS
  │     ├── Python envs
  │     └── temporary files
  │
  └── /mnt/ai-storage       HDD
        └── aigc_data/
              ├── models/
              ├── datasets/
              ├── hf_cache/
              ├── features/
              └── checkpoints/
```

This is preferable because your HDD has ~931.5 GB while your Mac does not need to hold the datasets at all.

The supplied hackathon specification explicitly allows public/properly licensed datasets and self-created transformed samples, and requires models below 2B parameters. fileciteturn1file0L36-L39

---

# 2. Your hardware changes our design

Your **RTX 3050 6 GB** is enough for a strong hackathon prototype, but it changes how we should train.

It is **not** a good idea to fine-tune CLIP-L + SigLIP2 + ConvNeXt simultaneously.

Instead:

### Large teacher

Use:

- CLIP ViT-L/14
- SigLIP2 Base
- SRM + Haar DWT
- ConvNeXt-Tiny
- dynamic fusion/gating

Keep the large semantic backbones frozen and run feature extraction sequentially.

### Small final student

Use:

- SigLIP2 Base, frozen or lightly adapted
- SRM + DWT
- ConvNeXt-Tiny
- compact fusion head
- temperature calibration

Then distill the teacher into the student.

The final detector is therefore much smaller and cheaper to run than the development teacher ensemble.

---

# 3. Proposed model architecture

```text
                         INPUT IMAGE
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
        CLIP ViT-L/14   SigLIP2 Base     RGB → SRM
             │                │                │
             │                │                ▼
             │                │             Haar DWT
             │                │                │
             │                │                ▼
             │                │          ConvNeXt-Tiny
             │                │                │
             └────────────────┼────────────────┘
                              │
                     Feature projection
                              │
                     Reliability gate
                              │
                              ▼
                     Teacher prediction
                              │
                              ▼
                     KNOWLEDGE DISTILLATION
                              │
                              ▼
                ┌─────────────────────────┐
                │     FINAL STUDENT       │
                │                         │
                │ SigLIP2 + forensic CNN  │
                │ + compact fusion head   │
                └────────────┬────────────┘
                             │
                             ▼
                     P(AIGC | image)
```

The research document supplied with the project describes the same central principle: semantic features survive many degradations better than high-frequency forensic traces, so the detector should dynamically balance semantic and frequency evidence. fileciteturn1file1L28-L46

It also specifically describes the hybrid architecture as a frozen foundation encoder plus an SRM/DWT frequency-residual network with dynamic gating. fileciteturn2file3L196-L212

---

# 4. Why not just use the biggest model?

Because the hackathon rewards more than raw parameter count.

The constraints explicitly say **<2B parameters**, and the project is supposed to be a hackathon-scale proof of concept. fileciteturn1file0L36-L39

A good final system should be:

- accurate
- robust
- reasonably fast
- memory-efficient
- reproducible
- explainable enough to defend in judging

A teacher/student design gives us a strong story:

> **A multi-domain forensic teacher transfers semantic and frequency-domain knowledge into a compact robustness-oriented detector.**

---

# 5. Models to download

## A. CLIP ViT-L/14 — teacher semantic stream

Repository:

`openai/clip-vit-large-patch14`

Purpose:

- global scene semantics
- composition
- object relationships
- semantic consistency
- robustness when pixel-level forensic evidence is degraded

Download directly on the server:

```bash
hf download openai/clip-vit-large-patch14 \
  --local-dir /mnt/ai-storage/aigc_data/models/clip-vit-large-patch14
```

---

## B. SigLIP2 Base 224 — primary semantic/student stream

Repository:

`google/siglip2-base-patch16-224`

Purpose:

- modern semantic visual representation
- smaller than CLIP-L
- suitable for the final student

Download:

```bash
hf download google/siglip2-base-patch16-224 \
  --local-dir /mnt/ai-storage/aigc_data/models/siglip2-base-patch16-224
```

---

## C. ConvNeXt-Tiny — forensic stream

Repository:

`timm/convnext_tiny.fb_in1k`

Purpose:

- process SRM high-pass residuals
- learn frequency-domain generation artifacts
- provide complementary evidence to semantic encoders

Download:

```bash
hf download timm/convnext_tiny.fb_in1k \
  --local-dir /mnt/ai-storage/aigc_data/models/convnext_tiny
```

---

# 6. Datasets — what we should actually download

Your HDD is ~931.5 GB, so **do not download everything in full**.

The most useful datasets are:

| Dataset | Role | Recommendation |
|---|---|---|
| Community Forensics-Small | broad generator diversity | **high priority** |
| SID_Set | social-media + synthetic/tampered data | **high priority** |
| WildFake | hackathon-specific distribution | **training partitions only** |
| GenImage | additional generator diversity | **selected subset** |
| CIFAKE | baseline/sanity check | optional |
| COCO val2017 | hackathon validation | **locked** |
| WildFake DALL-E Advanced | hackathon validation | **locked** |

---

# 7. Community Forensics-Small

Repository:

`OwensLab/CommunityForensics-Small`

This is particularly valuable because it contains about **278K generated images from 4,803 generator models**, paired with about 278K real images. citeturn1search1

That diversity is more important to us than simply having millions of images from a handful of generators.

However, the dataset card warns that downloading/indexing can consume up to about **600 GB** because of re-indexed Arrow data. citeturn1search1

Therefore:

### First experiment

Do **not** materialize all 556K images.

Stream a bounded slice:

```bash
source ~/.venvs/aigc_detect/bin/activate
python server/scripts/05_stream_community_forensics.py \
  --limit 50000 \
  --out /mnt/ai-storage/aigc_data/datasets/cf_slice_50k
```

Then increase to 100K or 200K if storage and training time permit.

The Hugging Face dataset explicitly supports streaming. citeturn1search1

---

# 8. SID_Set

Repository:

`saberzl/SID_Set`

The public dataset has 210K training examples and 30K validation examples, with real, full-synthetic and tampered categories. The public release is approximately 140 GB. citeturn1search0turn1search2

This is valuable because it is closer to our real-world problem than a simple clean synthetic-vs-real dataset.

Download:

```bash
hf download saberzl/SID_Set \
  --repo-type dataset \
  --local-dir /mnt/ai-storage/aigc_data/datasets/sid_set
```

---

# 9. GenImage

GenImage is useful as an additional generator-diversity source, but it should **not** be downloaded blindly in full on this 931 GB machine.

Use a selected subset after the core model is working.

Recommended order:

```text
Community Forensics slice
        ↓
SID_Set
        ↓
train/evaluate
        ↓
GenImage selected subset
        ↓
train/evaluate again
```

The purpose of adding GenImage is to test whether the model improves cross-generator generalisation rather than simply increasing the number of near-duplicate examples.

---

# 10. WildFake

WildFake is directly relevant to the hackathon.

Use the ModelScope WildFake repository for the **allowed training partitions**.

The hackathon benchmark is special:

- COCO val2017: 4,998 authentic images
- WildFake DALL-E Advanced: 8,843 AIGC images

These are **demonstration/validation only**. fileciteturn1file0L56-L63

Therefore create:

```text
/mnt/ai-storage/aigc_data/datasets/validation_LOCKED/
├── coco_val2017/
└── wildfake_dalle_advanced/
```

and never expose those directories to the training loader.

ModelScope provides a CLI download mechanism with `modelscope download --dataset ... --local_dir ...`. citeturn0search7

Example:

```bash
pip install modelscope
modelscope download \
  --dataset hy2628982280/WildFake \
  --local_dir /mnt/ai-storage/aigc_data/datasets/wildfake_train
```

**Before executing this for the entire repository, inspect the file listing and isolate the permitted training partitions.** Do not blindly place the complete repository into the training directory because the challenge's DALL-E Advanced validation subset must remain isolated.

---

# 11. CIFAKE

CIFAKE is useful for:

- debugging
- baseline comparison
- verifying that the inference pipeline works

It should **not** dominate training.

The detector should not learn that “CIFAR-like low-resolution artifacts = AI”.

Use it as a secondary sanity check after the main training pipeline works.

---

# 12. COCO validation

The challenge explicitly reserves COCO val2017 for validation.

The server script:

```bash
bash server/scripts/07_download_validation_coco.sh
```

stores it only under:

```text
/mnt/ai-storage/aigc_data/datasets/validation_LOCKED/coco_val2017
```

Do not symlink or copy it into `data/train`.

---

# 13. Storage plan

A reasonable target is:

```text
931 GB HDD
│
├── ~4–5 GB      pretrained model files
├── ~140 GB      SID_Set
├── ~50–200 GB   selected/streamed Community Forensics
├── variable     selected GenImage
├── variable     allowed WildFake training data
├── ~20–50 GB    checkpoints/features/logs
└── reserve       200+ GB if possible
```

The reserve is important because datasets can require temporary space for indexing, extraction and preprocessing.

**Do not run the machine at 99% disk usage.**

---

# 14. NVMe vs HDD

Use the **NVMe** for:

- OS
- Python virtual environment
- compiled libraries
- temporary extraction
- small caches
- active experiment files

Use `/mnt/ai-storage` for:

- datasets
- pretrained weights
- cached features
- checkpoints
- large experiment outputs

Example:

```bash
export AI_ROOT=/mnt/ai-storage/aigc_data
export HF_HOME=$AI_ROOT/hf_cache
```

---

# 15. Installation

From the project root on `buildabot`:

```bash
bash server/scripts/01_prepare_server.sh
```

Then:

```bash
source ~/.venvs/aigc_detect/bin/activate
```

Check:

```bash
bash server/scripts/08_server_status.sh
```

Then:

```bash
bash server/scripts/02_check_storage.sh
```

---

# 16. Download models

```bash
bash server/scripts/03_download_models.sh
```

The Hugging Face CLI supports direct local-directory downloads and a dry-run mode, and it maintains metadata in `.cache/huggingface` to avoid unnecessary re-downloads. citeturn0search0turn0search2

For a large download, leave the `.cache/huggingface` directory in place until everything is verified.

---

# 17. Dry-run before downloading large datasets

Always run:

```bash
bash server/scripts/04_download_datasets.sh --dry-run
```

This lets you inspect what Hugging Face intends to fetch before committing hundreds of GB.

---

# 18. Start with the core profile

```bash
bash server/scripts/04_download_datasets.sh --profile core
```

This downloads SID_Set and prepares a streaming workflow for Community Forensics rather than consuming the entire HDD.

Then:

```bash
python server/scripts/05_stream_community_forensics.py \
  --limit 50000 \
  --out /mnt/ai-storage/aigc_data/datasets/cf_slice_50k
```

---

# 19. When to materialize Community Forensics-Small

Only after the first experiment succeeds.

If you have enough free space:

```bash
bash server/scripts/04_download_datasets.sh --profile balanced
```

The dataset documentation warns that the small dataset can require around 600 GB after indexing. citeturn1search1

With your 931 GB HDD, I would **not** combine the full indexed Community Forensics-Small with large full copies of GenImage and WildFake.

---

# 20. Feature caching — essential for the RTX 3050

The teacher stage should cache features.

Instead of repeatedly doing:

```text
image → CLIP → SigLIP → frequency CNN → fusion
```

we do:

```text
image
 ├── CLIP → cached feature
 ├── SigLIP → cached feature
 └── frequency CNN → cached feature
```

Then fusion experiments use:

```text
cached CLIP feature
cached SigLIP feature
cached frequency feature
        ↓
     tiny gate
        ↓
     classifier
```

This allows dozens of experiments without repeatedly running the expensive encoders.

---

# 21. Teacher training

The teacher should initially use frozen backbones.

Train:

- projection layers
- reliability gate
- classification head

Use mixed precision.

Do not use the 6 GB GPU to train all foundation models end-to-end.

---

# 22. Distillation

After the teacher is stable:

```text
Teacher probability
        +
Teacher feature representation
        +
Ground truth label
        ↓
Student loss
```

Use a mixture of:

```text
L = BCE(student, label)
  + λ1 * KL(student, teacher)
  + λ2 * feature_loss(student, teacher)
```

The exact weights should be tuned experimentally.

The key idea is that the student learns:

1. whether an image is AIGC
2. how confident the teacher is
3. useful internal representations

---

# 23. Robustness training

The challenge requires robustness to transformations such as:

- JPEG quality 90/70/50/30
- Gaussian blur σ 0.5/1/2
- resize 0.5×/0.25× followed by upscaling
- Gaussian noise σ 0.02/0.05/0.10
- color jitter ±20%
- center crop 80%

These are explicitly specified by the challenge. fileciteturn1file0L21-L28

The model should therefore see transformations during training, but the final benchmark should apply them independently to held-out images.

---

# 24. Robustness evaluation

We should report:

| Condition | Accuracy | Balanced Acc. | F1 | AUROC |
|---|---:|---:|---:|---:|
| Clean | TODO | TODO | TODO | TODO |
| JPEG Q90 | TODO | TODO | TODO | TODO |
| JPEG Q70 | TODO | TODO | TODO | TODO |
| JPEG Q50 | TODO | TODO | TODO | TODO |
| JPEG Q30 | TODO | TODO | TODO | TODO |
| Blur σ0.5 | TODO | TODO | TODO | TODO |
| Blur σ1 | TODO | TODO | TODO | TODO |
| Blur σ2 | TODO | TODO | TODO | TODO |
| Resize 0.5× | TODO | TODO | TODO | TODO |
| Resize 0.25× | TODO | TODO | TODO | TODO |
| Noise σ0.02 | TODO | TODO | TODO | TODO |
| Noise σ0.05 | TODO | TODO | TODO | TODO |
| Noise σ0.10 | TODO | TODO | TODO | TODO |
| Color jitter | TODO | TODO | TODO | TODO |
| Crop 80% | TODO | TODO | TODO | TODO |

**Never invent these values.** Fill them only after running the actual benchmark.

---

# 25. Error analysis

Expected false-positive categories from the project research include:

- aggressive HDR/color grading
- digital artwork
- 3D/CGI content
- unusual high-ISO sensor noise

Expected false-negative categories include:

- localized inpainting
- severe repeated JPEG compression
- anti-forensic noise/grain matching

The supplied research document describes these specific failure regimes. fileciteturn3file0L106-L149

For the final report, save representative examples with:

```text
image
prediction
confidence
true label
transformation
reason for failure
```

---

# 26. Final inference requirement

The challenge requires a script that takes an image directory and outputs JSON with:

```json
[
  {
    "image_path": "/path/to/image.jpg",
    "pred": 0.9731
  }
]
```

The project already has the inference interface. The research document also specifies this exact mapping approach. fileciteturn3file0L10-L33

---

# 27. Recommended experiment sequence

## Experiment 0 — pipeline sanity

Dataset:

- small subset of CIFAKE

Model:

- student

Goal:

- confirm code and inference

---

## Experiment 1 — Community Forensics 50K

Dataset:

- 50K streamed examples

Goal:

- establish a diverse-generator baseline

---

## Experiment 2 — SID_Set

Add:

- SID_Set train

Goal:

- improve social-media/tampering robustness

---

## Experiment 3 — transformations

Add:

- JPEG
- blur
- resize
- noise
- color jitter
- crop

Goal:

- robustness

---

## Experiment 4 — teacher

Run:

- CLIP-L
- SigLIP2
- frequency branch

Goal:

- find whether the hybrid teacher actually beats the student

---

## Experiment 5 — distillation

Teacher → student.

Goal:

- retain teacher accuracy with much lower inference cost

---

## Experiment 6 — locked benchmark

Only now use:

- COCO val2017
- WildFake DALL-E Advanced

Goal:

- final hackathon demonstration

---

# 28. Commands at a glance

```bash
# SSH from Mac
ssh manan@buildabot

# Prepare environment
cd ~/aigc_robust_detection
bash server/scripts/01_prepare_server.sh

# Activate
source ~/.venvs/aigc_detect/bin/activate

# Check machine
bash server/scripts/08_server_status.sh

# Check disk
bash server/scripts/02_check_storage.sh

# Models
bash server/scripts/03_download_models.sh

# Dataset dry run
bash server/scripts/04_download_datasets.sh --dry-run

# Core datasets
bash server/scripts/04_download_datasets.sh --profile core

# Stream 50K Community Forensics examples
python server/scripts/05_stream_community_forensics.py \
  --limit 50000 \
  --out /mnt/ai-storage/aigc_data/datasets/cf_slice_50k

# Verify assets
bash server/scripts/06_verify_assets.sh

# COCO locked validation
bash server/scripts/07_download_validation_coco.sh
```

---

# 29. One-command server preparation

After copying the project to `buildabot`, the intended sequence is simply:

```bash
cd ~/aigc_robust_detection
bash server/scripts/01_prepare_server.sh
bash server/scripts/02_check_storage.sh
bash server/scripts/03_download_models.sh
bash server/scripts/04_download_datasets.sh --profile core
```

Then we inspect the storage and start the first 50K experiment.

---

# 30. Final strategy

The final system should not be:

> "three giant models glued together."

It should be:

> **A compact forensic detector distilled from a multi-domain semantic/frequency teacher and trained explicitly for real-world redistribution robustness.**

That gives us a strong answer to every major judging criterion:

### Technical Execution — 35%

- multi-stream architecture
- feature caching
- distillation
- robustness benchmark
- reproducible server setup

### Innovation & Problem Insight — 20%

- semantic/frequency complementarity
- degradation-aware gating
- teacher/student compression

### Impact & Relevance — 20%

- designed around social-media redistribution
- outputs a calibrated AIGC confidence score

### Feasibility — 15%

- final student fits substantially below 2B
- designed around 6 GB VRAM
- datasets stored on dedicated HDD

### Presentation — 10%

- clean architecture diagram
- robustness matrix
- error analysis
- teacher vs student comparison

---

# 31. Immediate next step

**Do not transfer the datasets through Tailscale.**

Transfer only the code repository/ZIP to `buildabot` if it is not already there.

Then run the server scripts directly on `buildabot`.

Tailscale is for:

- SSH
- code synchronization
- checkpoints
- results
- experiment management

The public internet connection of `buildabot` handles the multi-hundred-GB dataset downloads.
