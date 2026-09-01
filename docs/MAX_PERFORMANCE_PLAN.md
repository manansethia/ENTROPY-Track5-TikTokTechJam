# Maximum-Performance Plan — AIGC Robust Detection

## 1. Objective

Do not optimize for the smallest model first. Optimize for the **best generalizing detector that can actually be trained and demonstrated on the available hardware**, while respecting the hackathon's strict `<2B parameters` rule.

The server has:

- ~931.5 GB HDD mounted at `/mnt/ai-storage`
- ~475 GB NVMe root filesystem
- 32 GB RAM
- NVIDIA RTX 3050 6 GB VRAM
- Intel i5 12th Gen
- Fedora + `nvidia-smi`

Use the HDD for datasets/Hugging Face caches and the NVMe for the OS, Python environment, code, temporary feature shards, and the hottest working set. Do not fill the root filesystem just because it is available.

## 2. Important design change

The previous plan was too eager to commit to a ~400M handcrafted ensemble. This version deliberately treats the architecture as an **experiment**.

We download a pool of strong pretrained detectors/encoders first, benchmark them, and only then choose the final architecture.

The core candidate pool is:

1. AIDE fine-tuned checkpoint
2. AIDE 50-epoch checkpoint
3. DDA official checkpoint
4. UnivFD / CLIP ViT-L/14 baseline
5. SigLIP2 Large 384
6. SigLIP2 SO400M 384
7. SigLIP2 Base 224
8. DINOv2 Large
9. ConvNeXt-Tiny

The official AIDE project is a hybrid detector combining visual and forensic/noise evidence. DDA is a NeurIPS 2025 Spotlight method that explicitly aligns pixel and frequency domains and reports strong cross-benchmark generalization. These are therefore useful **baselines/teachers**, not assumptions about the final winner.

## 3. Why not simply use the largest available model?

Parameter count is not equivalent to forensic usefulness. A 1B semantic encoder can be excellent at recognizing scene content while still missing generator-specific frequency artifacts. Conversely, a tiny forensic model can collapse under JPEG/blur.

The winning architecture should be selected by a weighted score over:

- clean AUROC
- clean balanced accuracy
- cross-generator AUROC
- robustness AUROC/accuracy under each challenge transformation
- worst-case transformation performance
- calibration error / Brier score
- inference latency
- peak VRAM
- total parameter count

Do not select by clean accuracy alone.

## 4. Recommended architecture search

### Teacher ensemble

Start with a heterogeneous ensemble rather than a homogeneous stack:

```text
RGB image
 ├── SigLIP2 Large 384 ───────┐
 ├── CLIP ViT-L/14 ───────────┤
 ├── DINOv2 Large ────────────┤ semantic/self-supervised evidence
 │                            │
 └── SRM → Haar DWT → CNN ───┤ forensic evidence
                              ▼
                      reliability-aware fusion
                              ▼
                       teacher probability
```

Also evaluate pretrained AIDE and DDA as independent detector experts. Their outputs can be included as teacher logits/features if licensing and implementation allow it.

### Candidate final model

The final submitted model may be one of:

- a ~0.4B–0.9B semantic + forensic fusion model,
- a distilled ~100M–300M student,
- or a larger <2B model if it produces a **measurable** robustness gain worth the cost.

There is no requirement that the final model be small merely for aesthetics.

## 5. Distillation

Use a large/heterogeneous teacher only if it actually improves held-out performance.

Student loss:

```text
L = λ_label * BCE(student, y)
  + λ_kd * KL(student || teacher)
  + λ_feat * feature_alignment
  + λ_robust * consistency(student(x), student(T(x)))
```

where `T` is a stochastic post-processing transform.

This gives the student both the ground-truth decision and the teacher's softer uncertainty.

## 6. Robustness strategy

The challenge explicitly targets:

- JPEG quality 90/70/50/30
- Gaussian blur σ 0.5/1.0/2.0
- resize 0.5×/0.25× then upscale
- Gaussian noise σ 0.02/0.05/0.10
- color jitter ±20%
- center crop 80%

Training should use randomized mixtures, but evaluation should use **fixed deterministic levels** so the table is reproducible.

Add an additional stress suite for internal research only:

- JPEG Q20/Q10
- repeated JPEG
- mild sharpening
- screenshot-like resize + JPEG
- small rotations
- aspect-ratio crop
- grayscale
- mild webp recompression

Do not replace the official benchmark with the internal suite.

## 7. Data strategy

Use multiple distributions to reduce generator-specific shortcut learning.

Priority:

1. Community Forensics-Small — generator diversity is the major attraction. The released base dataset contains 2.7M generated images from 4,803 generator models; the Small release is about 11% of the base dataset and is designed to be easier to use.
2. SID_Set — useful social-image/manipulation diversity.
3. GenImage — large multi-generator benchmark/training source.
4. WildFake training partitions — only the portions not reserved by the challenge.
5. CIFAKE — debugging/baseline, not the main training distribution.

The exact challenge validation subset remains locked out of training.

## 8. Storage strategy

Recommended:

```text
/mnt/ai-storage/aigc_data/
  models/
  datasets/
  validation_LOCKED/
  hf_cache/
  features/
  checkpoints/
  benchmarks/
  manifests/
  logs/
  tmp/
```

Use the root/NVMe for:

- Python virtual environment
- Git repository
- source code
- temporary high-speed feature shards when needed

If the working dataset is too slow from HDD, copy only the current shard to NVMe rather than duplicating every dataset.

## 9. VRAM strategy for the RTX 3050 6 GB

The GPU is sufficient for a careful research pipeline but not for naïve end-to-end fine-tuning of several large encoders.

Use:

- frozen encoder feature extraction
- batch size 1–8 depending on model
- gradient accumulation
- AMP / FP16 or BF16 where supported
- activation checkpointing when fine-tuning
- CPU offload for sequential teacher extraction
- cached embeddings
- small trainable fusion heads
- parameter-efficient tuning before full fine-tuning

Most importantly, **do not load CLIP-L + SigLIP-L + DINO-L + a large forensic backbone simultaneously during training**. Extract/cache features sequentially.

## 10. Candidate evaluation order

1. GPU smoke test.
2. Verify every downloaded checkpoint loads.
3. Run all pretrained detector baselines on a small non-benchmark probe set.
4. Run each candidate against clean + transform probes.
5. Identify complementary errors.
6. Build the heterogeneous teacher.
7. Train the fusion head.
8. Attempt PEFT/LoRA only if it improves validation.
9. Distill to one or more students.
10. Select the best model by the full scorecard.
11. Freeze the final model.
12. Run the challenge validation benchmark exactly as specified.
13. Generate robustness table + error analysis.

## 11. What success looks like

A convincing result is not merely:

> 99% accuracy on clean images.

A stronger result is:

> high clean AUROC, small degradation under JPEG/blur/resize/noise, strong cross-generator transfer, calibrated confidence, and a clear explanation of failure cases.

That directly matches the problem's emphasis on realistic redistribution and robustness.
