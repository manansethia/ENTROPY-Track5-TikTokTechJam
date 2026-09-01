# Devpost Project Description — Draft

> Replace every `TODO` before submission. Do not claim an accuracy, AUROC, speed, or parameter count that has not been measured.

## Project title

**Multi-Domain Robust AIGC Forensics Detector**

## Inspiration

AI-generated images are increasingly realistic, but images rarely remain in their original form after creation. Social platforms, messaging apps, image hosts and editing tools may compress, resize, blur, crop or recolor an image. A detector that performs well only on pristine files can therefore fail precisely when forensic evidence is most needed.

Our project focuses on this gap: detecting AIGC while explicitly training and evaluating for common redistribution transformations.

## What we built

We built a tri-stream image detector that combines:

- a frozen **CLIP ViT-L/14** semantic encoder,
- a frozen **SigLIP** semantic encoder, and
- an active forensic stream using **SRM high-pass filtering → Haar wavelet detail extraction → ConvNeXt-Tiny**.

Their representations are projected into a shared 256-dimensional space. A learned softmax gate dynamically weights the three streams before a compact MLP produces `P(AIGC | image)`.

The model is designed under the hackathon's strict **<2B parameter** constraint. The exact instantiated count is printed by the training script and should be reported here:

**Measured total parameters: TODO**

## Why it should be robust

Semantic features can survive transformations that destroy fine pixel-level traces. Conversely, high-frequency residuals can expose subtle generative artifacts that semantic encoders miss.

Rather than committing to one source of evidence, the gate learns an input-dependent mixture. This is particularly useful when compression or blur weakens the residual stream.

## Robustness strategy

During training we apply stochastic versions of the benchmark transformations:

- JPEG compression
- Gaussian blur
- downscale/upscale
- Gaussian noise
- color jitter
- random crop

The reserved demonstration benchmark contains COCO val2017 authentic images and a WildFake DALL-E Advanced synthetic subset. These images are kept outside training and optimization.

## Results

### Clean vs. transformed

| Transform | Accuracy | Balanced Accuracy | F1 | AUROC |
|---|---:|---:|---:|---:|
| Clean | TODO | TODO | TODO | TODO |
| JPEG 90 | TODO | TODO | TODO | TODO |
| JPEG 70 | TODO | TODO | TODO | TODO |
| JPEG 50 | TODO | TODO | TODO | TODO |
| JPEG 30 | TODO | TODO | TODO | TODO |
| Blur 0.5 | TODO | TODO | TODO | TODO |
| Blur 1.0 | TODO | TODO | TODO | TODO |
| Blur 2.0 | TODO | TODO | TODO | TODO |
| Down/up 0.5× | TODO | TODO | TODO | TODO |
| Down/up 0.25× | TODO | TODO | TODO | TODO |
| Noise 0.02 | TODO | TODO | TODO | TODO |
| Noise 0.05 | TODO | TODO | TODO | TODO |
| Noise 0.10 | TODO | TODO | TODO | TODO |
| Color jitter | TODO | TODO | TODO | TODO |
| Center crop 80% | TODO | TODO | TODO | TODO |

## Technical stack

**Development:** VS Code / Jupyter / remote Linux GPU server — TODO: keep only tools actually used.

**Frameworks and libraries:**

- PyTorch
- torchvision
- Hugging Face Transformers
- OpenCLIP
- timm
- Albumentations (dependency retained for ecosystem compatibility)
- OpenCV
- scikit-learn
- pandas
- Pillow
- PyYAML

**Models:**

- OpenAI CLIP ViT-L/14
- Google SigLIP
- ConvNeXt-Tiny

## Datasets

Training sources should be listed with the exact subsets actually used. Candidate sources in the supplied project design are WildFake training partitions, SID_Set, balanced GenImage subsets and CIFAKE subsets.

**Exact training composition: TODO**

The reserved demonstration benchmark is:

- COCO val2017: 4,998 authentic images
- WildFake DALL-E Advanced: 8,843 synthetic images

## Error analysis

Our main expected failure modes are:

- heavily post-processed authentic photographs,
- digital art and CGI,
- high-ISO photography,
- localized synthetic edits,
- severe cascaded compression,
- deliberate camera-noise/film-grain matching.

**Actual examples and probabilities: TODO**

## Limitations

This is a research prototype, not a universal authenticity oracle. It can experience generator distribution shift, calibration error and ambiguity between synthetic imagery and unusual but authentic digital content. Extremely aggressive transformations can erase the evidence any detector depends on.

With more time, we would investigate stronger cross-generator training, explicit local-region reasoning, calibration, model distillation for faster inference, and evaluation on additional unseen generators.

## Impact

A robust image-level detector can act as one signal in content provenance, moderation, fraud investigation or media verification workflows. The key practical objective is not to declare every image "real" or "fake" with certainty, but to provide a calibrated forensic signal that remains useful after common redistribution transformations.

## Team contributions

- **TODO — Name:** TODO
- **TODO — Name:** TODO

## Demo

YouTube: **TODO**

Repository: **TODO**

## Responsible-use note

Predictions should be treated as probabilistic forensic evidence rather than proof of authorship, intent or authenticity. Human review and provenance information remain important for high-stakes decisions.
