# Error Analysis Note

The supplied design document identifies several expected boundary conditions. The final submission should replace the placeholders below with actual examples from the trained checkpoint.

## False positives

### 1. Heavy post-processing / HDR

Authentic photographs with aggressive tone mapping, clarity enhancement, or other strong post-processing can create residual patterns that resemble synthetic processing.

**Observed example:** `TODO`

**Why it matters:** A detector intended for online moderation must avoid treating professional photography as synthetic simply because it has unusual processing.

### 2. Digital illustration / CGI

Digital art and 3D renders do not contain ordinary camera-sensor noise in the same way as photographs. A residual detector can therefore interpret their signal as anomalous.

**Observed example:** `TODO`

### 3. High-ISO photography

Strong sensor noise can produce unusual residual distributions.

**Observed example:** `TODO`

## False negatives

### 1. Localized inpainting

When a synthetic edit occupies a small fraction of an otherwise authentic image, global features can be dominated by the authentic background.

**Observed example:** `TODO`

### 2. Extreme cascaded compression

Repeated aggressive compression/downsampling can remove both frequency evidence and fine texture.

**Observed example:** `TODO`

### 3. Anti-forensic noise matching

Synthetic images can deliberately add camera-like grain/noise, masking some high-frequency signatures.

**Observed example:** `TODO`

## What to record for each example

For every selected error, save:

- original image
- transformed version, if relevant
- ground-truth label
- model probability
- predicted class at threshold 0.5
- gate weights `[w_clip, w_siglip, w_freq]`
- short explanation

The gate weights are especially useful for investigating whether the model is relying on semantic or residual evidence in a failure case.

## Trade-off discussion

The final report should explicitly discuss:

- robustness vs. clean-set peak accuracy
- generalization to unseen generators
- false-positive cost on authentic processed media
- computational cost of multiple foundation encoders
- probability calibration vs. raw classification accuracy
