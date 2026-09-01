# Robustness Evaluation Protocol

## Required conditions

The evaluation script supports:

| Condition | Specification | Real-world analogy |
|---|---|---|
| Clean | Unmodified | Original image |
| JPEG_90 | JPEG quality 90 | Mild social re-encode |
| JPEG_70 | JPEG quality 70 | Moderate re-encode |
| JPEG_50 | JPEG quality 50 | Strong re-encode |
| JPEG_30 | JPEG quality 30 | Heavy re-encode |
| Blur_0.5 | Gaussian sigma 0.5 | Mild defocus/filtering |
| Blur_1.0 | Gaussian sigma 1.0 | Moderate blur |
| Blur_2.0 | Gaussian sigma 2.0 | Strong blur |
| Downscale_0.5x | Down/up 0.5× | Thumbnail/resampling |
| Downscale_0.25x | Down/up 0.25× | Severe thumbnailing |
| Noise_0.02 | Gaussian σ 0.02 in [0,1] | Mild sensor/channel noise |
| Noise_0.05 | Gaussian σ 0.05 in [0,1] | Moderate noise |
| Noise_0.10 | Gaussian σ 0.10 in [0,1] | Strong noise |
| ColorJitter | brightness/contrast/saturation ±20% | Auto-enhancement/filtering |
| CenterCrop_80 | Central 80% | Framing/profile crop |

## Metrics

Report at least:

- **Accuracy**
- **Balanced accuracy**
- **F1**
- **AUROC**

Balanced accuracy is particularly useful because the two reserved benchmark classes have different sample counts.

## Recommended presentation

A final results table should contain:

```text
Transform       Accuracy   Balanced Acc.   F1      AUROC
Clean           ...
JPEG 90         ...
JPEG 70         ...
...
Aggregate mean  ...
```

Do not invent values before actually running the evaluation.

## Important distinction

The robustness table should be generated from the isolated benchmark and should not be used to select the final model after the benchmark is exposed. If you tune on it, it is no longer a clean demonstration benchmark.
