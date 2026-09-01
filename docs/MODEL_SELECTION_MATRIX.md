# Model Selection Matrix

Fill this table from actual experiments. Never insert claimed results before running the test.

| Candidate | Params | Clean AUROC | Clean BAcc | JPEG-30 | Blur-2 | Resize-.25 | Noise-.10 | Crop-80 | Mean Robust | ECE | VRAM | ms/img | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| UnivFD / CLIP-L | | | | | | | | | | | | | |
| AIDE fine-tuned | | | | | | | | | | | | | |
| AIDE 50 epoch | | | | | | | | | | | | | |
| DDA | | | | | | | | | | | | | |
| SigLIP2 Large 384 | | | | | | | | | | | | | |
| SigLIP2 SO400M 384 | | | | | | | | | | | | | |
| DINOv2 Large | | | | | | | | | | | | | |
| Hybrid teacher | | | | | | | | | | | | | |
| Distilled student | | | | | | | | | | | | | |

## Selection rule

Use a pre-declared weighted score, for example:

- 20% clean AUROC
- 10% clean balanced accuracy
- 35% mean robustness AUROC
- 15% worst-case robustness AUROC
- 10% calibration
- 5% latency
- 5% VRAM/parameter efficiency

The exact weights can be changed before experiments, but must not be changed after seeing the final benchmark just to favor a preferred model.
