# Locked Out-of-Distribution (OOD) Evaluation Report

*Single-pass evaluation on external held-out benchmarks*

| OOD Dataset / Generator | Samples | Mean Predicted $P(\text{AIGC})$ | Detection Accuracy ($\tau=0.5$) |
| :--- | :--- | :--- | :--- |
| **Synthbuster_dalle2** | 100 | 0.1962 | **20.0%** |
| **Synthbuster_dalle3** | 100 | 0.9964 | **100.0%** |
| **Synthbuster_firefly** | 100 | 0.5625 | **56.0%** |
| **Synthbuster_glide** | 100 | 0.7436 | **77.0%** |
| **Synthbuster_midjourney-v5** | 100 | 0.8081 | **83.0%** |
| **Synthbuster_stable-diffusion-1-3** | 100 | 0.9059 | **92.0%** |
| **Synthbuster_stable-diffusion-1-4** | 100 | 0.9205 | **94.0%** |
| **Synthbuster_stable-diffusion-2** | 100 | 0.5686 | **57.0%** |
| **Synthbuster_stable-diffusion-xl** | 100 | 0.7827 | **80.0%** |

