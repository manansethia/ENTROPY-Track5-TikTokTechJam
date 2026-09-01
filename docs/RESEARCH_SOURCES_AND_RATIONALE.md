# Research Sources and Rationale

This project uses the supplied hackathon specification and the supplied AIGC Image Detection Pipeline as the primary project constraints.

External research used to expand the candidate pool:

- AIDE (ICLR 2025): hybrid AI-generated image detection using semantic and forensic evidence. Official repository: https://github.com/shilinyan99/AIDE
- DDA (NeurIPS 2025 Spotlight): aligns pixel and frequency domains to improve detector generalization. Official implementation: https://github.com/roy-ch/Dual-Data-Alignment
- Community Forensics (CVPR 2025): 2.7M generated images from 4,803 generator models; Small release is approximately 11% of the base dataset. https://huggingface.co/datasets/OwensLab/CommunityForensics
- UniversalFakeDetect (CVPR 2023): CLIP ViT-L/14 with a detector head and cross-generator evaluation. https://github.com/WisconsinAIVision/UniversalFakeDetect
- SigLIP 2 (2025): model family includes Base, Large, SO400M and Giant sizes, allowing an accuracy/compute sweep. https://huggingface.co/blog/siglip2

Important compliance note:

The hackathon still requires **strictly less than 2 billion parameters**. The SigLIP2 Giant checkpoint is not part of the default pool because its Hugging Face repository reports the full checkpoint around 2B parameters. Even though its vision tower is approximately 1B, using the full checkpoint introduces unnecessary ambiguity.
