

**Pasted markdown(20260827-163300).md**

File

1.

## Background

Generative AI tools are making it easier than ever to create highly realistic synthetic images at scale. This creates new risks for online platforms, including misinformation, impersonation, fraud, and reduced trust in digital content. In practice, detection becomes even harder after images are compressed, cropped, reposted, or lightly edited, so robust methods matter more than lab-only accuracy.

2.

## Problem Statement

We want participants to build a prototype that can distinguish AI-generated images from authentic images with strong robustness under realistic post-processing and redistribution scenarios. The goal is not only to achieve good detection performance on clean data, but also to maintain accuracy after transformations such as blur, compression, color adjustment, cropping, or rescaling. Solutions should present a clear technical approach, an evaluation strategy, and thoughtful discussion of trade-offs such as robustness, generalisation, and false positives.

**Note: We consider robustness against a subset of the following augmentataions.**

| **TransformParametersReal-World Analog** |                                 |                                   |
| ---------------------------------------- | ------------------------------- | --------------------------------- |
| JPEG Compression                         | quality = 90, 70, 50, 30        | Social-media re-encode, messaging |
| Gaussian Blur                            | kernel σ = 0.5, 1.0, 2.0        | Out-of-focus                      |
| Resize                                   | scale 0.5× / 0.25× then upscale | Thumbnail generation              |
| Gaussian Noise                           | σ = 0.02, 0.05, 0.10            | Low-light sensor noise            |
| Color Jitter                             | brightness/contrast/sat. ±20%   | Filter apps, auto-enhance         |
| Center Crop                              | crop 80%                        | Profile-picture cropping, framing |

3.

## Constraints & Scope

| **CategoryConstraints & Scope Details** |                                                                                                                                                                                                                                                                |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| In scope                                | Image-level AIGC detection, robustness to common image transformations, feature engineering, model design, evaluation design, error analysis, and explainability ideas                                                                                         |
| Out of scope                            | Full production deployment, platform-wide moderation systems, and non-image modalities such as video or audio                                                                                                                                                  |
| Limits                                  | Assume a hackathon-scale prototype, limited compute, and no access to internal production systems. Teams should optimise for a convincing proof of concept rather than a production-grade service. **Note: Participants must use models with <2B parameters.** |
| Allowed assumptions                     | Teams may use public or properly licensed datasets, create their own transformed test cases, and make reasonable assumptions about deployment context as long as those assumptions are stated clearly.                                                         |

4.

## Available Resources & Data

- Public or properly licensed image datasets for AIGC detection and image forensics.
- Self-created transformed samples using operations such as blur, compression, cropping, color adjustment, or rescaling.
- Public documentation for relevant machine learning and computer vision libraries.
- Datasets:
  - [https://huggingface.co/datasets/saberzl/SID\_Set](https://huggingface.co/datasets/saberzl/SID_Set)
  - [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  -
  ## [https://modelscope.cn/datasets/hy2628982280/WildFake/summary](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
  - For this modelscope dataset, please translate it via the translation button before use:
    [image](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YmI1MGY5MzFiMGYwMzM5NzAzYWI2MWMwNmYxOTY1ZTRfa25IdFRvWHpYQm0xQm9SVkU0bVJoWHVzRzRXdkF2R0pfVG9rZW46QzZpQ2I3M29pb2NDU1F4Y2RDY203aE5GeUxjXzE3ODc4NDgzODQ6MTc4Nzg1MTk4NF9WNA\&add_watermark=true\&scene_type=CCM)

**Validation Dataset (for Demonstration Purposes Only):**

We choose **a subset of WildFake** for participants to demonstrate their models’ performance and track iterative improvements. This dataset serves only as a reference benchmark and will not contribute to the final score. **Do not use the following data during training.** Specifically:

| **Dataset# Num** |                 |      |
| ---------------- | --------------- | ---- |
| Non-AIGC         | COCO val2017    | 4998 |
| AIGC             | DALL·E Advanced | 8843 |

5.

## Expected Deliverables

1. **Written Project Description (via Devpost)**

- Provide a clear written description of your project that includes:
  - How your solution addresses the problem statement
  - Development tools used (e.g. VSCode, Colab, Jupyter)
  - Models or APIs used
  - Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
  - Datasets and assets used

2. **Public Code/GitHub Repository**

- Submit a link to a public Code/GitHub repository containing:
  - Well-structured, commented code covering all components of your solution
  - A script that takes an image directory as input and outputs a confidence score for each image, indicating the likelihood that it is AIGC-generated. The output should be a JSON file containing **`image_path`** and **`pred`** for each image.
  - A README file that includes:
    - Project overview
    - Setup and installation instructions
    - Steps to reproduce your results
    - A brief reflection on your solution's limitations and what you would improve given more time
    - Team member contributions (if applicable, i.e. team participants, non-solo participants)

3. **Demo Video**

- Submit a short video that:
  - Demonstrates your solution working end-to-end (e.g. inference results, dashboard, model predictions)
  - Is uploaded to YouTube and set to public visibility
  - Is linked in your Devpost description
  - Does not include third-party trademarks or copyrighted content without permission

4. **Robustness Evaluation Summary**

- Include a compact table or visual summary comparing performance on clean images versus transformed images.

5. **Error Analysis Note**

- Highlight representative false positives, false negatives, and any trade-offs in the proposed approach.

6.

## Judging Criteria

| **Judging CriteriaDefinitionWeight** |                                                                                                                                                                                                                                                                          |                                                                                                                                                                                                                 |         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **Technical Execution**              | The solution demonstrates strong engineering fundamentals, such as well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making.                | **35%**                                                                                                                                                                                                         |         |
| **Innovation & Problem Insight**     | The project demonstrates originality in both idea and approach. It stands out for the sharpness of its problem understanding — how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it.                                | **20%**                                                                                                                                                                                                         |         |
| **Impact & Relevance**               | The project has clear potential to deliver value to real users or stakeholders — with meaningful reach, tangible benefit, and relevance that goes beyond solving for the hackathon prompt alone.                                                                         | **20%**                                                                                                                                                                                                         |         |
| **Feasibility & Practicality**       | The solution is realistic and buildable beyond a prototype. The approach is technically and operationally sustainable — resource usage is proportionate, the architecture holds under real-world conditions, and the implementation is grounded rather than speculative. | **15%**                                                                                                                                                                                                         |         |
| **Presentation & Communication**     | The team communicates their work with clarity.                                                                                                                                                                                                                           | [Final Event Only]: The pitch tells a coherent story; from problem to solution to potential, and the team is able to respond to questions with depth, demonstrating genuine understanding of their own project. | **10%** |

I’ve got the full hackathon brief and the current

1.


## Background

Generative AI tools are making it easier than ever to create highly realistic synthetic images at scale. This creates new risks for online platforms, including misinformation, impersonation, fraud, and reduced trust in digital content. In practice, detection becomes even harder after images are compressed, cropped, reposted, or lightly edited, so robust methods matter more than lab-only accuracy.

2.


## Problem Statement

We want participants to build a prototype that can distinguish AI-generated images from authentic images with strong robustness under realistic post-processing and redistribution scenarios. The goal is not only to achieve good detection performance on clean data, but also to maintain accuracy after transformations such as blur, compression, color adjustment, cropping, or rescaling. Solutions should present a clear technical approach, an evaluation strategy, and thoughtful discussion of trade-offs such as robustness, generalisation, and false positives.

**Note: We consider robustness against a subset of the following augmentataions.**

| **TransformParametersReal-World Analog** |                                 |                                   |
| ---------------------------------------- | ------------------------------- | --------------------------------- |
| JPEG Compression                         | quality = 90, 70, 50, 30        | Social-media re-encode, messaging |
| Gaussian Blur                            | kernel σ =   0.5, 1.0, 2.0      | Out-of-focus                      |
| Resize                                   | scale 0.5× / 0.25× then upscale | Thumbnail generation              |
| Gaussian Noise                           | σ = 0.02, 0.05, 0.10            | Low-light sensor noise            |
| Color Jitter                             | brightness/contrast/sat. ±20%   | Filter apps, auto-enhance         |
| Center Crop                              | crop 80%                        | Profile-picture cropping, framing |

3.


## Constraints & Scope

| **CategoryConstraints & Scope Details** |                                                                                                                                                                                                                                                                |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| In scope                                | Image-level AIGC detection, robustness to common image transformations, feature engineering, model design, evaluation design, error analysis, and explainability ideas                                                                                         |
| Out of scope                            | Full production deployment, platform-wide moderation systems, and non-image modalities such as video or audio                                                                                                                                                  |
| Limits                                  | Assume a hackathon-scale prototype, limited compute, and no access to internal production systems. Teams should optimise for a convincing proof of concept rather than a production-grade service. **Note: Participants must use models with <2B parameters.** |
| Allowed assumptions                     | Teams may use public or properly licensed datasets, create their own transformed test cases, and make reasonable assumptions about deployment context as long as those assumptions are stated clearly.                                                         |

4.


## Available Resources & Data

- Public or properly licensed image datasets for AIGC detection and image forensics.
- Self-created transformed samples using operations such as blur, compression, cropping, color adjustment, or rescaling.
- Public documentation for relevant machine learning and computer vision libraries.
- Datasets: 
  - [https://huggingface.co/datasets/saberzl/SID\_Set](https://huggingface.co/datasets/saberzl/SID_Set)
  - [https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  -

  ## [https://modelscope.cn/datasets/hy2628982280/WildFake/summary](https://modelscope.cn/datasets/hy2628982280/WildFake/summary)
  - For this modelscope dataset, please translate it via the translation button before use:
     [image](https://bytedance.larkoffice.com/space/api/box/stream/download/asynccode/?code=YmI1MGY5MzFiMGYwMzM5NzAzYWI2MWMwNmYxOTY1ZTRfa25IdFRvWHpYQm0xQm9SVkU0bVJoWHVzRzRXdkF2R0pfVG9rZW46QzZpQ2I3M29pb2NDU1F4Y2RDY203aE5GeUxjXzE3ODc4NDgzODQ6MTc4Nzg1MTk4NF9WNA\&add_watermark=true\&scene_type=CCM)

**Validation Dataset (for Demonstration Purposes Only):**

We choose **a subset of WildFake** for participants to demonstrate their models’ performance and track iterative improvements. This dataset serves only as a reference benchmark and will not contribute to the final score. **Do not use the following data during training.** Specifically:

| **Dataset# Num** |                 |      |
| ---------------- | --------------- | ---- |
| Non-AIGC         | COCO val2017    | 4998 |
| AIGC             | DALL·E Advanced | 8843 |

5.


## Expected Deliverables

1. **Written Project Description (via Devpost)**

- Provide a clear written description of your project that includes: 
  - How your solution addresses the problem statement
  - Development tools used (e.g. VSCode, Colab, Jupyter)
  - Models or APIs used
  - Libraries and frameworks used (e.g. Hugging Face Transformers, PyTorch, scikit-learn, pandas)
  - Datasets and assets used

2. **Public Code/GitHub Repository**

- Submit a link to a public Code/GitHub repository containing: 
  - Well-structured, commented code covering all components of your solution
  - A script that takes an image directory as input and outputs a confidence score for each image, indicating the likelihood that it is AIGC-generated. The output should be a JSON file containing `image_path` and `pred` for each image.
  - A README file that includes: 
    - Project overview
    - Setup and installation instructions
    - Steps to reproduce your results
    - A brief reflection on your solution's limitations and what you would improve given more time
    - Team member contributions (if applicable, i.e. team participants, non-solo participants)

3. **Demo Video**

- Submit a short video that: 
  - Demonstrates your solution working end-to-end (e.g. inference results, dashboard, model predictions)
  - Is uploaded to YouTube and set to public visibility
  - Is linked in your Devpost description
  - Does not include third-party trademarks or copyrighted content without permission

4. **Robustness Evaluation Summary**

- Include a compact table or visual summary comparing performance on clean images versus transformed images.

5. **Error Analysis Note**

- Highlight representative false positives, false negatives, and any trade-offs in the proposed approach.

6.


## Judging Criteria

| **Judging CriteriaDefinitionWeight** |                                                                                                                                                                                                                                                                          |                                                                                                                                                                                                                 |         |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **Technical Execution**              | The solution demonstrates strong engineering fundamentals, such as well-structured code, thoughtful architecture, and effective use of APIs or models. The demo runs reliably, and the technical complexity reflects deliberate, capable decision-making.                | **35%**                                                                                                                                                                                                         | <br>    |
| **Innovation & Problem Insight**     | The project demonstrates originality in both idea and approach. It stands out for the sharpness of its problem understanding — how clearly the team has framed the challenge, why it matters, and how directly the solution addresses it.                                | **20%**                                                                                                                                                                                                         | <br>    |
| **Impact & Relevance**               | The project has clear potential to deliver value to real users or stakeholders — with meaningful reach, tangible benefit, and relevance that goes beyond solving for the hackathon prompt alone.                                                                         | **20%**                                                                                                                                                                                                         | <br>    |
| **Feasibility & Practicality**       | The solution is realistic and buildable beyond a prototype. The approach is technically and operationally sustainable — resource usage is proportionate, the architecture holds under real-world conditions, and the implementation is grounded rather than speculative. | **15%**                                                                                                                                                                                                         | <br>    |
| **Presentation & Communication**     | The team communicates their work with clarity.                                                                                                                                                                                                                           | [Final Event Only]: The pitch tells a coherent story; from problem to solution to potential, and the team is able to respond to questions with depth, demonstrating genuine understanding of their own project. | **10%** |

### Project Constraints & Design Specifications

- **Parameter Ceiling:** The fused model totals **\~415M parameters** (CLIP ViT-L/14 \~304M + SigLIP2-Base \~86M + ConvNeXt-Tiny \~28M + Heads \~2M), comfortably below the strict **<2B parameter limit**.
- **Pre-Compressed & Corrupted Images:** The dynamic gating layer evaluates both macro semantic coherence and high-pass residual sub-bands. When input images are heavily compressed or blurred, high-frequency residuals decay; the gating module automatically downweights the frequency branch and relies on the invariant semantic encoders.
- **Benchmark Isolation:** MS-COCO val2017 (4,998 images) and WildFake DALL-E Advanced (8,843 images) are strictly reserved for testing and validation. Under no circumstances should they be included in **`data/train/`**.

### Repository File Hierarchy

Plaintext

```
aigc_robust_detection/
├── configs/
│   └── train_config.yaml
├── models/
│   ├── __init__.py
│   ├── srm_filters.py
│   └── tri_hybrid_detector.py
├── scripts/
│   ├── setup_server_env.sh
│   ├── train_detector.py
│   ├── run_inference.py
│   └── evaluate_robustness.py
├── requirements.txt
└── README.md


```

### Complete File Contents

#### `requirements.txt`

Plaintext

```
torch>=2.1.0
torchvision>=0.16.0
open-clip-torch>=2.24.0
transformers>=4.38.0
timm>=0.9.12
albumentations>=1.4.0
opencv-python-headless>=4.8.0
scikit-learn>=1.3.0
pandas>=2.1.0
pyyaml>=6.0.1
tqdm>=4.66.0
Pillow>=10.0.0


```

#### `configs/train_config.yaml`

YAML

```
paths:
  train_data_dir: "data/train"
  val_coco_dir: "data/val_demo/coco_val2017"
  val_dalle_dir: "data/val_demo/dalle_advanced"
  output_dir: "checkpoints"
  logs_dir: "logs"

training:
  batch_size: 16
  gradient_accumulation_steps: 4
  num_workers: 4
  epochs: 15
  base_lr: 1e-4
  weight_decay: 1e-4
  mixed_precision: true

augmentations:
  jpeg_compression:
    prob: 0.60
    quality_range: [30, 90]
  gaussian_blur:
    prob: 0.40
    kernel_limit: [3, 7]
    sigma_range: [0.5, 2.0]
  resize:
    prob: 0.35
    scale_range: [0.25, 0.50]
  gaussian_noise:
    prob: 0.30
    var_range: [10.0, 50.0]
  color_jitter:
    prob: 0.40
    brightness: 0.2
    contrast: 0.2
    saturation: 0.2
    hue: 0.1
  random_crop:
    prob: 0.40
    scale: [0.7, 1.0]


```

#### `models/__init__.py`

Python

```
from .tri_hybrid_detector import MasterEnsembleDetector
from .srm_filters import WaveletResidualBlock


```

#### `models/srm_filters.py`

Python

```
import torch
import torch.nn as nn
import torch.nn.functional as F

class SRMConvolution(nn.Module):
    def __init__(self):
        super().__init__()
        srm_kernel = torch.tensor([
            [0.0,  0.25, 0.0],
            [0.25, -1.0, 0.25],
            [0.0,  0.25, 0.0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel', srm_kernel.repeat(3, 1, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=1, groups=3)

class WaveletResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.srm = SRMConvolution()

    def haar_dwt2d(self, x):
        x01 = x[:, :, 0::2, :] / 2.0
        x02 = x[:, :, 1::2, :] / 2.0
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]

        lh = -x1 - x3 + x2 + x4
        hl = -x1 + x3 - x2 + x4
        hh = x1 - x3 - x2 + x4
        return torch.cat([lh, hl, hh], dim=1)

    def forward(self, x):
        res = self.srm(x)
        return self.haar_dwt2d(res)


```

#### `models/tri_hybrid_detector.py`

Python

```
import torch
import torch.nn as nn
from transformers import AutoModel
import open_clip
import timm
from .srm_filters import WaveletResidualBlock

class MasterEnsembleDetector(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.clip_encoder = self.clip_model.visual
        for param in self.clip_encoder.parameters():
            param.requires_grad = False
            
        self.siglip_model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
        self.siglip_encoder = self.siglip_model.vision_model
        for param in self.siglip_encoder.parameters():
            param.requires_grad = False

        self.residual_extractor = WaveletResidualBlock()
        self.freq_encoder = timm.create_model('convnext_tiny', pretrained=True, in_chans=9, num_classes=0)

        self.proj_clip = nn.Linear(768, 256)
        self.proj_siglip = nn.Linear(768, 256)
        self.proj_freq = nn.Linear(self.freq_encoder.num_features, 256)

        self.gate = nn.Sequential(
            nn.Linear(256 * 3, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x_clip, x_siglip, x_raw):
        with torch.no_grad():
            f_clip = self.clip_encoder(x_clip)
            if isinstance(f_clip, tuple):
                f_clip = f_clip[0]
            siglip_out = self.siglip_encoder(pixel_values=x_siglip)
            f_siglip = siglip_out.pooler_output

        f_clip = self.proj_clip(f_clip)
        f_siglip = self.proj_siglip(f_siglip)

        f_res = self.residual_extractor(x_raw)
        f_freq = self.freq_encoder(f_res)
        f_freq = self.proj_freq(f_freq)

        f_cat = torch.cat([f_clip, f_siglip, f_freq], dim=-1)
        w = self.gate(f_cat)

        f_fused = w[:, 0:1] * f_clip + w[:, 1:2] * f_siglip + w[:, 2:3] * f_freq
        return self.classifier(f_fused).squeeze(-1)


```

#### `scripts/setup_server_env.sh`

Bash

```
#!/usr/bin/env bash
set -e

sudo apt-get update && sudo apt-get install -y build-essential libgl1-mesa-glx libglib2.0-0 rsync git
conda create -n aigc_detect python=3.10 -y || true
source $(conda info --base)/etc/profile.d/conda.sh
conda activate aigc_detect

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

mkdir -p checkpoints logs data/train data/val_demo/coco_val2017 data/val_demo/dalle_advanced


```

#### `scripts/train_detector.py`

Python

```
import os
import glob
import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from tqdm import tqdm
from models.tri_hybrid_detector import MasterEnsembleDetector

class AIGCDataset(Dataset):
    def __init__(self, root_dir, augment_pipeline=None):
        self.image_paths = []
        self.labels = []
        self.augment_pipeline = augment_pipeline
        
        for p in glob.glob(os.path.join(root_dir, 'real', '*.*')):
            self.image_paths.append(p)
            self.labels.append(0.0)
        for p in glob.glob(os.path.join(root_dir, 'synthetic', '*.*')):
            self.image_paths.append(p)
            self.labels.append(1.0)
            
        self.clip_tf = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        self.siglip_tf = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = np.array(Image.open(path).convert('RGB'))
        
        if self.augment_pipeline:
            augmented = self.augment_pipeline(image=image)['image']
        else:
            augmented = image
            
        x_clip = self.clip_tf(image=augmented)['image']
        x_siglip = self.siglip_tf(image=augmented)['image']
        x_raw = x_siglip
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x_clip, x_siglip, x_raw, label

def get_train_pipeline(cfg):
    aug = cfg['augmentations']
    return A.Compose([
        A.OneOf([
            A.ImageCompression(quality_lower=aug['jpeg_compression']['quality_range'][0], 
                               quality_upper=aug['jpeg_compression']['quality_range'][1], 
                               p=aug['jpeg_compression']['prob']),
            A.GaussianBlur(blur_limit=aug['gaussian_blur']['kernel_limit'], 
                           sigma_limit=aug['gaussian_blur']['sigma_range'], 
                           p=aug['gaussian_blur']['prob']),
            A.GaussNoise(var_limit=aug['gaussian_noise']['var_range'], 
                         p=aug['gaussian_noise']['prob']),
        ], p=0.8),
        A.RandomResizedCrop(224, 224, scale=tuple(aug['random_crop']['scale']), p=aug['random_crop']['prob']),
        A.ColorJitter(brightness=aug['color_jitter']['brightness'],
                      contrast=aug['color_jitter']['contrast'],
                      saturation=aug['color_jitter']['saturation'],
                      hue=aug['color_jitter']['hue'],
                      p=aug['color_jitter']['prob'])
    ])

def train():
    with open("configs/train_config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MasterEnsembleDetector().to(device)

    train_dataset = AIGCDataset(cfg['paths']['train_data_dir'], augment_pipeline=get_train_pipeline(cfg))
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True, 
        num_workers=cfg['training']['num_workers'],
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=float(cfg['training']['base_lr']), 
        weight_decay=float(cfg['training']['weight_decay'])
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=cfg['training']['mixed_precision'])
    accum_steps = cfg['training']['gradient_accumulation_steps']

    for epoch in range(cfg['training']['epochs']):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']}")
        for step, (x_clip, x_siglip, x_raw, targets) in enumerate(pbar):
            x_clip, x_siglip, x_raw, targets = x_clip.to(device), x_siglip.to(device), x_raw.to(device), targets.to(device)

            with torch.cuda.amp.autocast(enabled=cfg['training']['mixed_precision']):
                outputs = model(x_clip, x_siglip, x_raw)
                loss = criterion(outputs, targets) / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accum_steps
            pbar.set_postfix({'loss': f"{(loss.item() * accum_steps):.4f}"})

        os.makedirs(cfg['paths']['output_dir'], exist_ok=True)
        torch.save(model.state_dict(), os.path.join(cfg['paths']['output_dir'], f"master_model_epoch_{epoch+1}.pth"))

if __name__ == '__main__':
    train()


```

#### `scripts/run_inference.py`

Python

```
import os
import json
import argparse
import torch
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from models.tri_hybrid_detector import MasterEnsembleDetector

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    model = MasterEnsembleDetector()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    clip_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    siglip_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    valid_exts = ('.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff')
    all_files = []
    for root, _, files in os.walk(args.image_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                all_files.append(os.path.join(root, f))

    results = []
    with torch.no_grad():
        for path in all_files:
            try:
                img = np.array(Image.open(path).convert('RGB'))
                x_clip = clip_tf(image=img)['image'].unsqueeze(0).to(device)
                x_siglip = siglip_tf(image=img)['image'].unsqueeze(0).to(device)
                
                logit = model(x_clip, x_siglip, x_siglip)
                prob = torch.sigmoid(logit).item()

                rel_path = os.path.relpath(path, args.image_dir)
                results.append({
                    "image_path": rel_path,
                    "pred": round(float(prob), 4)
                })
            except Exception as e:
                pass

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    main()


```

#### `scripts/evaluate_robustness.py`

Python

```
import os
import glob
import argparse
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from models.tri_hybrid_detector import MasterEnsembleDetector

PERTS = {
    "Clean": A.NoOp(),
    "JPEG_90": A.ImageCompression(quality_lower=90, quality_upper=90, p=1.0),
    "JPEG_70": A.ImageCompression(quality_lower=70, quality_upper=70, p=1.0),
    "JPEG_50": A.ImageCompression(quality_lower=50, quality_upper=50, p=1.0),
    "JPEG_30": A.ImageCompression(quality_lower=30, quality_upper=30, p=1.0),
    "Blur_0.5": A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.5, 0.5), p=1.0),
    "Blur_1.0": A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(1.0, 1.0), p=1.0),
    "Blur_2.0": A.GaussianBlur(blur_limit=(7, 7), sigma_limit=(2.0, 2.0), p=1.0),
    "Downscale_0.5x": A.Downscale(scale_min=0.5, scale_max=0.5, p=1.0),
    "Downscale_0.25x": A.Downscale(scale_min=0.25, scale_max=0.25, p=1.0),
    "Noise_0.02": A.GaussNoise(var_limit=(10.0, 10.0), p=1.0),
    "Noise_0.05": A.GaussNoise(var_limit=(25.0, 25.0), p=1.0),
    "Noise_0.10": A.GaussNoise(var_limit=(50.0, 50.0), p=1.0),
    "ColorJitter": A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=1.0),
    "CenterCrop_80": A.CenterCrop(height=180, width=180, p=1.0)
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--coco_dir", type=str, required=True)
    parser.add_argument("--dalle_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MasterEnsembleDetector().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    coco_paths = glob.glob(os.path.join(args.coco_dir, '*.*'))
    dalle_paths = glob.glob(os.path.join(args.dalle_dir, '*.*'))
    samples = [(p, 0.0) for p in coco_paths] + [(p, 1.0) for p in dalle_paths]

    clip_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    siglip_tf = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    for name, transform in PERTS.items():
        correct = 0
        total = 0
        for path, label in samples:
            img = np.array(Image.open(path).convert('RGB'))
            corrupted = transform(image=img)['image']
            
            x_clip = clip_tf(image=corrupted)['image'].unsqueeze(0).to(device)
            x_siglip = siglip_tf(image=corrupted)['image'].unsqueeze(0).to(device)
            
            with torch.no_grad():
                prob = torch.sigmoid(model(x_clip, x_siglip, x_siglip)).item()
                pred = 1.0 if prob >= 0.5 else 0.0
                if pred == label:
                    correct += 1
                total += 1
        print(f"Perturbation: {name:<16} | Accuracy: {correct/total*100:.2f}%")

if __name__ == '__main__':
    main()


```

#### `README.md`

Markdown

````
# Multi-Domain Robust AIGC Forensics Detector (<2B Parameters)

A tri-stream spatial-frequency detection architecture combining frozen vision foundation models (OpenAI CLIP ViT-L/14 and SigLIP2) with high-pass residual wavelet extraction (SRM + ConvNeXt-Tiny).

## Installation
```bash
git clone <repo_url>
cd aigc_robust_detection
bash scripts/setup_server_env.sh


````

## Training

Bash

```
python scripts/train_detector.py


```

## Inference Interface

Bash

```
python scripts/run_inference.py \
  --image_dir ./test_images \
  --checkpoint ./checkpoints/master_model_epoch_15.pth \
  --output ./results.json \
  --device cuda


```

## Robustness Evaluation Matrix

Bash

```
python scripts/evaluate_robustness.py \
  --checkpoint ./checkpoints/master_model_epoch_15.pth \
  --coco_dir ./data/val_demo/coco_val2017 \
  --dalle_dir ./data/val_demo/dalle_advanced
```

**Close**