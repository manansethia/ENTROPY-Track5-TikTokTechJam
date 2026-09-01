import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .augmentations import training_augment


VALID_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


class AIGCDataset(Dataset):
    """Dataset loader with strict validation benchmark isolation guard and robust image handling."""

    def __init__(self, root_dir: str, cfg: dict, target_size: int = 224):
        # Strict validation & out-of-distribution benchmark isolation guard
        abs_root = os.path.abspath(root_dir)
        LOCKED_BENCHMARKS = [
            "validation_LOCKED", "validation_locked", "aigibench_eval",
            "synthbuster", "wildrf", "synthwildx", "vct2", "chameleon"
        ]
        for locked in LOCKED_BENCHMARKS:
            if locked in abs_root:
                raise RuntimeError(
                    f"CRITICAL LEAKAGE DETECTED: Attempted to train on locked evaluation benchmark: {abs_root}. "
                    "All benchmark and out-of-distribution test suites must remain strictly isolated from training."
                )

        self.items = []
        for label_name, label in [("real", 0.0), ("synthetic", 1.0)]:
            pattern = os.path.join(root_dir, label_name, "**", "*")
            for p in glob.glob(pattern, recursive=True):
                if p.lower().endswith(VALID_EXTS):
                    abs_p = os.path.abspath(p)
                    for locked in LOCKED_BENCHMARKS:
                        if locked in abs_p:
                            raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: File {p} is part of locked benchmark '{locked}'!")
                    self.items.append((p, label))

        if not self.items:
            raise RuntimeError(
                f"No images found in {root_dir}. Expected subdirectories '{root_dir}/real' and '{root_dir}/synthetic'."
            )

        self.cfg = cfg
        self.target_size = target_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        try:
            pil_img = Image.open(path).convert("RGB")
            image = np.array(pil_img)
        except Exception as e:
            # Fallback for corrupted images: create neutral canvas
            image = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)

        # Apply stochastic training augmentations
        if self.cfg and "augmentations" in self.cfg:
            image = training_augment(image, self.cfg)

        # Ensure spatial dimensions match target_size for reliable batch collation
        h, w = image.shape[:2]
        if h != self.target_size or w != self.target_size:
            image = cv2.resize(
                image, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA if (h > self.target_size or w > self.target_size) else cv2.INTER_LINEAR
            )

        tensor_img = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return tensor_img, torch.tensor(label, dtype=torch.float32)
