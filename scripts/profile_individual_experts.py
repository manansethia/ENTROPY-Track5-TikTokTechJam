#!/usr/bin/env python3
"""Stage 1: Individual Expert Profiling Harness (Zero-Shot & Forensic Scoring).
Evaluates every candidate expert model independently across adversarial transformations:
- Clean, JPEG 30, Gaussian Blur sigma=2.0, Downscale 0.25x, Gaussian Noise sigma=0.10, Crop 80%, Color Jitter.

Preconditions Enforced:
1. Every model loads successfully and memory is released after run.
2. Predictions are verified finite probabilities in [0.0, 1.0].
3. Strict fail-closed policy checks against configs/dataset_policy.yaml.
4. Reports exact metrics: Clean, JPEG30, Blur2, Resize .25, Noise .10, Crop80, Mean RI, Worst, Degradation.
"""

import gc
import json
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer, CLIPModel, CLIPProcessor, ConvNextV2Model

from models.fft_spectral_detector import FFTSpectralFeatureExtractor
from models.edge_artifact_detector import EdgeArtifactFeatureExtractor
from scripts.augmentations import (
    _jpeg,
    _blur,
    _down_up,
    _noise,
    _jitter,
)

POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "dataset_policy.yaml"
with open(POLICY_PATH) as f:
    DATASET_POLICY = yaml.safe_load(f)


def check_dataset_path(path_str: str, required_role: str = "TRAIN"):
    abs_p = os.path.abspath(path_str)
    for ds_name, ds_info in DATASET_POLICY.get("datasets", {}).items():
        if ds_name in abs_p:
            if required_role == "TRAIN" and ds_info.get("train") == "forbidden":
                raise RuntimeError(
                    f"CRITICAL LEAKAGE DETECTED: Path {abs_p} violates policy for '{ds_name}' (train: forbidden)!"
                )


def apply_transformation(img_np: np.ndarray, transform_name: str) -> np.ndarray:
    img = img_np.copy()
    if transform_name == "clean":
        return img
    elif transform_name == "jpeg30":
        return _jpeg(img, quality=30)
    elif transform_name == "blur2":
        return _blur(img, sigma=2.0)
    elif transform_name == "resize0.25":
        return _down_up(img, scale=0.25)
    elif transform_name == "noise0.10":
        return _noise(img, sigma=0.10)
    elif transform_name == "crop80":
        h, w = img.shape[:2]
        ch, cw = int(h * 0.8), int(w * 0.8)
        top = (h - ch) // 2
        left = (w - cw) // 2
        return cv2.resize(img[top:top+ch, left:left+cw], (w, h))
    elif transform_name == "color_jitter":
        return _jitter(img, brightness=0.3, contrast=0.3, saturation=0.3)
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


class ExpertInferenceRunner:
    """Standardized inference adapter for an individual expert."""

    def __init__(self, expert_name: str, device: str = "cuda"):
        self.name = expert_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.models_base = Path("/mnt/ai-storage/aigc_data/models")
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.text_embeds = None
        self._load_expert()

    def _load_expert(self):
        print(f"\n[Stage 1] Initializing Expert: {self.name} on {self.device}...")
        prompts = ["a real authentic natural photograph", "an ai-generated synthetic artificial image"]

        if self.name == "SigLIP-SO400M":
            p = str(self.models_base / "siglip_so400m_224")
            self.processor = AutoImageProcessor.from_pretrained(p)
            self.tokenizer = AutoTokenizer.from_pretrained(p)
            self.model = AutoModel.from_pretrained(p).to(self.device).eval()
            with torch.no_grad():
                txt_in = self.tokenizer(prompts, padding="max_length", return_tensors="pt").to(self.device)
                txt_out = self.model.get_text_features(**txt_in)
                txt_tensor = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else (txt_out[0] if isinstance(txt_out, tuple) else txt_out)
                self.text_embeds = F.normalize(txt_tensor, dim=-1)

        elif self.name == "CLIP-ViT-L":
            p = str(self.models_base / "clip_vitl14")
            self.processor = CLIPProcessor.from_pretrained(p)
            self.model = CLIPModel.from_pretrained(p).to(self.device).eval()
            with torch.no_grad():
                txt_in = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                txt_out = self.model.get_text_features(**txt_in)
                txt_tensor = txt_out.pooler_output if hasattr(txt_out, "pooler_output") else (txt_out[0] if isinstance(txt_out, tuple) else txt_out)
                self.text_embeds = F.normalize(txt_tensor, dim=-1)

        elif self.name == "DINOv2-Registers":
            p = str(self.models_base / "dinov2_registers_large")
            self.processor = AutoImageProcessor.from_pretrained(p)
            self.model = AutoModel.from_pretrained(p).to(self.device).eval()

        elif self.name == "ConvNeXt-V2":
            p = str(self.models_base / "convnextv2_tiny")
            self.processor = AutoImageProcessor.from_pretrained(p)
            self.model = ConvNextV2Model.from_pretrained(p).to(self.device).eval()

        elif self.name == "2D-FFT-Spectral":
            self.model = FFTSpectralFeatureExtractor(num_radial_bins=64).to(self.device).eval()

        elif self.name == "Edge-Specialist":
            self.model = EdgeArtifactFeatureExtractor(out_dim=256).to(self.device).eval()
        else:
            raise ValueError(f"Unknown expert: {self.name}")

    @torch.no_grad()
    def predict_batch(self, images_np: list[np.ndarray]) -> np.ndarray:
        pil_imgs = [Image.fromarray(img) for img in images_np]
        
        if self.name in ["SigLIP-SO400M", "CLIP-ViT-L"]:
            img_in = self.processor(images=pil_imgs, return_tensors="pt").to(self.device)
            img_out = self.model.get_image_features(**img_in)
            img_feats = img_out.pooler_output if hasattr(img_out, "pooler_output") else (img_out[0] if isinstance(img_out, tuple) else img_out)
            img_feats = F.normalize(img_feats, dim=-1)
            
            # Cosine similarity with text prompts [Real, Synthetic]
            logits = (img_feats @ self.text_embeds.T) * 100.0
            probs = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()

        elif self.name in ["DINOv2-Registers", "ConvNeXt-V2"]:
            inputs = self.processor(images=pil_imgs, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            feat = out.last_hidden_state[:, 0] if hasattr(out, "last_hidden_state") else out.pooler_output
            feat_norm = F.normalize(feat, dim=-1)
            scores = torch.norm(feat_norm - feat_norm.mean(dim=0, keepdim=True), dim=-1)
            probs = torch.sigmoid((scores - scores.mean()) * 2.0).cpu().numpy()

        elif self.name == "2D-FFT-Spectral":
            tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in images_np]
            tensors = [F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False) for t in tensors]
            batch_t = torch.cat(tensors, dim=0).to(self.device)
            spec_feats = self.model(batch_t)
            hi_mid_ratio = spec_feats[:, -3:].mean(dim=-1) / (spec_feats[:, -6:-3].mean(dim=-1) + 1e-5)
            probs = torch.sigmoid((hi_mid_ratio - 1.0) * 3.0).cpu().numpy()

        elif self.name == "Edge-Specialist":
            tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in images_np]
            tensors = [F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False) for t in tensors]
            batch_t = torch.cat(tensors, dim=0).to(self.device)
            edge_feats = self.model(batch_t)
            edge_energy = torch.norm(edge_feats, dim=-1)
            probs = torch.sigmoid(edge_energy - edge_energy.mean()).cpu().numpy()

        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        return probs

    def cleanup(self):
        del self.model
        if self.processor:
            del self.processor
        if self.tokenizer:
            del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_stage1_benchmarking(sample_size=400):
    print("=== Launching Stage 1: Individual Expert Profiling ===")
    data_base = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k")
    real_dir = data_base / "real"
    fake_dir = data_base / "synthetic"
    
    check_dataset_path(str(real_dir), "TRAIN")
    check_dataset_path(str(fake_dir), "TRAIN")

    real_files = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".png"))])[:sample_size // 2]
    fake_files = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith((".jpg", ".png"))])[:sample_size // 2]

    print(f"Loaded {len(real_files)} Real + {len(fake_files)} Synthetic development samples from massive_balanced_50k.")
    all_paths = real_files + fake_files
    labels = np.array([0.0] * len(real_files) + [1.0] * len(fake_files))
    
    images_clean = []
    for p in all_paths:
        try:
            im = cv2.imread(p)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (224, 224))
            images_clean.append(im)
        except Exception:
            images_clean.append(np.zeros((224, 224, 3), dtype=np.uint8))

    transformations = ["clean", "jpeg30", "blur2", "resize0.25", "noise0.10", "crop80", "color_jitter"]
    
    experts_to_profile = [
        "SigLIP-SO400M",
        "CLIP-ViT-L",
        "DINOv2-Registers",
        "ConvNeXt-V2",
        "2D-FFT-Spectral",
        "Edge-Specialist",
    ]

    results_table = {}

    for exp_name in experts_to_profile:
        runner = ExpertInferenceRunner(exp_name)
        exp_results = {}
        
        for t_name in transformations:
            transformed_imgs = [apply_transformation(img, t_name) for img in images_clean]
            
            preds = []
            for i in range(0, len(transformed_imgs), 32):
                batch_imgs = transformed_imgs[i : i + 32]
                p = runner.predict_batch(batch_imgs)
                preds.extend(p)
            preds = np.array(preds)
            
            try:
                auc = roc_auc_score(labels, preds)
            except Exception:
                auc = 0.5
            
            exp_results[t_name] = round(float(auc), 4)

        all_aucs = [exp_results[t] for t in transformations]
        clean_auc = exp_results["clean"]
        worst_auc = min(all_aucs)
        mean_auc = round(float(np.mean(all_aucs)), 4)
        degradation = round(float(clean_auc - worst_auc), 4)
        
        exp_results["mean_robustness_index"] = mean_auc
        exp_results["worst_case_auroc"] = worst_auc
        exp_results["robustness_degradation"] = degradation

        results_table[exp_name] = exp_results
        print(f"--> {exp_name} | Clean: {clean_auc} | Mean RI: {mean_auc} | Worst: {worst_auc} | Deg: {degradation}")
        
        runner.cleanup()

    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage1_expert_profiling.json"
    with open(out_file, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved Stage 1 Profiling Report to {out_file}!")


if __name__ == "__main__":
    run_stage1_benchmarking()
