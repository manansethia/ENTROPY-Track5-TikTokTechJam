#!/usr/bin/env python3
"""Authoritative Master Protocol Section 8: Supervised Representation Probes.
Extracts frozen backbone features from Tier 3 (generic vision representations)
and Tier 4 (forensic handcrafted specialists), fits controlled linear probes
(0 = Real, 1 = AIGC), and evaluates across all 7 core adversarial conditions.
Saves authoritative benchmark to reports/supervised_representation_benchmark.json.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoModel,
    CLIPImageProcessor,
    CLIPModel,
)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(Path(".").resolve()))
MODELS_DIR = Path("/mnt/ai-storage/aigc_data/models")
DATA_DIR = Path("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k")
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calculate_ece(probs, labels, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper if i < n_bins - 1 else probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return round(float(ece), 4)


def apply_transformation(img_np: np.ndarray, transform_name: str) -> np.ndarray:
    if transform_name == "clean":
        return img_np
    elif transform_name == "jpeg30":
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, enc = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR), encode_param)
        dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
    elif transform_name == "blur2":
        return cv2.GaussianBlur(img_np, (7, 7), 2.0)
    elif transform_name == "resize0.25":
        h, w = img_np.shape[:2]
        small = cv2.resize(img_np, (max(16, int(w * 0.25)), max(16, int(h * 0.25))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    elif transform_name == "noise0.10":
        noise = np.random.normal(0, 0.10 * 255, img_np.shape).astype(np.float32)
        return np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif transform_name == "crop80":
        h, w = img_np.shape[:2]
        ch, cw = int(h * 0.8), int(w * 0.8)
        top, left = (h - ch) // 2, (w - cw) // 2
        crop = img_np[top : top + ch, left : left + cw]
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    elif transform_name == "color_jitter":
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 10) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.2, 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.9, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return img_np


class FeatureExtractorEngine:
    def __init__(self, name: str):
        self.name = name
        self.device = device
        self.vram_before = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
        self.vram_peak = self.vram_before
        self.param_count = 0
        self._load_backbone()

    def _load_backbone(self):
        torch.cuda.reset_peak_memory_stats()
        if self.name == "DINOv2-Registers":
            p = MODELS_DIR / "dinov2_registers_large"
            self.proc = AutoImageProcessor.from_pretrained(str(p))
            self.model = AutoModel.from_pretrained(str(p)).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 1024
        elif self.name == "EVA-02-Large-448":
            p = MODELS_DIR / "eva02_large_patch14_448"
            self.proc = AutoImageProcessor.from_pretrained(str(p))
            self.model = AutoModel.from_pretrained(str(p)).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 1024
        elif self.name == "ConvNeXt-V2":
            p = MODELS_DIR / "convnextv2_tiny"
            self.proc = AutoImageProcessor.from_pretrained(str(p))
            self.model = AutoModel.from_pretrained(str(p)).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 768
        elif self.name == "CLIP-ViT-L":
            p = MODELS_DIR / "clip_vitl14"
            self.proc = CLIPImageProcessor.from_pretrained(str(p))
            self.model = CLIPModel.from_pretrained(str(p)).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 768
        elif self.name == "SigLIP-SO400M":
            p = MODELS_DIR / "siglip_so400m_224"
            self.proc = AutoProcessor.from_pretrained(str(p))
            self.model = AutoModel.from_pretrained(str(p)).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 1152
        elif self.name == "2D-FFT-Spectral":
            from models.fft_spectral_detector import FFTSpectralFeatureExtractor
            self.model = FFTSpectralFeatureExtractor(num_radial_bins=64).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 201
        elif self.name == "SRM-DWT-Wavelet":
            from models.srm_filters import WaveletResidualBlock
            self.model = WaveletResidualBlock().eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 36
        elif self.name == "Edge-Specialist":
            from models.edge_artifact_detector import EdgeArtifactFeatureExtractor
            self.model = EdgeArtifactFeatureExtractor(out_dim=256).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 256
        elif self.name == "Patch-MIL":
            from models.patch_mil_expert import GatedAttentionMIL
            self.model = GatedAttentionMIL(in_dim=768, hidden_dim=256).eval().to(self.device)
            self.param_count = sum(p.numel() for p in self.model.parameters())
            self.feat_dim = 768

    @torch.no_grad()
    def extract_features(self, images_np_list: list) -> np.ndarray:
        feats_list = []
        bs = 32
        for i in range(0, len(images_np_list), bs):
            batch_np = images_np_list[i : i + bs]
            
            if self.name in ["DINOv2-Registers", "EVA-02-Large-448", "ConvNeXt-V2"]:
                pils = [Image.fromarray(im) for im in batch_np]
                inputs = self.proc(images=pils, return_tensors="pt").to(self.device)
                out = self.model(**inputs)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    feat = out.pooler_output
                elif hasattr(out, "last_hidden_state"):
                    feat = out.last_hidden_state.mean(dim=[-2, -1]) if out.last_hidden_state.ndim == 4 else out.last_hidden_state[:, 0]
                else:
                    raw = out[0] if isinstance(out, tuple) else out
                    feat = raw.mean(dim=[-2, -1]) if raw.ndim == 4 else raw[:, 0]
                feats_list.append(feat.cpu().numpy().reshape(len(batch_np), -1))

            elif self.name == "CLIP-ViT-L":
                pils = [Image.fromarray(im) for im in batch_np]
                inputs = self.proc(images=pils, return_tensors="pt").to(self.device)
                feat = self.model.get_image_features(**inputs)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    feat = feat.pooler_output
                elif hasattr(feat, "last_hidden_state"):
                    feat = feat.last_hidden_state[:, 0]
                elif hasattr(feat, "image_embeds"):
                    feat = feat.image_embeds
                elif isinstance(feat, tuple):
                    feat = feat[0]
                feats_list.append(feat.cpu().numpy().reshape(len(batch_np), -1))

            elif self.name == "SigLIP-SO400M":
                pils = [Image.fromarray(im) for im in batch_np]
                inputs = self.proc(images=pils, return_tensors="pt").to(self.device)
                if hasattr(self.model, "get_image_features"):
                    feat = self.model.get_image_features(**inputs)
                else:
                    feat = self.model.vision_model(**inputs) if hasattr(self.model, "vision_model") else self.model(**inputs)
                if hasattr(feat, "pooler_output") and feat.pooler_output is not None:
                    feat = feat.pooler_output
                elif hasattr(feat, "last_hidden_state"):
                    feat = feat.last_hidden_state[:, 0]
                elif hasattr(feat, "image_embeds"):
                    feat = feat.image_embeds
                elif isinstance(feat, tuple):
                    feat = feat[0]
                feats_list.append(feat.cpu().numpy().reshape(len(batch_np), -1))

            elif self.name == "2D-FFT-Spectral":
                tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in batch_np]
                tensors = [F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False) for t in tensors]
                batch_t = torch.cat(tensors, dim=0).to(self.device)
                spec_feats = self.model(batch_t)
                feats_list.append(spec_feats.cpu().numpy().reshape(len(batch_np), -1))

            elif self.name == "SRM-DWT-Wavelet":
                tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in batch_np]
                tensors = [F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False) for t in tensors]
                batch_t = torch.cat(tensors, dim=0).to(self.device)
                srm_map = self.model(batch_t)
                srm_mean = srm_map.mean(dim=[-2, -1])
                srm_std = srm_map.std(dim=[-2, -1])
                srm_max = srm_map.amax(dim=[-2, -1])
                srm_min = srm_map.amin(dim=[-2, -1])
                srm_vec = torch.cat([srm_mean, srm_std, srm_max, srm_min], dim=1)
                feats_list.append(srm_vec.cpu().numpy().reshape(len(batch_np), -1))

            elif self.name == "Edge-Specialist":
                tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in batch_np]
                tensors = [F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False) for t in tensors]
                batch_t = torch.cat(tensors, dim=0).to(self.device)
                edge_feats = self.model(batch_t)
                feats_list.append(edge_feats.cpu().numpy().reshape(len(batch_np), -1))

            elif self.name == "Patch-MIL":
                tensors = [torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 for img in batch_np]
                tensors = [F.interpolate(t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False) for t in tensors]
                batch_t = torch.cat(tensors, dim=0).to(self.device)
                # Extract 4 spatial quadrant crops as local instances
                c1 = batch_t[:, :, :128, :128].reshape(len(batch_np), 3, -1).mean(dim=-1)
                c2 = batch_t[:, :, :128, 128:].reshape(len(batch_np), 3, -1).mean(dim=-1)
                c3 = batch_t[:, :, 128:, :128].reshape(len(batch_np), 3, -1).mean(dim=-1)
                c4 = batch_t[:, :, 128:, 128:].reshape(len(batch_np), 3, -1).mean(dim=-1)
                patches_raw = torch.stack([c1, c2, c3, c4], dim=1) # B x 4 x 3
                patches_proj = F.pad(patches_raw, (0, 768 - 3)) # B x 4 x 768
                bag_feat, _ = self.model(patches_proj)
                feats_list.append(bag_feat.cpu().numpy().reshape(len(batch_np), -1))

        if torch.cuda.is_available():
            self.vram_peak = torch.cuda.max_memory_allocated() / (1024**3)

        return np.concatenate(feats_list, axis=0)

    def cleanup(self):
        del self.model
        if hasattr(self, "proc"):
            del self.proc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_supervised_probes():
    print("=== Launching Master Protocol Section 8: Supervised Probes ===")
    real_dir = DATA_DIR / "real"
    fake_dir = DATA_DIR / "synthetic"
    
    real_files = sorted([os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith((".jpg", ".png"))])[:200]
    fake_files = sorted([os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith((".jpg", ".png"))])[:200]
    
    # 75% Train (150 Real, 150 Fake = 300), 25% Validation (50 Real, 50 Fake = 100)
    train_paths = real_files[:150] + fake_files[:150]
    train_labels = np.array([0.0] * 150 + [1.0] * 150)
    
    val_paths = real_files[150:200] + fake_files[150:200]
    val_labels = np.array([0.0] * 50 + [1.0] * 50)
    
    print(f"--> Dataset Split: {len(train_paths)} Train (150R/150F) | {len(val_paths)} Val (50R/50F)")

    # Load clean images
    train_imgs_clean = []
    for p in train_paths:
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))
        train_imgs_clean.append(im)

    val_imgs_clean = []
    for p in val_paths:
        im = cv2.imread(p)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (224, 224))
        val_imgs_clean.append(im)

    transformations = ["clean", "jpeg30", "blur2", "resize0.25", "noise0.10", "crop80", "color_jitter"]
    
    candidate_experts = [
        "CLIP-ViT-L",
        "SigLIP-SO400M",
        "DINOv2-Registers",
        "EVA-02-Large-448",
        "ConvNeXt-V2",
        "2D-FFT-Spectral",
        "SRM-DWT-Wavelet",
        "Edge-Specialist",
        "Patch-MIL",
    ]

    benchmark_results = {
        "metadata": {
            "protocol_section": "Section 8 (Supervised Representation Probes)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_samples": len(train_labels),
            "val_samples": len(val_labels),
            "probe_classifier": "LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)",
            "label_convention": "0 = Real, 1 = AIGC (Output: P(AIGC))",
        },
        "supervised_probe_matrix": {},
        "vram_and_latency_audit": {},
    }

    all_val_clean_preds = {}
    all_val_cond_preds = {}

    for exp_name in candidate_experts:
        print(f"\n---> Probing Expert: {exp_name}")
        t0 = time.time()
        extractor = FeatureExtractorEngine(exp_name)

        # 1. Extract Train Features (Clean)
        X_train = extractor.extract_features(train_imgs_clean)
        
        # Normalize features
        mean_v = np.mean(X_train, axis=0, keepdims=True)
        std_v = np.std(X_train, axis=0, keepdims=True) + 1e-6
        X_train_norm = (X_train - mean_v) / std_v

        # 2. Fit Controlled Linear Probe
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        clf.fit(X_train_norm, train_labels)

        # 3. Evaluate across 7 Transformations on Validation Split
        cond_aucs = {}
        val_clean_preds = None

        cond_preds = {}
        for t_name in transformations:
            transformed_val_imgs = [apply_transformation(img, t_name) for img in val_imgs_clean]
            X_val = extractor.extract_features(transformed_val_imgs)
            X_val_norm = (X_val - mean_v) / std_v
            
            # P(AIGC) probability
            probs = clf.predict_proba(X_val_norm)[:, 1]
            auc = roc_auc_score(val_labels, probs) if len(np.unique(val_labels)) > 1 else 0.5
            cond_aucs[t_name] = round(float(auc), 4)
            cond_preds[t_name] = probs.tolist()

            if t_name == "clean":
                val_clean_preds = probs

        t_elapsed = time.time() - t0
        latency_ms_per_sample = round((t_elapsed / (len(val_paths) * len(transformations))) * 1000.0, 2)
        all_val_clean_preds[exp_name] = val_clean_preds
        all_val_cond_preds[exp_name] = cond_preds

        # Metrics computation
        all_aucs = [cond_aucs[t] for t in transformations]
        clean_auc = cond_aucs["clean"]
        worst_auc = min(all_aucs)
        mean_ri = round(float(np.mean(all_aucs)), 4)
        degrad = round(float(clean_auc - worst_auc), 4)
        
        auprc = round(float(average_precision_score(val_labels, val_clean_preds)), 4)
        brier = round(float(brier_score_loss(val_labels, val_clean_preds)), 4)
        ece = calculate_ece(val_clean_preds, val_labels)

        bin_preds = (val_clean_preds >= 0.5).astype(float)
        tn, fp, fn, tp = confusion_matrix(val_labels, bin_preds, labels=[0, 1]).ravel()
        fpr = round(float(fp / (fp + tn)), 4) if (fp + tn) > 0 else 0.0
        fnr = round(float(fn / (fn + tp)), 4) if (fn + tp) > 0 else 0.0
        acc = round(float((tp + tn) / len(val_labels)), 4)

        cond_aucs["mean_robustness_index"] = mean_ri
        cond_aucs["worst_case_auroc"] = worst_auc
        cond_aucs["robustness_degradation"] = degrad
        cond_aucs["clean_auprc"] = auprc
        cond_aucs["brier_score"] = brier
        cond_aucs["expected_calibration_error"] = ece
        cond_aucs["clean_accuracy"] = acc
        cond_aucs["clean_fpr"] = fpr
        cond_aucs["clean_fnr"] = fnr
        cond_aucs["feature_dimension"] = extractor.feat_dim
        cond_aucs["backbone_parameters"] = extractor.param_count

        benchmark_results["supervised_probe_matrix"][exp_name] = cond_aucs
        benchmark_results["vram_and_latency_audit"][exp_name] = {
            "peak_vram_gb": round(extractor.vram_peak, 3),
            "latency_ms_per_sample": latency_ms_per_sample,
            "parameter_count": extractor.param_count,
        }

        print(f"--> {exp_name} | Clean AUROC: {clean_auc} | Mean RI: {mean_ri} | Worst: {worst_auc} | Degrad: {degrad} | FPR: {fpr} | Latency: {latency_ms_per_sample}ms")
        extractor.cleanup()

    # Save validation labels and prediction vectors for downstream error analysis
    benchmark_results["val_labels"] = val_labels.tolist()
    benchmark_results["all_val_clean_predictions"] = {k: v.tolist() for k, v in all_val_clean_preds.items()}
    benchmark_results["all_val_condition_predictions"] = all_val_cond_preds

    # Save Authoritative Benchmark Report
    out_file = REPORTS_DIR / "supervised_representation_benchmark.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_results, f, indent=2)

    print(f"\nAuthoritative Supervised Representation Benchmark written to {out_file}!")


if __name__ == "__main__":
    run_supervised_probes()
