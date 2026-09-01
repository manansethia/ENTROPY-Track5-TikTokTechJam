#!/usr/bin/env python3
"""Unified Forensic Explainability & Diagnostic Attribution CLI & Engine.

Performs full-spectrum forensic diagnostics on input images:
1. Grad-CAM on ViT backbones (SigLIP, CLIP, DINOv2) & ConvNeXt-V2
2. Multi-Head ViT Attention Rollout across transformer depth
3. 2D FFT Frequency-Domain Power Spectrum & iFFT Spatial Anomaly Localization
4. Multiscale Edge & Boundary Residual Heatmaps (Sobel, Laplacian, SRM)
5. Patch-Level Localized Attribution Scores & Top Anomaly Ranking
6. Publication-grade 8-panel diagnostic dashboard export to reports/explainability/
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoProcessor, CLIPModel, ConvNextV2Model

from models.forensic_explainability import (
    CNNConvNeXtGradCAM,
    EdgeResidualExplainer,
    ForensicDiagnosticSuite,
    FrequencySpectralExplainer,
    PatchForensicScorer,
    ViTAttentionRollout,
    ViTGradCAM,
)
from models.quad_hybrid_detector import QuadHybridGatingHead
from models.srm_filters import SRMConvolution
from scripts.train_tri_hybrid_gating import DynamicGatingFusionHead


class UniversalForensicExplainer:
    """Full-Spectrum Forensic Explainer supporting Quad-Hybrid, Tri-Hybrid, and Standalone Backbones."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "checkpoints/quad_hybrid_v1/best_model.pt",
        siglip_dir: Optional[str] = None,
        clip_dir: Optional[str] = None,
        dinov2_dir: Optional[str] = None,
        convnext_dir: Optional[str] = None,
        device: str = "cuda",
    ):
        if torch.cuda.is_available() and device == "cuda":
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and device == "mps":
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        print(f"[UniversalForensicExplainer] Initializing on device: {self.device}")

        self.siglip_model = None
        self.clip_model = None
        self.dinov2_model = None
        self.convnext_model = None
        self.fusion_head = None
        self.model_type = "generic"

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

        # 1. Attempt loading SigLIP
        siglip_id = siglip_dir or "google/siglip-base-patch16-224"
        try:
            print(f"Loading SigLIP ({siglip_id}) in {dtype}...")
            self.siglip_model = AutoModel.from_pretrained(siglip_id, torch_dtype=dtype).to(self.device).eval()
        except Exception as e:
            print(f"[WARN] Could not load SigLIP from {siglip_id}: {e}")

        # 2. Attempt loading CLIP
        clip_id = clip_dir or "openai/clip-vit-base-patch32"
        try:
            print(f"Loading CLIP ({clip_id}) in {dtype}...")
            self.clip_model = CLIPModel.from_pretrained(clip_id, torch_dtype=dtype).to(self.device).eval()
        except Exception as e:
            print(f"[WARN] Could not load CLIP from {clip_id}: {e}")

        # 3. Attempt loading DINOv2
        if dinov2_dir and os.path.exists(dinov2_dir):
            try:
                print(f"Loading DINOv2 from {dinov2_dir} in {dtype}...")
                self.dinov2_model = AutoModel.from_pretrained(dinov2_dir, torch_dtype=dtype).to(self.device).eval()
            except Exception as e:
                print(f"[WARN] Could not load DINOv2: {e}")

        # 4. Attempt loading ConvNeXt-V2
        if convnext_dir and os.path.exists(convnext_dir):
            try:
                print(f"Loading ConvNeXt-V2 from {convnext_dir} in {dtype}...")
                self.convnext_model = ConvNextV2Model.from_pretrained(convnext_dir, torch_dtype=dtype).to(self.device).eval()
            except Exception as e:
                print(f"[WARN] Could not load ConvNeXt: {e}")

        # 5. Load Trained Checkpoint (Quad-Hybrid or Tri-Hybrid)
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                ckpt = torch.load(checkpoint_path, map_location=self.device)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                
                # Detect whether Quad-Hybrid or Tri-Hybrid
                if "proj_convnext.0.weight" in state_dict:
                    print(f"Detected Quad-Hybrid (4-stream) checkpoint: {checkpoint_path}")
                    self.fusion_head = QuadHybridGatingHead().to(self.device)
                    self.fusion_head.load_state_dict(state_dict)
                    self.fusion_head.eval()
                    self.model_type = "quad_hybrid"
                else:
                    print(f"Detected Tri-Hybrid (3-stream) checkpoint: {checkpoint_path}")
                    has_d = "proj_dinov2.0.weight" in state_dict
                    self.fusion_head = DynamicGatingFusionHead(has_dinov2=has_d).to(self.device)
                    self.fusion_head.load_state_dict(state_dict)
                    self.fusion_head.eval()
                    self.model_type = "tri_hybrid"
            except Exception as e:
                print(f"[WARN] Checkpoint loading note: {e}. Running in standalone attribution mode.")

        # 6. Preprocessing Normalizers
        self.siglip_norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.clip_norm = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        self.dino_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 7. Initialize Attribution Engines
        primary_vit = self.siglip_model or (self.clip_model.vision_model if self.clip_model else None) or self.dinov2_model
        
        self.vit_gradcam = ViTGradCAM(primary_vit, has_cls_token=(primary_vit == self.dinov2_model or primary_vit == (self.clip_model.vision_model if self.clip_model else None))) if primary_vit else None
        self.cnn_gradcam = CNNConvNeXtGradCAM(self.convnext_model) if self.convnext_model else None
        self.attention_rollout = ViTAttentionRollout(primary_vit) if primary_vit else None
        self.freq_explainer = FrequencySpectralExplainer(num_radial_bins=64)
        self.edge_explainer = EdgeResidualExplainer()
        self.patch_scorer = PatchForensicScorer(grid_size=(14, 14))

        self.suite = ForensicDiagnosticSuite(
            vit_gradcam=self.vit_gradcam,
            cnn_gradcam=self.cnn_gradcam,
            attention_rollout=self.attention_rollout,
            freq_explainer=self.freq_explainer,
            edge_explainer=self.edge_explainer,
            patch_scorer=self.patch_scorer,
        )

    def predict_and_explain(
        self,
        image_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Runs full inference and multi-signal forensic explainability pipeline."""
        raw_pil = Image.open(image_path).convert("RGB")
        orig_w, orig_h = raw_pil.size
        img_np = np.array(raw_pil)

        preprocess = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])
        t_raw = preprocess(raw_pil)

        prob_aigc = 0.5
        gates_list = []

        # 1. Run Gated Model Inference if models are loaded
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        if self.fusion_head is not None and self.siglip_model is not None and self.clip_model is not None:
            with torch.no_grad():
                t_sig = self.siglip_norm(t_raw).unsqueeze(0).to(device=self.device, dtype=dtype)
                t_clp = self.clip_norm(t_raw).unsqueeze(0).to(device=self.device, dtype=dtype)

                s_out = self.siglip_model(pixel_values=t_sig) if not hasattr(self.siglip_model, "vision_model") else self.siglip_model.vision_model(pixel_values=t_sig)
                s_feat = s_out.last_hidden_state[:, 0] if hasattr(s_out, "last_hidden_state") else s_out.pooler_output
                s_feat = (s_feat / s_feat.norm(dim=-1, keepdim=True)).float()

                c_out = self.clip_model.vision_model(pixel_values=t_clp)
                c_feat = c_out.last_hidden_state[:, 0] if hasattr(c_out, "last_hidden_state") else c_out.pooler_output
                c_feat = (c_feat / c_feat.norm(dim=-1, keepdim=True)).float()

                d_feat = torch.zeros(1, 1024, device=self.device, dtype=torch.float32)
                if self.dinov2_model is not None:
                    t_din = self.dino_norm(t_raw).unsqueeze(0).to(device=self.device, dtype=dtype)
                    d_out = self.dinov2_model(pixel_values=t_din)
                    d_feat = d_out.last_hidden_state[:, 0] if hasattr(d_out, "last_hidden_state") else d_out.pooler_output
                    d_feat = (d_feat / d_feat.norm(dim=-1, keepdim=True)).float()

                x_feat = torch.zeros(1, 768, device=self.device, dtype=torch.float32)
                if self.convnext_model is not None:
                    t_cnx = self.dino_norm(t_raw).unsqueeze(0).to(device=self.device, dtype=dtype)
                    x_out = self.convnext_model(pixel_values=t_cnx)
                    x_feat = x_out.last_hidden_state.mean(dim=[-2, -1])
                    x_feat = (x_feat / x_feat.norm(dim=-1, keepdim=True)).float()

                if self.model_type == "quad_hybrid":
                    logits, gates = self.fusion_head(s_feat, c_feat, d_feat, x_feat)
                else:
                    logits, gates = self.fusion_head(s_feat, c_feat, d_feat if self.fusion_head.has_dinov2 else None)

                prob_aigc = F.softmax(logits, dim=-1)[0, 1].item()
                gates_list = gates[0].cpu().numpy().tolist()

        # 2. Targeted forward functions for Grad-CAM
        def forward_siglip(x_tensor):
            if self.siglip_model is not None:
                x_in = x_tensor.to(dtype=dtype, device=self.device)
                s_o = self.siglip_model(pixel_values=x_in) if not hasattr(self.siglip_model, "vision_model") else self.siglip_model.vision_model(pixel_values=x_in)
                s_f = s_o.last_hidden_state[:, 0] if hasattr(s_o, "last_hidden_state") else s_o.pooler_output
                s_f = (s_f / s_f.norm(dim=-1, keepdim=True)).float()
                if self.fusion_head is not None and hasattr(self.fusion_head, "proj_siglip"):
                    p_s = self.fusion_head.proj_siglip(s_f)
                    return self.fusion_head.classifier(p_s)
                return s_f[:, :2] if s_f.shape[-1] >= 2 else s_f
            return x_tensor.mean(dim=(-2, -1))

        def forward_convnext(x_tensor):
            if self.convnext_model is not None:
                x_in = x_tensor.to(dtype=dtype, device=self.device)
                x_o = self.convnext_model(pixel_values=x_in)
                x_f = x_o.last_hidden_state.mean(dim=[-2, -1])
                x_f = (x_f / x_f.norm(dim=-1, keepdim=True)).float()
                if self.fusion_head is not None and hasattr(self.fusion_head, "proj_convnext"):
                    p_x = self.fusion_head.proj_convnext(x_f)
                    return self.fusion_head.classifier(p_x)
                return x_f[:, :2] if x_f.shape[-1] >= 2 else x_f
            return x_tensor.mean(dim=(-2, -1))

        # 3. Run Diagnostic Suite
        t_input = self.siglip_norm(t_raw).unsqueeze(0).to(device=self.device, dtype=dtype)
        report = self.suite.explain(
            image=img_np,
            input_tensor=t_input,
            forward_fn=forward_siglip,
            pred_prob_aigc=prob_aigc,
            model_gates=gates_list,
            output_path=output_path,
        )

        return report


def main():
    p = argparse.ArgumentParser(description="Full-Spectrum Forensic Explainability & Attribution Engine")
    p.add_argument("--image", required=True, help="Path to input test image or directory of images")
    p.add_argument("--checkpoint", default="checkpoints/quad_hybrid_v1/best_model.pt", help="Path to trained model checkpoint")
    p.add_argument("--siglip_dir", default=None, help="Directory/HF model ID for SigLIP")
    p.add_argument("--clip_dir", default=None, help="Directory/HF model ID for CLIP")
    p.add_argument("--dinov2_dir", default=None, help="Directory/HF model ID for DINOv2")
    p.add_argument("--convnext_dir", default=None, help="Directory/HF model ID for ConvNeXt-V2")
    p.add_argument("--output", default="reports/explainability/diagnosis.jpg", help="Output path for 8-panel diagnostic figure")
    p.add_argument("--output_json", default=None, help="Output path for structured JSON report")
    p.add_argument("--device", default="cuda", help="Target device (cuda, mps, cpu)")
    args = p.parse_args()

    explainer = UniversalForensicExplainer(
        checkpoint_path=args.checkpoint,
        siglip_dir=args.siglip_dir,
        clip_dir=args.clip_dir,
        dinov2_dir=args.dinov2_dir,
        convnext_dir=args.convnext_dir,
        device=args.device,
    )

    image_p = Path(args.image)
    if image_p.is_file():
        res = explainer.predict_and_explain(image_p, output_path=args.output)
        print("\n" + "=" * 60)
        print("FORENSIC DIAGNOSIS COMPLETE")
        print("=" * 60)
        print(f"Image:       {res['image_path']}")
        print(f"Verdict:     {res['verdict']} (Confidence: {res['prob_aigc']*100:.2f}%)")
        print(f"Spectral:    HF Energy Ratio: {res['spectral_metrics']['high_freq_energy_ratio']*100:.1f}%, Peak Z-Score: {res['spectral_metrics']['grid_peak_anomaly_score']:.2f}")
        print(f"Edge Score:  {res['edge_metrics']['edge_anomaly_score']:.3f}")
        print("Top Ranked Anomalous Patches:")
        for idx, patch in enumerate(res['top_anomalous_patches'][:3]):
            print(f"  #{idx+1} BBox {patch['bbox']}: Risk {patch['composite_risk']:.3f} [{patch['primary_anomaly_category']}]")
        print("=" * 60)

        if args.output_json:
            out_json = Path(args.output_json)
            out_json.parent.mkdir(parents=True, exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(f"Saved JSON report to {out_json}")
    elif image_p.is_dir():
        out_dir = Path(args.output).parent if Path(args.output).suffix else Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        img_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        files = [f for f in image_p.iterdir() if f.suffix.lower() in img_exts]
        print(f"Processing directory with {len(files)} images...")
        all_results = []
        for f in files:
            out_fig = out_dir / f"{f.stem}_diagnosis.jpg"
            r = explainer.predict_and_explain(f, output_path=out_fig)
            all_results.append(r)
        
        json_path = out_dir / "batch_diagnosis_summary.json"
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(all_results, jf, indent=2)
        print(f"Batch analysis complete. Summary saved to {json_path}")


if __name__ == "__main__":
    main()
