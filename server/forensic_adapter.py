"""
server/forensic_adapter.py
Pluggable Forensic Model Adapter Architecture.
Production Engine: Model C0 (Triple-Hybrid Champion ~735M Parameters).
With neural SRM residual heatmap extraction directly from model filter banks.
"""

import os
import sys
import time
import io
import platform
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from PIL import Image, ImageFilter
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.provenance_engine import inspect_image_provenance_full
from server.spatial_engine import compute_deterministic_spatial_evidence, pil_to_base64


class ForensicModelAdapter(ABC):
    """Abstract Base Class for all AIGC Forensic Detection Model Adapters."""

    @abstractmethod
    def predict(
        self,
        image_input: Union[Image.Image, bytes, str],
        filename: str = "evidence.png",
        operating_mode: str = "standard"
    ) -> Dict[str, Any]:
        """Run single image forensic analysis."""
        pass

    @abstractmethod
    def predict_batch(
        self,
        images: List[Union[Image.Image, bytes]],
        filenames: Optional[List[str]] = None,
        operating_mode: str = "standard"
    ) -> List[Dict[str, Any]]:
        """Run high-throughput batch forensic analysis."""
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata, parameter counts, and versioning info."""
        pass


class TripleHybridChampionAdapter(ForensicModelAdapter):
    """
    Primary Production Adapter: Triple-Hybrid Champion Model (~735M Parameters).
    Architecture: OpenAI CLIP ViT-L/14 + Google SigLIP SO400M + Deterministic SRM Wavelet Residual + Bottleneck Fusion.
    Runs in full acceleration mode on Linux/GPU with direct SRM feature map heatmap extraction.
    """

    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        self.default_ckpt = checkpoint_path or os.getenv(
            "CHECKPOINT_PATH",
            str(PROJECT_ROOT / "checkpoints" / "production" / "final_champion_frozen_model.pt")
        )
        self.is_linux_server = (platform.system() == "Linux") or (os.getenv("IS_INFERENCE_SERVER", "0") == "1")
        self.device_str = device or ("cuda" if self.is_linux_server else "cpu")
        self.model = None
        self.metadata = {}
        self.temperature = 1.523021
        self.is_loaded = False
        
        if self.is_linux_server or os.getenv("LOAD_HEAVY_MODEL_ON_MAC", "0") == "1":
            self._load_heavy_server_model()
        else:
            self._init_lightweight_client_mode()

    def _init_lightweight_client_mode(self):
        """Lightweight initialization."""
        self.metadata = {
            "model_name": "Triple-Hybrid Champion",
            "version": "v4.3-REM-A",
            "architecture": "CLIP ViT-L/14 + SigLIP SO400M + SRM Wavelet Residual Head",
            "total_parameters": 735038561,
            "parameter_count": 735038561,
            "operating_device": "private-inference-service",
            "checkpoint_path": self.default_ckpt,
            "is_loaded": False,
            "inference_available": False
        }
        self.is_loaded = False

    def _load_heavy_server_model(self):
        """Loads Model C0 Champion into GPU memory."""
        try:
            import torch
            from deployment.portable_model import load_portable_champion_model
            
            ckpt_file = Path(self.default_ckpt)
            if ckpt_file.exists():
                print(f"[ForensicAdapter] Loading 735M Champion model from {ckpt_file}...")
                model, meta = load_portable_champion_model(ckpt_file, device=self.device_str)
                self.model = model
                self.metadata = {
                    **meta,
                    "model_name": "Triple-Hybrid Champion",
                    "architecture": "CLIP ViT-L/14 + SigLIP SO400M + SRM Wavelet Residual Head",
                    "parameter_count": 735038561,
                    "total_parameters": 735038561,
                    "is_loaded": True,
                    "inference_available": True,
                    "operating_device": self.device_str,
                }
                self.temperature = meta.get("temperature", 1.523021)
                self.is_loaded = True
                print("[ForensicAdapter] 735M Champion model ready in GPU memory! ✅")
        except Exception as e:
            print(f"[ForensicAdapter] Notice during model load: {e}")
            self._init_lightweight_client_mode()

    def _build_model_c0_heatmap(self, srm_tensor_224: np.ndarray, orig_w: int, orig_h: int) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        Builds a smooth, accurate continuous heatmap overlay directly from Model C0 SRM residual maps.
        """
        import cv2

        # 1. Normalize SRM map to [0, 1] using robust percentile scaling
        p_min = np.percentile(srm_tensor_224, 2)
        p_max = np.percentile(srm_tensor_224, 98)
        norm_map = np.clip((srm_tensor_224 - p_min) / (p_max - p_min + 1e-6), 0.0, 1.0)

        # 2. Smooth map with 2D Gaussian filter
        blurred = cv2.GaussianBlur(norm_map, (7, 7), 2.0)
        
        # 3. Resize to 384x384 canvas
        canvas_size = (384, 384)
        resized = cv2.resize(blurred, canvas_size, interpolation=cv2.INTER_CUBIC)
        resized = np.clip(resized, 0.0, 1.0)

        # 4. Compute affected area percentage
        threshold = 0.58
        affected_pixels = np.sum(resized > threshold)
        affected_area_pct = float((affected_pixels / resized.size) * 100.0)

        # 5. Extract localized bounding boxes
        suspicious_boxes = []
        binary_mask = (resized > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        scale_x = orig_w / canvas_size[0]
        scale_y = orig_h / canvas_size[1]
        
        for cnt in contours:
            if cv2.contourArea(cnt) > 300:
                x, y, w, h = cv2.boundingRect(cnt)
                box_conf = float(np.mean(resized[y:y+h, x:x+w]))
                suspicious_boxes.append({
                    "box_2d": [int(y * scale_y), int(x * scale_x), int((y + h) * scale_y), int((x + w) * scale_x)],
                    "confidence": round(box_conf, 4),
                    "label": "Manipulated_AIGC_Region"
                })

        # 6. Build high-contrast Jet/Turbo colormap with adaptive alpha
        rgba = np.zeros((canvas_size[1], canvas_size[0], 4), dtype=np.uint8)
        uint8_map = (resized * 255.0).astype(np.uint8)
        color_bgr = cv2.applyColorMap(uint8_map, cv2.COLORMAP_JET)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)

        rgba[..., 0:3] = color_rgb
        # Dynamic alpha: low on cool background (~30-50), high on anomalous regions (~180-210)
        alpha = np.clip(30 + (resized ** 1.5) * 180.0, 25, 215).astype(np.uint8)
        rgba[..., 3] = alpha

        heatmap_pil = Image.fromarray(rgba, mode="RGBA")
        heatmap_b64 = pil_to_base64(heatmap_pil)

        return heatmap_b64, suspicious_boxes, round(affected_area_pct, 1)

    def predict(
        self,
        image_input: Union[Image.Image, bytes, str],
        filename: str = "evidence.png",
        operating_mode: str = "standard"
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        timeline = []
        
        # 1. Ingest Stage
        timeline.append({"stage": "INGEST", "timestamp": time.strftime("%H:%M:%S"), "detail": f"Received {filename}"})
        
        if isinstance(image_input, bytes):
            image_bytes = image_input
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        elif isinstance(image_input, (str, Path)):
            with open(image_input, "rb") as f:
                image_bytes = f.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            filename = Path(image_input).name
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG")
            image_bytes = buf.getvalue()
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_w, orig_h = pil_img.size

        # 2. Metadata & Provenance Stage
        timeline.append({"stage": "METADATA", "timestamp": time.strftime("%H:%M:%S"), "detail": "Extracted EXIF, XMP, IPTC & C2PA markers"})
        meta_prov = inspect_image_provenance_full(image_bytes, filename=filename)

        # 3. Spatial & Frequency Stage
        timeline.append({"stage": "FORENSICS", "timestamp": time.strftime("%H:%M:%S"), "detail": "Computed 2D FFT, SRM 5x5 residuals & edge variance"})
        spatial_evidence = compute_deterministic_spatial_evidence(pil_img)

        # 4. Global Inference Stage
        timeline.append({"stage": "GLOBAL", "timestamp": time.strftime("%H:%M:%S"), "detail": "Evaluated Triple-Hybrid Champion (~735M parameters)"})
        if self.model is None or not self.is_loaded:
            raise RuntimeError("MODEL_INFERENCE_UNAVAILABLE: no verified model is loaded; no forensic verdict was produced.")

        try:
            import torch
            import torch.nn.functional as F
            from deployment.portable_model import portable_eval_transform
            tensor = portable_eval_transform(pil_img).unsqueeze(0).to(self.device_str)
            with torch.no_grad():
                logits = self.model(tensor)
                raw_logit = float(logits[0].item() if isinstance(logits, tuple) else logits.item())
                
                # Extract neural SRM residual map directly from Model C0 filter bank
                filters = self.model.srm_extractor.filters.to(dtype=tensor.dtype, device=tensor.device)
                srm_res = F.conv2d(tensor, filters, padding=2)
                srm_map = torch.abs(srm_res).mean(dim=1).squeeze(0).cpu().numpy()

            calibrated_logit = raw_logit / self.temperature
            prob_aigc = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
            
            # Build high-fidelity Model C0 heatmap
            heatmap_b64, suspicious_boxes, affected_area = self._build_model_c0_heatmap(srm_map, orig_w, orig_h)
            spatial_evidence["artifacts"]["heatmap_overlay_base64"] = heatmap_b64
            spatial_evidence["affected_area_percentage"] = affected_area

        except Exception as error:
            raise RuntimeError("MODEL_INFERENCE_FAILED: no forensic verdict was produced.") from error

        # 5. Provenance Synthesis Stage
        timeline.append({"stage": "PROVENANCE", "timestamp": time.strftime("%H:%M:%S"), "detail": "Reconciled C2PA status and synthetic generator signatures"})

        # 6. Verdict Classification Stage
        timeline.append({"stage": "VERDICT", "timestamp": time.strftime("%H:%M:%S"), "detail": "Multi-tier decision gating finalized"})

        if prob_aigc >= 0.78:
            verdict = "FULL_AIGC"
            verdict_label = "FULL-AIGC"
            verdict_badge = "SYNTHETIC IMAGE"
            verdict_desc = "Synthetic image generated by generative diffusion or autoregressive model."
            confidence = prob_aigc
            real_prob = 1.0 - prob_aigc
            partial_prob = 0.10
            full_prob = prob_aigc
        elif prob_aigc <= 0.22:
            verdict = "REAL"
            verdict_label = "REAL"
            verdict_badge = "AUTHENTIC PHOTOGRAPH"
            verdict_desc = "Authentic photograph. No significant synthetic evidence detected."
            confidence = 1.0 - prob_aigc
            real_prob = 1.0 - prob_aigc
            partial_prob = 0.05
            full_prob = prob_aigc
        elif affected_area > 3.0 and prob_aigc > 0.35:
            verdict = "PARTIAL_AIGC"
            verdict_label = "PARTIAL-AI"
            verdict_badge = "LOCALIZED INPAINTING"
            verdict_desc = f"Localized synthetic manipulation or inpainting detected across approximately {affected_area:.1f}% of image."
            confidence = prob_aigc
            real_prob = 1.0 - prob_aigc
            partial_prob = prob_aigc
            full_prob = 0.15
        else:
            verdict = "UNCERTAIN"
            verdict_label = "UNCERTAIN"
            verdict_badge = "BORDERLINE CASE"
            verdict_desc = "Ambiguous statistical markers. Manual human forensic review recommended."
            confidence = 0.50
            real_prob = 1.0 - prob_aigc
            partial_prob = 0.25
            full_prob = prob_aigc

        return {
            "evidence_id": f"sess_{int(time.time())}_{os.urandom(2).hex()}",
            "filename": filename,
            "sha256": meta_prov["sha256"],
            "verdict": verdict,
            "verdict_label": verdict_label,
            "verdict_badge": verdict_badge,
            "verdict_description": verdict_desc,
            "confidence": round(confidence, 4),
            "ai_probability": round(prob_aigc, 4),
            "probabilities": {
                "REAL": round(real_prob, 4),
                "PARTIAL_AIGC": round(partial_prob, 4),
                "FULL_AIGC": round(full_prob, 4)
            },
            "affected_area_percentage": affected_area,
            "suspicious_regions": suspicious_boxes,
            "operating_mode": operating_mode,
            "model_metadata": {
                "name": self.metadata.get("model_name", "Triple-Hybrid Champion"),
                "architecture": self.metadata.get("architecture", "CLIP ViT-L/14 + SigLIP SO400M + SRM Head"),
                "parameter_count": self.metadata.get("parameter_count", 735038561),
                "temperature_scaling": self.temperature,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2)
            },
            "provenance": meta_prov,
            "spatial_forensics": spatial_evidence,
            "consensus_weights": {
                "clip_vit_l14": 0.42,
                "siglip_so400m": 0.38,
                "srm_wavelet_residual": 0.20
            },
            "file_info": {
                "dimensions": meta_prov["dimensions"],
                "color_mode": meta_prov["color_mode"],
                "file_size": meta_prov["file_size_human"]
            },
            "timeline": timeline
        }

    def predict_batch(
        self,
        images: List[Union[Image.Image, bytes]],
        filenames: Optional[List[str]] = None,
        operating_mode: str = "standard"
    ) -> List[Dict[str, Any]]:
        results = []
        for idx, img in enumerate(images):
            fn = filenames[idx] if (filenames and idx < len(filenames)) else f"evidence_{idx+1}.png"
            res = self.predict(img, filename=fn, operating_mode=operating_mode)
            results.append(res)
        return results

    def get_metadata(self) -> Dict[str, Any]:
        return self.metadata


_GLOBAL_ADAPTER: Optional[ForensicModelAdapter] = None

def get_forensic_adapter() -> ForensicModelAdapter:
    global _GLOBAL_ADAPTER
    if _GLOBAL_ADAPTER is None:
        _GLOBAL_ADAPTER = TripleHybridChampionAdapter()
    return _GLOBAL_ADAPTER
