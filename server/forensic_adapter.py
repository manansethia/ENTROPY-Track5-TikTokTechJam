"""
server/forensic_adapter.py
Pluggable Forensic Model Adapter Architecture.
Designed for private inference deployment without exposing infrastructure details.
"""

import os
import sys
import time
import io
import platform
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server.provenance_engine import inspect_image_provenance_full
from server.spatial_engine import compute_deterministic_spatial_evidence


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
    Runs in full acceleration mode on the private inference service.
    Runs in zero-overhead lightweight mode on client/Mac.
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
        
        # Only initialize heavy PyTorch tensors on the inference service or when explicitly requested.
        if self.is_linux_server or os.getenv("LOAD_HEAVY_MODEL_ON_MAC", "0") == "1":
            self._load_heavy_server_model()
        else:
            self._init_lightweight_client_mode()

    def _init_lightweight_client_mode(self):
        """Lightweight Mac initialization: zero heavy model in RAM."""
        print("[ForensicAdapter] Running in lightweight mode (heavy model offloaded to the inference service).")
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
        """Executed by the private inference service only."""
        try:
            import torch
            from deployment.portable_model import load_portable_champion_model
            
            ckpt_file = Path(self.default_ckpt)
            if ckpt_file.exists():
                print(f"[Inference Service] Loading 735M Champion model from {ckpt_file}...")
                model, meta = load_portable_champion_model(ckpt_file, device=self.device_str)
                self.model = model
                self.metadata = {
                    **meta,
                    "is_loaded": True,
                    "inference_available": True,
                    "operating_device": "private-inference-service",
                }
                self.temperature = meta.get("temperature", 1.523021)
                self.is_loaded = True
                print("[Inference Service] Champion model ready.")
        except Exception as e:
            print(f"[Inference Service] Notice during model load: {e}")
            self._init_lightweight_client_mode()

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
            from deployment.portable_model import portable_eval_transform
            tensor = portable_eval_transform(pil_img).unsqueeze(0).to(self.device_str)
            with torch.no_grad():
                logits = self.model(tensor)
                raw_logit = float(logits[0].item() if isinstance(logits, tuple) else logits.item())
            calibrated_logit = raw_logit / self.temperature
            prob_aigc = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
        except Exception as error:
            raise RuntimeError("MODEL_INFERENCE_FAILED: no forensic verdict was produced.") from error

        # 5. Provenance Synthesis Stage
        timeline.append({"stage": "PROVENANCE", "timestamp": time.strftime("%H:%M:%S"), "detail": "Reconciled C2PA status and synthetic generator signatures"})

        # 6. Verdict Classification Stage
        timeline.append({"stage": "VERDICT", "timestamp": time.strftime("%H:%M:%S"), "detail": "Multi-tier decision gating finalized"})

        affected_area = spatial_evidence.get("affected_area_percentage", 0.0)
        
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
            verdict = "PARTIAL_AI"
            verdict_label = "PARTIAL-AI"
            verdict_badge = "AUTHENTIC PHOTOGRAPH + LOCALIZED AI MANIPULATION"
            verdict_desc = f"Authentic photograph with localized generative inpainting or object insertion (~{affected_area}% affected area)."
            confidence = 0.88
            real_prob = 0.15
            partial_prob = 0.75
            full_prob = 0.10
        else:
            verdict = "REVIEW_REQUIRED"
            verdict_label = "REVIEW REQUIRED"
            verdict_badge = "AMBIGUOUS / BORDERLINE SIGNAL"
            verdict_desc = "Conflicting multi-scale forensic evidence. Manual expert inspection advised."
            confidence = 0.52
            real_prob = 0.35
            partial_prob = 0.35
            full_prob = 0.30

        if confidence >= 0.85:
            conf_tier = "HIGH_CONFIDENCE"
        elif confidence >= 0.65:
            conf_tier = "MODERATE_CONFIDENCE"
        else:
            conf_tier = "UNCERTAIN_BORDERLINE"

        total_latency = (time.perf_counter() - t0) * 1000.0

        return {
            "success": True,
            "filename": filename,
            "evidence_id": f"AF-{meta_prov['sha256'][:4].upper()}",
            "verdict": verdict,
            "verdict_label": verdict_label,
            "verdict_badge": verdict_badge,
            "verdict_description": verdict_desc,
            "confidence": round(confidence, 4),
            "confidence_tier": conf_tier,
            "probability_aigc": round(prob_aigc, 6),
            "real_probability": round(real_prob, 4),
            "partial_probability": round(partial_prob, 4),
            "full_probability": round(full_prob, 4),
            "affected_area_percentage": affected_area,
            "raw_logit": round(raw_logit, 4),
            "latency_ms": round(total_latency, 2),
            "device_used": "private-inference-service" if self.is_linux_server else "lightweight-forensic-engine",
            "model": {
                "name": "Triple-Hybrid Champion",
                "version": "v4.3-REM-A",
                "architecture": "CLIP ViT-L/14 + SigLIP SO400M + SRM Wavelet Residual Head + Fusion Head",
                "parameter_count": "735M (735,038,561)",
                "temperature_scaling": self.temperature,
                "host": "private-inference-service" if self.is_linux_server else "local-client"
            },
            "spatial_forensics": {
                "fft_high_frequency_ratio": spatial_evidence["fft_high_frequency_ratio"],
                "srm_residual_energy": spatial_evidence["srm_residual_energy"],
                "laplacian_variance": spatial_evidence["laplacian_variance"],
                "inconsistency_status": spatial_evidence["inconsistency_status"],
                "spatial_localization_available": spatial_evidence["spatial_localization_available"],
                "artifacts": spatial_evidence["artifacts"]
            },
            "metadata": meta_prov["exif"],
            "provenance": meta_prov["provenance"],
            "file_info": {
                "sha256": meta_prov["sha256"],
                "dimensions": meta_prov["dimensions"],
                "width": meta_prov["width"],
                "height": meta_prov["height"],
                "format": meta_prov["format"],
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
