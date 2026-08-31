"""
deployment/inference.py
Model-Agnostic High-Performance Inference Engine with Multi-Tier Confidence Routing
"""

import time
from typing import Union, List, Dict, Any, Optional
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deployment.config import config
from deployment.preprocess import preprocess_single_image, preprocess_batch_images, load_pil_image
from deployment.schemas import PredictResponse, ForensicBreakdown

def compute_forensic_metrics(pil_img: Image.Image) -> ForensicBreakdown:
    """Computes real-time FFT, Laplacian, and SRM residuals for explainability breakdown."""
    try:
        img_arr = np.array(pil_img.convert("L"), dtype=np.float32)
        h, w = img_arr.shape
        
        # 1. 2D FFT Radial Frequency Energy
        fft = np.fft.fftshift(np.fft.fft2(img_arr))
        mag = np.abs(fft)
        center_y, center_x = h // 2, w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        high_freq_mask = r > (min(h, w) * 0.35)
        high_freq_ratio = float(np.sum(mag * high_freq_mask) / (np.sum(mag) + 1e-8))
        
        # 2. Laplacian Edge Variance
        from scipy.ndimage import laplace
        lap = laplace(img_arr)
        lap_var = float(np.var(lap))
        
        # 3. Deterministic SRM Noise Residual Energy
        srm_filter = np.array([
            [-1, 2, -2, 2, -1],
            [2, -6, 8, -6, 2],
            [-2, 8, -12, 8, -2],
            [2, -6, 8, -6, 2],
            [-1, 2, -2, 2, -1]
        ], dtype=np.float32) / 12.0
        from scipy.signal import convolve2d
        srm_res = convolve2d(img_arr, srm_filter, mode="same", boundary="symm")
        srm_energy = float(np.mean(np.abs(srm_res)))
        
        status = "CLEAN"
        if high_freq_ratio > 0.18 or srm_energy > 4.5 or lap_var > 550.0:
            status = "ANOMALY_DETECTED"
        elif high_freq_ratio < 0.04 and lap_var < 80.0:
            status = "COMPRESSION_DEGRADED"
            
        return ForensicBreakdown(
            fft_high_frequency_ratio=round(high_freq_ratio, 4),
            srm_residual_energy=round(srm_energy, 4),
            laplacian_variance=round(lap_var, 2),
            inconsistency_status=status
        )
    except Exception:
        return ForensicBreakdown(
            fft_high_frequency_ratio=0.0,
            srm_residual_energy=0.0,
            laplacian_variance=0.0,
            inconsistency_status="CLEAN"
        )

class ForensicInferenceEngine:
    def __init__(self, model: nn.Module, metadata: Dict[str, Any], device: str = None):
        self.model = model
        self.metadata = metadata
        self.device = torch.device(device or metadata.get("device", config.device))
        self.temperature = getattr(config, "temperature_scaling", 1.523021)
        
    def get_threshold(self, mode: str) -> float:
        if mode == "low_fpr_01":
            return config.threshold_low_fpr_01
        elif mode == "low_fpr_001":
            return config.threshold_low_fpr_001
        elif mode == "low_fpr_05":
            return config.threshold_low_fpr_05
        elif mode == "low_fpr_10":
            return config.threshold_low_fpr_10
        elif mode == "low_fpr_005":
            return config.threshold_low_fpr_005
        else:
            return config.threshold_standard
            
    def predict(
        self,
        image_input: Union[Image.Image, bytes, str],
        threshold_mode: str = "standard",
        include_forensic_breakdown: bool = False
    ) -> PredictResponse:
        """
        Executes single-image inference and returns a structured PredictResponse.
        """
        t0 = time.perf_counter()
        
        # Preprocessing
        tensor = preprocess_single_image(image_input, device=str(self.device))
        
        # Inference
        with torch.no_grad():
            if "cuda" in str(self.device):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    res = self.model(tensor)
            else:
                res = self.model(tensor)
                
            if isinstance(res, tuple):
                logit_val = float(res[0].to(torch.float32).item())
            else:
                logit_val = float(res.to(torch.float32).item())
                
        # Temperature Calibrated Probability
        calibrated_logit = logit_val / self.temperature
        prob_aigc = float(1.0 / (1.0 + np.exp(-calibrated_logit)))
        
        threshold = self.get_threshold(threshold_mode)
        is_aigc = prob_aigc >= threshold
        pred_class = "AIGC_SYNTHETIC" if is_aigc else "AUTHENTIC_REAL"
        
        # Confidence Tiering
        if prob_aigc < 0.25:
            conf_tier = "HIGH_CONFIDENCE_REAL"
        elif prob_aigc > 0.85:
            conf_tier = "HIGH_CONFIDENCE_AIGC"
        else:
            conf_tier = "UNCERTAIN_BORDERLINE"
            
        forensic_info = None
        if include_forensic_breakdown:
            try:
                img = load_pil_image(image_input)
                forensic_info = compute_forensic_metrics(img)
            except Exception:
                pass
                
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        return PredictResponse(
            success=True,
            probability_aigc=round(prob_aigc, 6),
            raw_logit=round(logit_val, 6),
            predicted_class=pred_class,
            is_aigc=is_aigc,
            confidence_tier=conf_tier,
            threshold_used=threshold,
            threshold_mode=threshold_mode,
            latency_ms=round(latency_ms, 2),
            device_used=str(self.device),
            model_version=config.model_name,
            model_sha256=self.metadata.get("parameter_hash", "UNKNOWN"),
            forensic_breakdown=forensic_info
        )

    def predict_batch(
        self,
        images: List[Union[Image.Image, bytes, str]],
        threshold_mode: str = "standard"
    ) -> List[PredictResponse]:
        """
        High-throughput batch inference.
        """
        t0 = time.perf_counter()
        tensors = preprocess_batch_images(images, device=str(self.device))
        
        with torch.no_grad():
            if "cuda" in str(self.device):
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    res = self.model(tensors)
            else:
                res = self.model(tensors)
                
            if isinstance(res, tuple):
                logits = res[0].squeeze(-1).to(torch.float32).cpu().numpy()
            else:
                logits = res.squeeze(-1).to(torch.float32).cpu().numpy()
                
        calibrated_logits = logits / self.temperature
        probs = 1.0 / (1.0 + np.exp(-calibrated_logits))
        threshold = self.get_threshold(threshold_mode)
        
        total_latency = (time.perf_counter() - t0) * 1000.0
        per_item_lat = total_latency / max(1, len(images))
        
        responses = []
        for i in range(len(images)):
            p = float(probs[i])
            is_aigc = p >= threshold
            pred_class = "AIGC_SYNTHETIC" if is_aigc else "AUTHENTIC_REAL"
            
            if p < 0.25:
                conf = "HIGH_CONFIDENCE_REAL"
            elif p > 0.85:
                conf = "HIGH_CONFIDENCE_AIGC"
            else:
                conf = "UNCERTAIN_BORDERLINE"
                
            responses.append(PredictResponse(
                success=True,
                probability_aigc=round(p, 6),
                raw_logit=round(float(logits[i]), 6),
                predicted_class=pred_class,
                is_aigc=is_aigc,
                confidence_tier=conf,
                threshold_used=threshold,
                threshold_mode=threshold_mode,
                latency_ms=round(per_item_lat, 2),
                device_used=str(self.device),
                model_version=config.model_name,
                model_sha256=self.metadata.get("parameter_hash", "UNKNOWN"),
                forensic_breakdown=None
            ))
            
        return responses
