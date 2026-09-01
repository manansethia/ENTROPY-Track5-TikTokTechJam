"""
server/spatial_engine.py
Deterministic Spatial & Frequency Forensic Analysis Engine
Computes real-time FFT radial spectral decay, SRM noise residuals, edge variance,
and smooth, high-contrast forensic heatmaps.
"""

import io
import base64
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image, ImageFilter


def pil_to_base64(img: Image.Image, format: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def compute_deterministic_spatial_evidence(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Computes genuine mathematical forensic metrics from the image:
    1. 2D FFT Radial Frequency Energy & High-Frequency Power Ratio
    2. 5x5 SRM (Spatial-Rich Model) High-Pass Residual Energy
    3. Laplacian Edge Variance & Gradient Discontinuity
    4. Affected Area % estimation via thresholded residual variance
    5. Diagnostic visual artifact maps (SRM PNG, FFT power spectrum PNG, Smooth Heatmap PNG)
    """
    img_rgb = pil_img.convert("RGB")
    w_orig, h_orig = img_rgb.size
    
    # Standardize processing resolution for deterministic comparison
    proc_size = 256
    img_gray = img_rgb.convert("L").resize((proc_size, proc_size), Image.Resampling.BILINEAR)
    arr = np.array(img_gray, dtype=np.float32)
    h, w = arr.shape

    # 1. 2D FFT Radial Power Spectrum
    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(fft)
    log_mag = np.log1p(mag)
    
    # Normalize log spectrum to uint8 image
    norm_fft = ((log_mag - log_mag.min()) / (log_mag.max() - log_mag.min() + 1e-8) * 255.0).astype(np.uint8)
    fft_img = Image.fromarray(norm_fft, mode="L")
    
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    cutoff = min(h, w) * 0.35
    high_freq_mask = r > cutoff
    total_energy = float(np.sum(mag) + 1e-8)
    high_freq_energy = float(np.sum(mag * high_freq_mask))
    high_freq_ratio = float(high_freq_energy / total_energy)

    # 2. SRM (Spatial Rich Model) 5x5 High-Pass Noise Residuals
    srm_kernel = np.array([
        [-1,  2,  -2,  2, -1],
        [ 2, -6,   8, -6,  2],
        [-2,  8, -12,  8, -2],
        [ 2, -6,   8, -6,  2],
        [-1,  2,  -2,  2, -1]
    ], dtype=np.float32) / 12.0
    
    # 2D convolution with reflection padding
    pad_arr = np.pad(arr, 2, mode="reflect")
    srm_res = np.zeros_like(arr)
    for ky in range(5):
        for kx in range(5):
            srm_res += pad_arr[ky:ky+h, kx:kx+w] * srm_kernel[ky, kx]
            
    srm_abs = np.abs(srm_res)
    srm_energy = float(np.mean(srm_abs))
    
    # Normalize SRM residual for visualization
    srm_vis = np.clip(srm_abs * 6.0, 0, 255).astype(np.uint8)
    srm_img = Image.fromarray(srm_vis, mode="L")

    # 3. Laplacian Edge Variance
    pad1 = np.pad(arr, 1, mode="reflect")
    lap = (pad1[0:-2, 1:-1] + pad1[2:, 1:-1] + pad1[1:-1, 0:-2] + pad1[1:-1, 2:] - 4.0 * pad1[1:-1, 1:-1])
    lap_var = float(np.var(lap))

    # 4. Multi-Scale Continuous Anomaly Map
    # Combine SRM residual with localized Laplacian magnitude
    lap_abs = np.abs(lap)
    norm_srm = (srm_abs - np.min(srm_abs)) / (np.percentile(srm_abs, 98) - np.min(srm_abs) + 1e-6)
    norm_lap = (lap_abs - np.min(lap_abs)) / (np.percentile(lap_abs, 98) - np.min(lap_abs) + 1e-6)
    combined_anomaly = np.clip(0.60 * norm_srm + 0.40 * norm_lap, 0.0, 1.0)

    # Smooth the continuous anomaly map with a 2D Gaussian filter
    anomaly_pil = Image.fromarray((combined_anomaly * 255.0).astype(np.uint8), mode="L")
    anomaly_smoothed = anomaly_pil.filter(ImageFilter.GaussianBlur(radius=3.5))
    hm_arr = np.array(anomaly_smoothed, dtype=np.float32) / 255.0

    # Estimate affected area percentage
    threshold = 0.52
    affected_cells = hm_arr > threshold
    affected_percentage = float(np.sum(affected_cells) / hm_arr.size * 100.0)

    # 5. Build Attribution Heatmap RGBA overlay (Jet Color Mapping)
    rgba_heatmap = np.zeros((proc_size, proc_size, 4), dtype=np.uint8)
    
    # Red: High on upper half, zero on low
    r_chan = np.clip((hm_arr - 0.35) / 0.65 * 255.0, 0, 255).astype(np.uint8)
    # Green: Peaks around mid-level (0.45 - 0.70)
    g_chan = np.clip((1.0 - np.abs(hm_arr - 0.50) * 2.2) * 220.0, 0, 220).astype(np.uint8)
    # Blue: High on lower half, drops on high anomaly
    b_chan = np.clip((0.65 - hm_arr) / 0.65 * 240.0, 0, 240).astype(np.uint8)
    # Alpha: Low on cool regions (50-70), vivid on anomalous regions (180-220)
    a_chan = np.clip(45 + hm_arr * 175.0, 30, 220).astype(np.uint8)

    rgba_heatmap[..., 0] = r_chan
    rgba_heatmap[..., 1] = g_chan
    rgba_heatmap[..., 2] = b_chan
    rgba_heatmap[..., 3] = a_chan

    heatmap_img = Image.fromarray(rgba_heatmap, mode="RGBA")

    # Inconsistency Classification
    if high_freq_ratio > 0.22 or srm_energy > 5.5 or lap_var > 600.0:
        inconsistency_status = "ANOMALY_DETECTED"
    elif high_freq_ratio < 0.035 and lap_var < 75.0:
        inconsistency_status = "COMPRESSION_DEGRADED"
    else:
        inconsistency_status = "CLEAN_PASS"

    return {
        "fft_high_frequency_ratio": round(high_freq_ratio, 4),
        "srm_residual_energy": round(srm_energy, 4),
        "laplacian_variance": round(lap_var, 2),
        "inconsistency_status": inconsistency_status,
        "affected_area_percentage": round(affected_percentage, 1),
        "spatial_localization_available": True,
        "artifacts": {
            "srm_residual_base64": pil_to_base64(srm_img),
            "fft_spectrum_base64": pil_to_base64(fft_img),
            "heatmap_overlay_base64": pil_to_base64(heatmap_img)
        }
    }
