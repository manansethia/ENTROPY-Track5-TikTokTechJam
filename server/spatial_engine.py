"""
server/spatial_engine.py
Deterministic Spatial & Frequency Forensic Analysis Engine
Computes real-time FFT radial spectral decay, SRM noise residuals, and edge variance.
"""

import io
import base64
from typing import Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image


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
    5. Diagnostic visual artifact maps (SRM PNG, FFT power spectrum PNG, Heatmap PNG)
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
    # Vectorized 5x5 window conv
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
    # Discrete Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    pad1 = np.pad(arr, 1, mode="reflect")
    lap = (pad1[0:-2, 1:-1] + pad1[2:, 1:-1] + pad1[1:-1, 0:-2] + pad1[1:-1, 2:] - 4.0 * pad1[1:-1, 1:-1])
    lap_var = float(np.var(lap))

    # 4. Localized Inpainting / Area % Estimation
    # Compute local spatial variance grid (16x16 cells = 16px per block)
    block_size = 16
    n_blocks = proc_size // block_size
    grid_scores = np.zeros((n_blocks, n_blocks), dtype=np.float32)
    
    for by in range(n_blocks):
        for bx in range(n_blocks):
            patch_srm = srm_abs[by*block_size:(by+1)*block_size, bx*block_size:(bx+1)*block_size]
            grid_scores[by, bx] = float(np.mean(patch_srm))
            
    # Baseline threshold for anomalous noise
    median_score = np.median(grid_scores)
    std_score = np.std(grid_scores) + 1e-6
    z_scores = (grid_scores - median_score) / std_score
    anomalous_cells = z_scores > 2.2
    affected_percentage = float(np.sum(anomalous_cells) / anomalous_cells.size * 100.0)

    # 5. Build Attribution Heatmap RGBA overlay
    # Upsample grid to image size
    heatmap_grid = Image.fromarray((np.clip(z_scores * 50, 0, 255)).astype(np.uint8), mode="L")
    heatmap_upscaled = heatmap_grid.resize((proc_size, proc_size), Image.Resampling.BICUBIC)
    hm_arr = np.array(heatmap_upscaled, dtype=np.float32) / 255.0
    
    # Generate Turbo/Inferno-like RGBA colormap
    rgba_heatmap = np.zeros((proc_size, proc_size, 4), dtype=np.uint8)
    rgba_heatmap[..., 0] = np.clip(hm_arr * 255 * 1.4, 0, 255).astype(np.uint8)  # Red
    rgba_heatmap[..., 1] = np.clip((1.0 - np.abs(hm_arr - 0.5) * 2) * 180, 0, 255).astype(np.uint8) # Gold/Green
    rgba_heatmap[..., 2] = np.clip((1.0 - hm_arr) * 120, 0, 255).astype(np.uint8) # Blue
    rgba_heatmap[..., 3] = np.clip(hm_arr * 210, 0, 210).astype(np.uint8) # Alpha

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
