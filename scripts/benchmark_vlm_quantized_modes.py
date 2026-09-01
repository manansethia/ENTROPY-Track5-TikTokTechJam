#!/usr/bin/env python3
"""
scripts/benchmark_vlm_quantized_modes.py
Systematic Benchmark of Moondream2 Execution Modes & Forensic Output Quality
"""

import os
import sys
import time
import gc
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from scipy.ndimage import laplace
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEST_IMAGES = [
    ("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real/train-00008-of-00249_real_00050.jpg", 0, "Authentic_Photography"),
    ("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/aigc/train-00000-of-00249_aigc_00000.jpg", 1, "Latent_Diffusion"),
    ("/mnt/ai-storage/aigc_data/datasets/massive_balanced_50k/real/train-00000-of-00249_real_00001.jpg", 0, "Real_Complex_Texture")
]

REPORT_PATH = Path("/home/manan/aigc_robust_detection/reports/vlm_quantization_benchmark.json")
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

FORENSIC_TAGS = [
    "noise", "sensor_noise", "texture", "micro_texture", "boundary", "halo",
    "smoothness", "blur", "frequency", "compression", "artifact", "grain"
]

def get_mem_telemetry():
    vram_peak = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    vram_curr = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0.0
    ram_curr = psutil.Process().memory_info().rss / (1024**2)
    return vram_curr, vram_peak, ram_curr

def compute_physical_signals(pil_img):
    img_arr = np.array(pil_img.convert("L"), dtype=np.float32)
    h, w = img_arr.shape
    
    # 2D FFT
    fft = np.fft.fftshift(np.fft.fft2(img_arr))
    mag = np.abs(fft)
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    high_freq_mask = r > (min(h, w) * 0.35)
    high_freq_ratio = float(np.sum(mag * high_freq_mask) / (np.sum(mag) + 1e-8))
    
    # Laplacian Var
    lap_var = float(np.var(laplace(img_arr)))
    
    # SRM Residual
    srm_filter = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1]
    ], dtype=np.float32) / 12.0
    srm_res = convolve2d(img_arr, srm_filter, mode="same", boundary="symm")
    srm_energy = float(np.mean(np.abs(srm_res)))
    
    return {
        "high_freq_ratio": high_freq_ratio,
        "laplacian_var": lap_var,
        "srm_energy": srm_energy
    }

def extract_forensic_tags(text):
    text_lower = text.lower()
    found = [tag for tag in FORENSIC_TAGS if tag in text_lower]
    return list(set(found))

def evaluate_claim_validity(text, physical_signals, true_label):
    text_lower = text.lower()
    unsupported_claims = 0
    total_claims = 0
    
    if "high noise" in text_lower or "grainy" in text_lower:
        total_claims += 1
        if physical_signals["srm_energy"] < 2.5:
            unsupported_claims += 1
            
    if "smooth" in text_lower or "blur" in text_lower:
        total_claims += 1
        if physical_signals["laplacian_var"] > 500.0:
            unsupported_claims += 1
            
    if "frequency anomaly" in text_lower or "unnatural pattern" in text_lower:
        total_claims += 1
        if physical_signals["high_freq_ratio"] < 0.08 and true_label == 0:
            unsupported_claims += 1
            
    unsupported_rate = (unsupported_claims / max(1, total_claims)) if total_claims > 0 else 0.0
    return {
        "total_claims": total_claims,
        "unsupported_claims": unsupported_claims,
        "unsupported_rate": unsupported_rate
    }

def run_benchmark_mode(mode_name, load_fn):
    print(f"\n=======================================================")
    print(f"  TESTING MODE: {mode_name}")
    print(f"=======================================================")
    
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    vlm_id = "vikhyatk/moondream2"
    vlm_rev = "2024-08-26"
    
    t0_load = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(vlm_id, revision=vlm_rev, trust_remote_code=True)
        model = load_fn(vlm_id, vlm_rev)
        model.eval()
        load_time = time.perf_counter() - t0_load
        print(f"  >>> Loaded model in {load_time:.2f}s")
    except Exception as e:
        print(f"  [FAIL] Model Loading Failed: {e}")
        return {
            "mode": mode_name,
            "load_time_s": 0.0,
            "cuda_oom": True if "CUDA out of memory" in str(e) else False,
            "error": str(e),
            "valid": False,
            "mean_seconds_per_image": 0.0,
            "throughput_images_per_sec": 0.0,
            "output_validity": False
        }
        
    latencies = []
    responses = []
    tag_counts = []
    unsupported_rates = []
    cuda_oom = False
    valid_output = True
    
    for idx, (img_path, lbl, dom) in enumerate(TEST_IMAGES):
        try:
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")
                signals = compute_physical_signals(img)
                
                # Resize thumbnail to 378x378 for single-patch fast encoding
                thumb = img.copy()
                thumb.thumbnail((378, 378))
                
                prompt = f"Forensic analysis: Source is {dom} (Label={lbl}). Identify micro-textures, sensor noise, compression patterns, and boundary artifacts."
                
                t0 = time.perf_counter()
                with torch.inference_mode():
                    enc = model.encode_image(thumb)
                    ans = model.answer_question(enc, prompt, tokenizer)
                dt = time.perf_counter() - t0
                latencies.append(dt)
                responses.append(ans)
                
                tags = extract_forensic_tags(ans)
                tag_counts.append(len(tags))
                
                claim_eval = evaluate_claim_validity(ans, signals, lbl)
                unsupported_rates.append(claim_eval["unsupported_rate"])
                
                print(f"  [Img {idx+1}/3] ({dom[:15]}) Latency: {dt:.2f}s | Tags: {tags} | Text: {ans[:60]}...")
                if not ans or len(ans.strip()) < 8:
                    valid_output = False
        except Exception as e:
            print(f"  [ERROR] Inference failed on image {idx+1}: {e}")
            if "CUDA out of memory" in str(e):
                cuda_oom = True
            valid_output = False
            break
            
    vram_curr, vram_peak, ram_curr = get_mem_telemetry()
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    throughput = (1.0 / mean_lat) if mean_lat > 0 else 0.0
    mean_tags = float(np.mean(tag_counts)) if tag_counts else 0.0
    mean_unsupported = float(np.mean(unsupported_rates)) if unsupported_rates else 0.0
    
    res = {
        "mode": mode_name,
        "load_time_s": round(load_time, 2),
        "vram_peak_mb": round(vram_peak, 1),
        "ram_peak_mb": round(ram_curr, 1),
        "mean_seconds_per_image": round(mean_lat, 3),
        "throughput_images_per_sec": round(throughput, 2),
        "cuda_oom": cuda_oom,
        "output_validity": valid_output,
        "mean_forensic_tags_extracted": round(mean_tags, 1),
        "mean_unsupported_claim_rate": round(mean_unsupported, 3),
        "sample_responses": responses
    }
    print(f"  >>> Summary: Peak VRAM: {res['vram_peak_mb']} MB | Latency: {res['mean_seconds_per_image']}s/img | Tags: {res['mean_forensic_tags_extracted']} | Unsupp Rate: {res['mean_unsupported_claim_rate']} | Valid: {res['output_validity']}")
    return res

def main():
    print("=" * 80)
    print("  MOONDREAM2 HARDWARE & FORENSIC QUALITY BENCHMARK (RTX 3050 6GB)")
    print("=" * 80)
    
    results = {}
    
    # 1. FP16 GPU
    results["1_FP16_GPU"] = run_benchmark_mode(
        "1_FP16_GPU",
        lambda v_id, v_rev: AutoModelForCausalLM.from_pretrained(
            v_id, trust_remote_code=True, revision=v_rev,
            torch_dtype=torch.float16, device_map="cuda:0"
        )
    )
    
    # 2. INT8 GPU (bitsandbytes)
    results["2_INT8_GPU"] = run_benchmark_mode(
        "2_INT8_GPU",
        lambda v_id, v_rev: AutoModelForCausalLM.from_pretrained(
            v_id, trust_remote_code=True, revision=v_rev,
            load_in_8bit=True, device_map="cuda:0"
        )
    )
    
    # 3. 4-bit NF4 GPU (bitsandbytes)
    bnb_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    results["3_4BIT_NF4_GPU"] = run_benchmark_mode(
        "3_4BIT_NF4_GPU",
        lambda v_id, v_rev: AutoModelForCausalLM.from_pretrained(
            v_id, trust_remote_code=True, revision=v_rev,
            quantization_config=bnb_4bit, device_map="cuda:0"
        )
    )
    
    # 4. CPU Reference (FP32)
    results["4_CPU_FP32"] = run_benchmark_mode(
        "4_CPU_FP32",
        lambda v_id, v_rev: AutoModelForCausalLM.from_pretrained(
            v_id, trust_remote_code=True, revision=v_rev,
            torch_dtype=torch.float32, device_map="cpu"
        )
    )
    
    candidate_modes = [
        v for v in results.values()
        if v["output_validity"] and not v["cuda_oom"]
        and v["mean_seconds_per_image"] > 0
        and v.get("mean_unsupported_claim_rate", 1.0) <= 0.25
        and v.get("mean_forensic_tags_extracted", 0) >= 1.0
    ]
    
    if candidate_modes:
        selected_winner = min(candidate_modes, key=lambda x: x["mean_seconds_per_image"])
    else:
        selected_winner = results["4_CPU_FP32"]
        
    out_data = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "benchmark_modes": results,
        "selected_champion_mode": selected_winner["mode"],
        "decision_rationale": f"Selected {selected_winner['mode']} with speed {selected_winner['mean_seconds_per_image']}s/img, peak VRAM {selected_winner['vram_peak_mb']}MB, and unsupported claim rate {selected_winner.get('mean_unsupported_claim_rate', 0.0)}."
    }
    
    with open(REPORT_PATH, "w") as f:
        json.dump(out_data, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"  BENCHMARK & QUALITY AUDIT COMPLETE.")
    print(f"  Selected Champion Mode: {out_data['selected_champion_mode']}")
    print(f"  Report written to {REPORT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    main()
