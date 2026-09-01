"""
deployment/benchmark.py
Production Latency & Throughput Benchmark Harness for AIGC Vision Detector
Measures:
  1. Cold-start latency (model loading + initial forward pass)
  2. Warm-state single-image latency (batch=1)
  3. Batched throughput (batch=16, 32, 64)
  4. Memory footprint (RAM & VRAM)
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from PIL import Image
import numpy as np
import psutil
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from deployment.config import config
from deployment.model_loader import load_production_model
from deployment.inference import ForensicInferenceEngine

def create_synthetic_test_image(size=(512, 512)):
    """Generates a synthetic RGB image for benchmarking."""
    arr = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    return Image.fromarray(arr)

def run_benchmark(device_str: str = "cpu", iterations: int = 50):
    print(f"\n{'='*70}")
    print(f"  BENCHMARKING AIGC DETECTOR ON DEVICE: {device_str.upper()}")
    print(f"{'='*70}")
    
    dev = torch.device(device_str)
    
    # 1. Cold Start Benchmark
    t0_cold = time.perf_counter()
    model, metadata = load_production_model(device=device_str)
    engine = ForensicInferenceEngine(model, metadata, device=device_str)
    test_img = create_synthetic_test_image()
    
    # First forward pass
    _ = engine.predict(test_img)
    if "cuda" in device_str:
        torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - t0_cold) * 1000.0
    print(f"[1] Cold Start Latency (Load + 1st Pass): {cold_start_ms:.2f} ms")
    
    # 2. Warm-State Single-Image Latency (Batch=1)
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = engine.predict(test_img)
        if "cuda" in device_str:
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)
        
    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    print(f"[2] Warm Single-Image Latency (Batch=1, N={iterations}):")
    print(f"    - Mean: {mean_lat:.2f} ms | P50: {p50_lat:.2f} ms | P95: {p95_lat:.2f} ms | P99: {p99_lat:.2f} ms")
    
    # 3. Batched Throughput (Batch=32)
    batch_sizes = [8, 16, 32]
    throughput_results = {}
    for bs in batch_sizes:
        batch_imgs = [create_synthetic_test_image() for _ in range(bs)]
        t0_batch = time.perf_counter()
        _ = engine.predict_batch(batch_imgs)
        if "cuda" in device_str:
            torch.cuda.synchronize()
        batch_time = time.perf_counter() - t0_batch
        fps = bs / max(0.0001, batch_time)
        throughput_results[f"batch_{bs}_fps"] = round(fps, 2)
        print(f"[3] Batched Throughput (Batch={bs}): {fps:.1f} images/sec (Total: {batch_time*1000.0:.2f} ms)")
        
    # 4. Memory Footprint
    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    vram_mb = float(torch.cuda.memory_allocated(0) / (1024 * 1024)) if "cuda" in device_str and torch.cuda.is_available() else 0.0
    print(f"[4] Memory Footprint:")
    print(f"    - Host System RAM: {ram_mb:.1f} MB")
    if "cuda" in device_str:
        print(f"    - Dedicated GPU VRAM: {vram_mb:.1f} MB")
        
    report = {
        "device": device_str,
        "cold_start_latency_ms": round(cold_start_ms, 2),
        "batch_1_latency_mean_ms": round(mean_lat, 2),
        "batch_1_latency_p50_ms": round(p50_lat, 2),
        "batch_1_latency_p95_ms": round(p95_lat, 2),
        "batch_1_latency_p99_ms": round(p99_lat, 2),
        "throughput_fps": throughput_results,
        "host_ram_used_mb": round(ram_mb, 1),
        "gpu_vram_used_mb": round(vram_mb, 1)
    }
    
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--save-report", action="store_true")
    args = parser.parse_args()
    
    res = run_benchmark(device_str=args.device, iterations=args.iterations)
    if args.save_report:
        report_file = PROJECT_ROOT / "reports" / f"deployment_benchmark_{args.device.replace(':', '_')}.json"
        with open(report_file, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n>>> Saved benchmark report to {report_file}")
