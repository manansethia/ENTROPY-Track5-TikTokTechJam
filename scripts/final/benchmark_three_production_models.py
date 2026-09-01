#!/usr/bin/env python3
"""
benchmark_three_production_models.py
------------------------------------
Executes comprehensive comparative validation across all THREE production models:
  1. FP32 Master Model (6.79 GB)
  2. FP16 Master Model (3.39 GB)
  3. FP8 / INT8 Quantized Master Model (1.70 GB)
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
import torch

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.production_master_engine import ProductionMasterEngine

def run_benchmark():
    print("=" * 115)
    print("   BENCHMARKING THREE FINAL PRODUCTION CANDIDATE MODELS (FP32, FP16, FP8)")
    print("=" * 115)

    test_images = [
        ("/home/manan/aigc_robust_detection/test_inputs/final_user_test/9872345-mia-khalifa-big-tit-brunette-loves-hard-cock-133-3883013410.jpg", "REAL (Portrait)"),
        ("/home/manan/aigc_robust_detection/test_inputs/4women.webp", "PARTIAL / COLLAGE"),
        ("/home/manan/aigc_robust_detection/test_inputs/final_user_test/a8887a3acfa7159c298b2a6de446db77-1200536355.png", "FULL_AIGC (Flux/SD3)"),
        ("/home/manan/aigc_robust_detection/test_inputs/final_user_test/mia-khalifa-blowjob-675545-3390259016.jpg", "PARTIAL_AIGC (Inpainted)")
    ]

    precisions = [
        ("FP16", "cuda:0"),
        ("FP8", "cuda:0"),
        ("FP32", "cpu")
    ]
    results = {}

    for prec, dev in precisions:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n👉 Initializing Production Model [{prec}] on {dev}...")
        engine = ProductionMasterEngine(precision=prec, device=dev)
        results[prec] = {}

        for path, tag in test_images:
            fname = Path(path).name
            print(f"\n   [{prec}] Evaluating: {fname[:35]} ({tag})...")
            res = engine.predict(path, save_heatmap=True)
            results[prec][fname] = res

            print(f"      Verdict:                 {res['verdict']:<14} | Confidence: {res['confidence']:.4f}")
            print(f"      Probabilities:           Real: {res['probabilities']['real']:.4f} | Partial-AI: {res['probabilities']['partial_ai']:.4f} | Full-AIGC: {res['probabilities']['full_aigc']:.4f}")
            print(f"      Affected Area:           {res['affected_area_percentage']:.2f}% ({res['suspicious_regions_count']} suspicious regions)")
            print(f"      Latency:                 {res['runtime_telemetry']['latency_ms']:.1f} ms | Device: {res['runtime_telemetry']['device']}")
            print(f"      Heatmap:                 {res['heatmap_path']}")

        del engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 115)
    print("                     THREE PRODUCTION MODELS COMPARATIVE BENCHMARK MATRIX")
    print("=" * 115)
    print(f"{'Target Test Case':<32} | {'Precision':<8} | {'Verdict':<14} | {'Confidence':<10} | {'Area %':<8} | {'Latency':<10}")
    print("-" * 115)
    for path, tag in test_images:
        fname = Path(path).name
        for prec in ["FP32", "FP16", "FP8"]:
            r = results[prec][fname]
            print(f"{fname[:30]:<32} | {prec:<8} | {r['verdict']:<14} | {r['confidence']:<10.4f} | {r['affected_area_percentage']:<7.1f}% | {r['runtime_telemetry']['latency_ms']:<7.1f} ms")
        print("-" * 115)
    print("=" * 115)

    out_file = "/home/manan/aigc_robust_detection/reports/final/three_production_models_audit.json"
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Full Production Benchmark Audit saved -> {out_file} ✅")

if __name__ == "__main__":
    run_benchmark()
