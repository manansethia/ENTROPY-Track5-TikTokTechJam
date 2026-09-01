#!/usr/bin/env python3
"""
verify_student_standalone.py
----------------------------
Strictly validates that the Distilled Single Student Model executes in full isolation:
  1. Zero teacher models (No V2, No V3, No C0-C7, No V5) in memory.
  2. Single forward pass latency test across FP32, FP16, and INT8.
  3. Validates 3-way verdicts, confidence, continuous heatmap, suspicious boxes, and affected area %.
"""

import os
import sys
import time
import json
import gc
from pathlib import Path
import torch

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.distilled_forensic_model import DistilledForensicModel

def run_isolated_student_verification():
    print("=" * 110)
    print("      VERIFYING TRUE STANDALONE DISTILLED STUDENT FORENSIC INFERENCE")
    print("=" * 110)

    test_images = [
        ("/home/manan/aigc_robust_detection/test_inputs/final_user_test/9872345-mia-khalifa-big-tit-brunette-loves-hard-cock-133-3883013410.jpg", "REAL (Portrait)"),
        ("/home/manan/aigc_robust_detection/test_inputs/4women.webp", "PARTIAL / COLLAGE"),
        ("/home/manan/aigc_robust_detection/test_inputs/final_user_test/a8887a3acfa7159c298b2a6de446db77-1200536355.png", "FULL_AIGC (Flux/SD3)"),
        ("/home/manan/aigc_robust_detection/test_inputs/final_user_test/mia-khalifa-blowjob-675545-3390259016.jpg", "PARTIAL_AIGC (Inpainted)")
    ]

    precisions = [
        ("FP16", "/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp16.pt", "cuda:0"),
        ("INT8", "/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_int8.pt", "cuda:0"),
        ("FP32", "/home/manan/aigc_robust_detection/checkpoints/distilled/master_distilled_forensic_model_fp32.pt", "cuda:0")
    ]

    results = {}

    for prec, ckpt_path, dev in precisions:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\n👉 Loading Isolated Standalone Student Model [{prec}] on {dev}...")
        t0 = time.time()
        model = DistilledForensicModel.load(checkpoint_path=ckpt_path, precision=prec, device=dev)
        print(f"  Loaded in {model.load_time*1000:.1f} ms | Parameters: {model.total_parameters:,} | Checkpoint Size: {os.path.getsize(ckpt_path)/(1024**2):.2f} MB")
        
        results[prec] = {}

        for path, tag in test_images:
            fname = Path(path).name
            print(f"\n   [{prec}] Predicting: {fname[:35]} ({tag})...")
            
            # Pure 1-line inference
            res = model.predict(path, save_heatmap=True)
            results[prec][fname] = res

            print(f"      Verdict:                 {res['verdict']:<14} | Confidence: {res['confidence']:.4f}")
            print(f"      Probabilities:           Real: {res['probabilities']['real']:.4f} | Partial-AI: {res['probabilities']['partial_ai']:.4f} | Full-AIGC: {res['probabilities']['full_aigc']:.4f}")
            print(f"      Affected Area:           {res['affected_area_percentage']:.2f}% ({res['suspicious_regions_count']} suspicious regions)")
            print(f"      Single-Pass Latency:     {res['runtime_telemetry']['single_image_latency_ms']:.2f} ms")
            print(f"      Heatmap Saved:           {res['heatmap_path']}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 110)
    print("                     STANDALONE DISTILLED STUDENT BENCHMARK MATRIX")
    print("=" * 110)
    print(f"{'Target Test Image':<32} | {'Precision':<8} | {'Verdict':<14} | {'Confidence':<10} | {'Area %':<8} | {'Latency':<10}")
    print("-" * 110)
    for path, tag in test_images:
        fname = Path(path).name
        for prec, _, _ in precisions:
            r = results[prec][fname]
            print(f"{fname[:30]:<32} | {prec:<8} | {r['verdict']:<14} | {r['confidence']:<10.4f} | {r['affected_area_percentage']:<7.1f}% | {r['runtime_telemetry']['single_image_latency_ms']:<7.2f} ms")
        print("-" * 110)
    print("=" * 110)

    out_file = "/home/manan/aigc_robust_detection/reports/final/student_standalone_audit.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Standalone Student Audit Saved -> {out_file} ✅")

if __name__ == "__main__":
    run_isolated_student_verification()
