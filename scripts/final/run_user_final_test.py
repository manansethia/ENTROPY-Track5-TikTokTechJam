#!/usr/bin/env python3
"""
run_user_final_test.py
----------------------
Executes the Definitive Unified Master AIGC Forensic Detection System
(1.88 Billion Parameters) on the user's 6 test files.
Generates full forensic diagnoses, visual attribution heatmaps, and audit logs.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.final_unified_forensic_pipeline import FinalUnifiedForensicPipeline

def main():
    print("=" * 105)
    print("      EXECUTING FINAL USER BENCHMARK TEST ON 1.88 BILLION PARAMETER MULTI-SPECIALIST SYSTEM")
    print("=" * 105)

    pipeline = FinalUnifiedForensicPipeline()
    test_dir = "/home/manan/aigc_robust_detection/test_inputs/final_user_test"
    report_dir = "/home/manan/aigc_robust_detection/reports/final"
    os.makedirs(report_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(test_dir) if not f.startswith(".")])
    print(f"Discovered {len(files)} test images in {test_dir}:\n")
    for i, f in enumerate(files, 1):
        print(f"  [{i}] {f}")

    results = {}

    for i, f in enumerate(files, 1):
        img_path = os.path.join(test_dir, f)
        print("\n" + "-" * 105)
        print(f"👉 [{i}/{len(files)}] Running 1.88B Ensemble on: {f}")
        print("-" * 105)
        
        t0 = time.time()
        res = pipeline.analyze(img_path, save_heatmap=True)
        dur = time.time() - t0
        
        results[f] = res
        
        print(f"   Verdict:                 {res['verdict']:<15} | Confidence: {res['confidence']:.4f}")
        print(f"   Probabilities:           Real: {res['real_probability']:.4f} | Partial-AI: {res['partial_ai_probability']:.4f} | Full-AIGC: {res['full_aigc_probability']:.4f}")
        print(f"   Affected Area:           {res['affected_area_percentage']:.2f}% | Max Patch Anomaly: {res['max_patch_anomaly']:.4f} ({res['suspicious_regions_count']} suspicious regions)")
        print(f"   V2 AIDE Spectral Score:  {res['evidence_breakdown']['V2_AIDE_Spectral_Score']:.4f}")
        print(f"   V3 Gated Ensemble Score: {res['evidence_breakdown']['V3_Ensemble_Gated_Score']:.4f}")
        print(f"   V5 Spatial Engine Score: {res['evidence_breakdown']['V5_CAG_Spatial_Score']:.4f}")
        print(f"   Specialist Breakdown:    C0 (Anchor): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C0_TripleHybrid_Champion', 0.0):.4f} | C1 (Portrait): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C1_Portrait_Remediation', 0.0):.4f} | C2 (SPAI): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C2_SPAI_MultiFreq_ViT', 0.0):.4f}")
        print(f"                            C3 (Community): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C3_CommunityForensics_ViT', 0.0):.4f} | C4 (HighRes): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C4_ConvNeXt_HighRes', 0.0):.4f} | C5 (divine2k): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C5_ConvNeXt_Tiny_divine2k', 0.0):.4f}")
        print(f"                            C6 (EfficientNet): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C6_EfficientNet_B0', 0.0):.4f} | C7 (ResNet50): {res['evidence_breakdown']['V3_Specialist_Scores'].get('C7_ResNet50_Deep', 0.0):.4f}")
        print(f"   Provenance Findings:     C2PA: {res['provenance']['c2pa_manifest_detected']} | Signatures: {res['provenance']['ai_software_signatures']} | EXIF Tags: {list(res['provenance']['exif_metadata'].keys())}")
        print(f"   Telemetry:               Peak VRAM: {res['runtime_telemetry']['peak_vram_allocated_mib']} MiB | Latency: {dur:.2f}s")
        print(f"   Heatmap Output:          {res['heatmap_path']}")

    print("\n" + "=" * 105)
    print("                               FINAL USER BENCHMARK SUMMARY TABLE")
    print("=" * 105)
    print(f"{'Filename':<40} | {'Verdict':<14} | {'Confidence':<10} | {'V2 Spectral':<11} | {'V3 Gated':<10} | {'V5 Spatial':<10} | {'Max Anomaly':<11} | {'Area %':<7}")
    print("-" * 115)
    for f in files:
        r = results[f]
        print(f"{f[:38]:<40} | {r['verdict']:<14} | {r['confidence']:<10.4f} | {r['evidence_breakdown']['V2_AIDE_Spectral_Score']:<11.4f} | {r['evidence_breakdown']['V3_Ensemble_Gated_Score']:<10.4f} | {r['evidence_breakdown']['V5_CAG_Spatial_Score']:<10.4f} | {r['max_patch_anomaly']:<11.4f} | {r['affected_area_percentage']:<6.1f}%")
    print("=" * 115)

    out_file = os.path.join(report_dir, "user_final_test_audit.json")
    with open(out_file, "w") as fp:
        json.dump(results, fp, indent=2)
    print(f"\n  Final User Test Audit Report saved to: {out_file} ✅")

if __name__ == "__main__":
    main()
