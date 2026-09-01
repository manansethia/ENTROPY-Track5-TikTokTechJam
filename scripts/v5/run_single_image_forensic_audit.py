#!/usr/bin/env python3
"""
run_single_image_forensic_audit.py
----------------------------------
Runs the complete multi-specialist forensic pipeline on an arbitrary image:
  1. Decoupled Provenance Engine (EXIF, IPTC, C2PA, Watermarks, Software Signatures)
  2. V5-CAG Spatial Engine (Tri-Class Probabilities, Patch Heatmap, Affected Area)
  3. Master Fused Engine (Dynamic Calibrated Consensus)
"""

import os
import sys
import json
import time
from pathlib import Path
from PIL import Image
import numpy as np

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.v5.v5_provenance_engine import V5ProvenanceEngine
from scripts.v5.v5_inference_engine import V5ForensicInferenceEngine
from scripts.fused.run_fused_ai_teaching_feedback_loop import MasterFusedForensicPipeline

def audit(image_path: str):
    print("=" * 95)
    print(f"  EXECUTING MULTI-SPECIALIST FORENSIC AUDIT: {os.path.basename(image_path)}")
    print("=" * 95)
    
    img = Image.open(image_path)
    w, h = img.size
    print(f"  Image Resolution: {w} x {h} ({w*h/1e6:.2f} Megapixels, Format: {img.format})")
    
    # 1. Provenance Engine
    prov_engine = V5ProvenanceEngine()
    prov_res = prov_engine.analyze_provenance(image_path)
    print("\n  [Provenance Subsystem]")
    print(f"    - C2PA Manifest Present:  {prov_res.get('c2pa_findings', {}).get('manifest_detected', False)}")
    print(f"    - Software / AI Flags:    {prov_res.get('provenance_flags', [])}")
    print(f"    - Exif Camera:            {prov_res.get('exif_metadata', {}).get('Make', 'None')} {prov_res.get('exif_metadata', {}).get('Model', '')}")
    
    # 2. V5-CAG Spatial Engine
    v5_engine = V5ForensicInferenceEngine()
    v5_res = v5_engine.analyze(image_path)
    print("\n  [V5-CAG Spatial Engine]")
    print(f"    - Verdict:                {v5_res['verdict']}")
    print(f"    - Confidence:             {v5_res['confidence']:.4f}")
    print(f"    - Tri-Class Distribution: Real={v5_res['class_probabilities']['REAL']:.4f}, Partial={v5_res['class_probabilities']['PARTIAL_AIGC']:.4f}, Full={v5_res['class_probabilities']['FULL_AIGC']:.4f}")
    print(f"    - Affected Area:          {v5_res['affected_area_percentage']:.2f}%")
    print(f"    - Suspicious Patches:     {len(v5_res['suspicious_regions'])}")
    print(f"    - Heatmap Saved:          {v5_res['localization_heatmap_path']}")
    
    # 3. Master Fused Engine
    fused_engine = MasterFusedForensicPipeline()
    fused_res = fused_engine.analyze_image(image_path)
    print("\n  [Master Fused Consensus]")
    print(f"    - Fused Verdict:          {fused_res['verdict']}")
    print(f"    - Fused Confidence:       {fused_res['confidence']:.4f}")
    print(f"    - Fused AI Probability:   {fused_res['fused_ai_probability']:.4f}")
    print(f"    - Specialist Scores:      {fused_res['specialist_scores']}")
    
    # Combined Audit Report
    full_report = {
        "image_name": os.path.basename(image_path),
        "dimensions": {"width": w, "height": h, "megapixels": round((w*h)/1e6, 2)},
        "provenance": prov_res,
        "v5_cag_spatial": v5_res,
        "master_fused": fused_res
    }
    
    out_json = "/home/manan/aigc_robust_detection/reports/single_image_audit.json"
    with open(out_json, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"\n  Complete Audit Report written to: {out_json} ✅")
    print("=" * 95)

if __name__ == "__main__":
    target = "/home/manan/aigc_robust_detection/test_inputs/milky_way_test.webp"
    if len(sys.argv) > 1: target = sys.argv[1]
    audit(target)
