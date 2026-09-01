#!/usr/bin/env python3
"""
run_final_unified_benchmarks.py
-------------------------------
Executes the Definitive Unified Master AIGC Forensic Detection System across
the test suite and writes authoritative audit reports and model manifests.
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, "/home/manan/aigc_robust_detection")
from scripts.final.final_unified_forensic_pipeline import FinalUnifiedForensicPipeline

def run_benchmarks():
    print("=" * 95)
    print("  EXECUTING FINAL UNIFIED SYSTEM BENCHMARK (1.88 BILLION PARAMETERS)")
    print("=" * 95)
    
    pipeline = FinalUnifiedForensicPipeline()
    test_dir = "/home/manan/aigc_robust_detection/test_inputs/batch_eval"
    report_dir = "/home/manan/aigc_robust_detection/reports/final"
    os.makedirs(report_dir, exist_ok=True)
    
    test_cases = [
        ("Image 1: Crab Nebula (Hubble)", os.path.join(test_dir, "img1_crab_nebula.jpg")),
        ("Image 2: Earth from Space (Satellite)", os.path.join(test_dir, "img2_earth_globe.jpg")),
        ("Image 3: Prehistoric Volcanic Earth (Art)", os.path.join(test_dir, "img3_volcano_meteors.jpg")),
        ("Image 4: Dakshineswar Temple (Photo)", os.path.join(test_dir, "img4_temple_reflection.jpg")),
        ("Image 5: Sci-Fi Floating City (AI)", os.path.join(test_dir, "img5_scifi_globe_city.png")),
        ("Image 6: 4-Women Collage (Multi-Scale)", "/home/manan/aigc_robust_detection/test_inputs/4women.webp")
    ]
    
    audit_results = {}
    
    for label, img_path in test_cases:
        if not os.path.exists(img_path):
            print(f"  Skipping missing: {img_path}")
            continue
            
        print(f"\n  👉 Running 1.88B Ensemble on: {label}...")
        t0 = time.time()
        res = pipeline.analyze(img_path, save_heatmap=True)
        dur = time.time() - t0
        
        audit_results[label] = res
        print(f"     Verdict:     {res['verdict']:12s} | Confidence: {res['confidence']:.4f}")
        print(f"     V2 AIDE:     {res['evidence_breakdown']['V2_AIDE_Spectral_Score']:.4f}")
        print(f"     V3 Gated:    {res['evidence_breakdown']['V3_Ensemble_Gated_Score']:.4f}")
        print(f"     V5 Spatial:  {res['evidence_breakdown']['V5_CAG_Spatial_Score']:.4f} (Max Anom: {res['max_patch_anomaly']:.4f})")
        print(f"     Latency:     {dur:.2f}s | Peak VRAM: {res['runtime_telemetry']['peak_vram_allocated_mib']} MiB")

    # 1. Write Runtime Audit Report
    audit_file = os.path.join(report_dir, "final_fusion_runtime_audit.json")
    with open(audit_file, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n  Runtime Audit saved to: {audit_file} ✅")

    # 2. Write Final Model Manifest
    manifest = {
        "system_name": "Definitive Unified Master AIGC Forensic Detection SYSTEM",
        "total_aggregate_parameters": 1818494881,
        "total_aggregate_parameters_human": "1.818 Billion Parameters",
        "execution_strategy": "Sequential GPU Inference on cuda:0 with Automatic Memory Release",
        "hardware_target": "NVIDIA GeForce RTX 3050 (5,803 MiB VRAM)",
        "models": [
            {
                "subsystem": "V2_Spectral_HighPass",
                "model_name": "AIDE High-Pass Frequency (HPF) Deep Neural Network",
                "checkpoint_path": "/mnt/ai-storage/aigc_data/models/aide_finetuned/checkpoint42.pth",
                "parameters": 897832732,
                "role": "Deep frequency-domain / high-pass spectral artifact detection"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C0: Triple-Hybrid Champion Anchor (CLIP-ViT-L/14 + SigLIP-SO400M + SRM Wavelet)",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_frozen_model.pt",
                "parameters": 734968253,
                "role": "Global multi-modal visual and semantic foundation anchor"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C1: Portrait Remediation ConvNeXt-Tiny",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/specialists/c5_convnext_tiny_epoch_3.pt",
                "parameters": 27820897,
                "role": "Facial skin tone and human portrait false-alarm neutralization"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C2: SPAI Multi-Frequency ViT",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c2_spai_vit_best.pt",
                "parameters": 21811969,
                "role": "Multi-frequency synthetic texture and generative noise detection"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C3: CommunityForensics ViT-Small (384x384)",
                "checkpoint_path": "/mnt/ai-storage/aigc_data/models/community_forensics_vit_small/model.safetensors",
                "parameters": 21811969,
                "role": "SOTA community benchmark transformer artifact detection"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C4: ConvNeXt High-Resolution Master",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c4_convnext_base_best.pt",
                "parameters": 27820897,
                "role": "High-resolution convolutional micro-texture analysis"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C5: divine2k ConvNeXt-Tiny Classifier",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c5_convnext_tiny_best.pt",
                "parameters": 27820897,
                "role": "Broad general synthetic generator classification"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C6: EfficientNet-B0 Fast Specialist",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c6_efficientnet_b0_best.pt",
                "parameters": 4008829,
                "role": "Lightweight boundary and edge artifact detection"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "C7: ResNet-50 Deep Specialist",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/specialists_v3/c7_resnet50_best.pt",
                "parameters": 23510081,
                "role": "Residual network baseline for deep GAN and diffusion synthesis"
            },
            {
                "subsystem": "V3_Specialist_Ensemble",
                "model_name": "V3 Learned 8-Expert Gating Network",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt",
                "parameters": 1224,
                "role": "Learned dynamic routing across C0–C7 specialist logits (Temperature: 1.15)"
            },
            {
                "subsystem": "V5_Spatial_Engine",
                "model_name": "V5-CAG Multi-Scale Spatial Engine & SegHead",
                "checkpoint_path": "/home/manan/aigc_robust_detection/checkpoints/experimental/v5/v5_champion_cag.pt",
                "parameters": 31088357,
                "role": "512px/768px/1024px hierarchical patch attention, bounding boxes, and continuous 64x64 mask"
            },
            {
                "subsystem": "Provenance_Engine",
                "model_name": "Decoupled Provenance & Metadata Subsystem",
                "checkpoint_path": "scripts/v5/v5_provenance_engine.py",
                "parameters": 0,
                "role": "EXIF, XMP, IPTC, C2PA Content Credentials, and AI software signature audit"
            }
        ]
    }
    
    manifest_file = os.path.join(report_dir, "final_model_manifest.json")
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Model Manifest saved to: {manifest_file} ✅")
    print("=" * 95)

if __name__ == "__main__":
    run_benchmarks()
