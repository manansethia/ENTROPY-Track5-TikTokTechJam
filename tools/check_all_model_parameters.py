#!/usr/bin/env python3
"""Programmatic Parameter Audit Tool.
Accurately counts total and trainable parameters for every backbone and forensic module.
Saves results to reports/model_parameter_audit.json.
"""

import json
import os
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from transformers import AutoModel, CLIPModel, ConvNextV2Model

from models.fft_spectral_detector import FFTSpectralFeatureExtractor, FFTEnergyClassifierHead
from models.edge_artifact_detector import EdgeArtifactFeatureExtractor
from models.patch_mil_expert import PatchMILExpert
from models.dual_evidence_router import DualEvidenceReliabilityRouter
from models.srm_filters import WaveletResidualBlock


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main():
    print("=== Running Authoritative Model Parameter Audit ===")
    audit = {}
    models_base = Path("/mnt/ai-storage/aigc_data/models")
    
    # 1. Foundation Models
    models_to_check = [
        ("SigLIP-Base-224", models_base / "siglip_base_224", "hf_automodel"),
        ("SigLIP-SO400M-224", models_base / "siglip_so400m_224", "hf_automodel"),
        ("CLIP-ViT-L-14", models_base / "clip_vitl14", "hf_clip"),
        ("DINOv2-Large", models_base / "dinov2_large", "hf_automodel"),
        ("DINOv2-Large-Registers", models_base / "dinov2_registers_large", "hf_automodel"),
        ("ConvNeXt-V2-Tiny", models_base / "convnextv2_tiny", "hf_convnext"),
        ("EVA-02-Large-448", models_base / "eva02_large_patch14_448", "hf_automodel"),
    ]

    for name, path, mtype in models_to_check:
        if path.exists():
            try:
                print(f"Loading {name} from {path}...")
                if mtype == "hf_automodel":
                    m = AutoModel.from_pretrained(str(path))
                elif mtype == "hf_clip":
                    m = CLIPModel.from_pretrained(str(path))
                elif mtype == "hf_convnext":
                    m = ConvNextV2Model.from_pretrained(str(path))
                
                tot, train = count_params(m)
                audit[name] = {
                    "path": str(path),
                    "total_parameters": tot,
                    "trainable_parameters": train,
                    "total_million": round(tot / 1e6, 2),
                    "status": "LOADED_VERIFIED",
                }
                print(f"  --> {name}: {tot:,} parameters ({tot/1e6:.2f}M)")
                del m
                torch.cuda.empty_cache()
            except Exception as e:
                audit[name] = {"path": str(path), "status": f"LOAD_ERROR: {str(e)}"}
                print(f"  --> {name} Error: {e}")
        else:
            audit[name] = {"path": str(path), "status": "NOT_ON_DISK"}
            print(f"  --> {name}: Not found on disk at {path}")

    # 2. Specialized Forensic Modules
    print("\nAuditing Forensic Modules...")
    forensic_modules = [
        ("FFT-Spectral-Extractor", FFTSpectralFeatureExtractor(num_radial_bins=64)),
        ("FFT-Energy-Classifier-Head", FFTEnergyClassifierHead()),
        ("Edge-Artifact-Extractor", EdgeArtifactFeatureExtractor(out_dim=256)),
        ("Patch-MIL-Expert", PatchMILExpert(patch_dim=768, out_dim=512)),
        ("Dual-Evidence-Reliability-Router", DualEvidenceReliabilityRouter()),
        ("Wavelet-SRM-Residual-Block", WaveletResidualBlock()),
    ]

    for name, module in forensic_modules:
        tot, train = count_params(module)
        audit[name] = {
            "total_parameters": tot,
            "trainable_parameters": train,
            "total_million": round(tot / 1e6, 4),
            "status": "INITIALIZED_VERIFIED",
        }
        print(f"  --> {name}: {tot:,} parameters ({tot/1e6:.4f}M)")

    # 3. Compute Active 4-Stream + 6-Stream Ensemble Parameter Totals
    active_pool_models = ["SigLIP-SO400M-224", "DINOv2-Large-Registers", "CLIP-ViT-L-14", "ConvNeXt-V2-Tiny", "FFT-Spectral-Extractor", "Edge-Artifact-Extractor", "Patch-MIL-Expert", "Dual-Evidence-Reliability-Router"]
    total_active_params = sum(audit[k].get("total_parameters", 0) for k in active_pool_models if k in audit and "total_parameters" in audit[k])

    audit["ACTIVE_FOUNDATION_POOL_TOTAL"] = {
        "total_parameters": total_active_params,
        "total_billion": round(total_active_params / 1e9, 4),
        "hard_cap_billion": 2.0,
        "under_cap": total_active_params < 2.0e9,
    }
    print(f"\n========================================================")
    print(f"TOTAL ACTIVE ENSEMBLE PARAMETERS: {total_active_params:,} ({total_active_params/1e9:.4f}B)")
    print(f"HARD CAP: 2.0000B | PASSES CAP CONSTRAINT: {total_active_params < 2.0e9}")
    print(f"========================================================")

    # Save output
    out_dir = Path("reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "model_parameter_audit.json"
    with open(out_file, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\nSaved authoritative audit report to {out_file}!")


if __name__ == "__main__":
    main()
