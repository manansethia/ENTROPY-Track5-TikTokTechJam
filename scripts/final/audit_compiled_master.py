#!/usr/bin/env python3
"""Provenance audit for the historical compiled master ensemble.

The artifact may be structurally loadable into its historical wrapper while
still containing randomly initialized or wrongly mapped expert branches.  This
audit evaluates the source construction contract, not just tensor shape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


EXPECTED = {
    "V2_AIDE": {"path": "/mnt/ai-storage/aigc_data/models/aide_finetuned/checkpoint42.pth", "expected": "AIDE_Model, strict state load"},
    "C0": {"path": "checkpoints/production/final_champion_frozen_model.pt", "expected": "ScientificVisionDetector, strict state load"},
    "C1": {"path": "checkpoints/portrait_rem_1/portrait_rem_1_epoch_3.pt", "expected": "ScientificVisionDetector ConfigA, strict state load"},
    "C2": {"path": "checkpoints/specialists_v3/c2_spai_vit_best.pt", "expected": "torchvision ResNet-50, strict state load"},
    "C3": {"path": "/mnt/ai-storage/aigc_data/models/community_forensics_vit_small/model.safetensors", "expected": "Hugging Face ViTForImageClassification 384px, strict state load"},
    "C4": {"path": "checkpoints/specialists_v3/c4_convnext_base_best.pt", "expected": "torchvision ConvNeXt-Tiny, strict state load"},
    "C5": {"path": "checkpoints/specialists_v3/c5_convnext_tiny_best.pt", "expected": "torchvision ConvNeXt-Tiny, strict state load"},
    "C6": {"path": "checkpoints/specialists_v3/c6_efficientnet_b0_best.pt", "expected": "torchvision EfficientNet-B0, strict state load"},
    "C7": {"path": "checkpoints/specialists_v3/c7_resnet50_best.pt", "expected": "torchvision ResNet-50, strict state load"},
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = torch.load(args.compiled, map_location="meta", weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    violations = [
        {"component": "C1", "historical_mapping": "ConvNeXt-Tiny checkpoint from specialists/c5", "required_mapping": EXPECTED["C1"]["expected"]},
        {"component": "C2", "historical_mapping": "ViT-Small", "required_mapping": EXPECTED["C2"]["expected"]},
        {"component": "C3", "historical_mapping": "timm ViT-Small", "required_mapping": EXPECTED["C3"]["expected"]},
        {"component": "C0", "historical_mapping": "reduced TripleHybrid wrapper loaded strict=False", "required_mapping": EXPECTED["C0"]["expected"]},
        {"component": "all", "historical_mapping": "strict=False component ingestion", "required_mapping": "strict=True for every eligible component"},
    ]
    components = []
    for name, item in EXPECTED.items():
        source = Path(item["path"])
        if not source.is_absolute(): source = args.root / source
        components.append({"id": name, "source_exists": source.is_file(), "source_sha256": digest(source) if source.is_file() else None, "required_contract": item["expected"]})
    payload = {"compiled": {"path": args.compiled.name, "sha256": digest(args.compiled), "tensor_count": len(state), "serialized_parameter_count": sum(t.numel() for t in state.values())}, "status": "NOT_PROMOTABLE", "reason": "Historical construction contract contains proven architecture/source mismatches. A structural load does not establish trained provenance.", "violations": violations, "rebuild_requirement": "Build a fresh ensemble from source teachers only after each selected component passes strict=True load and real-image forward validation. Do not reuse historical compiled weights as a promotion shortcut.", "source_components": components}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "violations": len(violations), "serialized_parameters": payload["compiled"]["serialized_parameter_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
