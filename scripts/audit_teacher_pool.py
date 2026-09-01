#!/usr/bin/env python3
"""Fail-closed inventory and validation harness for candidate teacher artifacts.

This utility intentionally separates *inspection* from a successful teacher
qualification.  A checkpoint is never called usable merely because it can be
deserialised.  Strict loading and a real-image forward pass are required when
an explicit, reviewed factory is supplied in the audit manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import sys

import torch


DEFAULT_TEACHERS = [
    ("V1", "checkpoints/tri_hybrid_v1/best_model.pt"),
    ("V2", "checkpoints/tri_hybrid_3stream_v2/best_model.pt"),
    ("V3", "checkpoints/tri_hybrid_45k_v3/best_model.pt"),
    ("V4", "checkpoints/experimental/v4_3_champion_config_c.pt"),
    ("V5", "checkpoints/experimental/v5/v5_champion_cag.pt"),
    ("C0", "checkpoints/production/final_champion_frozen_model.pt"),
    ("C1", "checkpoints/portrait_rem_1/portrait_rem_1_epoch_3.pt"),
    ("C2", "checkpoints/specialists_v3/c2_spai_vit_best.pt"),
    ("C3", "/mnt/ai-storage/aigc_data/models/community_forensics_vit_small/model.safetensors"),
    ("C4", "checkpoints/specialists_v3/c4_convnext_base_best.pt"),
    ("C5", "checkpoints/specialists_v3/c5_convnext_tiny_best.pt"),
    ("C6", "checkpoints/specialists_v3/c6_efficientnet_b0_best.pt"),
    ("C7", "checkpoints/specialists_v3/c7_resnet50_best.pt"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_state(payload: Any) -> tuple[dict[str, torch.Tensor] | None, str | None]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "gating_head_state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
                return value, key
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload, "root"
    return None, None


def tensor_summary(state: dict[str, torch.Tensor] | None) -> dict[str, Any]:
    if state is None:
        return {"state_tensor_count": 0, "state_parameter_count": 0, "sample_keys": []}
    return {
        "state_tensor_count": len(state),
        "state_parameter_count": sum(value.numel() for value in state.values()),
        "sample_keys": list(state)[:12],
        "dtypes": sorted({str(value.dtype) for value in state.values()}),
    }


def build_factory(spec: str, kwargs: dict[str, Any]):
    module_name, separator, attr_name = spec.partition(":")
    if not separator:
        raise ValueError("factory must be written as module:callable")
    factory = getattr(importlib.import_module(module_name), attr_name)
    return factory(**kwargs)


def strict_probe(entry: dict[str, Any], checkpoint: Path, image: Path | None) -> dict[str, Any]:
    """Strictly validates only explicitly reviewed architecture factories."""
    factory = entry.get("factory")
    if not factory:
        return {"status": "NOT_QUALIFIED", "reason": "no reviewed architecture factory supplied"}
    if image is None:
        return {"status": "NOT_QUALIFIED", "reason": "no real probe image supplied"}
    try:
        if entry.get("loader") == "hf_vit_binary":
            from safetensors.torch import load_file
            from transformers import ViTConfig, ViTForImageClassification
            model = ViTForImageClassification(ViTConfig(**entry["vit_config"])) 
            state, state_key = load_file(str(checkpoint), device="cpu"), "safetensors"
        else:
            model = build_factory(factory, entry.get("factory_kwargs", {}))
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state, state_key = resolve_state(payload)
        if state is None:
            return {"status": "NOT_QUALIFIED", "reason": "no tensor state dict"}
        model.load_state_dict(state, strict=True)
        model.eval()
        from PIL import Image
        from torchvision.transforms import Compose, Normalize, Resize, ToTensor
        normalization = entry.get("normalization", {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})
        resolution = int(entry.get("input_resolution", 224))
        sample = Compose([Resize((resolution, resolution)), ToTensor(), Normalize(**normalization)])(Image.open(image).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            output = model(sample)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return {"status": "QUALIFIED", "state_key": state_key, "output_shape": list(output.shape), "output_sample": output.detach().float().flatten()[:8].tolist()}
    except Exception as error:  # report the precise failed qualification, never fall back.
        return {"status": "NOT_QUALIFIED", "reason": str(error), "traceback": traceback.format_exc(limit=3)}


def inspect(entry: dict[str, Any], root: Path, image: Path | None) -> dict[str, Any]:
    raw_path = Path(entry["path"])
    checkpoint = (raw_path if raw_path.is_absolute() else root / raw_path).resolve()
    record: dict[str, Any] = {"id": entry["id"], "requested_path": entry["path"], "exists": checkpoint.is_file()}
    if not checkpoint.is_file():
        record.update({"qualification": {"status": "NOT_QUALIFIED", "reason": "artifact missing"}})
        return record
    record.update({"file_size_bytes": checkpoint.stat().st_size, "sha256": sha256(checkpoint)})
    if checkpoint.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys()}
        record.update({"format": "safetensors", **tensor_summary(state), "qualification": strict_probe(entry, checkpoint, image)})
        return record
    try:
        payload = torch.load(checkpoint, map_location="meta", weights_only=True)
        state, state_key = resolve_state(payload)
        record.update({"format": "torch", "payload_type": type(payload).__name__, "payload_keys": list(payload)[:30] if isinstance(payload, dict) else [], "state_key": state_key, **tensor_summary(state)})
        record["qualification"] = strict_probe(entry, checkpoint, image)
    except Exception as error:
        record["qualification"] = {"status": "NOT_QUALIFIED", "reason": f"safe inspection failed: {error}"}
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="project root; never published in the report")
    parser.add_argument("--manifest", type=Path, help="reviewed JSON list of teachers and optional factories")
    parser.add_argument("--probe-image", type=Path, help="one real image for qualified strict forward probes")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    # Reviewed factories live in the project; executing this script puts
    # scripts/ on sys.path, not the project root.
    sys.path.insert(0, str(args.root.resolve()))
    entries = json.loads(args.manifest.read_text()) if args.manifest else [{"id": ident, "path": path} for ident, path in DEFAULT_TEACHERS]
    report = {"generated_at": datetime.now(timezone.utc).isoformat(), "schema_version": 1, "policy": "Inspection is not teacher qualification. QUALIFIED requires strict=True and a real-image forward pass.", "teachers": [inspect(entry, args.root, args.probe_image) for entry in entries]}
    report["summary"] = {"qualified": sum(item.get("qualification", {}).get("status") == "QUALIFIED" for item in report["teachers"]), "not_qualified": sum(item.get("qualification", {}).get("status") != "QUALIFIED" for item in report["teachers"])}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
