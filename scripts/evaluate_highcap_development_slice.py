#!/usr/bin/env python3
"""Small, declared non-organizer development-slice evaluation for HighCap.

This is a smoke evaluation, not a release benchmark. It purposefully uses only
the supplied local corpus roots and records exactly which files were sampled.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import torch
from PIL import Image


def images(root: Path, limit: int) -> list[Path]:
    return sorted(path for path in root.glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})[:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument("--partial-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("highcap", args.model_source)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    model = module.HighCapacityStudentForensicModel()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    transform = module.T.Compose([module.T.Resize((384, 384)), module.T.ToTensor(), module.T.Normalize([.485, .456, .406], [.229, .224, .225])])
    records = []
    for expected, root in (("REAL", args.real_root), ("PARTIAL_AI", args.partial_root)):
        for path in images(root, args.limit):
            tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0)
            with torch.inference_mode():
                result = model(tensor)
            probs = result["probabilities"].squeeze().tolist()
            predicted = ("REAL", "PARTIAL_AI", "FULL_AIGC")[int(max(range(3), key=lambda index: probs[index]))]
            records.append({"relative_path": str(path.relative_to(root)), "expected": expected, "predicted": predicted, "probabilities": [round(float(value), 6) for value in probs]})
    total = len(records)
    correct = sum(record["expected"] == record["predicted"] for record in records)
    report = {"scope": "Non-organizer development smoke slice only; not release-quality evidence.", "model": {"checkpoint": args.checkpoint.name, "strict_load": "PASS", "parameters": sum(p.numel() for p in model.parameters())}, "slice": {"real_root_label": args.real_root.name, "partial_root_label": args.partial_root.name, "per_class_limit": args.limit}, "metrics": {"accuracy": round(correct / total, 6) if total else None, "total": total, "correct": correct}, "records": records}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["metrics"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
