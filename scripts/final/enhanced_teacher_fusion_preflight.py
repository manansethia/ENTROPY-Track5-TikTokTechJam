#!/usr/bin/env python3
"""Build a fail-closed training manifest for the next ENTROPY teacher fusion.

This deliberately does not load the previously compiled 1.82B checkpoint.  That
artifact was assembled with non-strict, mismatched modules, so using it as a
teacher would propagate arbitrary weights.  A training run may use only teachers
that pass their architecture-specific strict-load audit and only rows whose
provenance is declared independent of the protected challenge benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Teacher:
    key: str
    checkpoint: str
    role: str
    audit: str
    eligible: bool
    reason: str


TEACHERS = (
    Teacher("C0", "checkpoints/production/final_champion_frozen_model.pt", "general semantic + residual anchor", "strict-pass", True, "verified standalone forward pass"),
    Teacher("C1", "checkpoints/portrait_rem_1/portrait_rem_1_epoch_3.pt", "portrait remediation", "strict-pass", True, "verified standalone forward pass"),
    Teacher("C2", "checkpoints/specialists_v3/c2_spai_vit_best.pt", "frequency specialist", "strict-pass", True, "verified standalone forward pass"),
    Teacher("C3", "community_forensics_vit_small/model.safetensors", "community ViT", "architecture-recovery", False, "installed Transformers version does not strictly load the checkpoint"),
    Teacher("C4", "checkpoints/specialists/c4_convnext_base_fp32.pt", "ConvNeXt specialist", "strict-pass", True, "verified standalone forward pass"),
    Teacher("C5", "checkpoints/specialists/c5_convnext_tiny_fp32.pt", "ConvNeXt specialist", "strict-pass", True, "verified standalone forward pass"),
    Teacher("C6", "checkpoints/specialists/c6_efficientnet_b0_epoch_3.pt", "edge specialist", "strict-pass", True, "verified standalone forward pass"),
    Teacher("C7", "checkpoints/specialists/c7_resnet50_epoch_3.pt", "ResNet specialist", "strict-pass", True, "verified standalone forward pass"),
    Teacher("V1", "checkpoints/tri_hybrid_v1/best_model.pt", "early hybrid", "component-unverified", False, "architecture and strict load not yet reconciled"),
    Teacher("V2", "models/aide_finetuned/checkpoint42.pth", "AIDE spectral", "architecture-recovery", False, "requires the exact OpenCLIP ConvNeXt-XXL environment and strict load"),
    Teacher("V3", "checkpoints/fusion/learned_multi_expert_gating_head.pt", "routing head", "component-pass", True, "eligible only after score-vector schema verification"),
    Teacher("V4", "checkpoints/experimental/v4_3_champion_config_c.pt", "partial-AI prototype", "component-unverified", False, "architecture and mask contract not yet reconciled"),
    Teacher("V5", "checkpoints/experimental/v5/v5_champion_cag.pt", "spatial localizer", "component-unverified", False, "backbone provenance is not a trained teacher"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lines(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL rows with path, label, source, split and sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-train", action="store_true", help="emit train-ready only when all release gates pass")
    args = parser.parse_args()

    required = {"path", "label", "source", "split", "sha256", "benchmark_isolated"}
    selected = [teacher for teacher in TEACHERS if teacher.eligible]
    rejected = [teacher for teacher in TEACHERS if not teacher.eligible]
    records = list(lines(args.manifest))
    missing = sorted({field for row in records for field in required - row.keys()})
    non_isolated = [row.get("path", "<unknown>") for row in records if not row.get("benchmark_isolated", False)]
    wrong_split = [row.get("path", "<unknown>") for row in records if row.get("split") not in {"train", "val", "test"}]

    state = "ready_for_teacher_cache"
    blockers: list[str] = []
    if missing:
        blockers.append("manifest missing required fields: " + ", ".join(missing))
    if non_isolated:
        blockers.append(f"{len(non_isolated)} rows lack benchmark-isolation provenance")
    if wrong_split:
        blockers.append(f"{len(wrong_split)} rows have an invalid split")
    if args.allow_train and rejected:
        blockers.append("all-teacher training requested but unqualified teachers remain: " + ", ".join(t.key for t in rejected))
    if blockers:
        state = "blocked"

    report = {
        "state": state,
        "manifest": str(args.manifest),
        "records": len(records),
        "selected_teachers": [asdict(teacher) for teacher in selected],
        "held_teachers": [asdict(teacher) for teacher in rejected],
        "manifest_sha256": sha256(args.manifest),
        "blockers": blockers,
        "next_step": "cache logits and localization outputs from selected teachers" if state == "ready_for_teacher_cache" else "resolve each listed blocker before any release training",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"state": state, "records": len(records), "selected": [t.key for t in selected], "held": [t.key for t in rejected], "blockers": blockers}, indent=2))
    return 0 if state == "ready_for_teacher_cache" else 2


if __name__ == "__main__":
    raise SystemExit(main())
