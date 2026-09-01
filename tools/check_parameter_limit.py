#!/usr/bin/env python3
"""Automated Parameter Compliance Validator (< 2 Billion Parameters).

Verifies that model architectures or saved checkpoints strictly comply with the
competition limit of < 2,000,000,000 total parameters.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from models.tri_hybrid_detector import MasterEnsembleDetector

PARAMETER_LIMIT = 2_000_000_000


def analyze_model(model: torch.nn.Module, name: str = "MasterEnsembleDetector"):
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    module_breakdown = {}

    for mod_name, module in model.named_children():
        mod_total = sum(p.numel() for p in module.parameters())
        mod_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        mod_frozen = mod_total - mod_trainable
        module_breakdown[mod_name] = {
            "total": mod_total,
            "trainable": mod_trainable,
            "frozen": mod_frozen,
        }
        total_params += mod_total
        trainable_params += mod_trainable
        frozen_params += mod_frozen

    # Handle parameters not caught in immediate named_children
    all_total = sum(p.numel() for p in model.parameters())
    all_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_frozen = all_total - all_trainable

    print("=" * 70)
    print(f"  AIGC DETECTOR PARAMETER COMPLIANCE AUDIT")
    print("=" * 70)
    print(f"Model Name / Architecture: {name}")
    print(f"Parameter Limit Ceiling:   {PARAMETER_LIMIT:,} (<2.0B)")
    print("-" * 70)
    print(f"{'Module / Sub-Network':<30} | {'Total Params':<15} | {'Trainable':<12} | {'Frozen':<12}")
    print("-" * 70)
    for mod_name, stats in module_breakdown.items():
        print(
            f"{mod_name:<30} | {stats['total']:>15,} | {stats['trainable']:>12,} | {stats['frozen']:>12,}"
        )
    print("-" * 70)
    print(f"{'TOTAL PARAMETERS':<30} | {all_total:>15,} | {all_trainable:>12,} | {all_frozen:>12,}")
    print("=" * 70)

    # Memory footprint estimates
    fp32_mb = (all_total * 4) / (1024 ** 2)
    fp16_mb = (all_total * 2) / (1024 ** 2)
    int8_mb = (all_total * 1) / (1024 ** 2)
    print(f"Estimated Model Memory Footprint:")
    print(f"  FP32: {fp32_mb:,.2f} MB ({fp32_mb / 1024:.2f} GB)")
    print(f"  FP16 / BF16: {fp16_mb:,.2f} MB ({fp16_mb / 1024:.2f} GB)")
    print(f"  INT8: {int8_mb:,.2f} MB ({int8_mb / 1024:.2f} GB)")
    print("-" * 70)

    status = "PASS" if all_total < PARAMETER_LIMIT else "FAIL"
    pct_used = (all_total / PARAMETER_LIMIT) * 100.0
    print(f"STATUS: {status} ({pct_used:.2f}% of 2B parameter budget used)")
    print("=" * 70)

    if status == "FAIL":
        print(f"\n[FATAL ERROR] Model exceeds limit by {all_total - PARAMETER_LIMIT:,} parameters!")
        sys.exit(1)
    else:
        print(f"\n[SUCCESS] Model complies with competition parameter constraints.")
        return 0


def main():
    p = argparse.ArgumentParser(description="Check model parameter compliance")
    p.add_argument("--config", default="configs/train_config.yaml", help="Path to model config")
    p.add_argument("--checkpoint", default=None, help="Optional checkpoint file to inspect")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print(f"Instantiating model from config: {args.config}...")
    model = MasterEnsembleDetector(**cfg.get("models", {}))

    if args.checkpoint:
        print(f"Loading weights from checkpoint: {args.checkpoint}...")
        state = torch.load(args.checkpoint, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)

    analyze_model(model, name="Tri-Stream MasterEnsembleDetector")


if __name__ == "__main__":
    main()
