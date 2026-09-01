import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.tri_hybrid_detector import MasterEnsembleDetector
from scripts.data import AIGCDataset
from scripts.transforms import prepare_inputs


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train Tri-Stream Robust AIGC Detector")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg.get("seed", 42))

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = MasterEnsembleDetector(**cfg.get("models", {})).to(device)
    param_report = model.parameter_report()
    print("Parameter report:", param_report)
    assert param_report["total"] < 2_000_000_000, "FATAL: Total parameters exceed 2B limit!"

    train_data_dir = cfg["paths"]["train_data_dir"]
    train_dataset = AIGCDataset(train_data_dir, cfg, target_size=224)
    loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=float(cfg["training"]["base_lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(cfg["training"]["mixed_precision"]) and device.type == "cuda"
    scaler = GradScaler(enabled=amp_enabled)
    accum = int(cfg["training"]["gradient_accumulation_steps"])
    os.makedirs(cfg["paths"]["output_dir"], exist_ok=True)

    print(f"Starting training for {cfg['training']['epochs']} epochs on {len(train_dataset)} images...")

    for epoch in range(int(cfg["training"]["epochs"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{cfg['training']['epochs']}")

        for step, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            clip_in, siglip_in, raw_in = prepare_inputs(images)

            with autocast(enabled=amp_enabled):
                logits = model(clip_in, siglip_in, raw_in)
                loss = criterion(logits, targets) / accum

            scaler.scale(loss).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.trainable_parameters(), float(cfg["training"].get("max_grad_norm", 1.0))
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item() * accum
            pbar.set_postfix(loss=f"{(loss.item() * accum):.4f}")

        epoch_loss = running_loss / max(len(loader), 1)
        checkpoint = {
            "state_dict": model.state_dict(),
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "config": cfg,
            "parameter_report": param_report,
        }
        ckpt_path = os.path.join(
            cfg["paths"]["output_dir"], f"master_model_epoch_{epoch + 1}.pth"
        )
        torch.save(checkpoint, ckpt_path)
        print(f"Epoch {epoch + 1} complete. Mean loss: {epoch_loss:.5f}. Saved checkpoint -> {ckpt_path}")


if __name__ == "__main__":
    main()
