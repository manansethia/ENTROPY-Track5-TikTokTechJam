#!/usr/bin/env python3
"""High-throughput multi-worker foundation feature extractor.
Utilizes full 12-thread CPU DataLoader with pinned memory and FP16 AMP on GPU (VRAM target: 4.8–5.4 GB).
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, CLIPModel

from scripts.data import VALID_EXTS

torch.backends.cudnn.benchmark = True


def find_image_files(root_dir):
    LOCKED_BENCHMARKS = [
        "validation_LOCKED", "validation_locked", "aigibench_eval",
        "synthbuster", "wildrf", "synthwildx", "vct2", "chameleon"
    ]
    abs_root = os.path.abspath(root_dir)
    for locked in LOCKED_BENCHMARKS:
        if locked in abs_root:
            raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: Attempted to extract features from locked benchmark: {abs_root}!")

    files = []
    for base, _, names in os.walk(root_dir):
        for n in sorted(names):
            if n.lower().endswith(VALID_EXTS):
                full_p = os.path.abspath(os.path.join(base, n))
                for locked in LOCKED_BENCHMARKS:
                    if locked in full_p:
                        raise RuntimeError(f"CRITICAL LEAKAGE DETECTED: File {full_p} is part of locked benchmark '{locked}'!")
                files.append(os.path.join(base, n))
    return sorted(files)


class ImagePathsDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                return self.transform(img)
            return img
        except Exception:
            # Fallback to black image if corrupted
            black = Image.new("RGB", (224, 224), (0, 0, 0))
            if self.transform:
                return self.transform(black)
            return black


def get_vram_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        res = torch.cuda.memory_reserved() / (1024 ** 3)
        return f"[VRAM: {alloc:.2f}GB alloc / {res:.2f}GB res]"
    return ""


def extract_siglip_features(image_paths, model_dir, device="cuda", batch_size=128, num_workers=8):
    print(f"\n[1/3] Extracting SigLIP features from {model_dir} (Batch: {batch_size}, Workers: {num_workers})...")
    # Standard SigLIP transform
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ds = ImagePathsDataset(image_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=3)

    model = AutoModel.from_pretrained(model_dir).to(device)
    model.eval()

    features_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"SigLIP Extraction {get_vram_str()}"):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                vision_out = model.vision_model(pixel_values=batch)
                feats = vision_out.pooler_output
            features_list.append(feats.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(features_list, axis=0)


def extract_clip_features(image_paths, clip_dir="/mnt/ai-storage/aigc_data/models/clip_vitl14", device="cuda", batch_size=96, num_workers=8):
    print(f"\n[2/3] Extracting CLIP ViT-L/14 features from {clip_dir} (Batch: {batch_size}, Workers: {num_workers})...")
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
    ])
    ds = ImagePathsDataset(image_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=3)

    model = CLIPModel.from_pretrained(clip_dir).to(device)
    model.eval()

    features_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"CLIP Extraction {get_vram_str()}"):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                vision_out = model.vision_model(pixel_values=batch)
                feats = vision_out.pooler_output
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(features_list, axis=0)


def extract_dinov2_features(image_paths, dinov2_dir="/mnt/ai-storage/aigc_data/models/dinov2_large", device="cuda", batch_size=96, num_workers=8):
    print(f"\n[3/3] Extracting DINOv2-Large features from {dinov2_dir} (Batch: {batch_size}, Workers: {num_workers})...")
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImagePathsDataset(image_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=3)

    model = AutoModel.from_pretrained(dinov2_dir).to(device)
    model.eval()

    features_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"DINOv2 Extraction {get_vram_str()}"):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(batch)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state[:, 0]
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(features_list, axis=0)


def extract_convnext_features(image_paths, convnext_dir, device="cuda", batch_size=128, num_workers=8):
    print(f"\n[4/4] Extracting ConvNeXt-V2-Tiny features from {convnext_dir} (Batch: {batch_size}, Workers: {num_workers})...")
    transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = ImagePathsDataset(image_paths, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=3)

    from transformers import ConvNextV2Model
    model = ConvNextV2Model.from_pretrained(convnext_dir).to(device)
    model.eval()

    features_list = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"ConvNeXt Extraction {get_vram_str()}"):
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                outputs = model(batch)
                feats = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=[-2, -1])
                feats = feats / feats.norm(dim=-1, keepdim=True)
            features_list.append(feats.float().cpu().numpy())

    del model
    torch.cuda.empty_cache()
    return np.concatenate(features_list, axis=0)


def main():
    p = argparse.ArgumentParser(description="High-Throughput Foundation Feature Extractor")
    p.add_argument("--data_dir", default="/mnt/ai-storage/aigc_data/datasets/balanced_scaled_train")
    p.add_argument("--output_h5", default="/mnt/ai-storage/aigc_data/cache/balanced_features_4stream.h5")
    p.add_argument("--siglip_dir", default="/mnt/ai-storage/aigc_data/models/siglip_base_224")
    p.add_argument("--clip_dir", default="/mnt/ai-storage/aigc_data/models/clip_vitl14")
    p.add_argument("--dinov2_dir", default="/mnt/ai-storage/aigc_data/models/dinov2_large")
    p.add_argument("--convnext_dir", default="/mnt/ai-storage/aigc_data/models/convnextv2_tiny")
    p.add_argument("--include_dinov2", action="store_true", default=True, help="Include DINOv2-Large stream")
    p.add_argument("--include_convnext", action="store_true", default=True, help="Include ConvNeXt-V2 stream")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"High-Throughput Feature Extraction initialized on {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    real_paths = find_image_files(os.path.join(args.data_dir, "real"))
    synthetic_paths = find_image_files(os.path.join(args.data_dir, "synthetic"))
    all_paths = real_paths + synthetic_paths
    labels = np.array([0] * len(real_paths) + [1] * len(synthetic_paths), dtype=np.int64)

    print(f"Total dataset: {len(all_paths):,} images ({len(real_paths):,} real, {len(synthetic_paths):,} synthetic)")
    if not all_paths:
        print("ERROR: No images found.")
        sys.exit(1)

    out_path = Path(args.output_h5)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    siglip_feats = extract_siglip_features(all_paths, args.siglip_dir, device=device, batch_size=args.batch_size, num_workers=args.num_workers)
    clip_feats = extract_clip_features(all_paths, clip_dir=args.clip_dir, device=device, batch_size=min(args.batch_size, 96), num_workers=args.num_workers)
    dinov2_feats = None
    if args.include_dinov2 and os.path.isdir(args.dinov2_dir):
        dinov2_feats = extract_dinov2_features(all_paths, dinov2_dir=args.dinov2_dir, device=device, batch_size=min(args.batch_size, 96), num_workers=args.num_workers)

    convnext_feats = None
    if args.include_convnext and os.path.isdir(args.convnext_dir):
        convnext_feats = extract_convnext_features(all_paths, convnext_dir=args.convnext_dir, device=device, batch_size=args.batch_size, num_workers=args.num_workers)

    print(f"\nWriting compressed HDF5 cache to {out_path}...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("siglip_features", data=siglip_feats, compression="gzip")
        f.create_dataset("clip_features", data=clip_feats, compression="gzip")
        if dinov2_feats is not None:
            f.create_dataset("dinov2_features", data=dinov2_feats, compression="gzip")
        if convnext_feats is not None:
            f.create_dataset("convnext_features", data=convnext_feats, compression="gzip")
        f.create_dataset("labels", data=labels)
        f.create_dataset("paths", data=[str(p).encode("utf-8") for p in all_paths])

    print(f"\nSUCCESS: Cached {len(all_paths):,} feature vectors to {out_path}!")
    shape_str = f"SigLIP: {siglip_feats.shape}, CLIP: {clip_feats.shape}"
    if dinov2_feats is not None:
        shape_str += f", DINOv2: {dinov2_feats.shape}"
    if convnext_feats is not None:
        shape_str += f", ConvNeXt: {convnext_feats.shape}"
    print(shape_str)


if __name__ == "__main__":
    main()
