# =====================================================================================
# PRODUCTION HEAVY SPECIALIST TRAINING ENGINE (FULL CONVNEXT BACKBONE)
# Full 28.6M Parameter Backbone Training on 20,000 Balanced Images (236k Manifest)
# Hardware Target: Buildabot RTX 3050 (4.5 - 5.5 GB VRAM Target, 8 CPU Workers)
# =====================================================================================

import os, sys, time, json, random, hashlib, gc, subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Deterministic Seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 85)
print("  PRODUCTION HEAVY SPECIALIST TRAINING (CONVNEXT BACKBONE)")
print("=" * 85)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Hardware Device : {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"Total VRAM Available   : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

# 1. LOAD 236K MANIFEST & CONSTRUCT 20K BALANCED PHYSICAL POOL
MANIFEST_PATH = "/home/manan/aigc_robust_detection/reports/master_highres_train_manifest.json"
with open(MANIFEST_PATH, "r") as f:
    manifest_data = json.load(f)

samples = manifest_data.get("samples", [])
real_pool = [s for s in samples if s["label"] == 0]
aigc_pool = [s for s in samples if s["label"] == 1]
random.shuffle(real_pool)
random.shuffle(aigc_pool)

target_n = min(len(real_pool), len(aigc_pool), 10000)
train_samples = real_pool[:target_n] + aigc_pool[:target_n]
random.shuffle(train_samples)

print(f"Total Manifest Corpus  : {len(samples)} images")
print(f"Balanced Training Pool : {len(train_samples)} physical images (REAL: {target_n}, AIGC: {target_n})\n")

# 2. HIGH-THROUGHPUT DATASET WITH COLOR JITTER & RESOLUTION RESCALE
class RobustHighResDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = s["canonical_path"]
        label = float(s["label"])
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                tensor = self.transform(img)
            return tensor, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)

dataset = RobustHighResDataset(train_samples)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    prefetch_factor=4,
    persistent_workers=True
)

# 3. INSTANTIATE FULL CONVNEXT BACKBONE & LOAD PRETRAINED SPECIALIST WEIGHTS
print("--- Loading ConvNeXt-Tiny Pretrained Weights (28.6M Parameters) ---")
model = models.convnext_tiny(num_classes=1)

init_weight_path = "/mnt/ai-storage/aigc_data/models/divine2k_ensemble/convNext_final.pth"
sd = torch.load(init_weight_path, map_location="cpu", weights_only=False)
model.load_state_dict(sd)
print(f"Loaded initialization checkpoint: {init_weight_path}")

# Unfreeze entire feature extractor and classifier
for param in model.parameters():
    param.requires_grad = True

model = model.to(DEVICE)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Active Trainable Parameters: {total_params:,} (100% unconstrained)\n")

# 4. OPTIMIZER, DIFFERENTIAL LR, AND MIXED PRECISION SCALER
optimizer = torch.optim.AdamW([
    {"params": model.features.parameters(), "lr": 2e-5, "weight_decay": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-4, "weight_decay": 1e-4}
])
criterion = nn.BCEWithLogitsLoss()
scaler = torch.amp.GradScaler('cuda')

# 5. LIVE GPU TRAINING LOOP WITH COMPUTE TELEMETRY
EPOCHS = 3
total_batches = len(loader)
print(f"--- Starting Full-Scale GPU Training ({EPOCHS} Epochs, {total_batches} Batches/Epoch, Batch Size 32) ---")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    
    for batch_idx, (images, targets) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda'):
            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        if batch_idx % 10 == 0 or batch_idx == total_batches:
            avg_loss = running_loss / batch_idx
            elapsed = time.time() - epoch_start
            samples_sec = (batch_idx * 32) / elapsed
            vram_gb = torch.cuda.memory_allocated(0) / (1024**3)
            vram_res_gb = torch.cuda.memory_reserved(0) / (1024**3)
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx:04d}/{total_batches}] | Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) | {samples_sec:.1f} img/s | VRAM Allocated: {vram_gb:.2f} GB (Reserved: {vram_res_gb:.2f} GB)")
            sys.stdout.flush()

    epoch_time = time.time() - epoch_start
    print(f"\n>> Epoch [{epoch}/{EPOCHS}] Completed in {epoch_time:.1f}s | Final Avg Loss: {running_loss/total_batches:.4f}\n")
    
    # Save Checkpoint
    os.makedirs("/home/manan/aigc_robust_detection/checkpoints/specialists", exist_ok=True)
    out_ckpt = f"/home/manan/aigc_robust_detection/checkpoints/specialists/convnext_heavy_epoch_{epoch}.pt"
    torch.save(model.state_dict(), out_ckpt)
    print(f"   Saved Checkpoint: {out_ckpt}\n")

print("=" * 85)
print("  FULL-SCALE CONVNEXT SPECIALIST TRAINING COMPLETE")
print("=" * 85)
