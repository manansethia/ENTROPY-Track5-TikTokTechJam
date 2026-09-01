# =====================================================================================
# PRODUCTION MASTER HIGH-RESOLUTION SPECIALIST TRAINING ENGINE
# Active Multi-Epoch GPU Training on 20,000 Balanced Physical Images (236k Corpus)
# Hardware: Buildabot RTX 3050 (6GB VRAM) with Mixed Precision & PyTorch DataLoader
# =====================================================================================

import os, sys, time, json, random, hashlib, gc
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Set deterministic seed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

print("=" * 85)
print("  PRODUCTION MASTER HIGH-RESOLUTION SPECIALIST TRAINING")
print("=" * 85)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Active Compute Device : {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"Total VRAM Available  : {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\n")

# 1. LOAD MASTER MANIFEST & BALANCE TRAINING SAMPLES
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

print(f"Loaded Master Pool: {len(samples)} total images")
print(f"Balanced Training Split: {len(train_samples)} physical images (Real: {target_n}, AIGC: {target_n})\n")

# 2. METHOD C MULTI-CROP DATASET
class HighResMultiCropDataset(Dataset):
    def __init__(self, samples: List[Dict]):
        self.samples = samples
        self.tf_global = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])
        self.tf_crop = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        path = s["canonical_path"]
        label = float(s["label"])
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size
            g_img = self.tf_global(img)
            
            cw, ch = min(w, 224), min(h, 224)
            tl = self.tf_crop(img.crop((0, 0, cw, ch)).resize((224, 224)))
            tr = self.tf_crop(img.crop((w - cw, 0, w, ch)).resize((224, 224)))
            bl = self.tf_crop(img.crop((0, h - ch, cw, h)).resize((224, 224)))
            br = self.tf_crop(img.crop((w - cw, h - ch, w, h)).resize((224, 224)))
            crops = torch.stack([tl, tr, bl, br]) # [4, 3, 224, 224]
            return g_img, crops, torch.tensor(label, dtype=torch.float32)
        except Exception:
            return torch.zeros(3, 224, 224), torch.zeros(4, 3, 224, 224), torch.tensor(label, dtype=torch.float32)

dataset = HighResMultiCropDataset(train_samples)
loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

# 3. SPECIALIST ARCHITECTURE
class HighResSpecialistModel(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        self.crop_proj = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, embed_dim)
        )
        self.fusion_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, global_img: torch.Tensor, crops: torch.Tensor) -> torch.Tensor:
        B = global_img.size(0)
        g_feat = self.global_proj(global_img).unsqueeze(1)
        crops_flat = crops.view(B * 4, 3, 224, 224)
        c_feat = self.crop_proj(crops_flat).view(B, 4, -1)
        attn_out, _ = self.fusion_attn(query=g_feat, key=c_feat, value=c_feat)
        fused = torch.cat([g_feat.squeeze(1), attn_out.squeeze(1)], dim=-1)
        return self.classifier(fused).squeeze(-1)

# 4. MULTI-EPOCH ACTIVE TRAINING LOOP
model = HighResSpecialistModel().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scaler = torch.amp.GradScaler('cuda')
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 3
total_batches = len(loader)
print(f"--- Starting Active GPU Training ({EPOCHS} Epochs, {total_batches} Batches/Epoch) ---")

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_start = time.time()
    running_loss = 0.0
    
    for batch_idx, (g_img, crops, targets) in enumerate(loader, 1):
        g_img = g_img.to(DEVICE, non_blocking=True)
        crops = crops.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            preds = model(g_img, crops)
            loss = criterion(preds, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        if batch_idx % 25 == 0 or batch_idx == total_batches:
            avg_loss = running_loss / batch_idx
            samples_sec = (batch_idx * 16) / (time.time() - epoch_start)
            vram_mb = torch.cuda.memory_allocated(0) / (1024**2)
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{batch_idx:04d}/{total_batches}] | Loss: {loss.item():.4f} (Avg: {avg_loss:.4f}) | {samples_sec:.1f} samples/s | VRAM: {vram_mb:.1f} MB")
            sys.stdout.flush()

    epoch_time = time.time() - epoch_start
    print(f"\n>> Epoch [{epoch}/{EPOCHS}] Completed in {epoch_time:.1f}s | Final Avg Loss: {running_loss/total_batches:.4f}\n")
    
    # Save per-epoch checkpoint
    os.makedirs("/home/manan/aigc_robust_detection/checkpoints/specialists", exist_ok=True)
    ckpt_path = f"/home/manan/aigc_robust_detection/checkpoints/specialists/specialist_epoch_{epoch}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"   Saved Checkpoint: {ckpt_path}\n")

print("=" * 85)
print("  ACTIVE GPU TRAINING COMPLETE")
print("=" * 85)
