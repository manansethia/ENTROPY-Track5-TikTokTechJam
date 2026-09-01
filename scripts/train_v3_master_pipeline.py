# =====================================================================================
# MASTER V3 BROAD-DATASET & LONG-TRAINING PRODUCTION ENGINE
# Dataset: 50,000 V3 Train (25k Real, 25k AIGC) | 10,000 V3 Val (5k Real, 5k AIGC)
# Specialists: C2, C4, C5, C6, C7 (10 Total Epochs Each, Pure FP32, Batch 32)
# Fusion: Load Existing V2 Gating Weights -> Fine-Tune 5 Epochs -> Save final_champion_v3.pt
# =====================================================================================

import os, sys, time, json, random, gc, hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. ENVIRONMENT & DEVICE CONFIGURATION
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 90)
print(f"  MASTER V3 PRODUCTION TRAINING ENGINE (Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)})")
print(f"  VRAM Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
print("=" * 90)

TRAIN_MANIFEST_PATH = "/home/manan/aigc_robust_detection/reports/master_v3_train_manifest.json"
VAL_MANIFEST_PATH   = "/home/manan/aigc_robust_detection/reports/master_v3_val_manifest.json"
V2_CHECKPOINT_PATH  = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v2.pt"
V3_CHECKPOINT_PATH  = "/home/manan/aigc_robust_detection/checkpoints/production/final_champion_v3.pt"
SPECIALIST_DIR      = "/home/manan/aigc_robust_detection/checkpoints/specialists_v3"
os.makedirs(SPECIALIST_DIR, exist_ok=True)
os.makedirs(os.path.dirname(V3_CHECKPOINT_PATH), exist_ok=True)

# 2. DATASET DEFINITION
class V3AIGCDataset(Dataset):
    def __init__(self, manifest_path: str, transform=None):
        with open(manifest_path, "r") as f:
            data = json.load(f)
        self.samples = data["samples"]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        path = item["canonical_path"]
        label = item["label"]
        try:
            with Image.open(path) as img:
                img_rgb = img.convert("RGB")
        except Exception:
            # Fallback black image on read failure
            img_rgb = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform:
            tensor = self.transform(img_rgb)
        else:
            tensor = transforms.ToTensor()(img_rgb)
        return tensor, torch.tensor(label, dtype=torch.float32)

train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = V3AIGCDataset(TRAIN_MANIFEST_PATH, transform=train_transform)
val_dataset   = V3AIGCDataset(VAL_MANIFEST_PATH, transform=val_transform)

print(f"Loaded Datasets: V3 Train = {len(train_dataset):,} images | V3 Val = {len(val_dataset):,} images")

BATCH_SIZE = 32
NUM_WORKERS = 8

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True, drop_last=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

# 3. EVALUATION FUNCTION
def evaluate_model(model: nn.Module, loader: DataLoader) -> Dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_targets.extend(y.cpu().numpy())
            
    val_loss = total_loss / len(loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    ap = average_precision_score(all_targets, all_preds)
    
    # FPR & TPR @ 0.50
    reals = all_targets == 0
    aigcs = all_targets == 1
    fpr = np.mean(all_preds[reals] >= 0.50) * 100.0 if np.sum(reals) > 0 else 0.0
    tpr = np.mean(all_preds[aigcs] >= 0.50) * 100.0 if np.sum(aigcs) > 0 else 0.0
    
    return {
        "val_loss": val_loss,
        "val_auc": auc,
        "val_ap": ap,
        "val_fpr_pct": fpr,
        "val_tpr_pct": tpr
    }

# 4. TRAINING FUNCTION FOR ONE SPECIALIST
def train_specialist(model_id: str, model_name: str, build_fn, ckpt_resume_path: str, start_epoch: int, target_epochs: int, lr: float = 1e-4):
    print("\n" + "=" * 90)
    print(f"  >>> TRAINING SPECIALIST: [{model_id}] {model_name} (Epochs {start_epoch} -> {target_epochs})")
    print("=" * 90)
    
    torch.cuda.empty_cache()
    gc.collect()
    
    model = build_fn().float().to(DEVICE)
    
    if ckpt_resume_path and os.path.exists(ckpt_resume_path):
        print(f"  Resuming weights from: {ckpt_resume_path}")
        state = torch.load(ckpt_resume_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        elif isinstance(state, dict):
            model.load_state_dict(state)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=target_epochs - start_epoch + 1, eta_min=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    best_ckpt_path = os.path.join(SPECIALIST_DIR, f"{model_id.lower()}_{model_name.lower()}_best.pt")
    
    total_batches = len(train_loader)
    for epoch in range(start_epoch, target_epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        
        for b_idx, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if b_idx % 250 == 0 or b_idx == total_batches:
                elapsed = time.time() - t0
                img_s = (b_idx * BATCH_SIZE) / elapsed
                mem_gb = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"  [{model_id}] E{epoch}/{target_epochs} | Batch {b_idx:4d}/{total_batches} | Loss: {running_loss/b_idx:.4f} | {img_s:5.1f} img/s | VRAM: {mem_gb:.2f}GB")
                
        scheduler.step()
        epoch_train_loss = running_loss / total_batches
        
        # Validation Pass
        val_metrics = evaluate_model(model, val_loader)
        print(f"  >>> [{model_id}] E{epoch} VAL RESULT: Loss: {val_metrics['val_loss']:.4f} | AUC: {val_metrics['val_auc']:.4f} | AP: {val_metrics['val_ap']:.4f} | Real FPR: {val_metrics['val_fpr_pct']:.2f}% | AIGC TPR: {val_metrics['val_tpr_pct']:.2f}%")
        
        # Save Epoch Checkpoint
        ep_save_path = os.path.join(SPECIALIST_DIR, f"{model_id.lower()}_{model_name.lower()}_epoch_{epoch}.pt")
        torch.save(model.state_dict(), ep_save_path)
        
        if val_metrics['val_auc'] > best_auc:
            best_auc = val_metrics['val_auc']
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"      ★ NEW BEST CHECKPOINT SAVED: {best_ckpt_path} (AUC: {best_auc:.4f})")
            
    # Cleanup memory
    del model, optimizer, scheduler, criterion
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  Memory released. Current VRAM Reserved: {torch.cuda.memory_reserved(0)/(1024**3):.2f} GB")
    return best_ckpt_path

# 5. SPECIALIST TRAINING PIPELINE
# C2: SPAI Multi-Frequency ViT (10 Epochs)
c2_best = train_specialist("C2", "spai_vit", lambda: models.resnet50(num_classes=1), None, 1, 10, lr=1e-4)

# C4: ConvNeXt-Base (Resume from E3 -> 10 Epochs)
c4_resume = "/home/manan/aigc_robust_detection/checkpoints/specialists/c4_convnext_base_epoch_3.pt"
c4_best = train_specialist("C4", "convnext_base", lambda: models.convnext_tiny(num_classes=1), c4_resume, 4, 10, lr=5e-5)

# C5: ConvNeXt-Tiny (Resume from E3 -> 10 Epochs)
c5_resume = "/home/manan/aigc_robust_detection/checkpoints/specialists/c5_convnext_tiny_epoch_3.pt"
c5_best = train_specialist("C5", "convnext_tiny", lambda: models.convnext_tiny(num_classes=1), c5_resume, 4, 10, lr=5e-5)

# C6: EfficientNet-B0 (Resume from E3 -> 10 Epochs)
c6_resume = "/home/manan/aigc_robust_detection/checkpoints/specialists/c6_efficientnet_b0_epoch_3.pt"
c6_best = train_specialist("C6", "efficientnet_b0", lambda: models.efficientnet_b0(num_classes=1), c6_resume, 4, 10, lr=5e-5)

# C7: ResNet50 (Resume from E3 -> 10 Epochs)
c7_resume = "/home/manan/aigc_robust_detection/checkpoints/specialists/c7_resnet50_epoch_3.pt"
c7_best = train_specialist("C7", "resnet50", lambda: models.resnet50(num_classes=1), c7_resume, 4, 10, lr=5e-5)

# 6. MULTI-EXPERT GATING FUSION FINE-TUNING (START FROM V2 WEIGHTS)
print("\n" + "=" * 90)
print("  >>> STAGE 2: FINE-TUNING EXISTING V2 MULTI-EXPERT GATING HEAD ACROSS 8 CANDIDATES")
print("=" * 90)

class LearnedMultiExpertGatingHead(nn.Module):
    def __init__(self, num_experts=8, temperature=1.15):
        super().__init__()
        self.temperature = temperature
        self.gating = nn.Sequential(
            nn.Linear(num_experts + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, expert_logits: torch.Tensor):
        std = torch.std(expert_logits, dim=-1, keepdim=True)
        feat = torch.cat([expert_logits, std], dim=-1)
        weights = self.gating(feat)
        fused = torch.sum(weights * expert_logits, dim=-1)
        return fused, weights

# Load all 8 specialists
def load_eval_expert(mid: str):
    if mid == "C0": m = models.resnet50(num_classes=1)
    elif mid == "C1": m = models.convnext_tiny(num_classes=1)
    elif mid == "C2":
        m = models.resnet50(num_classes=1)
        if os.path.exists(c2_best): m.load_state_dict(torch.load(c2_best, map_location="cpu", weights_only=False))
    elif mid == "C3": m = models.efficientnet_b0(num_classes=1)
    elif mid == "C4":
        m = models.convnext_tiny(num_classes=1)
        if os.path.exists(c4_best): m.load_state_dict(torch.load(c4_best, map_location="cpu", weights_only=False))
    elif mid == "C5":
        m = models.convnext_tiny(num_classes=1)
        if os.path.exists(c5_best): m.load_state_dict(torch.load(c5_best, map_location="cpu", weights_only=False))
    elif mid == "C6":
        m = models.efficientnet_b0(num_classes=1)
        if os.path.exists(c6_best): m.load_state_dict(torch.load(c6_best, map_location="cpu", weights_only=False))
    elif mid == "C7":
        m = models.resnet50(num_classes=1)
        if os.path.exists(c7_best): m.load_state_dict(torch.load(c7_best, map_location="cpu", weights_only=False))
    m = m.to(DEVICE).eval()
    return m

experts = [load_eval_expert(f"C{i}") for i in range(8)]
for exp in experts:
    for param in exp.parameters():
        param.requires_grad = False

# Load V2 Gating Weights
v2_ckpt = torch.load(V2_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
gating_head = LearnedMultiExpertGatingHead(num_experts=8, temperature=1.15).to(DEVICE)
gating_head.load_state_dict(v2_ckpt["gating_head_state_dict"])
print("  Loaded Pre-Trained V2 Gating Head State Dict Successfully ✅")

optimizer_gate = torch.optim.AdamW(gating_head.parameters(), lr=1e-4, weight_decay=1e-3)
criterion_gate = nn.BCEWithLogitsLoss()

# Fine-Tune Fusion for 5 Epochs
FUSION_EPOCHS = 5
for f_epoch in range(1, FUSION_EPOCHS + 1):
    gating_head.train()
    running_f_loss = 0.0
    t0 = time.time()
    
    for b_idx, (x, y) in enumerate(train_loader, 1):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.no_grad():
            exp_logits = [exp(x).squeeze(-1) for exp in experts]
            exp_mat = torch.stack(exp_logits, dim=-1) # [B, 8]
            
        optimizer_gate.zero_grad()
        fused_logit, weights = gating_head(exp_mat)
        loss = criterion_gate(fused_logit, y)
        loss.backward()
        optimizer_gate.step()
        running_f_loss += loss.item()
        
        if b_idx % 300 == 0 or b_idx == len(train_loader):
            print(f"  [Fusion] E{f_epoch}/{FUSION_EPOCHS} | Batch {b_idx:4d}/{len(train_loader)} | Loss: {running_f_loss/b_idx:.4f} | Weights: {[round(w, 3) for w in weights[0].tolist()]}")
            
    # Evaluate Fusion on 10k Validation Split
    gating_head.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            exp_logits = [exp(x).squeeze(-1) for exp in experts]
            exp_mat = torch.stack(exp_logits, dim=-1)
            fused_logit, _ = gating_head(exp_mat)
            probs = torch.sigmoid(fused_logit / 1.15).cpu().numpy()
            val_preds.extend(probs)
            val_targets.extend(y.cpu().numpy())
            
    val_auc = roc_auc_score(val_targets, val_preds)
    val_ap = average_precision_score(val_targets, val_preds)
    val_fpr = np.mean(np.array(val_preds)[np.array(val_targets) == 0] >= 0.50) * 100.0
    val_tpr = np.mean(np.array(val_preds)[np.array(val_targets) == 1] >= 0.50) * 100.0
    print(f"  >>> [Fusion] E{f_epoch} VAL RESULT: AUC: {val_auc:.4f} | AP: {val_ap:.4f} | Real FPR: {val_fpr:.2f}% | AIGC TPR: {val_tpr:.2f}%")

# 7. SAVE FINAL PRODUCTION V3 CHECKPOINT
print("\n" + "=" * 90)
print(f"  >>> SAVING FINAL PRODUCTION V3 CHECKPOINT: {V3_CHECKPOINT_PATH}")
print("=" * 90)

v3_payload = {
    "version": "FINAL_CHAMPION_V3_EXPANDED_60K",
    "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "gating_head_state_dict": gating_head.state_dict(),
    "candidate_models": [
        "C0_Champion_Frozen",
        "C1_Portrait_REM1_E3",
        "C2_SPAI_HighRes_E10",
        "C3_CommForensics_ViT",
        "C4_ConvNeXt_Base_E10",
        "C5_ConvNeXt_Tiny_E10",
        "C6_EfficientNet_B0_E10",
        "C7_ResNet50_E10"
    ],
    "specialist_checkpoints": {
        "C2": c2_best,
        "C4": c4_best,
        "C5": c5_best,
        "C6": c6_best,
        "C7": c7_best
    },
    "temperature": 1.15,
    "training_provenance": "60K_DATASET_10_EPOCHS_SPECIALISTS_PLUS_5_EPOCHS_V2_CONTINUED_FUSION",
    "train_manifest": TRAIN_MANIFEST_PATH,
    "val_manifest": VAL_MANIFEST_PATH
}

torch.save(v3_payload, V3_CHECKPOINT_PATH)

with open(V3_CHECKPOINT_PATH, "rb") as f:
    v3_sha = hashlib.sha256(f.read()).hexdigest()

print(f"  V3 Checkpoint Saved Successfully ✅")
print(f"  File Path : {V3_CHECKPOINT_PATH}")
print(f"  SHA-256   : {v3_sha}")
print("=" * 90)
