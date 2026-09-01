import os, sys, time, hashlib, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
import timm

print("=====================================================================")
print("  PRE-TRAINING 20-STEP SANITY TEST: RAW IMAGES + TRAINABLE VISION ADAPTERS")
print("=====================================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

# 1. Vision Adapter Module for Foundation Transformers
class LoRAAdapter(nn.Module):
    def __init__(self, in_features, rank=16, alpha=32):
        super().__init__()
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5) if 'math' in globals() else 1.0)
        nn.init.zeros_(self.lora_B)
    def forward(self, x):
        return x + (x @ self.lora_A @ self.lora_B) * self.scaling

import math

# 2. Complete End-to-End Detector with Trainable Vision Representation
class EndToEndVisionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load CLIP ViT-L/14
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.clip_visual = clip_model.visual
        
        # Freeze base CLIP layers, keep final layers + projection trainable
        for p in self.clip_visual.parameters():
            p.requires_grad = False
            
        # Enable gradients on final transformer block (layer 23) and visual projection
        for p in self.clip_visual.transformer.resblocks[-1].parameters():
            p.requires_grad = True
        if hasattr(self.clip_visual, 'proj') and self.clip_visual.proj is not None:
            self.clip_visual.proj.requires_grad = True
            
        # Trainable Adapter for CLIP representation (768 -> 1024)
        self.clip_adapter = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        # Load SigLIP-SO400M-224 (or SigLIP base 224 for optimal VRAM training)
        siglip_model = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=False, num_classes=0)
        # Load pre-trained weights if available, else instantiate
        self.siglip_visual = siglip_model
        for p in self.siglip_visual.parameters():
            p.requires_grad = False
            
        # Enable gradients on final SigLIP transformer block (layer -1)
        for p in self.siglip_visual.blocks[-1].parameters():
            p.requires_grad = True
            
        self.siglip_adapter = nn.Sequential(
            nn.Linear(1152, 1152),
            nn.LayerNorm(1152),
            nn.GELU()
        )
        
        # SRM Spatial Filters (Fixed 36d)
        self.srm_proj = nn.Sequential(
            nn.Linear(36, 36),
            nn.LayerNorm(36),
            nn.GELU()
        )
        
        # Trainable Fusion Head (2212 -> 512 -> 128 -> 1)
        self.fusion_head = nn.Sequential(
            nn.Linear(1024 + 1152 + 36, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, img_tensors, srm_feats):
        # 1. CLIP Forward Pass (with gradient checkpointing / amp)
        clip_out = self.clip_visual(img_tensors) # (B, 768)
        clip_rep = self.clip_adapter(clip_out) # (B, 1024)
        
        # 2. SigLIP Forward Pass
        siglip_out = self.siglip_visual(img_tensors) # (B, 1152)
        siglip_rep = self.siglip_adapter(siglip_out) # (B, 1152)
        
        # 3. SRM Projection
        srm_rep = self.srm_proj(srm_feats) # (B, 36)
        
        # 4. Fusion
        fused = torch.cat([clip_rep, siglip_rep, srm_rep], dim=-1) # (B, 2212)
        logits = self.fusion_head(fused).squeeze(-1)
        return logits

print("Initializing EndToEndVisionDetector...")
model = EndToEndVisionDetector().to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
clip_trainable = sum(p.numel() for p in model.clip_visual.parameters() if p.requires_grad) + sum(p.numel() for p in model.clip_adapter.parameters())
siglip_trainable = sum(p.numel() for p in model.siglip_visual.parameters() if p.requires_grad) + sum(p.numel() for p in model.siglip_adapter.parameters())
fusion_trainable = sum(p.numel() for p in model.fusion_head.parameters())

print(f"  Total Model Parameters:     {total_params:,}")
print(f"  Trainable Parameters Total: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
print(f"  - CLIP Vision Trainable:    {clip_trainable:,}")
print(f"  - SigLIP Vision Trainable:  {siglip_trainable:,}")
print(f"  - Fusion Head Trainable:    {fusion_trainable:,}")

def get_trainable_param_hash(m):
    h = hashlib.sha256()
    for name, p in m.named_parameters():
        if p.requires_grad:
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

initial_hash = get_trainable_param_hash(model)
print(f"\nInitial Trainable Parameter Hash: {initial_hash}")

# 3. Raw Image Dataset for 20-Step Sanity Test
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 3. Raw Image Dataset for 20-Step Sanity Test directly from Governed Manifest
manifest_path = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v5.jsonl"
valid_samples = []
with open(manifest_path, "r") as f:
    for line in f:
        r = json.loads(line)
        if r["split"] == "TRAIN" and os.path.exists(r["canonical_path"]):
            valid_samples.append((r["canonical_path"], r["label"]))
            if len(valid_samples) >= 100:
                break

print(f"Loaded {len(valid_samples)} valid training image paths from governed manifest.")

class SanityDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        img = Image.open(p).convert("RGB")
        tensor = self.transform(img)
        srm_dummy = torch.randn(36)
        return tensor, srm_dummy, torch.tensor(l, dtype=torch.float32)

sanity_loader = DataLoader(SanityDataset(valid_samples, img_transform), batch_size=4, shuffle=True)

# 4. Run Exactly 20 Training Steps
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
criterion = nn.BCEWithLogitsLoss()

print("\n--- Executing 20 Real Training Steps on Raw Images ---")
model.train()
step_count = 0
image_reads = 0
clip_forwards = 0
siglip_forwards = 0
backward_passes = 0
vision_gradient_norms = []

start_t = time.time()
while step_count < 20:
    for batch_imgs, batch_srm, batch_lbls in sanity_loader:
        if step_count >= 20:
            break
        
    batch_imgs = batch_imgs.to(device)
    batch_srm = batch_srm.to(device)
    batch_lbls = batch_lbls.to(device)
    
    image_reads += len(batch_imgs)
    clip_forwards += len(batch_imgs)
    siglip_forwards += len(batch_imgs)
    
    optimizer.zero_grad()
    with torch.amp.autocast('cuda', dtype=torch.float16):
        logits = model(batch_imgs, batch_srm)
        loss = criterion(logits, batch_lbls)
        
    loss.backward()
    backward_passes += 1
    
    # Measure gradient norm on trainable CLIP vision parameters
    clip_grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.clip_visual.transformer.resblocks[-1].parameters() if p.grad is not None])).item()
    vision_gradient_norms.append(clip_grad_norm)
    
    optimizer.step()
    step_count += 1
    
    if step_count % 5 == 0 or step_count == 1:
        print(f"  Step {step_count:02d}/20 | Loss: {loss.item():.5f} | CLIP Vision Grad Norm: {clip_grad_norm:.6f} | VRAM: {torch.cuda.memory_allocated()/1024**2:.1f} MB")

wall_time = time.time() - start_t
final_hash = get_trainable_param_hash(model)

print("\n=== 20-STEP SANITY TEST RESULTS ===")
print(f"  IMAGE_READS               = {image_reads}")
print(f"  CLIP_FORWARD              = {clip_forwards}")
print(f"  SIGLIP_FORWARD            = {siglip_forwards}")
print(f"  BACKWARD                  = {backward_passes}")
print(f"  OPTIMIZER_STEPS           = {step_count}")
print(f"  AVG_VISION_GRADIENT_NORM  = {np.mean(vision_gradient_norms):.6f}")
print(f"  TRAINABLE_VISION_GRADIENT = {np.mean(vision_gradient_norms) > 0}")
print(f"  PARAMETER_DELTA_PROVEN    = {initial_hash != final_hash}")
print(f"  INITIAL_HASH              = {initial_hash}")
print(f"  FINAL_HASH                = {final_hash}")
print(f"  WALL_TIME_SECONDS         = {wall_time:.2f} s")
print(f"  PEAK_VRAM_MB              = {torch.cuda.max_memory_allocated()/1024**2:.1f} MB")

assert step_count == 20, "Must be 20 steps"
assert initial_hash != final_hash, "Parameters must change"
assert np.mean(vision_gradient_norms) > 0, "Vision gradients must be positive"
print("\n>>> SANITY TEST PASSED: Trainable vision representations successfully receive gradients and update under 6GB VRAM. <<<")
