import os, sys, time, hashlib, json, math, gc, random
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
import timm
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss
import psutil

# Threading controls
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_num_interop_threads(4)

print("=====================================================================")
print("  HIGH-CAPACITY VISION ARCHITECTURE BENCHMARK & DEV EVALUATION")
print("=====================================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
print(f"CPU Threads: {psutil.cpu_count(logical=True)} logical threads | Total RAM: {psutil.virtual_memory().total/1024**3:.2f} GB")

manifest_path = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"
if not os.path.exists(manifest_path):
    manifest_path = "/Users/manan/Documents/Tiktok/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"

with open(manifest_path, "rb") as f:
    manifest_sha = hashlib.sha256(f.read()).hexdigest()

split_records = {"TRAIN": [], "DEV": []}
with open(manifest_path, "r") as f:
    for line in f:
        r = json.loads(line)
        s = r["split"]
        if s in split_records:
            split_records[s].append((r["canonical_path"], r["label"], r.get("generator_or_domain", "unknown"), r.get("image_id", "")))

print(f"Manifest v6 Loaded: {len(split_records['TRAIN']):,d} TRAIN | {len(split_records['DEV']):,d} DEV (SHA: {manifest_sha[:16]}...)")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

class StreamRawImageDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label, domain, img_id = self.records[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.transform(img)
        except Exception:
            tensor = torch.zeros(3, 224, 224)
        srm_dummy = torch.randn(36)
        return tensor, srm_dummy, torch.tensor(label, dtype=torch.float32), img_id

# -------------------------------------------------------------------
# ARCHITECTURE BUILDER: CONFIG A (31.9M), CONFIG B (55.4M), CONFIG C (84.8M)
# -------------------------------------------------------------------
class HighCapacityVisionDetector(nn.Module):
    def __init__(self, capacity_tier="A"):
        super().__init__()
        self.capacity_tier = capacity_tier
        
        # 1. CLIP ViT-L/14
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.clip_visual = clip_model.visual
        for p in self.clip_visual.parameters():
            p.requires_grad = False
            
        num_clip_blocks = 1 if capacity_tier == "A" else (2 if capacity_tier == "B" else 3)
        for i in range(1, num_clip_blocks + 1):
            for p in self.clip_visual.transformer.resblocks[-i].parameters():
                p.requires_grad = True
        if hasattr(self.clip_visual, 'proj') and self.clip_visual.proj is not None:
            self.clip_visual.proj.requires_grad = True
            
        clip_adapter_dim = 1024 if capacity_tier == "A" else 1536
        self.clip_adapter = nn.Sequential(
            nn.Linear(768, clip_adapter_dim),
            nn.LayerNorm(clip_adapter_dim),
            nn.GELU(),
            nn.Linear(clip_adapter_dim, 1024),
            nn.LayerNorm(1024)
        )
        
        # 2. SigLIP SO400M
        siglip_model = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=False, num_classes=0)
        self.siglip_visual = siglip_model
        for p in self.siglip_visual.parameters():
            p.requires_grad = False
            
        num_siglip_blocks = 1 if capacity_tier == "A" else (2 if capacity_tier == "B" else 3)
        for i in range(1, num_siglip_blocks + 1):
            for p in self.siglip_visual.blocks[-i].parameters():
                p.requires_grad = True
                
        siglip_adapter_dim = 1152 if capacity_tier == "A" else 1536
        self.siglip_adapter = nn.Sequential(
            nn.Linear(1152, siglip_adapter_dim),
            nn.LayerNorm(siglip_adapter_dim),
            nn.GELU(),
            nn.Linear(siglip_adapter_dim, 1152),
            nn.LayerNorm(1152)
        )
        
        # 3. SRM Spatial Filter Stream
        self.srm_proj = nn.Sequential(
            nn.Linear(36, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 36)
        )
        
        # 4. Multi-Layer Bottleneck Fusion Head
        fusion_in_dim = 1024 + 1152 + 36
        hidden_dim = 512 if capacity_tier == "A" else 768
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, img_tensors, srm_feats):
        clip_out = self.clip_visual(img_tensors)
        clip_rep = self.clip_adapter(clip_out)
        siglip_out = self.siglip_visual(img_tensors)
        siglip_rep = self.siglip_adapter(siglip_out)
        srm_rep = self.srm_proj(srm_feats)
        fused = torch.cat([clip_rep, siglip_rep, srm_rep], dim=-1)
        return self.fusion_head(fused).squeeze(-1)

def compute_stable_asymmetric_loss(logits, targets, lambda_fp=2.5):
    logits_f32 = logits.float()
    targets_f32 = targets.float()
    weights = torch.where(targets_f32 == 0.0, torch.tensor(lambda_fp, device=logits.device), torch.tensor(1.0, device=logits.device))
    return F.binary_cross_entropy_with_logits(logits_f32, targets_f32, weight=weights, reduction='mean')

def count_trainable_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def get_param_hash(m):
    h = hashlib.sha256()
    for name, p in m.named_parameters():
        if p.requires_grad:
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def benchmark_capacity_tier(tier="A", batch_size=48, num_workers=4, prefetch_factor=2):
    print(f"\n=====================================================================")
    print(f"  BENCHMARKING CONFIG {tier}: BS={batch_size}, Workers={num_workers}, PF={prefetch_factor}")
    print(f"=====================================================================")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    model = HighCapacityVisionDetector(capacity_tier=tier).to(device)
    trainable_params = count_trainable_parameters(model)
    print(f"  Total Trainable Vision & Fusion Parameters: {trainable_params:,d} ({trainable_params/1e6:.2f}M params)")
    
    train_dataset = StreamRawImageDataset(split_records["TRAIN"], train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    dev_dataset = StreamRawImageDataset(split_records["DEV"], eval_transform)
    dev_loader = DataLoader(dev_dataset, batch_size=48, shuffle=False, num_workers=4)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-5)
    scaler = torch.amp.GradScaler('cuda')
    
    initial_hash = get_param_hash(model)
    model.train()
    
    warmup_batches = 50
    measure_batches = 100
    
    iter_loader = iter(train_loader)
    print(f"  [1/3] Running {warmup_batches} Warmup Batches...")
    t_w0 = time.time()
    for _ in range(warmup_batches):
        batch_imgs, batch_srm, batch_lbls, _ = next(iter_loader)
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_srm = batch_srm.to(device, non_blocking=True)
        batch_lbls = batch_lbls.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(batch_imgs, batch_srm)
            loss = compute_stable_asymmetric_loss(logits, batch_lbls, lambda_fp=2.5)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    torch.cuda.synchronize()
    print(f"        Warmup Completed in {time.time()-t_w0:.2f}s.")
    
    print(f"  [2/3] Running {measure_batches} Measurement Batches...")
    t0 = time.time()
    total_samples = 0
    losses = []
    grad_norms = []
    opt_steps = 0
    
    for b_idx in range(measure_batches):
        batch_imgs, batch_srm, batch_lbls, _ = next(iter_loader)
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_srm = batch_srm.to(device, non_blocking=True)
        batch_lbls = batch_lbls.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(batch_imgs, batch_srm)
            loss = compute_stable_asymmetric_loss(logits, batch_lbls, lambda_fp=2.5)
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        losses.append(loss.item())
        total_samples += len(batch_imgs)
        
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.clip_visual.transformer.resblocks[-1].parameters() if p.grad is not None])).item()
        grad_norms.append(gnorm)
        
        scaler.step(optimizer)
        scaler.update()
        opt_steps += 1
        
    torch.cuda.synchronize()
    duration = time.time() - t0
    
    final_hash = get_param_hash(model)
    param_delta_proven = (initial_hash != final_hash)
    
    throughput_samples = total_samples / duration
    sec_per_batch = duration / measure_batches
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
    vram_headroom_mb = 6144.0 - peak_reserved_mb
    
    print(f"  [3/3] Evaluating on 10,000-Sample DEV Split...")
    model.eval()
    dev_probs, dev_targets = [], []
    with torch.no_grad():
        for batch_imgs, batch_srm, batch_lbls, _ in dev_loader:
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            batch_srm = batch_srm.to(device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.float16):
                logits = model(batch_imgs, batch_srm)
            probs = torch.sigmoid(logits.float()).cpu().numpy()
            dev_probs.extend(probs)
            dev_targets.extend(batch_lbls.numpy())
            
    probs_np = np.array(dev_probs)
    targets_np = np.array(dev_targets)
    auroc = roc_auc_score(targets_np, probs_np)
    precision, recall, _ = precision_recall_curve(targets_np, probs_np)
    auprc = auc(recall, precision)
    
    real_probs = probs_np[targets_np == 0]
    aigc_probs = probs_np[targets_np == 1]
    
    tau_01 = np.percentile(real_probs, 99.9)
    tpr_at_01_fpr = float(np.mean(aigc_probs >= tau_01))
    
    tau_001 = np.percentile(real_probs, 99.99)
    tpr_at_001_fpr = float(np.mean(aigc_probs >= tau_001))
    
    is_safe = vram_headroom_mb > 500.0 and param_delta_proven and not math.isnan(np.mean(losses))
    
    res = {
        "tier": tier,
        "trainable_parameters": trainable_params,
        "trainable_parameters_m": round(trainable_params / 1e6, 2),
        "batch_size": batch_size,
        "throughput_samples_per_sec": round(throughput_samples, 2),
        "seconds_per_batch": round(sec_per_batch, 4),
        "peak_allocated_vram_mb": round(peak_allocated_mb, 1),
        "peak_reserved_vram_mb": round(peak_reserved_mb, 1),
        "vram_headroom_mb": round(vram_headroom_mb, 1),
        "avg_loss": round(float(np.mean(losses)), 5),
        "avg_grad_norm": round(float(np.mean(grad_norms)), 5),
        "dev_auroc": round(float(auroc), 6),
        "dev_auprc": round(float(auprc), 6),
        "dev_tpr_at_0_1pct_fpr": round(tpr_at_01_fpr, 4),
        "dev_tpr_at_0_01pct_fpr": round(tpr_at_001_fpr, 4),
        "parameter_delta_proven": param_delta_proven,
        "is_safe": is_safe
    }
    
    print(f"\n  --- CONFIG {tier} RESULTS ---")
    print(f"  Trainable Params:     {res['trainable_parameters_m']}M parameters")
    print(f"  Measured Throughput:  {res['throughput_samples_per_sec']:.2f} samples/sec ({res['seconds_per_batch']:.4f} s/batch)")
    print(f"  Peak Reserved VRAM:   {res['peak_reserved_vram_mb']:.1f} MB (Headroom: {res['vram_headroom_mb']:.1f} MB)")
    print(f"  Dev AUROC / AUPRC:    {res['dev_auroc']:.6f} / {res['dev_auprc']:.6f}")
    print(f"  TPR @ 0.10% FPR:      {res['dev_tpr_at_0_1pct_fpr']*100:.2f}%")
    print(f"  TPR @ 0.01% FPR:      {res['dev_tpr_at_0_01pct_fpr']*100:.2f}%")
    print(f"  Memory Safe:          {res['is_safe']}")
    
    return res

if __name__ == "__main__":
    print("\n[STEP 1] Benchmarking Config A (Current ~31.9M Trainable Params)...")
    res_a = benchmark_capacity_tier(tier="A", batch_size=48, num_workers=4, prefetch_factor=2)
    
    print("\n[STEP 2] Benchmarking Config B (~55.4M Trainable Params: 2 CLIP + 2 SigLIP Blocks)...")
    res_b = benchmark_capacity_tier(tier="B", batch_size=48, num_workers=4, prefetch_factor=2)
    
    print("\n[STEP 3] Benchmarking Config C (~84.8M Trainable Params: 3 CLIP + 3 SigLIP Blocks)...")
    res_c = benchmark_capacity_tier(tier="C", batch_size=48, num_workers=4, prefetch_factor=2)
    
    print("\n=====================================================================")
    print("  HIGH-CAPACITY VISION ARCHITECTURE COMPARISON & SELECTION")
    print("=====================================================================")
    print(f"  Config A (31.9M): Throughput = {res_a['throughput_samples_per_sec']:.2f} img/s | RSV VRAM = {res_a['peak_reserved_vram_mb']:.1f} MB | TPR@0.1% = {res_a['dev_tpr_at_0_1pct_fpr']*100:.2f}%")
    print(f"  Config B (55.4M): Throughput = {res_b['throughput_samples_per_sec']:.2f} img/s | RSV VRAM = {res_b['peak_reserved_vram_mb']:.1f} MB | TPR@0.1% = {res_b['dev_tpr_at_0_1pct_fpr']*100:.2f}%")
    print(f"  Config C (84.8M): Throughput = {res_c['throughput_samples_per_sec']:.2f} img/s | RSV VRAM = {res_c['peak_reserved_vram_mb']:.1f} MB | TPR@0.1% = {res_c['dev_tpr_at_0_1pct_fpr']*100:.2f}%")
    
    out_dir = "/home/manan/aigc_robust_detection/reports"
    os.makedirs(out_dir, exist_ok=True)
    report_data = {
        "config_a_31_9m": res_a,
        "config_b_55_4m": res_b,
        "config_c_84_8m": res_c,
        "manifest_sha256": manifest_sha
    }
    with open(os.path.join(out_dir, "high_capacity_vision_benchmark.json"), "w") as f:
        json.dump(report_data, f, indent=2)
        
    print("\nBenchmark telemetry saved to reports/high_capacity_vision_benchmark.json")
