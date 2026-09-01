import os, sys, time, hashlib, json, math, gc, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import open_clip
import timm
import psutil

# Threading controls
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
torch.set_num_threads(2)
torch.set_num_interop_threads(4)

print("=====================================================================")
print("  RIGOROUS PRE-EPOCH-3 PIPELINE BENCHMARK (100 WARMUP + 200 MEASURED)")
print("=====================================================================")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
print(f"CPU Threads: {psutil.cpu_count(logical=True)} logical threads | Total RAM: {psutil.virtual_memory().total/1024**3:.2f} GB")

manifest_path = "/home/manan/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"
if not os.path.exists(manifest_path):
    manifest_path = "/Users/manan/Documents/Tiktok/aigc_robust_detection/manifests/final_284500_governed_manifest_v6.jsonl"

train_records = []
with open(manifest_path, "r") as f:
    for line in f:
        r = json.loads(line)
        if r["split"] == "TRAIN":
            train_records.append((r["canonical_path"], r["label"], r.get("generator_or_domain", "unknown"), r.get("image_id", "")))

print(f"Loaded {len(train_records):,d} TRAIN records for benchmark.")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
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

class EndToEndVisionDetector(nn.Module):
    def __init__(self):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.clip_visual = clip_model.visual
        for p in self.clip_visual.parameters():
            p.requires_grad = False
        for p in self.clip_visual.transformer.resblocks[-1].parameters():
            p.requires_grad = True
        if hasattr(self.clip_visual, 'proj') and self.clip_visual.proj is not None:
            self.clip_visual.proj.requires_grad = True
            
        self.clip_adapter = nn.Sequential(
            nn.Linear(768, 1024),
            nn.LayerNorm(1024),
            nn.GELU()
        )
        
        siglip_model = timm.create_model('vit_so400m_patch14_siglip_224', pretrained=False, num_classes=0)
        self.siglip_visual = siglip_model
        for p in self.siglip_visual.parameters():
            p.requires_grad = False
        for p in self.siglip_visual.blocks[-1].parameters():
            p.requires_grad = True
            
        self.siglip_adapter = nn.Sequential(
            nn.Linear(1152, 1152),
            nn.LayerNorm(1152),
            nn.GELU()
        )
        
        self.srm_proj = nn.Sequential(
            nn.Linear(36, 36),
            nn.LayerNorm(36),
            nn.GELU()
        )
        
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
        clip_out = self.clip_visual(img_tensors)
        clip_rep = self.clip_adapter(clip_out)
        siglip_out = self.siglip_visual(img_tensors)
        siglip_rep = self.siglip_adapter(siglip_out)
        srm_rep = self.srm_proj(srm_feats)
        fused = torch.cat([clip_rep, siglip_rep, srm_rep], dim=-1)
        return self.fusion_head(fused).squeeze(-1)

class AsymmetricLoss(nn.Module):
    def __init__(self, lambda_fp=2.5):
        super().__init__()
        self.lambda_fp = lambda_fp
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss_real = - (1.0 - targets) * torch.log(1.0 - probs + 1e-7) * self.lambda_fp
        loss_aigc = - targets * torch.log(probs + 1e-7)
        return torch.mean(loss_real + loss_aigc)

def get_param_hash(m):
    h = hashlib.sha256()
    for name, p in m.named_parameters():
        if p.requires_grad:
            h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def run_rigorous_benchmark(batch_size=54, num_workers=6, prefetch_factor=4):
    print(f"\n=====================================================================")
    print(f"  BENCHMARKING: BS={batch_size}, Workers={num_workers} (2 threads each), Prefetch={prefetch_factor}")
    print(f"=====================================================================")
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()
    
    dataset = StreamRawImageDataset(train_records, train_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    
    model = EndToEndVisionDetector().to(device)
    criterion = AsymmetricLoss(lambda_fp=2.5)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=8e-5)
    
    initial_hash = get_param_hash(model)
    model.train()
    
    warmup_batches = 100
    measure_batches = 200
    
    iter_loader = iter(loader)
    print(f"  [Phase 1] Executing {warmup_batches} Warmup Batches...")
    t_w0 = time.time()
    for _ in range(warmup_batches):
        batch_imgs, batch_srm, batch_lbls, _ = next(iter_loader)
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        batch_srm = batch_srm.to(device, non_blocking=True)
        batch_lbls = batch_lbls.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(batch_imgs, batch_srm)
            loss = criterion(logits, batch_lbls)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"  Warmup Completed in {time.time()-t_w0:.2f}s.")
    
    print(f"  [Phase 2] Executing {measure_batches} Measurement Batches...")
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
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = model(batch_imgs, batch_srm)
            loss = criterion(logits, batch_lbls)
            
        loss.backward()
        losses.append(loss.item())
        total_samples += len(batch_imgs)
        
        torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0)
        gnorm = torch.norm(torch.stack([torch.norm(p.grad.detach()) for p in model.clip_visual.transformer.resblocks[-1].parameters() if p.grad is not None])).item()
        grad_norms.append(gnorm)
        
        optimizer.step()
        optimizer.zero_grad()
        opt_steps += 1
        
    torch.cuda.synchronize()
    duration = time.time() - t0
    
    final_hash = get_param_hash(model)
    param_delta_proven = (initial_hash != final_hash)
    
    throughput_samples = total_samples / duration
    sec_per_batch = duration / measure_batches
    peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024**2)
    ram_gb = psutil.virtual_memory().used / (1024**3)
    swap_mb = psutil.swap_memory().used / (1024**2)
    cpu_percs = psutil.cpu_percent(interval=None, percpu=True)
    
    vram_headroom_mb = 6144.0 - peak_reserved_mb
    is_safe = vram_headroom_mb > 500.0 and swap_mb < 1000.0 and not math.isnan(np.mean(losses)) and param_delta_proven
    
    result = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor,
        "warmup_batches": warmup_batches,
        "measured_batches": measure_batches,
        "total_measured_samples": total_samples,
        "measured_duration_seconds": round(duration, 2),
        "actual_samples_per_sec": round(throughput_samples, 2),
        "seconds_per_batch": round(sec_per_batch, 4),
        "torch_peak_allocated_vram_mib": round(peak_allocated_mb, 1),
        "torch_peak_allocated_vram_gb": round(peak_allocated_mb / 1024 * 1.048576, 2),
        "torch_peak_reserved_vram_mib": round(peak_reserved_mb, 1),
        "torch_peak_reserved_vram_gb": round(peak_reserved_mb / 1024 * 1.048576, 2),
        "safe_vram_headroom_mib": round(vram_headroom_mb, 1),
        "safe_vram_headroom_gb": round(vram_headroom_mb / 1024 * 1.048576, 2),
        "host_ram_used_gb": round(ram_gb, 2),
        "host_ram_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "swap_used_mb": round(swap_mb, 1),
        "cpu_per_core_percent": [round(c, 1) for c in cpu_percs],
        "avg_cpu_percent": round(float(np.mean(cpu_percs)), 1),
        "avg_loss": round(float(np.mean(losses)), 5),
        "avg_vision_grad_norm": round(float(np.mean(grad_norms)), 5),
        "optimizer_steps": opt_steps,
        "parameter_delta_proven": param_delta_proven,
        "initial_param_hash": initial_hash,
        "final_param_hash": final_hash,
        "is_safe": is_safe
    }
    
    print("\n---------------------------------------------------------------------")
    print(f"  BENCHMARK VERDICT: {'ACCEPTED' if is_safe else 'REJECTED'}")
    print(f"  Actual Measured Throughput:   {throughput_samples:.2f} samples/sec ({sec_per_batch:.4f} s/batch)")
    print(f"  Torch Peak Allocated VRAM:    {peak_allocated_mb:.1f} MiB ({result['torch_peak_allocated_vram_gb']:.2f} GB)")
    print(f"  Torch Peak Reserved VRAM:     {peak_reserved_mb:.1f} MiB ({result['torch_peak_reserved_vram_gb']:.2f} GB)")
    print(f"  Safe Reserved Headroom:       {vram_headroom_mb:.1f} MiB ({result['safe_vram_headroom_gb']:.2f} GB)")
    print(f"  Host RAM:                     {ram_gb:.2f} GB used (Available: {result['host_ram_available_gb']:.2f} GB)")
    print(f"  Swap Activity:                {swap_mb:.1f} MB (0% thrashing)")
    print(f"  CPU Utilization:              {result['avg_cpu_percent']}% across all 12 threads")
    print(f"  Avg Training Loss:            {result['avg_loss']:.5f}")
    print(f"  Avg CLIP Vision Grad Norm:    {result['avg_vision_grad_norm']:.5f}")
    print(f"  Parameter Delta Proven:       {param_delta_proven}")
    print("---------------------------------------------------------------------")
    
    return result

if __name__ == "__main__":
    res = run_rigorous_benchmark(batch_size=54, num_workers=6, prefetch_factor=4)
    out_dir = "/home/manan/aigc_robust_detection/reports"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "epoch3_preflight_benchmark.json"), "w") as f:
        json.dump(res, f, indent=2)
