# =====================================================================================
# FP32 BATCH SIZE BENCHMARK FOR RTX 3050 (6GB VRAM)
# Evaluates forward + backward + optimizer step across batch sizes: [32, 48, 64, 96, 128]
# Pure torch.float32 (No AMP / No FP16)
# =====================================================================================

import os, sys, time, gc
import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("=" * 80)
print(f"  FP32 BATCH SIZE BENCHMARK ON {DEVICE} ({torch.cuda.get_device_name(0)})")
print("=" * 80)

candidate_batch_sizes = [32, 48, 64, 96, 128]
benchmark_results = []

for bs in candidate_batch_sizes:
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model: ConvNeXt-Tiny (Representative backbone, 28M parameters)
    model = models.convnext_tiny(num_classes=1).float().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Warmup
    try:
        dummy_x = torch.randn(bs, 3, 224, 224, dtype=torch.float32, device=DEVICE)
        dummy_y = torch.ones(bs, dtype=torch.float32, device=DEVICE)
        
        for _ in range(3):
            optimizer.zero_grad()
            out = model(dummy_x).squeeze(-1)
            loss = criterion(out, dummy_y)
            loss.backward()
            optimizer.step()
            
        torch.cuda.synchronize()
        
        # Timed Benchmark (10 iterations)
        t0 = time.time()
        for _ in range(10):
            optimizer.zero_grad()
            out = model(dummy_x).squeeze(-1)
            loss = criterion(out, dummy_y)
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        
        vram_allocated = torch.cuda.memory_allocated(0) / (1024**3)
        vram_reserved = torch.cuda.memory_reserved(0) / (1024**3)
        img_per_sec = (bs * 10) / elapsed
        batch_time_ms = (elapsed / 10) * 1000
        
        res = {
            "batch_size": bs,
            "status": "PASSED ✅",
            "vram_allocated_gb": f"{vram_allocated:.2f} GB",
            "vram_reserved_gb": f"{vram_reserved:.2f} GB",
            "throughput_img_sec": f"{img_per_sec:.1f} img/s",
            "batch_time_ms": f"{batch_time_ms:.1f} ms"
        }
        benchmark_results.append(res)
        print(f"  Batch Size: {bs:3d} | Status: PASSED ✅ | VRAM Reserved: {vram_reserved:.2f} GB | Throughput: {img_per_sec:6.1f} img/s | Batch Time: {batch_time_ms:5.1f} ms")
        
    except torch.cuda.OutOfMemoryError:
        print(f"  Batch Size: {bs:3d} | Status: FAILED (CUDA OOM) ❌")
        benchmark_results.append({"batch_size": bs, "status": "CUDA OOM ❌"})
        torch.cuda.empty_cache()
    finally:
        del model, optimizer, criterion, dummy_x, dummy_y
        torch.cuda.empty_cache()
        gc.collect()

out_bench_path = "/home/manan/aigc_robust_detection/reports/v3_batch_size_benchmark.json"
with open(out_bench_path, "w") as f:
    import json
    json.dump(benchmark_results, f, indent=2)

print("\nBenchmark report saved to:", out_bench_path)
print("=" * 80)
