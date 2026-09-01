# Kaggle Worker GPU & Network Probe (with sm_60 compatibility check)
import os, sys, subprocess

# Check if sm_60 (P100) and fix PyTorch compatibility
try:
    import torch
    if torch.cuda.is_available() and "P100" in torch.cuda.get_device_name(0):
        print("Detected Tesla P100 (sm_60). Installing PyTorch 2.4.0 with sm_60 support...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "torch==2.4.0+cu121", "torchvision==0.19.0+cu121",
            "--extra-index-url", "https://download.pytorch.org/whl/cu121"
        ])
except Exception as e:
    print(f"P100 fix error: {e}")

import torch
print("=" * 60)
print("=== KAGGLE WORKER HARDWARE & ENVIRONMENT PROBE ===")
print("=" * 60)

cuda_avail = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "NONE"
vram_mb = round(torch.cuda.get_device_properties(0).total_memory / (1024**2), 1) if cuda_avail else 0

# Torch CUDA matmul test
torch_cuda_ok = False
if cuda_avail:
    try:
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.float32)
        y = x @ x
        torch.cuda.synchronize()
        torch_cuda_ok = bool(y.sum().item() != 0)
    except Exception as e:
        print(f"Torch CUDA test error: {e}")

print(f"CUDA_AVAILABLE = {cuda_avail}")
print(f"GPU_COUNT = {gpu_count}")
print(f"GPU_NAME = {gpu_name}")
print(f"VRAM_MB = {vram_mb}")
print(f"TORCH_CUDA = {'PASS' if torch_cuda_ok else 'FAIL'}")
print(f"GPU_SELF_TEST = {'PASS' if (cuda_avail and torch_cuda_ok) else 'FAIL'}")
print("=" * 60)
