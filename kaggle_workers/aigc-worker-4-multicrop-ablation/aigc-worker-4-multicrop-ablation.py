# Worker 4: Multi-Crop Resolution Ablation
import os, sys, time, json
import torch
print("=== KAGGLE WORKER 4: MULTI-CROP ABLATION EXPERIMENT ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
os.system("pip install -q open_clip_torch torchvision timm pillow")
print("Environment ready for Multi-Crop ablation experiment.")
