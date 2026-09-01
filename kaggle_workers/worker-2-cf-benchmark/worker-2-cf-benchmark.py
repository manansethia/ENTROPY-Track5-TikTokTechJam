# Worker 2: CommunityForensics ViT-Small Benchmark
import os, sys, time, json
import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
print("=== KAGGLE WORKER 2: COMMUNITY FORENSICS 21.8M BENCHMARK ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
model_id = "buildborderless/CommunityForensics-DeepfakeDet-ViT"
processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(model_id).cuda()
model.eval()
print("CommunityForensics ViT-Small (21.8M) initialized successfully on Kaggle GPU!")
