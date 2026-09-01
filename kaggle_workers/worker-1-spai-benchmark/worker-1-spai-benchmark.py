# Worker 1: SPAI / TFG-Model Benchmark
import os, sys, time, json
import torch
print("=== KAGGLE WORKER 1: SPAI / TFG BENCHMARK ===")
print("CUDA Available:", torch.cuda.is_available(), "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
os.system("pip install -q timm yacs albumentations ftfy filetype lmdb")
os.system("git clone https://huggingface.co/aminasifar1/TFG-model /tmp/spai_tfg")
sys.path.insert(0, "/tmp/spai_tfg")
os.chdir("/tmp/spai_tfg")
from inference import EndpointHandler
from PIL import Image
handler = EndpointHandler("/tmp/spai_tfg")
print("SPAI initialized successfully on Kaggle GPU!")
