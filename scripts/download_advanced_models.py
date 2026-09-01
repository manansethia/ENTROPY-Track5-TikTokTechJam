import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODELS_DIR = Path('/mnt/ai-storage/aigc_data/models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

ADVANCED_MODELS = [
    {
        'name': 'siglip_so400m_224',
        'repo_id': 'google/siglip-so400m-patch14-224',
        'desc': 'Google SigLIP-SO400M (400M params, 1152-d) - Shape-Optimized Vision Transformer',
    },
    {
        'name': 'dinov2_registers_large',
        'repo_id': 'facebook/dinov2-with-registers-large',
        'desc': 'Meta DINOv2-Large with 4 Registers (300M params, 1024-d) - Dense 3D Geometry',
    },
]

for m in ADVANCED_MODELS:
    dest = MODELS_DIR / m['name']
    print(f'[Model Downloader] Fetching {m["desc"]}...')
    try:
        snapshot_download(
            repo_id=m['repo_id'],
            local_dir=str(dest),
            max_workers=6,
            ignore_patterns=['*.msgpack', '*.h5', '*.tflite', '*.ot'],
        )
        print(f'--> {m["name"]} downloaded successfully to {dest}!')
    except Exception as e:
        print(f'Error downloading {m["name"]}: {e}')
