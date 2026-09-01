import os, subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

HDD_BASE = Path('/mnt/ai-storage/aigc_data/datasets')

# 1. ArtBench-10
artbench_dir = HDD_BASE / 'artbench_hard_negatives'
artbench_dir.mkdir(parents=True, exist_ok=True)
print('[Downloader] Downloading ArtBench-10...')
try:
    snapshot_download(repo_id='civitai/artbench-10', repo_type='dataset', local_dir=str(artbench_dir), max_workers=6)
    print('[Downloader] ArtBench-10 completed.')
except Exception as e:
    print('Note on ArtBench-10:', e)

# 2. Historical & Vintage Archival Photography
vintage_dir = HDD_BASE / 'vintage_archival_photos'
vintage_dir.mkdir(parents=True, exist_ok=True)
print('[Downloader] Downloading Vintage Archival Photography...')
try:
    snapshot_download(repo_id='dalle-mini/vintage-photos', repo_type='dataset', local_dir=str(vintage_dir), max_workers=6)
    print('[Downloader] Vintage photography completed.')
except Exception as e:
    print('Note on Vintage Photography:', e)

# 3. CIFAKE
cifake_dir = HDD_BASE / 'cifake'
cifake_dir.mkdir(parents=True, exist_ok=True)
print('[Downloader] Downloading CIFAKE...')
try:
    snapshot_download(repo_id='roberta/cifake', repo_type='dataset', local_dir=str(cifake_dir), max_workers=6)
    print('[Downloader] CIFAKE completed.')
except Exception as e:
    print('Note on CIFAKE:', e)

print('=== All Remaining Hard-Negative Datasets Completed ===')
