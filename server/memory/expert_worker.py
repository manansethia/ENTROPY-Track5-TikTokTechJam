"""Subprocess isolation pattern for large vision experts.

The parent process should launch one worker per expert. When this process exits,
its CUDA context disappears, providing a stronger reset than cache clearing alone.
This file is a framework/example; model-specific adapters live in scripts/adapters.
"""
from __future__ import annotations
import argparse, gc, json, os
from pathlib import Path
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-id', required=True)
    ap.add_argument('--image-dir', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    print(json.dumps({
        'model_id': args.model_id,
        'image_dir': args.image_dir,
        'output': args.output,
        'cuda': torch.cuda.is_available(),
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }, indent=2))
    # Intentionally no generic model loader here: custom checkpoints require adapters.
    # A real adapter should write embeddings/results and then let the process exit.

if __name__ == '__main__':
    main()
