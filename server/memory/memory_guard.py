"""CUDA memory lifecycle helpers for sequential large-model vision experiments."""
from __future__ import annotations
import gc, json, os, subprocess, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch


def cuda_snapshot(tag: str = "snapshot") -> dict[str, Any]:
    out = {"tag": tag, "time": time.time()}
    if not torch.cuda.is_available():
        out["cuda"] = False
        return out
    out.update({
        "cuda": True,
        "device": torch.cuda.get_device_name(0),
        "allocated": torch.cuda.memory_allocated(0),
        "reserved": torch.cuda.memory_reserved(0),
        "max_allocated": torch.cuda.max_memory_allocated(0),
        "max_reserved": torch.cuda.max_memory_reserved(0),
        "free_total": torch.cuda.mem_get_info(0)[0],
        "total": torch.cuda.mem_get_info(0)[1],
    })
    return out


def hard_cleanup(*objects: Any, ipc: bool = True) -> dict[str, Any]:
    """Delete explicit objects, run Python GC, release cached CUDA blocks, optionally IPC collect."""
    for obj in objects:
        try:
            del obj
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() if ipc else None
        gc.collect()
    return cuda_snapshot("after_cleanup")


@contextmanager
def gpu_model_scope(name: str, cleanup_ipc: bool = True):
    """Context manager for expert lifecycle. The caller should still avoid lingering references."""
    before = cuda_snapshot(f"{name}:before")
    model = None
    try:
        yield before
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if cleanup_ipc:
                torch.cuda.ipc_collect()
        gc.collect()


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(snapshot, indent=2))


def nvidia_smi_snapshot() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
             "--format=csv,noheader"], text=True, stderr=subprocess.STDOUT
        ).strip()
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"
