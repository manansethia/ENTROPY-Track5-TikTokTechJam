#!/usr/bin/env python3
"""Audit and safety gate for the final AIGC Forensics product.

This tool intentionally does not train a model by itself. It creates a
verifiable record of what is present, blocks unsafe training inputs, and keeps
the production decision separate from unverified experiments.

Run from the repository root:
    python scripts/final_product_control.py audit --hash-checkpoints
    python scripts/final_product_control.py training-preflight --manifest manifests/...
    python scripts/final_product_control.py benchmark-isolation --benchmark-hashes hashes.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"}
PUBLIC_SURFACES = ("app/static", "frontend", "README.md")
PRIVATE_TERMS = (
    "buildabot", "cuda", "rtx", "100.69.", "tailscale", "ssh root@",
    "ssh manan@", "/home/manan", "/mnt/ai-storage",
)
UNSAFE_TRAINING_PATTERNS = {
    "synthetic PIL fallback": r"Image\.new\s*\(",
    "zero tensor fallback": r"torch\.zeros\s*\(",
    "random tensor fallback": r"torch\.randn?\s*\(",
    "synthetic fusion logits": r"np\.random\.(normal|rand|randn)",
    "mock data path": r"mock[_-]?highres|/tmp/mock",
    "filename-derived label": r"(label|target)\s*=.*(filename|file_name|path).*",
}


@dataclass
class CheckpointRecord:
    name: str
    version: str | None
    architecture: str | None
    parameters: int | None
    checkpoint_size: int | None
    precision: str | None
    sha256: str | None
    purpose: str | None
    training_status: str | None
    date: str | None
    source_dataset: str | None
    validation_dataset: str | None
    current_role: str | None
    superseded_by: str | None
    production_candidate: bool | None
    standalone_or_ensemble: str | None


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def safe_relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def checkpoint_precision(name: str) -> str | None:
    match = re.search(r"\b(fp32|fp16|fp8|int8)\b", name, flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


def inventory_checkpoints(root: Path, hash_checkpoints: bool) -> list[CheckpointRecord]:
    records: list[CheckpointRecord] = []
    checkpoint_root = root / "checkpoints"
    if checkpoint_root.exists():
        for path in sorted(checkpoint_root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in CHECKPOINT_SUFFIXES:
                continue
            stat = path.stat()
            records.append(CheckpointRecord(
                name=safe_relative(root, path), version=None, architecture=None,
                parameters=None, checkpoint_size=stat.st_size,
                precision=checkpoint_precision(path.name),
                sha256=sha256_file(path) if hash_checkpoints else None,
                purpose=None, training_status="PRESENT_ON_DISK",
                date=datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
                source_dataset=None, validation_dataset=None, current_role=None,
                superseded_by=None, production_candidate=None,
                standalone_or_ensemble=None,
            ))

    freeze = read_json(root / "reports/final_production_freeze_report.json")
    if freeze:
        counts = freeze.get("parameter_counts", {})
        records.append(CheckpointRecord(
            name="production/final_champion_frozen_model.pt",
            version="frozen production report",
            architecture=" + ".join(freeze.get("architecture", {}).get("backbones", [])) or None,
            parameters=counts.get("total_parameters"), checkpoint_size=None,
            precision="FP32", sha256=freeze.get("file_sha256"),
            purpose="Frozen production candidate reported by final production freeze record.",
            training_status="REPORTED_FROZEN", date=freeze.get("frozen_timestamp"),
            source_dataset=None, validation_dataset="Authoritative DEV split reported as 10,000 images.",
            current_role="production candidate", superseded_by=None,
            production_candidate=True, standalone_or_ensemble="standalone model class claimed by report",
        ))
    return records


def scan_public_privacy(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for surface in PUBLIC_SURFACES:
        path = root / surface
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            if not candidate.is_file() or candidate.suffix.lower() not in {".html", ".js", ".css", ".md", ".json", ".py"}:
                continue
            try:
                content = candidate.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                continue
            for term in PRIVATE_TERMS:
                if term in content:
                    findings.append({"file": safe_relative(root, candidate), "term": term})
    return findings


def scan_training_safety(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in sorted((root / "scripts").glob("*.py")):
        if not any(token in path.name for token in ("train", "training", "fusion", "master")):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for label, pattern in UNSAFE_TRAINING_PATTERNS.items():
            if re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
                findings.append({"file": safe_relative(root, path), "issue": label})
    return findings


def iter_manifest_records(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    for path in paths:
        if path.suffix.lower() != ".jsonl":
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    yield path, record


def load_hashes(path: Path | None) -> set[str]:
    if path is None or not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if re.fullmatch(r"[0-9a-fA-F]{64}", line.strip())}


def benchmark_isolation(root: Path, benchmark_hash_file: Path | None) -> dict[str, Any]:
    manifests = sorted((root / "manifests").glob("*.jsonl"))
    hashes = load_hashes(benchmark_hash_file)
    matches: list[dict[str, str]] = []
    sample_count = 0
    split_counts: Counter[str] = Counter()
    for path, record in iter_manifest_records(manifests):
        sample_count += 1
        split_counts[str(record.get("split", "UNKNOWN"))] += 1
        item_hash = str(record.get("sha256", "")).lower()
        if hashes and item_hash in hashes:
            matches.append({"manifest": safe_relative(root, path), "sha256": item_hash, "split": str(record.get("split", "UNKNOWN"))})
    return {
        "generated_at": utc_now(),
        "manifest_files_scanned": len(manifests),
        "records_scanned": sample_count,
        "split_counts": dict(split_counts),
        "official_benchmark_hashes_provided": len(hashes),
        "matching_records": matches,
        "status": (
            "PASS_NO_HASH_OVERLAP" if hashes and not matches else
            "FAIL_BENCHMARK_HASH_OVERLAP" if matches else
            "BLOCKED_OFFICIAL_BENCHMARK_HASH_LIST_REQUIRED"
        ),
        "note": "Dataset names and filenames are not sufficient proof of organizer-benchmark isolation. Supply the official benchmark SHA-256 list.",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_audit(root: Path, registry: list[CheckpointRecord], isolation: dict[str, Any], privacy: list[dict[str, str]], training: list[dict[str, str]]) -> Path:
    report = root / "reports/final_project_audit.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        "# Final Project Audit",
        "",
        f"Generated: {utc_now()}",
        "",
        "## Evidence-based status",
        "",
        f"- Model/checkpoint records: {len(registry)}",
        f"- Benchmark isolation: `{isolation['status']}`",
        f"- Public privacy findings: {len(privacy)}",
        f"- Unsafe training-script findings: {len(training)}",
        "",
        "## Frontend",
        "",
        "The active application is served from `app/static`. The current homepage must remain a targeted polish pass, not a redesign.",
        "",
        "## Backend",
        "",
        "Two API implementations are present (`app/server.py` and `deployment/api.py`). They expose overlapping but incompatible contracts and need consolidation before public deployment.",
        "",
        "## Model pipeline and history",
        "",
        "The frozen production report describes a 735,038,561-parameter model. Separate reports also describe multi-expert and high-capacity candidates. These are competing artifacts, not proof of one final model, until the checkpoint, inference class, and independent held-out evaluation are reconciled.",
        "",
        "## Datasets, training, and validation",
        "",
        "Training manifests contain real record-level hashes, but the official organizer benchmark hash list is not present in this repository. Isolation therefore remains blocked rather than assumed.",
        "",
        "## Robustness, provenance, and reporting",
        "",
        "Spatial and provenance modules exist. Public claims must be limited to metrics and artifacts that the final selected inference pipeline can reproduce.",
        "",
        "## Deployment and GitHub readiness",
        "",
        "A root public Git repository is not initialized. The nested `app` repository has independent history. Consolidate deliberately before publishing so the public repository contains the complete application without secrets or private paths.",
        "",
        "## Blocking findings",
        "",
    ]
    if privacy:
        summary.extend([f"- Public privacy scan: `{item['file']}` contains `{item['term']}`." for item in privacy])
    if training:
        summary.extend([f"- Training safety: `{item['file']}` contains `{item['issue']}`." for item in training])
    if not privacy and not training:
        summary.append("- No blocking static findings from this pass.")
    report.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return report


def run_audit(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    registry = inventory_checkpoints(root, args.hash_checkpoints)
    isolation = benchmark_isolation(root, args.benchmark_hashes)
    privacy = scan_public_privacy(root)
    training = scan_training_safety(root)
    write_json(root / "reports/model_registry.json", [asdict(record) for record in registry])
    write_json(root / "reports/benchmark_isolation_audit.json", isolation)
    write_audit(root, registry, isolation, privacy, training)
    payload = {
        "checkpoint_records": len(registry),
        "benchmark_isolation": isolation["status"],
        "privacy_findings": len(privacy),
        "unsafe_training_findings": len(training),
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


def run_preflight(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    isolation = benchmark_isolation(root, args.benchmark_hashes)
    safety = scan_training_safety(root)
    manifest = args.manifest.resolve()
    result = {
        "generated_at": utc_now(),
        "manifest": safe_relative(root, manifest),
        "manifest_exists": manifest.exists(),
        "benchmark_isolation": isolation["status"],
        "unsafe_training_findings": safety,
        "status": "READY" if manifest.exists() and isolation["status"] == "PASS_NO_HASH_OVERLAP" and not safety else "BLOCKED",
    }
    write_json(root / "reports/training_preflight.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY" else 2


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--root", type=Path, default=Path.cwd())
    command.add_argument("--benchmark-hashes", type=Path)
    subparsers = command.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--hash-checkpoints", action="store_true")
    audit.set_defaults(func=run_audit)
    preflight = subparsers.add_parser("training-preflight")
    preflight.add_argument("--manifest", type=Path, required=True)
    preflight.set_defaults(func=run_preflight)
    isolation = subparsers.add_parser("benchmark-isolation")
    isolation.set_defaults(func=lambda args: (write_json(args.root.resolve() / "reports/benchmark_isolation_audit.json", benchmark_isolation(args.root.resolve(), args.benchmark_hashes)), 0)[1])
    return command


if __name__ == "__main__":
    parsed = parser().parse_args()
    sys.exit(parsed.func(parsed))
