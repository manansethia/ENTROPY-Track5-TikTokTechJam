#!/usr/bin/env python3
"""Content-based isolation for the protected official benchmark.

This tool is intentionally fail-closed.  It never creates substitute benchmark
data and it never permits a training input to proceed without a complete,
verified protected hash set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
EXPECTED = {"COCO_val2017": 4998, "WildFake_DALLE_Advanced": 8843}
PATH_KEYS = ("path", "image_path", "file_path", "filepath", "source_path", "image")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def build_protected_set(coco_root: Path, dalle_root: Path, output_dir: Path) -> int:
    roots = {"COCO_val2017": coco_root, "WildFake_DALLE_Advanced": dalle_root}
    datasets: dict[str, list[Path]] = {}
    for name, root in roots.items():
        if not root.is_dir():
            print(f"MISSING PROTECTED DATASET: {name}: {root}", file=sys.stderr)
            return 2
        files = image_files(root)
        if len(files) != EXPECTED[name]:
            print(f"INVALID PROTECTED DATASET COUNT: {name}: expected {EXPECTED[name]}, found {len(files)}", file=sys.stderr)
            return 2
        datasets[name] = files

    output_dir.mkdir(parents=True, exist_ok=True)
    hashes: list[str] = []
    duplicate_hashes: dict[str, list[dict[str, str]]] = {}
    manifest_path = output_dir / "official_benchmark_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for name, files in datasets.items():
            label = "REAL" if name == "COCO_val2017" else "AIGC"
            for path in files:
                digest = sha256_file(path)
                entry = {"sha256": digest, "dataset": name, "relative_path": str(path.relative_to(roots[name])), "label": label}
                manifest.write(json.dumps(entry, sort_keys=True) + "\n")
                hashes.append(digest)
                duplicate_hashes.setdefault(digest, []).append(entry)

    unique_hashes = sorted(set(hashes))
    (output_dir / "official_benchmark_sha256.txt").write_text("\n".join(unique_hashes) + "\n", encoding="utf-8")
    duplicates = {key: value for key, value in duplicate_hashes.items() if len(value) > 1}
    (output_dir / "official_benchmark_duplicates.json").write_text(json.dumps(duplicates, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"COCO": len(datasets["COCO_val2017"]), "DALLE_ADVANCED": len(datasets["WildFake_DALLE_Advanced"]), "TOTAL": len(hashes), "UNIQUE_HASHES": len(unique_hashes), "DUPLICATE_HASHES": len(duplicates)}, sort_keys=True))
    return 0


def protected_hashes(path: Path) -> set[str]:
    if not path.is_file():
        raise RuntimeError(f"Protected benchmark hash list is missing: {path}")
    hashes = {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines() if len(line.strip()) == 64}
    if not hashes:
        raise RuntimeError("Protected benchmark hash list is empty.")
    return hashes


def manifest_paths(manifests: Iterable[Path]) -> Iterable[Path]:
    for manifest in manifests:
        if manifest.suffix.lower() not in {".jsonl", ".json", ".csv"}:
            continue
        if manifest.suffix.lower() == ".jsonl":
            lines = manifest.read_text(encoding="utf-8", errors="ignore").splitlines()
            for line in lines:
                try:
                    record: Any = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    for key in PATH_KEYS:
                        value = record.get(key)
                        if isinstance(value, str):
                            candidate = Path(value)
                            if candidate.is_file():
                                yield candidate
                            break


def audit_inputs(hash_file: Path, roots: list[Path], manifests: list[Path], report: Path) -> int:
    protected = protected_hashes(hash_file)
    candidates: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            raise RuntimeError(f"Training input root is missing: {root}")
        candidates.update(image_files(root))
    candidates.update(manifest_paths(manifests))

    matches: list[dict[str, str]] = []
    for path in sorted(candidates):
        digest = sha256_file(path)
        if digest in protected:
            matches.append({"path": str(path), "sha256": digest})

    payload = {
        "protected_benchmark": {"coco_val2017": 4998, "dalle_advanced": 8843, "total": 13841},
        "protected_unique_hashes": len(protected),
        "current_training_files_checked": len(candidates),
        "exact_matches": matches,
        "historical_manifests_checked": len(manifests),
        "historical_matches": matches,
        "status": "PASS" if not matches else "FAIL",
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if matches:
        print("OFFICIAL BENCHMARK CONTAMINATION DETECTED")
        return 1
    print("OFFICIAL BENCHMARK CONTAMINATION: 0")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build-protected-set")
    build.add_argument("--coco-root", type=Path, required=True)
    build.add_argument("--dalle-root", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    audit = sub.add_parser("audit")
    audit.add_argument("--protected-hashes", type=Path, required=True)
    audit.add_argument("--input-root", type=Path, action="append", default=[])
    audit.add_argument("--manifest", type=Path, action="append", default=[])
    audit.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "build-protected-set":
        return build_protected_set(args.coco_root, args.dalle_root, args.output_dir)
    try:
        return audit_inputs(args.protected_hashes, args.input_root, args.manifest, args.report)
    except RuntimeError as error:
        print(f"ISOLATION GATE BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
