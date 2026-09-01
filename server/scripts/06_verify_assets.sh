#!/usr/bin/env bash
set -euo pipefail
AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"

printf '\n=== Filesystem ===\n'
df -h "$AI_ROOT"
printf '\n=== Model pool ===\n'
find "$AI_ROOT/models" -maxdepth 2 -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pth' -o -name '*.pt' \) -printf '%p %s bytes\n' 2>/dev/null | sort | head -100
printf '\n=== Dataset roots ===\n'
find "$AI_ROOT/datasets" -maxdepth 2 -type d -print 2>/dev/null | sort | head -100
printf '\n=== Validation lock ===\n'
ls -ld "$AI_ROOT/validation_LOCKED" 2>/dev/null || true
cat "$AI_ROOT/validation_LOCKED/DO_NOT_TRAIN.txt" 2>/dev/null || true
