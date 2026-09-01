#!/usr/bin/env bash
set -euo pipefail
AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
OUT="$AI_ROOT/logs/memory"
mkdir -p "$OUT"
STAMP=$(date +%Y%m%d_%H%M%S)
{
  echo "timestamp=$STAMP"
  echo "--- nvidia-smi ---"
  nvidia-smi
  echo "--- processes ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null || true
  echo "--- memory ---"
  free -h
} | tee "$OUT/diagnostic_$STAMP.txt"
