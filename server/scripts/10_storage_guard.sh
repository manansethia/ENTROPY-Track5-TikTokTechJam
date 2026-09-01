#!/usr/bin/env bash
set -euo pipefail
AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
MIN_FREE_GB="${MIN_FREE_GB:-120}"
FREE=$(df -BG "$AI_ROOT" | awk 'NR==2 {gsub(/G/,"",$4);print $4}')
echo "Free: ${FREE}G"
if (( FREE < MIN_FREE_GB )); then
  echo "STOP: below ${MIN_FREE_GB}G safety floor." >&2
  exit 10
fi
