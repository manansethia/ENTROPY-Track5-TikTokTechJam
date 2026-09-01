#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
MIN_HDD_GB="${MIN_HDD_GB:-700}"

printf '\n=== AIGC detector hardware audit ===\n'
printf 'AI_ROOT: %s\n' "$AI_ROOT"

if [[ -f /etc/fedora-release ]]; then
  echo "OS: $(cat /etc/fedora-release)"
else
  echo "WARNING: /etc/fedora-release not found; expected Fedora."
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo '--- NVIDIA ---'
  nvidia-smi --query-gpu=name,memory.total,driver_version,pstate --format=csv,noheader
  echo
  nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || true
else
  echo 'ERROR: nvidia-smi not found.' >&2
  exit 1
fi

echo
 echo '--- CPU / RAM ---'
lscpu | grep -E 'Model name|CPU\(s\)|Thread|Core|Socket' | head -20 || true
free -h

echo
 echo '--- Disks ---'
lsblk -o NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS

echo
 echo '--- AI storage ---'
mkdir -p "$AI_ROOT"
df -h "$AI_ROOT"
AVAIL_GB=$(df -BG "$AI_ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}')
if (( AVAIL_GB < MIN_HDD_GB )); then
  echo "WARNING: only ${AVAIL_GB}G free at AI_ROOT; recommended >= ${MIN_HDD_GB}G for the full experiment." >&2
else
  echo "Storage headroom: ${AVAIL_GB}G free."
fi
