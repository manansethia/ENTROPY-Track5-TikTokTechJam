#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
DATA_ROOT="$AI_ROOT/datasets"
mkdir -p "$DATA_ROOT" "$AI_ROOT/validation_LOCKED" "$AI_ROOT/manifests"

source "${ENV_DIR:-$HOME/.venvs/aigc-detector}/bin/activate"

# Keep caches on the large HDD, not the system NVMe.
export HF_HOME="$AI_ROOT/hf_cache"
export HF_DATASETS_CACHE="$AI_ROOT/hf_cache/datasets"
export MODELSCOPE_CACHE="$AI_ROOT/hf_cache/modelscope"

usage() {
  cat <<USAGE
Usage:
  $0 --community-small   Download Community Forensics-Small
  $0 --sid               Download SID_Set
  $0 --cifake            Download CIFAKE metadata/files where supported
  $0 --wildfake          Attempt ModelScope WildFake download
  $0 --all               Run all non-benchmark downloads

The official challenge validation subset is NOT downloaded by this script.
Use 05_lock_validation.sh after obtaining the exact challenge-provided files.
USAGE
}

[[ $# -gt 0 ]] || { usage; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --community-small)
      echo 'Downloading Community Forensics-Small into parquet/cache form...'
      hf download OwensLab/CommunityForensics-Small --repo-type dataset --local-dir "$DATA_ROOT/community_forensics_small"
      ;;
    --sid)
      echo 'Downloading SID_Set...'
      hf download saberzl/SID_Set --repo-type dataset --local-dir "$DATA_ROOT/sid_set"
      ;;
    --cifake)
      echo 'CIFAKE is Kaggle-hosted. Install/configure Kaggle credentials, then place the archive under datasets/cifake.'
      echo 'Example: kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p "$DATA_ROOT/cifake" --unzip'
      ;;
    --wildfake)
      echo 'Downloading WildFake from ModelScope...'
      python - <<PY
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download('hy2628982280/WildFake', cache_dir='$DATA_ROOT/wildfake_cache')
PY
      ;;
    --all)
      "$0" --community-small
      "$0" --sid
      "$0" --cifake
      "$0" --wildfake
      ;;
    *) usage; exit 2;;
  esac
  shift
done

echo 'Dataset stage complete. Verify licenses and isolate challenge validation before training.'
