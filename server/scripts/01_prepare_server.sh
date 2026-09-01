#!/usr/bin/env bash
set -euo pipefail

AI_ROOT="${AI_ROOT:-/mnt/ai-storage/aigc_data}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ENV_DIR:-$HOME/.venvs/aigc-detector}"

if [[ ! -f /etc/fedora-release ]]; then
  echo 'WARNING: this script was designed for Fedora.'
fi

command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is required.' >&2; exit 1; }

mkdir -p "$AI_ROOT"/{models,datasets,validation_LOCKED,features,checkpoints,manifests,logs,hf_cache,tmp,benchmarks}
mkdir -p "$(dirname "$ENV_DIR")"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# Minimal control-plane dependencies. Full ML dependencies are installed by 02_install_ml_stack.sh.
pip install --upgrade huggingface_hub datasets modelscope pyyaml tqdm psutil safetensors

cat > "$AI_ROOT/manifests/environment.txt" <<ENV
project_root=$PROJECT_ROOT
env_dir=$ENV_DIR
ai_root=$AI_ROOT
os=$(cat /etc/fedora-release 2>/dev/null || echo unknown)
nvidia_smi=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true)
ENV

echo
 echo 'Prepared control environment.'
echo "Activate with: source $ENV_DIR/bin/activate"
echo "Next: bash server/scripts/02_install_ml_stack.sh"
