#!/usr/bin/env bash
set -e
source /home/manan/.venvs/aigc-detector/bin/activate
cd /home/manan/aigc_robust_detection
export PYTHONPATH=/home/manan/aigc_robust_detection
export CHECKPOINT_PATH=/home/manan/aigc_robust_detection/checkpoints/production/final_champion_frozen_model.pt
export IS_BUILDABOT=1
export CORS_ALLOWED_ORIGINS="https://techjam.manansethia.com,https://tiktoktechjam2026.manansethia.com,https://manansethia.com,https://www.manansethia.com,http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000,http://127.0.0.1:8000"
exec python -m uvicorn app.server:app --host 127.0.0.1 --port 8000 --workers 1

