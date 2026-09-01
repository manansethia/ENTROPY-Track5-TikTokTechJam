#!/usr/bin/env bash
# ==============================================================================
# Buildabot Production Setup Script for AIGC Robust Forensic Detection Server
# Run on: root@buildabot.lykoi-typhon.ts.net
# ==============================================================================

set -euo pipefail

echo "========================================================="
echo "  Deploying AIGC Forensic Detection Platform on Buildabot "
echo "========================================================="

APP_DIR="/home/manan/aigc_robust_detection"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# 1. Verify NVIDIA Driver & CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "[CUDA Check] GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "[WARNING] nvidia-smi not found. Proceeding with CPU inference."
fi

# 2. Check Systemd Service Installation
echo "[Service] Installing systemd service unit..."
cp "$APP_DIR/deployment/aigc_forensics.service" /etc/systemd/system/aigc_forensics.service
systemctl daemon-reload
systemctl enable aigc_forensics.service
systemctl restart aigc_forensics.service

echo "[Service Status] Checking service health..."
sleep 2
systemctl status aigc_forensics.service --no-pager

# 3. Health Check
echo "[Health] Querying local REST health endpoint on localhost:8000..."
curl -s http://localhost:8000/health || true

echo ""
echo "========================================================="
echo "  Buildabot Deployment Successfully Configured!          "
echo "  API Endpoint: http://buildabot.lykoi-typhon.ts.net:8000"
echo "========================================================="
