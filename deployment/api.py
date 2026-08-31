"""
deployment/api.py
FastAPI High-Throughput REST API Server for AIGC Vision Detector
"""

import os
import sys
import time
from pathlib import Path
from typing import List
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from deployment.config import config
from deployment.schemas import (
    PredictRequest, PredictResponse, BatchPredictResponse,
    ModelMetadataResponse, OperatingThreshold, HealthResponse
)
from deployment.model_loader import load_production_model
from deployment.inference import ForensicInferenceEngine
from deployment.preprocess import load_image_from_bytes

app = FastAPI(
    title="AIGC Robust Forensic Vision Detector",
    version="1.0.0",
    description="High-Accuracy Forensic AIGC Image Detector with Multi-Domain Foundation Fusion (<2B Parameters)"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000").split(",") if origin.strip()],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

global_engine: ForensicInferenceEngine = None
global_metadata = {}

@app.on_event("startup")
def startup_event():
    global global_engine, global_metadata
    print("[STARTUP] Initializing Model & Inference Engine...", flush=True)
    model, metadata = load_production_model(device=config.device)
    global_metadata = metadata
    global_engine = ForensicInferenceEngine(model, metadata, device=config.device)
    print(f"[STARTUP] Active Device: {global_engine.device} | Params: {metadata['total_parameters']:,}", flush=True)

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=global_engine is not None,
        inference_available=global_engine is not None,
    )

@app.get("/v1/metadata", response_model=ModelMetadataResponse)
def get_metadata():
    thresholds = [
        OperatingThreshold(mode="standard", threshold=config.threshold_standard, target_fpr="1.00%", description="Standard 50% boundary"),
        OperatingThreshold(mode="low_fpr_10", threshold=config.threshold_low_fpr_10, target_fpr="1.00%", description="Calibrated FPR <= 1.00%"),
        OperatingThreshold(mode="low_fpr_05", threshold=config.threshold_low_fpr_05, target_fpr="0.50%", description="Calibrated FPR <= 0.50%"),
        OperatingThreshold(mode="low_fpr_01", threshold=config.threshold_low_fpr_01, target_fpr="0.10%", description="Ultra-Reliable Enterprise Gate (FPR <= 0.10%)"),
        OperatingThreshold(mode="low_fpr_005", threshold=config.threshold_low_fpr_005, target_fpr="0.05%", description="High-Security Gate (FPR <= 0.05%)"),
        OperatingThreshold(mode="low_fpr_001", threshold=config.threshold_low_fpr_001, target_fpr="0.01%", description="Zero-False-Alarm Critical Gate (FPR <= 0.01%)")
    ]
    
    return ModelMetadataResponse(
        app_name="AIGC Forensic Vision Detector",
        app_version="1.0.0",
        model_name=config.model_name,
        architecture="CLIP-ViT-L/14 + SigLIP-SO400M-14 + Wavelet SRM Residual Head",
        parameter_count=global_metadata.get("total_parameters", config.total_parameters),
        model_sha256=global_metadata.get("parameter_hash", "UNKNOWN"),
        temperature_scaling=config.temperature_scaling,
        operating_thresholds=thresholds,
        inference_service="private-inference-service"
    )

@app.post("/v1/predict", response_model=PredictResponse)
async def predict_image(
    file: UploadFile = File(None),
    request_body: PredictRequest = None,
    threshold_mode: str = Query("standard", description="Operating mode: 'standard', 'low_fpr_01', 'low_fpr_001'"),
    include_forensic_breakdown: bool = Query(True, description="Include Fourier and Laplacian metrics")
):
    if file:
        image_bytes = await file.read()
        return global_engine.predict(
            image_bytes,
            threshold_mode=threshold_mode,
            include_forensic_breakdown=include_forensic_breakdown
        )
    elif request_body and request_body.image_base64:
        return global_engine.predict(
            request_body.image_base64,
            threshold_mode=request_body.threshold_mode or threshold_mode,
            include_forensic_breakdown=request_body.include_forensic_breakdown or include_forensic_breakdown
        )
    else:
        raise HTTPException(status_code=400, detail="Must provide either an uploaded image file or image_base64 string.")

@app.post("/v1/predict/batch", response_model=BatchPredictResponse)
async def predict_batch(
    files: List[UploadFile] = File(...),
    threshold_mode: str = Query("standard")
):
    if len(files) > 128:
        raise HTTPException(status_code=400, detail="Maximum batch size is 128 images.")
        
    t0 = time.perf_counter()
    image_bytes_list = [await f.read() for f in files]
    predictions = global_engine.predict_batch(image_bytes_list, threshold_mode=threshold_mode)
    total_latency = (time.perf_counter() - t0) * 1000.0
    fps = len(files) / max(0.001, (total_latency / 1000.0))
    
    return BatchPredictResponse(
        success=True,
        total_images=len(files),
        predictions=predictions,
        total_latency_ms=round(total_latency, 2),
        throughput_fps=round(fps, 1),
        device_used=str(global_engine.device)
    )

frontend_dir = Path(__file__).resolve().parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")
    
    @app.get("/", response_class=HTMLResponse)
    def index():
        index_file = frontend_dir / "index.html"
        if index_file.exists():
            return HTMLResponse(content=open(index_file).read())
        return HTMLResponse(content="<h1>AIGC Detector API Active</h1><p>Visit /docs for Swagger UI.</p>")
