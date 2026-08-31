"""
deployment/schemas.py
Pydantic Schemas for Inference Requests, Responses, and System Metadata
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    image_base64: Optional[str] = Field(None, description="Base64-encoded image string")
    threshold_mode: Optional[str] = Field("standard", description="Operating mode: 'standard', 'low_fpr_01', 'low_fpr_001'")
    include_forensic_breakdown: Optional[bool] = Field(False, description="Include FFT and Laplacian forensic metrics")

class ForensicBreakdown(BaseModel):
    fft_high_frequency_ratio: float = Field(..., description="High-frequency Fourier energy ratio")
    srm_residual_energy: float = Field(..., description="Spatial Rich Model noise residual energy")
    laplacian_variance: float = Field(..., description="Laplacian edge variance")
    inconsistency_status: str = Field(..., description="'CLEAN', 'ANOMALY_DETECTED', or 'COMPRESSION_DEGRADED'")

class PredictResponse(BaseModel):
    success: bool = True
    probability_aigc: float = Field(..., description="Calibrated probability that the image is synthetic [0.0..1.0]")
    raw_logit: float = Field(..., description="Uncalibrated model output logit")
    predicted_class: str = Field(..., description="'AIGC_SYNTHETIC' or 'AUTHENTIC_REAL'")
    is_aigc: bool = Field(..., description="True if probability exceeds the selected threshold")
    confidence_tier: str = Field(..., description="'HIGH_CONFIDENCE_REAL', 'UNCERTAIN_BORDERLINE', 'HIGH_CONFIDENCE_AIGC'")
    threshold_used: float = Field(..., description="Decision threshold applied")
    threshold_mode: str = Field(..., description="Threshold mode applied")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    device_used: str = Field(..., description="Execution device ('cuda:0' or 'cpu')")
    model_version: str = Field(..., description="Model identifier")
    model_sha256: str = Field(..., description="SHA-256 hash of active model weights")
    forensic_breakdown: Optional[ForensicBreakdown] = None

class BatchPredictResponse(BaseModel):
    success: bool = True
    total_images: int
    predictions: List[PredictResponse]
    total_latency_ms: float
    throughput_fps: float
    device_used: str

class OperatingThreshold(BaseModel):
    mode: str
    threshold: float
    target_fpr: str
    description: str

class ModelMetadataResponse(BaseModel):
    app_name: str
    app_version: str
    model_name: str
    architecture: str
    parameter_count: int
    model_sha256: str
    temperature_scaling: float
    operating_thresholds: List[OperatingThreshold]
    inference_service: str

class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool = True
    inference_available: bool
