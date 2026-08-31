from .tri_hybrid_detector import MasterEnsembleDetector
from .quad_hybrid_detector import QuadHybridGatingHead
from .srm_filters import WaveletResidualBlock, SRMConvolution
from .fft_spectral_detector import FFTSpectralFeatureExtractor, FFTEnergyClassifierHead
from .edge_artifact_detector import EdgeArtifactFeatureExtractor
from .forensic_explainability import (
    ViTGradCAM,
    CNNConvNeXtGradCAM,
    ViTAttentionRollout,
    FrequencySpectralExplainer,
    EdgeResidualExplainer,
    PatchForensicScorer,
    ForensicDiagnosticSuite,
    SpectralAnalysisResult,
    EdgeResidualResult,
    PatchAttribution,
)

__all__ = [
    "MasterEnsembleDetector",
    "QuadHybridGatingHead",
    "WaveletResidualBlock",
    "SRMConvolution",
    "FFTSpectralFeatureExtractor",
    "FFTEnergyClassifierHead",
    "EdgeArtifactFeatureExtractor",
    "ViTGradCAM",
    "CNNConvNeXtGradCAM",
    "ViTAttentionRollout",
    "FrequencySpectralExplainer",
    "EdgeResidualExplainer",
    "PatchForensicScorer",
    "ForensicDiagnosticSuite",
    "SpectralAnalysisResult",
    "EdgeResidualResult",
    "PatchAttribution",
]
