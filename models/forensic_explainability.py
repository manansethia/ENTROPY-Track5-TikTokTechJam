"""Full-Spectrum Forensic Explainability & Diagnostic Attribution Suite for AIGC Detection.

Provides mathematically principled, memory-safe attribution engines:
1. ViTGradCAM: Gradient-weighted Class Activation Mapping for Vision Transformers (CLIP, SigLIP, DINOv2).
2. CNNConvNeXtGradCAM: Stage-aware Grad-CAM for ConvNeXt-V2 and CNN trunks.
3. ViTAttentionRollout: Multi-Head Self-Attention Rollout across transformer depth.
4. FrequencySpectralExplainer: 2D FFT power spectrum, radial/azimuthal decay, and iFFT spatial anomaly reconstruction.
5. EdgeResidualExplainer: Multiscale Sobel, Laplacian, and SRM high-pass boundary anomaly localization.
6. PatchForensicScorer: Localized patch-level attribution ranking and risk categorization.
7. ForensicDiagnosticSuite: Unified orchestrator producing comprehensive multi-panel visual diagnostic dashboards.
"""

from __future__ import annotations

import io
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/script safety
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ==============================================================================
# 1. Vision Transformer Grad-CAM Engine
# ==============================================================================

class ViTGradCAM:
    """High-Resolution Grad-CAM for Vision Transformers (CLIP, SigLIP, DINOv2, timm ViT).
    
    Correctly handles:
    - [CLS] token slicing vs patch-only token grids
    - LayerNorm and self-attention block hooks
    - Frozen backbone parameter sets with activation-level gradient propagation
    - Memory-safe hook removal to prevent tensor retention leaks
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        has_cls_token: bool = True,
        grid_size: Optional[Tuple[int, int]] = None,
    ):
        self.model = model
        self.has_cls_token = has_cls_token
        self.grid_size = grid_size
        self.target_layer = target_layer or self._auto_locate_target_layer(model)
        
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []

    def _auto_locate_target_layer(self, model: nn.Module) -> nn.Module:
        """Heuristically finds the optimal final normalization or encoder layer."""
        # 1. HuggingFace CLIP / SigLIP vision_model
        if hasattr(model, "vision_model"):
            vm = model.vision_model
            if hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
                return vm.encoder.layers[-1]
            if hasattr(vm, "post_layernorm"):
                return vm.post_layernorm
        # 2. HuggingFace Dinov2
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return model.encoder.layer[-1]
        # 3. OpenCLIP / timm VisionTransformer
        if hasattr(model, "transformer") and hasattr(model.transformer, "resblocks"):
            return model.transformer.resblocks[-1]
        if hasattr(model, "blocks") and len(model.blocks) > 0:
            return model.blocks[-1]
        if hasattr(model, "norm"):
            return model.norm
        # Fallback to last child module
        children = list(model.children())
        if children:
            return children[-1]
        return model

    def _register_hooks(self):
        self._remove_hooks()
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0]
            elif isinstance(output, torch.Tensor):
                self.activations = output
            elif hasattr(output, "last_hidden_state"):
                self.activations = output.last_hidden_state
            else:
                self.activations = output

            # Enable gradient retention on intermediate activation tensor
            if isinstance(self.activations, torch.Tensor) and not self.activations.requires_grad:
                self.activations.requires_grad_(True)
            if isinstance(self.activations, torch.Tensor):
                self.activations.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradients = grad_output[0]
            elif isinstance(grad_output, torch.Tensor):
                self.gradients = grad_output

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h_fwd, h_bwd])

    def _remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_class_idx: int = 1,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Computes Grad-CAM spatial heatmap.
        
        Args:
            input_tensor: [1, 3, H, W] normalized input tensor
            forward_fn: Callable returning classification logits [1, C] or score
            target_class_idx: Class index to attribute (1 for AIGC/Synthetic)
            target_shape: Output (H, W) to resize the final heatmap to
        
        Returns:
            2D numpy array [H, W] normalized in [0, 1]
        """
        self._register_hooks()
        try:
            if hasattr(self.model, "parameters"):
                try:
                    dev = next(self.model.parameters()).device
                    input_tensor = input_tensor.to(dev)
                except (StopIteration, Exception):
                    pass

            # Forward pass with gradients enabled for explainability
            with torch.set_grad_enabled(True):
                # Ensure input requires grad if backbone is frozen
                x = input_tensor.clone().detach().requires_grad_(True)
                if forward_fn is not None:
                    logits = forward_fn(x)
                else:
                    out = self.model(x)
                    if hasattr(out, "logits"):
                        logits = out.logits
                    elif isinstance(out, torch.Tensor):
                        logits = out
                    elif hasattr(out, "last_hidden_state"):
                        logits = out.last_hidden_state.mean(dim=1)
                    else:
                        logits = out

                if logits.dim() == 1:
                    score = logits[target_class_idx] if len(logits) > 1 else logits[0]
                elif logits.dim() == 2:
                    score = logits[0, target_class_idx] if logits.shape[1] > 1 else logits[0, 0]
                else:
                    score = logits.sum()

                self.model.zero_grad(set_to_none=True)
                score.backward(retain_graph=False)

            # Retrieve activations and gradients
            act = self.activations
            grad = self.gradients
            if grad is None and act is not None and act.grad is not None:
                grad = act.grad

            if act is None or grad is None:
                # Graceful fallback: return uniform or activation norm heatmap
                h = target_shape[0] if target_shape else input_tensor.shape[-2]
                w = target_shape[1] if target_shape else input_tensor.shape[-1]
                return np.zeros((h, w), dtype=np.float32)

            # Process ViT tokens
            # act: [B, N, D] or [B, D, H, W]
            if act.dim() == 3:
                # Token sequence [B, N, D]
                tokens_act = act[0]  # [N, D]
                tokens_grad = grad[0]  # [N, D]

                if self.has_cls_token and tokens_act.shape[0] > 1:
                    tokens_act = tokens_act[1:]  # [N-1, D]
                    tokens_grad = tokens_grad[1:]  # [N-1, D]

                # Channel-wise importance weights alpha_k = mean_i(dScore/dA_ik)
                weights = torch.mean(tokens_grad, dim=0, keepdim=True)  # [1, D]
                cam_tokens = F.relu(torch.sum(weights * tokens_act, dim=-1))  # [N_patches]

                # Reshape tokens to 2D grid
                num_patches = cam_tokens.shape[0]
                if self.grid_size is not None:
                    gh, gw = self.grid_size
                else:
                    side = int(np.sqrt(num_patches))
                    gh, gw = side, side

                if gh * gw == num_patches:
                    cam_2d = cam_tokens.view(gh, gw).detach().cpu().float().numpy()
                else:
                    cam_2d = np.ones((14, 14), dtype=np.float32)
            elif act.dim() == 4:
                # 2D spatial feature map [B, C, H', W']
                weights = torch.mean(grad[0], dim=(-2, -1), keepdim=True)  # [C, 1, 1]
                cam_tensor = F.relu(torch.sum(weights * act[0], dim=0))  # [H', W']
                cam_2d = cam_tensor.detach().cpu().float().numpy()
            else:
                cam_2d = np.ones((14, 14), dtype=np.float32)

            # Min-Max normalize
            cam_min, cam_max = cam_2d.min(), cam_2d.max()
            if cam_max > cam_min + 1e-8:
                cam_norm = (cam_2d - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_2d)

            # Resize to target shape
            out_h = target_shape[0] if target_shape else input_tensor.shape[-2]
            out_w = target_shape[1] if target_shape else input_tensor.shape[-1]
            cam_resized = cv2.resize(cam_norm, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            return np.clip(cam_resized, 0.0, 1.0)

        finally:
            self._remove_hooks()
            self.activations = None
            self.gradients = None


# ==============================================================================
# 2. ConvNeXt / CNN Stage Grad-CAM Engine
# ==============================================================================

class CNNConvNeXtGradCAM:
    """Stage-aware Grad-CAM for ConvNeXt-V2, ConvNeXt-Tiny, ResNet, and CNN backbones."""

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.target_layer = target_layer or self._auto_locate_conv_stage(model)
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.hooks: List[Any] = []

    def _auto_locate_conv_stage(self, model: nn.Module) -> nn.Module:
        """Finds final convolutional stage or norm layer."""
        if hasattr(model, "stages") and len(model.stages) > 0:
            return model.stages[-1]
        if hasattr(model, "encoder") and hasattr(model.encoder, "stages"):
            return model.encoder.stages[-1]
        if hasattr(model, "convnext") and hasattr(model.convnext, "encoder"):
            return model.convnext.encoder.stages[-1]
        if hasattr(model, "layer4"):
            return model.layer4
        if hasattr(model, "features"):
            return model.features[-1]
        children = list(model.children())
        return children[-1] if children else model

    def _register_hooks(self):
        self._remove_hooks()
        self.activations = None
        self.gradients = None

        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.activations = output[0]
            elif hasattr(output, "last_hidden_state"):
                self.activations = output.last_hidden_state
            else:
                self.activations = output
            if isinstance(self.activations, torch.Tensor):
                if not self.activations.requires_grad:
                    self.activations.requires_grad_(True)
                self.activations.retain_grad()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                self.gradients = grad_output[0]
            elif isinstance(grad_output, torch.Tensor):
                self.gradients = grad_output

        h_fwd = self.target_layer.register_forward_hook(forward_hook)
        h_bwd = self.target_layer.register_full_backward_hook(backward_hook)
        self.hooks.extend([h_fwd, h_bwd])

    def _remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def generate(
        self,
        input_tensor: torch.Tensor,
        forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        target_class_idx: int = 1,
        target_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        self._register_hooks()
        try:
            if hasattr(self.model, "parameters"):
                try:
                    dev = next(self.model.parameters()).device
                    input_tensor = input_tensor.to(dev)
                except (StopIteration, Exception):
                    pass

            with torch.set_grad_enabled(True):
                x = input_tensor.clone().detach().requires_grad_(True)
                if forward_fn is not None:
                    logits = forward_fn(x)
                else:
                    out = self.model(x)
                    logits = out.logits if hasattr(out, "logits") else out

                if logits.dim() == 1:
                    score = logits[target_class_idx] if len(logits) > 1 else logits[0]
                elif logits.dim() == 2:
                    score = logits[0, target_class_idx] if logits.shape[1] > 1 else logits[0, 0]
                else:
                    score = logits.sum()

                self.model.zero_grad(set_to_none=True)
                score.backward(retain_graph=False)

            act = self.activations
            grad = self.gradients
            if grad is None and act is not None and act.grad is not None:
                grad = act.grad

            out_h = target_shape[0] if target_shape else input_tensor.shape[-2]
            out_w = target_shape[1] if target_shape else input_tensor.shape[-1]

            if act is None or grad is None:
                return np.zeros((out_h, out_w), dtype=np.float32)

            if act.dim() == 4:
                # [B, C, H', W'] or [B, H', W', C]
                if act.shape[1] != grad.shape[1] and act.shape[-1] == grad.shape[-1]:
                    act = act.permute(0, 3, 1, 2)
                    grad = grad.permute(0, 3, 1, 2)
                weights = torch.mean(grad[0], dim=(-2, -1), keepdim=True)
                cam_tensor = F.relu(torch.sum(weights * act[0], dim=0))
                cam_2d = cam_tensor.detach().cpu().float().numpy()
            elif act.dim() == 3:
                # [B, N, C]
                weights = torch.mean(grad[0], dim=0, keepdim=True)
                cam_tokens = F.relu(torch.sum(weights * act[0], dim=-1))
                side = int(np.sqrt(cam_tokens.shape[0]))
                cam_2d = cam_tokens.view(side, side).detach().cpu().float().numpy()
            else:
                cam_2d = np.ones((14, 14), dtype=np.float32)

            c_min, c_max = cam_2d.min(), cam_2d.max()
            if c_max > c_min + 1e-8:
                cam_norm = (cam_2d - c_min) / (c_max - c_min)
            else:
                cam_norm = np.zeros_like(cam_2d)

            cam_resized = cv2.resize(cam_norm, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            return np.clip(cam_resized, 0.0, 1.0)
        finally:
            self._remove_hooks()
            self.activations = None
            self.gradients = None


# ==============================================================================
# 3. ViT Multi-Head Self-Attention Rollout Engine
# ==============================================================================

class ViTAttentionRollout:
    """Computes Multi-Head Self-Attention Rollout across ViT transformer depth (Abnar & Zuidema, 2020).
    
    Recursively rolls out attention flow:
    R^{(l)} = (0.5 * A^{(l)} + 0.5 * I) * R^{(l-1)}
    """

    def __init__(self, model: nn.Module, discard_ratio: float = 0.1, head_fusion: str = "mean"):
        self.model = model
        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion
        self.attentions: List[torch.Tensor] = []
        self.hooks: List[Any] = []

    def _get_attention_modules(self) -> List[nn.Module]:
        """Locates all self-attention modules in the vision encoder."""
        modules = []
        # HuggingFace CLIP / SigLIP
        if hasattr(self.model, "vision_model"):
            enc = getattr(self.model.vision_model, "encoder", None)
            if enc and hasattr(enc, "layers"):
                for layer in enc.layers:
                    if hasattr(layer, "self_attn"):
                        modules.append(layer.self_attn)
        # HuggingFace DINOv2
        elif hasattr(self.model, "encoder") and hasattr(self.model.encoder, "layer"):
            for layer in self.model.encoder.layer:
                if hasattr(layer, "attention"):
                    modules.append(layer.attention)
        # OpenCLIP / timm ViT
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "resblocks"):
            for block in self.model.transformer.resblocks:
                if hasattr(block, "attn"):
                    modules.append(block.attn)
        elif hasattr(self.model, "blocks"):
            for block in self.model.blocks:
                if hasattr(block, "attn"):
                    modules.append(block.attn)
        return modules

    def _register_hooks(self):
        self._remove_hooks()
        self.attentions.clear()

        attn_modules = self._get_attention_modules()
        for mod in attn_modules:
            def hook_fn(m, inp, outp):
                # Check for attention matrix in outputs
                if isinstance(outp, tuple) and len(outp) > 1 and outp[1] is not None:
                    self.attentions.append(outp[1].detach().cpu())
                elif isinstance(outp, tuple) and isinstance(outp[0], torch.Tensor):
                    # Compute approximate attention from q, k if available
                    pass
            self.hooks.append(mod.register_forward_hook(hook_fn))

    def _remove_hooks(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    @torch.no_grad()
    def generate(
        self,
        input_tensor: torch.Tensor,
        target_shape: Optional[Tuple[int, int]] = None,
        forward_fn: Optional[Callable[[torch.Tensor], Any]] = None,
    ) -> np.ndarray:
        """Calculates patch-level attention rollout map."""
        self._register_hooks()
        try:
            if hasattr(self.model, "parameters"):
                try:
                    dev = next(self.model.parameters()).device
                    input_tensor = input_tensor.to(dev)
                except (StopIteration, Exception):
                    pass

            # Enable output_attentions if supported
            orig_output_attentions = getattr(self.model.config, "output_attentions", False) if hasattr(self.model, "config") else False
            if hasattr(self.model, "config") and hasattr(self.model.config, "output_attentions"):
                try:
                    self.model.config.output_attentions = True
                except Exception:
                    pass

            if forward_fn is not None:
                try:
                    out = forward_fn(input_tensor)
                except Exception:
                    out = self.model(input_tensor)
            else:
                out = self.model(input_tensor)

            if hasattr(out, "attentions") and out.attentions is not None and len(out.attentions) > 0:
                self.attentions = [a.detach().cpu() for a in out.attentions]

            if hasattr(self.model, "config") and hasattr(self.model.config, "output_attentions"):
                try:
                    self.model.config.output_attentions = orig_output_attentions
                except Exception:
                    pass

            out_h = target_shape[0] if target_shape else input_tensor.shape[-2]
            out_w = target_shape[1] if target_shape else input_tensor.shape[-1]

            if not self.attentions:
                # Robust Fallback: extract patch energy from hidden state
                if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    tokens = out.last_hidden_state[0].detach().cpu().float()
                    if tokens.shape[0] > 196:
                        tokens = tokens[1:]
                    patch_energy = torch.norm(tokens, dim=-1).numpy()
                    side = int(np.sqrt(len(patch_energy)))
                    if side * side == len(patch_energy):
                        mask = patch_energy.reshape((side, side))
                    else:
                        mask = np.ones((14, 14), dtype=np.float32) * 0.5
                else:
                    return np.ones((out_h, out_w), dtype=np.float32) * 0.5
            else:
                # Rollout computation across layers
                result = torch.eye(self.attentions[0].shape[-1])
                for attn in self.attentions:
                    if attn.dim() == 4:
                        if self.head_fusion == "mean":
                            attn_fused = torch.mean(attn[0], dim=0)
                        elif self.head_fusion == "max":
                            attn_fused = torch.max(attn[0], dim=0)[0]
                        elif self.head_fusion == "min":
                            attn_fused = torch.min(attn[0], dim=0)[0]
                        else:
                            attn_fused = torch.mean(attn[0], dim=0)
                    else:
                        attn_fused = attn[0]

                    if self.discard_ratio > 0:
                        flat = attn_fused.flatten()
                        k = int(len(flat) * self.discard_ratio)
                        if k > 0:
                            val, _ = torch.topk(flat, k, largest=False)
                            thresh = val[-1]
                            attn_fused = torch.where(attn_fused <= thresh, torch.zeros_like(attn_fused), attn_fused)

                    i_matrix = torch.eye(attn_fused.shape[-1])
                    a_hat = 0.5 * attn_fused + 0.5 * i_matrix
                    a_hat = a_hat / (a_hat.sum(dim=-1, keepdim=True) + 1e-8)
                    result = torch.matmul(a_hat, result)

                num_tokens = result.shape[0]
                side = int(np.sqrt(num_tokens))
                if side * side == num_tokens:
                    mask = result.mean(dim=0).view(side, side).numpy()
                else:
                    mask_tokens = result[0, 1:]
                    side = int(np.sqrt(len(mask_tokens)))
                    if side * side == len(mask_tokens):
                        mask = mask_tokens.view(side, side).numpy()
                    else:
                        mask = np.ones((14, 14), dtype=np.float32)

            m_min, m_max = mask.min(), mask.max()
            if m_max > m_min + 1e-8:
                mask_norm = (mask - m_min) / (m_max - m_min)
            else:
                mask_norm = np.zeros_like(mask)

            mask_resized = cv2.resize(mask_norm, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            return np.clip(mask_resized, 0.0, 1.0)
        finally:
            self._remove_hooks()
            self.attentions.clear()


# ==============================================================================
# 4. Frequency Domain Spectral Power & Spatial iFFT Anomaly Engine
# ==============================================================================

@dataclass
class SpectralAnalysisResult:
    log_power_spectrum: np.ndarray          # 2D [H, W] log power magnitude
    radial_profile: np.ndarray             # 1D radial energy decay curve
    natural_power_law_fit: np.ndarray       # Theoretical 1/f^alpha curve
    azimuthal_profile: np.ndarray          # 1D angular energy profile
    spectral_difference_map: np.ndarray    # 2D frequency anomaly magnitude
    spatial_frequency_anomaly_map: np.ndarray  # 2D [H, W] iFFT spatial anomaly heatmap
    high_freq_energy_ratio: float          # Ratio of high-frequency power to total
    grid_peak_anomaly_score: float         # Z-score peak magnitude (checkerboard indicator)
    is_frequency_anomalous: bool           # Forensic flag for generative upsampling artifacts


class FrequencySpectralExplainer:
    """Analyzes 2D FFT magnitude spectrum, radial 1/f^alpha decay, and spatial frequency anomalies.
    
    Generative models (GANs, Diffusion latent decoders) exhibit:
    1. Abnormal high-frequency spectral roll-offs (deviating from natural 1/f^2 law)
    2. Periodic spectral spikes from transposed convolutions and sub-pixel upsampling
    3. Checkerboard phase incoherence
    """

    def __init__(self, num_radial_bins: int = 64, num_angular_bins: int = 36):
        self.num_radial_bins = num_radial_bins
        self.num_angular_bins = num_angular_bins

    def analyze(self, image_np: np.ndarray) -> SpectralAnalysisResult:
        """Runs full 2D FFT spectral analysis on RGB uint8 or float image.
        
        Args:
            image_np: [H, W, 3] or [H, W] numpy array
            
        Returns:
            SpectralAnalysisResult dataclass
        """
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image_np.astype(np.float32)

        h, w = gray.shape
        cy, cx = h // 2, w // 2

        # 1. 2D Fast Fourier Transform & Shift
        fft_2d = np.fft.fft2(gray)
        fft_shifted = np.fft.fftshift(fft_2d)
        magnitude = np.abs(fft_shifted)
        log_power = np.log1p(magnitude)

        # 2. Radial & Azimuthal Coordinate Grids
        y_indices, x_indices = np.ogrid[:h, :w]
        r_grid = np.hypot(y_indices - cy, x_indices - cx)
        theta_grid = np.arctan2(y_indices - cy, x_indices - cx)  # [-pi, pi]

        max_radius = min(cy, cx)
        r_bins = np.linspace(0, max_radius, self.num_radial_bins + 1)
        radial_energies = np.zeros(self.num_radial_bins, dtype=np.float32)

        for i in range(self.num_radial_bins):
            mask = (r_grid >= r_bins[i]) & (r_grid < r_bins[i + 1])
            if np.any(mask):
                radial_energies[i] = np.mean(log_power[mask])

        # 3. Natural 1/f^alpha Power Law Baseline Fitting
        # Natural images follow P(f) = A / (1 + f/f0)^alpha where alpha approx 2.0
        frequencies = (r_bins[:-1] + r_bins[1:]) / 2.0
        valid_idx = frequencies > 2
        f_val = frequencies[valid_idx]
        e_val = radial_energies[valid_idx]

        if len(f_val) > 5 and np.all(e_val > 0):
            try:
                # Log-log linear fit: log(E) = -alpha * log(f) + c
                log_f = np.log(f_val)
                log_e = np.log(np.maximum(e_val, 1e-6))
                poly = np.polyfit(log_f, log_e, 1)
                alpha_fit = -poly[0]
                fit_log_e = poly[0] * np.log(np.maximum(frequencies, 1.0)) + poly[1]
                natural_fit = np.exp(fit_log_e)
            except Exception:
                alpha_fit = 2.0
                natural_fit = radial_energies[0] / (1.0 + (frequencies / 5.0) ** 1.5)
        else:
            alpha_fit = 2.0
            natural_fit = radial_energies[0] / (1.0 + (frequencies / 5.0) ** 1.5)

        # 4. Azimuthal Angular Distribution
        theta_bins = np.linspace(-np.pi, np.pi, self.num_angular_bins + 1)
        azimuthal_energies = np.zeros(self.num_angular_bins, dtype=np.float32)
        for j in range(self.num_angular_bins):
            mask_th = (theta_grid >= theta_bins[j]) & (theta_grid < theta_bins[j + 1]) & (r_grid > 5)
            if np.any(mask_th):
                azimuthal_energies[j] = np.mean(log_power[mask_th])

        # 5. High-Frequency Grid Peak & Periodic Anomaly Detection
        # Subtract smooth background to locate discrete spectral spikes (GAN checkerboard peaks)
        smooth_bg = cv2.GaussianBlur(log_power, (21, 21), 5.0)
        spec_diff = np.maximum(log_power - smooth_bg, 0.0)
        
        # Exclude DC center component
        center_mask = r_grid < 10
        spec_diff[center_mask] = 0.0

        diff_mean = np.mean(spec_diff)
        diff_std = np.std(spec_diff) + 1e-8
        z_scores = (spec_diff - diff_mean) / diff_std
        max_peak_z = float(np.max(z_scores))

        # 6. High-Frequency Energy Ratio
        high_freq_mask = r_grid >= (max_radius * 0.60)
        total_energy = np.sum(magnitude) + 1e-8
        hf_energy = np.sum(magnitude[high_freq_mask])
        hf_ratio = float(hf_energy / total_energy)

        # 7. Spatial Frequency Anomaly Heatmap (iFFT High-Pass / Bandpass Reconstruction)
        # High-pass filter in Fourier domain: suppress low frequencies, retain high-frequency residuals
        highpass_filter = 1.0 - np.exp(- (r_grid ** 2) / (2.0 * (max_radius * 0.25) ** 2))
        filtered_fft = fft_shifted * highpass_filter
        
        # Inverse 2D FFT
        ifft_shift = np.fft.ifftshift(filtered_fft)
        spatial_hf_residual = np.abs(np.fft.ifft2(ifft_shift))

        # Normalize spatial anomaly map
        s_min, s_max = spatial_hf_residual.min(), spatial_hf_residual.max()
        if s_max > s_min + 1e-8:
            spatial_anomaly_norm = (spatial_hf_residual - s_min) / (s_max - s_min)
        else:
            spatial_anomaly_norm = np.zeros_like(spatial_hf_residual)

        # Smooth slightly for clean forensic heatmap visualization
        spatial_anomaly_vis = cv2.GaussianBlur(spatial_anomaly_norm, (7, 7), 2.0)
        spatial_anomaly_vis = (spatial_anomaly_vis - spatial_anomaly_vis.min()) / (spatial_anomaly_vis.max() - spatial_anomaly_vis.min() + 1e-8)

        is_anomalous = bool(max_peak_z > 4.5 or hf_ratio > 0.35 or abs(alpha_fit - 2.0) > 0.85)

        # Normalize 2D power spectrum for display
        lp_min, lp_max = log_power.min(), log_power.max()
        lp_norm = (log_power - lp_min) / (lp_max - lp_min + 1e-8)

        return SpectralAnalysisResult(
            log_power_spectrum=lp_norm,
            radial_profile=radial_energies,
            natural_power_law_fit=natural_fit,
            azimuthal_profile=azimuthal_energies,
            spectral_difference_map=spec_diff / (spec_diff.max() + 1e-8),
            spatial_frequency_anomaly_map=spatial_anomaly_vis,
            high_freq_energy_ratio=hf_ratio,
            grid_peak_anomaly_score=max_peak_z,
            is_frequency_anomalous=is_anomalous,
        )


# ==============================================================================
# 5. Multiscale Edge & Boundary Residual Explainer
# ==============================================================================

@dataclass
class EdgeResidualResult:
    sobel_magnitude: np.ndarray           # Normalized [H, W] 1st-order gradient
    laplacian_residual: np.ndarray        # Normalized [H, W] 2nd-order differential
    srm_residual: np.ndarray              # Normalized [H, W] SRM high-pass residual
    gradient_inconsistency_map: np.ndarray  # [H, W] localized boundary discontinuity heatmap
    edge_anomaly_score: float             # Global edge artifact metric


class EdgeResidualExplainer:
    """Multiscale gradient & boundary residual localization (Sobel, Laplacian, SRM).
    
    Detects:
    - Generative blending discontinuities along object contours
    - Inconsistent edge smoothing (DALL-E / FLUX skin vs background sharpness)
    - Deconvolution edge ringing and boundary halos
    """

    def __init__(self, srm_kernel_type: str = "laplacian"):
        # Normalized 3x3 SRM High-Pass Kernel
        self.srm_kernel = np.array(
            [[0.0, 0.25, 0.0],
             [0.25, -1.0, 0.25],
             [0.0, 0.25, 0.0]], dtype=np.float32
        )

    def analyze(self, image_np: np.ndarray) -> EdgeResidualResult:
        if image_np.ndim == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            gray = image_np.astype(np.float32)

        # 1. 1st-Order Sobel Gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_mag = np.hypot(sobel_x, sobel_y)
        sobel_norm = (sobel_mag - sobel_mag.min()) / (sobel_mag.max() - sobel_mag.min() + 1e-8)

        # 2. 2nd-Order Laplacian Residual
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
        lap_norm = (laplacian - laplacian.min()) / (laplacian.max() - laplacian.min() + 1e-8)

        # 3. SRM High-Pass Filter Residual
        if image_np.ndim == 3:
            srm_channels = [
                np.abs(cv2.filter2D(image_np[:, :, c].astype(np.float32), -1, self.srm_kernel))
                for c in range(3)
            ]
            srm_mag = np.sqrt(np.sum([c ** 2 for c in srm_channels], axis=0))
        else:
            srm_mag = np.abs(cv2.filter2D(gray, -1, self.srm_kernel))

        srm_norm = (srm_mag - srm_mag.min()) / (srm_mag.max() - srm_mag.min() + 1e-8)

        # 4. Localized Gradient Inconsistency / Boundary Discontinuity Map
        # Compute local variance of gradient directions along edges
        angles = np.arctan2(sobel_y, sobel_x + 1e-8)
        local_mean_angle = cv2.blur(angles, (9, 9))
        angle_diff = np.abs(angles - local_mean_angle)
        angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)

        # Weight angular variance by edge strength to focus on real boundaries
        boundary_discontinuity = angle_diff * sobel_norm
        boundary_smoothed = cv2.GaussianBlur(boundary_discontinuity, (11, 11), 3.0)
        bd_norm = (boundary_smoothed - boundary_smoothed.min()) / (boundary_smoothed.max() - boundary_smoothed.min() + 1e-8)

        # Composite Edge Anomaly Map
        composite_edge = 0.4 * srm_norm + 0.3 * lap_norm + 0.3 * bd_norm
        composite_norm = (composite_edge - composite_edge.min()) / (composite_edge.max() - composite_edge.min() + 1e-8)

        edge_score = float(np.mean(composite_norm[composite_norm > 0.5]) if np.any(composite_norm > 0.5) else np.mean(composite_norm))

        return EdgeResidualResult(
            sobel_magnitude=sobel_norm,
            laplacian_residual=lap_norm,
            srm_residual=srm_norm,
            gradient_inconsistency_map=composite_norm,
            edge_anomaly_score=edge_score,
        )


# ==============================================================================
# 6. Patch-Level Localized Attribution & Risk Scorer
# ==============================================================================

@dataclass
class PatchAttribution:
    patch_idx: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    gradcam_score: float
    attention_score: float
    frequency_score: float
    edge_score: float
    composite_risk: float
    primary_anomaly_category: str


class PatchForensicScorer:
    """Partitions images into an M x N grid and calculates localized attribution scores."""

    def __init__(self, grid_size: Tuple[int, int] = (14, 14)):
        self.grid_size = grid_size

    def score_patches(
        self,
        image_shape: Tuple[int, int],
        gradcam_map: np.ndarray,
        attention_map: np.ndarray,
        frequency_map: np.ndarray,
        edge_map: np.ndarray,
        weights: Tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20),
    ) -> List[PatchAttribution]:
        h, w = image_shape
        gh, gw = self.grid_size
        patch_h = h / gh
        patch_w = w / gw

        w_cam, w_att, w_freq, w_edge = weights
        patches: List[PatchAttribution] = []

        idx = 0
        for i in range(gh):
            y1 = int(round(i * patch_h))
            y2 = int(round((i + 1) * patch_h)) if i < gh - 1 else h
            for j in range(gw):
                x1 = int(round(j * patch_w))
                x2 = int(round((j + 1) * patch_w)) if j < gw - 1 else w

                cam_s = float(np.mean(gradcam_map[y1:y2, x1:x2]))
                att_s = float(np.mean(attention_map[y1:y2, x1:x2]))
                freq_s = float(np.mean(frequency_map[y1:y2, x1:x2]))
                edge_s = float(np.mean(edge_map[y1:y2, x1:x2]))

                composite = (
                    w_cam * cam_s + w_att * att_s + w_freq * freq_s + w_edge * edge_s
                )

                # Determine primary anomaly category
                signals = [
                    (cam_s, "Semantic Saliency"),
                    (att_s, "ViT Patch Focus"),
                    (freq_s, "High-Freq Spectral Anomaly"),
                    (edge_s, "Edge Boundary Inconsistency"),
                ]
                primary_cat = max(signals, key=lambda s: s[0])[1]

                patches.append(
                    PatchAttribution(
                        patch_idx=idx,
                        bbox=(x1, y1, x2, y2),
                        gradcam_score=cam_s,
                        attention_score=att_s,
                        frequency_score=freq_s,
                        edge_score=edge_s,
                        composite_risk=composite,
                        primary_anomaly_category=primary_cat,
                    )
                )
                idx += 1

        # Sort by composite risk descending
        patches.sort(key=lambda p: p.composite_risk, reverse=True)
        return patches


# ==============================================================================
# 7. Unified Forensic Diagnostic Suite & Visualization Engine
# ==============================================================================

class ForensicDiagnosticSuite:
    """Unified Orchestrator: Runs all explainability engines and outputs publication-grade reports."""

    def __init__(
        self,
        vit_gradcam: Optional[ViTGradCAM] = None,
        cnn_gradcam: Optional[CNNConvNeXtGradCAM] = None,
        attention_rollout: Optional[ViTAttentionRollout] = None,
        freq_explainer: Optional[FrequencySpectralExplainer] = None,
        edge_explainer: Optional[EdgeResidualExplainer] = None,
        patch_scorer: Optional[PatchForensicScorer] = None,
    ):
        self.vit_gradcam = vit_gradcam
        self.cnn_gradcam = cnn_gradcam
        self.attention_rollout = attention_rollout
        self.freq_explainer = freq_explainer or FrequencySpectralExplainer()
        self.edge_explainer = edge_explainer or EdgeResidualExplainer()
        self.patch_scorer = patch_scorer or PatchForensicScorer()

    def explain(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        input_tensor: Optional[torch.Tensor] = None,
        forward_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        pred_prob_aigc: Optional[float] = None,
        model_gates: Optional[List[float]] = None,
        output_path: Optional[Union[str, Path]] = None,
        dpi: int = 160,
    ) -> Dict[str, Any]:
        """Runs full-spectrum forensic explanation and renders 8-panel diagnostic dashboard.
        
        Returns:
            Dict with all structured forensic scores, maps, and top anomalous patch metadata.
        """
        # 1. Load image
        if isinstance(image, (str, Path)):
            pil_img = Image.open(image).convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image, Image.Image):
            pil_img = image.convert("RGB")
            img_np = np.array(pil_img)
        elif isinstance(image, np.ndarray):
            img_np = image.copy()
            pil_img = Image.fromarray(img_np)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        orig_h, orig_w = img_np.shape[:2]

        # 2. Determine model device & prepare PyTorch tensor
        target_device = torch.device("cpu")
        if self.vit_gradcam is not None and hasattr(self.vit_gradcam.model, "parameters"):
            try:
                target_device = next(self.vit_gradcam.model.parameters()).device
            except (StopIteration, Exception):
                pass
        elif self.cnn_gradcam is not None and hasattr(self.cnn_gradcam.model, "parameters"):
            try:
                target_device = next(self.cnn_gradcam.model.parameters()).device
            except (StopIteration, Exception):
                pass

        if input_tensor is None:
            preprocess = transforms.Compose([
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            t_img = preprocess(pil_img).unsqueeze(0).to(target_device)
        else:
            t_img = input_tensor.to(target_device)

        # 3. Compute ViT Grad-CAM
        if self.vit_gradcam is not None:
            gradcam_map = self.vit_gradcam.generate(
                t_img, forward_fn=forward_fn, target_shape=(orig_h, orig_w)
            )
        else:
            gradcam_map = np.zeros((orig_h, orig_w), dtype=np.float32)

        # 4. Compute ConvNeXt / CNN Grad-CAM
        if self.cnn_gradcam is not None:
            cnn_cam_map = self.cnn_gradcam.generate(
                t_img, forward_fn=forward_fn, target_shape=(orig_h, orig_w)
            )
            # Combine or blend with ViT Grad-CAM
            if np.max(gradcam_map) > 0:
                gradcam_map = 0.6 * gradcam_map + 0.4 * cnn_cam_map
            else:
                gradcam_map = cnn_cam_map

        # 5. Compute ViT Attention Rollout
        if self.attention_rollout is not None:
            attention_map = self.attention_rollout.generate(
                t_img, target_shape=(orig_h, orig_w), forward_fn=forward_fn
            )
        else:
            attention_map = np.ones((orig_h, orig_w), dtype=np.float32) * 0.5

        # 6. Compute Frequency Domain Spectral Analysis
        spectral_res = self.freq_explainer.analyze(img_np)

        # 7. Compute Multiscale Edge Residuals
        edge_res = self.edge_explainer.analyze(img_np)

        # 8. Compute Patch-Level Localized Attribution Scores
        patches = self.patch_scorer.score_patches(
            image_shape=(orig_h, orig_w),
            gradcam_map=gradcam_map,
            attention_map=attention_map,
            frequency_map=spectral_res.spatial_frequency_anomaly_map,
            edge_map=edge_res.gradient_inconsistency_map,
        )

        top_patches = patches[:5]

        # 9. Create 8-Panel Forensic Visual Dashboard
        fig = plt.figure(figsize=(24, 12), dpi=dpi)
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.25, wspace=0.15)

        # Panel 1: Original Image with Anomaly Bounding Boxes
        ax1 = fig.add_subplot(gs[0, 0])
        img_boxed = img_np.copy()
        for idx, p in enumerate(top_patches):
            x1, y1, x2, y2 = p.bbox
            color = (255, 50, 50) if p.composite_risk > 0.5 else (255, 180, 0)
            cv2.rectangle(img_boxed, (x1, y1), (x2, y2), color, max(int(orig_w / 200), 2))
            cv2.putText(
                img_boxed,
                f"#{idx+1} ({p.composite_risk:.2f})",
                (x1 + 4, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        ax1.imshow(img_boxed)
        ax1.set_title(f"1. Input Image & Top Anomaly Bounding Boxes\n({orig_w}x{orig_h})", fontsize=11, fontweight="bold")
        ax1.axis("off")

        # Panel 2: ViT / CNN Grad-CAM Heatmap
        ax2 = fig.add_subplot(gs[0, 1])
        cam_colored = cv2.applyColorMap((gradcam_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        blend_cam = cv2.addWeighted(img_np, 0.5, cam_colored, 0.5, 0)
        ax2.imshow(blend_cam)
        ax2.set_title("2. Gradient Class Activation Map (Grad-CAM)\n[Semantic & Saliency Attribution]", fontsize=11, fontweight="bold")
        ax2.axis("off")

        # Panel 3: ViT Multi-Head Attention Rollout
        ax3 = fig.add_subplot(gs[0, 2])
        att_colored = cv2.applyColorMap((attention_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        att_colored = cv2.cvtColor(att_colored, cv2.COLOR_BGR2RGB)
        blend_att = cv2.addWeighted(img_np, 0.5, att_colored, 0.5, 0)
        ax3.imshow(blend_att)
        ax3.set_title("3. ViT Attention Rollout\n[Transformer Patch-to-Patch Attention Flow]", fontsize=11, fontweight="bold")
        ax3.axis("off")

        # Panel 4: 2D FFT Power Spectrum & Radial Decay
        ax4 = fig.add_subplot(gs[0, 3])
        # Inset 2D power spectrum and plot radial curve
        ax4.plot(spectral_res.radial_profile, color="#00e5ff", lw=2, label="Observed Spectral Decay")
        ax4.plot(spectral_res.natural_power_law_fit, color="#ff9100", lw=2, linestyle="--", label="Natural 1/f² Baseline")
        ax4.set_title(f"4. 2D FFT Radial Energy Fall-Off\n(Peak Z-Score: {spectral_res.grid_peak_anomaly_score:.2f} | HF Ratio: {spectral_res.high_freq_energy_ratio*100:.1f}%)", fontsize=11, fontweight="bold")
        ax4.set_xlabel("Radial Frequency Bin", fontsize=9)
        ax4.set_ylabel("Log Power Magnitude", fontsize=9)
        ax4.legend(loc="upper right", fontsize=8)
        ax4.grid(True, linestyle=":", alpha=0.6)

        # Panel 5: Spatial Frequency Anomaly Map (iFFT High-Pass Reconstruction)
        ax5 = fig.add_subplot(gs[1, 0])
        freq_colored = cv2.applyColorMap((spectral_res.spatial_frequency_anomaly_map * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        freq_colored = cv2.cvtColor(freq_colored, cv2.COLOR_BGR2RGB)
        ax5.imshow(freq_colored)
        ax5.set_title("5. Frequency Domain Spatial Anomaly Map\n(iFFT Bandpass Residual Reconstruction)", fontsize=11, fontweight="bold")
        ax5.axis("off")

        # Panel 6: Edge & Boundary Residual Heatmap
        ax6 = fig.add_subplot(gs[1, 1])
        edge_colored = cv2.applyColorMap((edge_res.gradient_inconsistency_map * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        edge_colored = cv2.cvtColor(edge_colored, cv2.COLOR_BGR2RGB)
        ax6.imshow(edge_colored)
        ax6.set_title(f"6. Multiscale Edge & Boundary Residuals\n(Sobel + Laplacian + SRM | Score: {edge_res.edge_anomaly_score:.2f})", fontsize=11, fontweight="bold")
        ax6.axis("off")

        # Panel 7: Patch-Level Composite Anomaly Risk Grid
        ax7 = fig.add_subplot(gs[1, 2])
        gh, gw = self.patch_scorer.grid_size
        risk_grid = np.zeros((gh, gw), dtype=np.float32)
        for p in patches:
            # Map index back to grid
            r = p.patch_idx // gw
            c = p.patch_idx % gw
            risk_grid[r, c] = p.composite_risk
        risk_upscaled = cv2.resize(risk_grid, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        risk_colored = cv2.applyColorMap((risk_upscaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        risk_colored = cv2.cvtColor(risk_colored, cv2.COLOR_BGR2RGB)
        blend_risk = cv2.addWeighted(img_np, 0.4, risk_colored, 0.6, 0)
        ax7.imshow(blend_risk)
        ax7.set_title("7. Localized Patch Composite Risk Grid\n[Multi-Paradigm Risk Attribution]", fontsize=11, fontweight="bold")
        ax7.axis("off")

        # Panel 8: Forensic Diagnostic Summary & Decision Card
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis("off")
        prob_val = pred_prob_aigc if pred_prob_aigc is not None else 0.5
        is_synth = prob_val >= 0.5
        verdict = "SYNTHETIC AIGC" if is_synth else "AUTHENTIC REAL"
        v_color = "#d50000" if is_synth else "#00c853"

        card_text = (
            f"FORENSIC DIAGNOSIS SUMMARY\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Classification:  {verdict}\n"
            f"Confidence:      {prob_val*100:.2f}%\n"
            f"Anomaly Level:   {'HIGH RISK' if prob_val > 0.7 else 'MODERATE' if prob_val >= 0.5 else 'LOW RISK'}\n\n"
            f"Orthogonal Evidence Breakdown:\n"
            f" • ViT Saliency Focus:      {float(np.mean(gradcam_map)):.3f}\n"
            f" • Attention Dispersion:    {float(np.std(attention_map)):.3f}\n"
            f" • Spectral Anomaly Peak:   {spectral_res.grid_peak_anomaly_score:.2f} σ\n"
            f" • High-Freq Energy Ratio:  {spectral_res.high_freq_energy_ratio*100:.1f}%\n"
            f" • Edge Gradient Residual:  {edge_res.edge_anomaly_score:.3f}\n\n"
            f"Top Ranked Localized Anomalies:\n"
        )
        for i, p in enumerate(top_patches[:3]):
            card_text += f" {i+1}. BBox {p.bbox}: Risk {p.composite_risk:.2f} [{p.primary_anomaly_category}]\n"

        if model_gates:
            card_text += f"\nDynamic Router Gate Weights:\n"
            gate_names = ["SigLIP", "CLIP", "DINOv2", "ConvNeXt"]
            for name, g in zip(gate_names[:len(model_gates)], model_gates):
                card_text += f" • {name:8s}: {g*100:5.1f}%\n"

        ax8.text(
            0.05,
            0.95,
            card_text,
            transform=ax8.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#1e1e24", edgecolor=v_color, linewidth=2, alpha=0.9),
            color="#f0f0f0",
        )

        title_prob = f"AIGC Likelihood: {prob_val*100:.2f}% ({verdict})"
        fig.suptitle(f"Multi-Paradigm Forensic Diagnostic Suite — {title_prob}", fontsize=14, fontweight="bold", y=0.98)

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(out_p, bbox_inches="tight", dpi=dpi)
            print(f"[ForensicSuite] Saved 8-panel diagnostic dashboard to {out_p}")
        plt.close(fig)

        return {
            "image_path": str(image) if isinstance(image, (str, Path)) else "in_memory_image",
            "prob_aigc": float(prob_val),
            "verdict": verdict,
            "is_synthetic": bool(is_synth),
            "spectral_metrics": {
                "high_freq_energy_ratio": float(spectral_res.high_freq_energy_ratio),
                "grid_peak_anomaly_score": float(spectral_res.grid_peak_anomaly_score),
                "is_frequency_anomalous": bool(spectral_res.is_frequency_anomalous),
            },
            "edge_metrics": {
                "edge_anomaly_score": float(edge_res.edge_anomaly_score),
            },
            "top_anomalous_patches": [asdict(p) for p in top_patches],
            "model_gates": [float(g) for g in model_gates] if model_gates else [],
            "dashboard_output_path": str(output_path) if output_path else None,
        }
