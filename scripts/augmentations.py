"""Stable image augmentations matching the hackathon robustness specification."""

import io
import random

import cv2
import numpy as np
from PIL import Image, ImageEnhance


def _jpeg(img, quality):
    pil = Image.fromarray(img)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def _blur(img, sigma):
    k = max(3, int(round(sigma * 4)) | 1)
    return cv2.GaussianBlur(img, (k, k), sigmaX=float(sigma))


def _down_up(img, scale):
    h, w = img.shape[:2]
    nh, nw = max(2, int(h * scale)), max(2, int(w * scale))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _noise(img, sigma):
    x = img.astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, x.shape).astype(np.float32)
    return np.clip((x + noise) * 255.0, 0, 255).astype(np.uint8)


def _jitter(img, brightness=0.2, contrast=0.2, saturation=0.2):
    pil = Image.fromarray(img)
    if brightness:
        factor = random.uniform(1 - brightness, 1 + brightness)
        pil = ImageEnhance.Brightness(pil).enhance(factor)
    if contrast:
        factor = random.uniform(1 - contrast, 1 + contrast)
        pil = ImageEnhance.Contrast(pil).enhance(factor)
    if saturation:
        factor = random.uniform(1 - saturation, 1 + saturation)
        pil = ImageEnhance.Color(pil).enhance(factor)
    return np.array(pil)


def _crop(img, scale):
    h, w = img.shape[:2]
    ch, cw = max(2, int(h * scale)), max(2, int(w * scale))
    top = random.randint(0, h - ch) if h > ch else 0
    left = random.randint(0, w - cw) if w > cw else 0
    return img[top:top + ch, left:left + cw]


def _ai_enhancement_sharpen(img, strength=1.5):
    """Simulates Remini/Topaz AI texture enhancement and unsharp masking."""
    pil = Image.fromarray(img)
    enhancer = ImageEnhance.Sharpness(pil)
    pil = enhancer.enhance(float(strength))
    # Slight contrast boost mimicking restorative tone mapping
    pil = ImageEnhance.Contrast(pil).enhance(1.1)
    return np.array(pil)


def _super_resolution_upscale(img, factor=2.0):
    """Simulates 2x/4x AI Super-Resolution (Real-ESRGAN / SwinIR / Bicubic upscale)."""
    h, w = img.shape[:2]
    # Downscale then high-order Lanczos/Bicubic upscale
    down_h, down_w = max(4, int(h / factor)), max(4, int(w / factor))
    small = cv2.resize(img, (down_w, down_h), interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
    # Apply sub-pixel unsharp mask
    gaussian = cv2.GaussianBlur(upscaled, (0, 0), 2.0)
    sharpened = cv2.addWeighted(upscaled, 1.3, gaussian, -0.3, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def training_augment(img, cfg):
    """Apply stochastic training corruption while preserving RGB uint8 output."""
    out = img
    a = cfg["augmentations"]

    # One major redistribution-style degradation.
    choices = []
    if random.random() < a["jpeg_compression"]["prob"]:
        choices.append("jpeg")
    if random.random() < a["gaussian_blur"]["prob"]:
        choices.append("blur")
    if random.random() < a["resize"]["prob"]:
        choices.append("resize")
    if choices:
        kind = random.choice(choices)
        if kind == "jpeg":
            q0, q1 = a["jpeg_compression"]["quality_range"]
            out = _jpeg(out, random.uniform(q0, q1))
        elif kind == "blur":
            s0, s1 = a["gaussian_blur"]["sigma_range"]
            out = _blur(out, random.uniform(s0, s1))
        else:
            s0, s1 = a["resize"]["scale_range"]
            out = _down_up(out, random.uniform(s0, s1))

    if random.random() < a["gaussian_noise"]["prob"]:
        s0, s1 = a["gaussian_noise"]["sigma_range"]
        out = _noise(out, random.uniform(s0, s1))

    if random.random() < a["color_jitter"]["prob"]:
        out = _jitter(
            out,
            a["color_jitter"]["brightness"],
            a["color_jitter"]["contrast"],
            a["color_jitter"]["saturation"],
        )

    if random.random() < a["random_crop"]["prob"]:
        s0, s1 = a["random_crop"]["scale"]
        out = _crop(out, random.uniform(s0, s1))

    return out


def evaluation_transform(name, img):
    """Deterministic transform for the published robustness matrix."""
    if name == "Clean":
        return img
    if name.startswith("JPEG_"):
        return _jpeg(img, int(name.split("_")[1]))
    if name.startswith("Blur_"):
        return _blur(img, float(name.split("_")[1]))
    if name.startswith("Downscale_"):
        return _down_up(img, float(name.split("_")[1].replace("x", "")))
    if name.startswith("Noise_"):
        return _noise(img, float(name.split("_")[1]))
    if name == "ColorJitter":
        return _jitter(img, 0.2, 0.2, 0.2)
    if name == "CenterCrop_80":
        h, w = img.shape[:2]
        ch, cw = int(h * 0.8), int(w * 0.8)
        top, left = (h - ch) // 2, (w - cw) // 2
        return img[top:top + ch, left:left + cw]
    raise ValueError(f"Unknown transform: {name}")


PERTURBATIONS = [
    "Clean", "JPEG_90", "JPEG_70", "JPEG_50", "JPEG_30",
    "Blur_0.5", "Blur_1.0", "Blur_2.0",
    "Downscale_0.5x", "Downscale_0.25x",
    "Noise_0.02", "Noise_0.05", "Noise_0.10",
    "ColorJitter", "CenterCrop_80",
]
