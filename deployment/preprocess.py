"""
deployment/preprocess.py
Deterministic Image Preprocessing Pipeline for AIGC Vision Detector
Guarantees 100% numerical parity with training and validation evaluation.
"""

import os
import io
import base64
from typing import Union, List
from PIL import Image, ImageOps
import numpy as np
import torch
from torchvision import transforms

# Pre-registered ImageNet/CLIP Normalization Constants
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]

# Production PyTorch Evaluation Transform (Exact match with Research Pipeline)
production_eval_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
])

def load_image_from_bytes(image_bytes: bytes) -> Image.Image:
    """Safely loads and validates image bytes into an RGB PIL Image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to decode image: {str(e)}")

def load_image_from_base64(b64_string: str) -> Image.Image:
    """Decodes a base64 string into an RGB PIL Image."""
    if "," in b64_string:
        b64_string = b64_string.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_string)
    return load_image_from_bytes(image_bytes)

def load_pil_image(image_input: Union[Image.Image, bytes, str]) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    elif isinstance(image_input, bytes):
        return load_image_from_bytes(image_input)
    elif isinstance(image_input, str):
        if os.path.exists(image_input):
            with open(image_input, "rb") as f:
                return load_image_from_bytes(f.read())
        else:
            return load_image_from_base64(image_input)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

def preprocess_single_image(image_input: Union[Image.Image, bytes, str], device: str = "cpu") -> torch.Tensor:
    """
    Transforms a single image into a normalized tensor ready for inference.
    Returns: Tensor of shape (1, 3, 224, 224)
    """
    pil_img = load_pil_image(image_input)
    tensor = production_eval_transform(pil_img).unsqueeze(0).to(device)
    return tensor

def preprocess_batch_images(images: List[Union[Image.Image, bytes, str]], device: str = "cpu") -> torch.Tensor:
    """
    Transforms a list of images into a single batched tensor.
    Returns: Tensor of shape (B, 3, 224, 224)
    """
    tensors = []
    for img in images:
        pil_img = load_pil_image(img)
        tensors.append(production_eval_transform(pil_img))
    return torch.stack(tensors).to(device)
