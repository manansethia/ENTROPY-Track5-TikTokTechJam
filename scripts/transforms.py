import torch
import torch.nn.functional as F

CLIP_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
SIGLIP_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
SIGLIP_STD = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def prepare_inputs(rgb_01: torch.Tensor):
    """Convert Bx3xHxW RGB tensors in [0,1] to model inputs."""
    x = F.interpolate(rgb_01, size=(224, 224), mode="bilinear", align_corners=False)
    clip = (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
    siglip = (x - SIGLIP_MEAN.to(x.device)) / SIGLIP_STD.to(x.device)
    return clip, siglip, x
