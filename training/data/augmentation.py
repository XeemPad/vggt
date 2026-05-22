# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict, Sequence, Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms


def get_image_augmentation(
    color_jitter: Optional[Dict[str, float]] = None,
    gray_scale: bool = True,
    gau_blur: bool = False
) -> Optional[transforms.Compose]:
    """Create a composition of image augmentations.

    Args:
        color_jitter: Dictionary containing color jitter parameters:
            - brightness: float (default: 0.5)
            - contrast: float (default: 0.5)
            - saturation: float (default: 0.5)
            - hue: float (default: 0.1)
            - p: probability of applying (default: 0.9)
            If None, uses default values
        gray_scale: Whether to apply random grayscale (default: True)
        gau_blur: Whether to apply gaussian blur (default: False)

    Returns:
        A Compose object of transforms or None if no transforms are added
    """
    transform_list = []
    default_jitter = {
        "brightness": 0.5,
        "contrast": 0.5,
        "saturation": 0.5,
        "hue": 0.1,
        "p": 0.9
    }

    # Handle color jitter
    if color_jitter is not None:
        # Merge with defaults for missing keys
        effective_jitter = {**default_jitter, **color_jitter}
    else:
        effective_jitter = default_jitter

    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=effective_jitter["brightness"],
                    contrast=effective_jitter["contrast"],
                    saturation=effective_jitter["saturation"],
                    hue=effective_jitter["hue"],
                )
            ],
            p=effective_jitter["p"],
        )
    )

    if gray_scale:
        transform_list.append(transforms.RandomGrayscale(p=0.05))

    if gau_blur:
        transform_list.append(
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.0))], p=0.05
            )
        )

    return transforms.Compose(transform_list) if transform_list else None


def _undistort_simple_radial(x_distorted, y_distorted, k1, num_iters):
    """Invert x_d = x_u * (1 + k1 * r_u^2) with fixed-point iterations."""
    x = x_distorted
    y = y_distorted
    for _ in range(num_iters):
        r2 = x * x + y * y
        scale = 1.0 + k1 * r2
        scale = scale.clamp(min=1e-6)
        x = x_distorted / scale
        y = y_distorted / scale
    return x, y


def apply_random_simple_radial_augmentation(
    images: torch.Tensor,
    intrinsics: torch.Tensor,
    distortions: torch.Tensor,
    probability: float = 0.5,
    delta_range: Sequence[float] = (-0.05, 0.05),
    shared: bool = True,
    clamp_range: Optional[Sequence[float]] = (-0.3, 0.3),
    num_iters: int = 8,
    padding_mode: str = "border",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Randomly changes SIMPLE_RADIAL k1 and warps images to match the new target.

    For every output pixel, the function first maps the pixel through the new
    k1 to an undistorted ray, then projects that ray with the old k1 and samples
    from the original image. Intrinsics are kept unchanged.
    """
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [S, C, H, W], got {tuple(images.shape)}")
    if intrinsics.ndim != 3 or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"Expected intrinsics with shape [S, 3, 3], got {tuple(intrinsics.shape)}")
    if distortions.ndim != 2 or distortions.shape[-1] < 1:
        raise ValueError(f"Expected distortions with shape [S, 1+], got {tuple(distortions.shape)}")
    if torch.rand((), device=images.device) >= probability:
        return images, distortions

    num_frames, _, height, width = images.shape
    dtype = images.dtype
    device = images.device

    low, high = float(delta_range[0]), float(delta_range[1])
    if shared:
        delta = torch.empty(1, device=device, dtype=dtype).uniform_(low, high).expand(num_frames)
    else:
        delta = torch.empty(num_frames, device=device, dtype=dtype).uniform_(low, high)

    old_k1 = distortions[:, 0].to(device=device, dtype=dtype)
    new_k1 = old_k1 + delta
    if clamp_range is not None:
        new_k1 = new_k1.clamp(float(clamp_range[0]), float(clamp_range[1]))

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x = x.unsqueeze(0).expand(num_frames, -1, -1)
    y = y.unsqueeze(0).expand(num_frames, -1, -1)

    fx = intrinsics[:, 0, 0].to(device=device, dtype=dtype).view(num_frames, 1, 1)
    fy = intrinsics[:, 1, 1].to(device=device, dtype=dtype).view(num_frames, 1, 1)
    cx = intrinsics[:, 0, 2].to(device=device, dtype=dtype).view(num_frames, 1, 1)
    cy = intrinsics[:, 1, 2].to(device=device, dtype=dtype).view(num_frames, 1, 1)

    x_new = (x - cx) / fx
    y_new = (y - cy) / fy
    x_undist, y_undist = _undistort_simple_radial(
        x_new,
        y_new,
        new_k1.view(num_frames, 1, 1),
        num_iters=num_iters,
    )

    r2 = x_undist * x_undist + y_undist * y_undist
    old_scale = 1.0 + old_k1.view(num_frames, 1, 1) * r2
    sample_x = fx * (x_undist * old_scale) + cx
    sample_y = fy * (y_undist * old_scale) + cy

    grid_x = 2.0 * sample_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(height - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)

    warped_images = F.grid_sample(
        images,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=True,
    )

    augmented_distortions = distortions.clone()
    augmented_distortions[:, 0] = new_k1.to(dtype=augmented_distortions.dtype, device=augmented_distortions.device)
    return warped_images, augmented_distortions
