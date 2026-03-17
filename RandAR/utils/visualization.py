import torch
import torchvision
import cv2
import numpy as np


def make_grid(imgs, scale=0.5, row_first=True):
    """
    Supports input in either:
      - [B, C, H, W]
      - [B, H, W, C]

    Returns:
      np.ndarray of shape [H', W', C]
    """

    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs)
    elif torch.is_tensor(imgs):
        imgs = imgs.detach().cpu()
    else:
        raise TypeError(f"Unsupported type for imgs: {type(imgs)}")

    if imgs.ndim != 4:
        raise ValueError(f"Expected 4D tensor/array, got shape {tuple(imgs.shape)}")

    # Convert HWC -> CHW if needed
    if imgs.shape[1] in (1, 3):
        # already [B, C, H, W]
        pass
    elif imgs.shape[-1] in (1, 3):
        # [B, H, W, C] -> [B, C, H, W]
        imgs = imgs.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(
            f"Cannot infer channel dimension from shape {tuple(imgs.shape)}"
        )

    B = imgs.shape[0]

    # Convert to float in [0, 1] for torchvision.make_grid
    if imgs.dtype == torch.uint8:
        imgs = imgs.float() / 255.0
    else:
        imgs = imgs.float()
        imgs = imgs.clamp(0.0, 1.0)

    num_row = int(np.sqrt(B / 2))
    if num_row < 1:
        num_row = 1
    num_col = int(np.ceil(B / num_row))

    if row_first:
        img_grid = torchvision.utils.make_grid(imgs, nrow=num_col, padding=0)
    else:
        img_grid = torchvision.utils.make_grid(imgs, nrow=num_row, padding=0)

    img_grid = img_grid.permute(1, 2, 0).cpu().numpy()

    # optional resize
    if scale != 1.0:
        img_grid = cv2.resize(img_grid, None, fx=scale, fy=scale)

    return img_grid