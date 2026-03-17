import torch
from typing import Optional

def extract_latent_tokens(batch_x: torch.Tensor) -> torch.Tensor:
    """
    Supports:
      (B, 1, T) -> (B, T)
      (B, T)    -> (B, T)
    Returns long tensor.
    """
    if batch_x.dim() == 3:
        # (B, 1, T)
        if batch_x.shape[1] != 1:
            raise ValueError(
                f"Expected x shape (B,1,T) but got (B,{batch_x.shape[1]},{batch_x.shape[2]}). "
                "If you actually have multiple streams, you must change dataset to output the correct stream."
            )
        return batch_x[:, 0, :].long()
    if batch_x.dim() == 2:
        return batch_x.long()
    raise ValueError(f"Unexpected latent x shape: {tuple(batch_x.shape)}")

def make_token_order(batch_size: int, T: int, device: torch.device, mode: str) -> Optional[torch.Tensor]:
    """
    mode:
      - "config": do not pass token_order (model decides internally by self.position_order)
      - "raster": force raster order
      - "random": force per-sample random permutation
    """
    if mode == "config":
        return None
    if mode == "raster":
        return torch.arange(T, device=device).unsqueeze(0).repeat(batch_size, 1)
    if mode == "random":
        return torch.stack([torch.randperm(T, device=device) for _ in range(batch_size)], dim=0)
    raise ValueError(f"Unknown order mode: {mode}")