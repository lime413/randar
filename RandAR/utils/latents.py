import torch
from typing import Optional
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
sys.path.append("./")

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

def make_token_order(
        batch_size: int,
        T: int,
        device: torch.device,
        mode: str,
        shuffle_ratio: Optional[float] = None,) -> Optional[torch.Tensor]:
    """
    Generate token order for RandAR-style training.
    
    Args:
        batch_size: number of samples in batch
        T: number of tokens (block_size)
        device: torch device
        mode: "config" | "raster" | "random" | "adaptive"
        shuffle_ratio: fraction of tokens to shuffle (0.0–1.0), only used for "adaptive"
    
    Returns:
        token_order: [B, T] permutation tensor, or None for "config" mode
    """
    if mode == "config":
        return None
    if mode == "raster":
        return torch.arange(T, device=device).unsqueeze(0).repeat(batch_size, 1)
    if mode == "random":
        return torch.stack([torch.randperm(T, device=device) for _ in range(batch_size)], dim=0)
    
    if mode == "adaptive":
        if shuffle_ratio is None:
            # Fallback to random if shuffle_ratio not provided
            return torch.stack([torch.randperm(T, device=device) for _ in range(batch_size)], dim=0)
        
        if shuffle_ratio <= 0.0:
            # Raster order
            return torch.arange(T, device=device).unsqueeze(0).repeat(batch_size, 1)
        
        if shuffle_ratio >= 1.0:
            # Fully random
            return torch.stack([torch.randperm(T, device=device) for _ in range(batch_size)], dim=0)
        
        # Partial shuffling: shuffle only shuffle_ratio fraction of tokens
        token_orders = []
        for _ in range(batch_size):
            perm = torch.arange(T, device=device)
            num_to_shuffle = int(T * shuffle_ratio)
            
            if num_to_shuffle == 0:
                token_orders.append(perm)
            else:
                
                shuffle_indices = torch.randperm(T, device=device)[:num_to_shuffle]
                
                shuffled_values = perm[shuffle_indices][torch.randperm(num_to_shuffle, device=device)]
                perm[shuffle_indices] = shuffled_values
                
                token_orders.append(perm)
        
        return torch.stack(token_orders, dim=0)
    
    raise ValueError(f"Unknown order mode: {mode}")


def resolve_order_mode(model: torch.nn.Module, mode: str) -> str:
    if mode == "config":
        return getattr(model, "position_order", "raster")
    return mode


def resolve_shuffle_ratio(
    resolved_order_mode: str,
    config_max_shuffle_ratio: Optional[float] = None,
    explicit_shuffle_ratio: Optional[float] = None,
) -> Optional[float]:
    if resolved_order_mode != "adaptive":
        return None

    if explicit_shuffle_ratio is not None:
        return float(explicit_shuffle_ratio)
    if config_max_shuffle_ratio is not None:
        return float(config_max_shuffle_ratio)

    raise ValueError(
        "Adaptive evaluation requires a shuffle ratio. "
        "Provide it explicitly or store max_shuffle_ratio in the config."
    )


# -------------------------
# Atomic save
# -------------------------

def _atomic_save_npy(dst_path: Path, array: np.ndarray) -> None:
    """
    Atomically save ndarray to .npy (temp file in same dir + os.replace).
    """
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(array, dtype=np.int64)

    tmp_path = dst_path.with_name(dst_path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, array, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, dst_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


# -------------------------
# Encoding: images -> indices
# -------------------------

@torch.inference_mode()
def encode_images(
    vq_model: torch.nn.Module,
    x: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Returns indices of shape (B, T) as torch.long.
    """
    def _encode_one(batch: torch.Tensor) -> torch.Tensor:
        out = vq_model.encode_indices(batch)
        out = out.to(dtype=torch.long)
        if out.ndim != 2:
            out = out.view(out.shape[0], -1)
        return out

    if chunk_size is None or x.shape[0] <= chunk_size:
        return _encode_one(x)

    parts = []
    for s in range(0, x.shape[0], chunk_size):
        parts.append(_encode_one(x[s:s + chunk_size]))
    return torch.cat(parts, dim=0)


# -------------------------
# Saving
# -------------------------
def _save_payload(payload: Dict[str, Any], out_dir: Path) -> None:
    codes = payload["codes"]     # (B, 1, T)
    labels = payload["labels"]   # (B,)
    indices = payload["indices"] # (B,)

    assert codes.ndim == 3, f"Expected (B,1,T), got {codes.shape}"
    assert codes.shape[1] == 1, f"Expected num_aug=1 (no aug), got {codes.shape}"
    assert labels.ndim == 1 and indices.ndim == 1
    assert codes.shape[0] == labels.shape[0] == indices.shape[0]

    for i in range(codes.shape[0]):
        cls_id = int(labels[i])
        sample_id = int(indices[i])
        dst = out_dir / str(cls_id) / f"{sample_id}.npy"
        if dst.exists():
            continue
        # Save per-sample array of shape (1, T)
        _atomic_save_npy(dst, codes[i])
