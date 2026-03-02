from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from omegaconf import OmegaConf
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import sys
sys.path.append("./")

from RandAR.dataset.builder import build_dataset
from RandAR.util import instantiate_from_config

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
# Checkpoint loading
# -------------------------
def load_tokenizer_from_yaml(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    config = OmegaConf.load(args.config)
    vq_model = instantiate_from_config(config.model).to(device)
    vq_model.load_state_dict(torch.load(args.vq_ckpt, map_location=device, weights_only=True), strict=True)
    vq_model.eval()
    for p in vq_model.parameters():
        p.requires_grad_(False)
    return vq_model


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
# Transform pipeline
# -------------------------
def build_image_transform(args: argparse.Namespace) -> transforms.Compose:
    if args.dataset == "cifar10":
        return transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[1.0] * 3, inplace=True),
        ])
    raise ValueError(f"Unsupported dataset: {args.dataset}")


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


# -------------------------
# Main
# -------------------------
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Requires at least one CUDA GPU.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"device={device}, seed={seed}")

    # ---- tokenizer (VQ-VAE) ----
    vq_model = load_tokenizer_from_yaml(args, device)
    print(f"Loaded VQ-VAE tokenizer from {args.vq_ckpt}")

    # ---- dataset + transforms ----
    transform = build_image_transform(args)
    dataset = build_dataset(is_train=args.is_train, args=args, transform=transform)

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # ---- output folder ----
    out_dir = Path(args.latents_path) / f"{args.dataset}-{args.tokenizer_name}-{args.image_size}_codes"
    out_dir.mkdir(parents=True, exist_ok=True)
    for c in range(int(dataset.nb_classes)):
        (out_dir / str(c)).mkdir(parents=True, exist_ok=True)

    pbar = tqdm(data_loader, desc="Encoding images")

    for x, y, index in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        index = index.to(device, non_blocking=True)

        # ---- tokenize ----
        codes_bt = encode_images(vq_model, x, chunk_size=args.encode_chunk_size)  # (B, T)

        if codes_bt.ndim != 2:
            raise ValueError(f"Tokenizer encode_indices produced unexpected shape: {tuple(codes_bt.shape)}")

        B, T = codes_bt.shape

        # must be square for tokenizer decoder and AR model
        grid = int(math.isqrt(T))
        if grid * grid != T:
            raise ValueError(
                f"Tokenizer produced T={T} tokens per image, which is not a perfect square. "
                "Tokenizer decoder and AR model expect square token grids."
            )

        # Save as (B, 1, T): num_aug=1 (no augmentation)
        codes = codes_bt.unsqueeze(1)  # (B, 1, T)

        payload = {
            "codes": codes.detach().cpu().numpy().astype(np.int64, copy=False),
            "labels": y.detach().cpu().numpy().astype(np.int64, copy=False),
            "indices": index.detach().cpu().numpy().astype(np.int64, copy=False),
        }
        _save_payload(payload, out_dir)

    pbar.close()
    print(f"Done. Saved latents to: {out_dir}")


if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/vq-vae.yaml")
    parser.add_argument("--tokenizer-name", type=str, default="vq-vae-512")
    parser.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-path", type=str, default="data/cifar10")
    parser.add_argument("--latents-path", type=str, default="data/latents_cifar_10")

    parser.add_argument("--is_train", type=bool, default=True)

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, choices=[32, 128, 256], default=32)
    parser.add_argument("--encode-chunk-size", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=0)

    args = parser.parse_args()
    main(args)