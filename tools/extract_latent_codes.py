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

def load_tokenizer_from_yaml(args, device):
    config = OmegaConf.load(args.config)
    vq_model = instantiate_from_config(config.model).to(device)
    vq_model.load_state_dict(torch.load(args.vq_ckpt, map_location=device))
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
    def _encode_one(batch: torch.Tensor) -> torch.Tensor:
        out = vq_model.encode_indices(batch)
        out = out.to(dtype=torch.long)
        if out.ndim == 3:
            out = out.reshape(out.shape[0], -1).reshape(-1)
        elif out.ndim == 2:
            out = out.reshape(-1)
        elif out.ndim != 1:
            out = out.reshape(-1)
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
        transform = transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[1.0]*3, inplace=True)
        ])
        return transform

    raise ValueError(f"Unsupported dataset: {args.dataset}")


# -------------------------
# Saving (rank0 only)
# -------------------------

def _save_payload(payload: Dict[str, Any], out_dir: Path) -> None:
    codes = payload["codes"]
    labels = payload["labels"]
    indices = payload["indices"]

    assert codes.ndim == 3, f"Expected (B,num_aug,num_tokens), got {codes.shape}"
    assert labels.ndim == 1 and indices.ndim == 1
    assert codes.shape[0] == labels.shape[0] == indices.shape[0]

    for i in range(codes.shape[0]):
        cls_id = int(labels[i])
        sample_id = int(indices[i])
        dst = out_dir / str(cls_id) / f"{sample_id}.npy"

        if dst.exists():
            continue

        _atomic_save_npy(dst, codes[i])


# -------------------------
# Main
# -------------------------

def main(args: argparse.Namespace) -> None:
    assert torch.cuda.is_available(), "Requires at least one GPU."

    # Set GPU directly
    device = torch.device("cuda:0")

    # Set seed for reproducibility
    seed = args.global_seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"device={device}, seed={seed}")

    # ---- tokenizer (VQ-VAE) ----
    vq_model = load_tokenizer_from_yaml(args, device)
    print(f"Loaded VQ-VAE tokenizer from {args.vq_ckpt}")

    # ---- dataset + transforms ----
    transform = build_image_transform(args)
    is_train = not args.debug
    dataset = build_dataset(is_train=is_train, args=args, transform=transform)

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

    printed_tokens = False

    for x, y, index in pbar:
        bs = x.shape[0]

        # flatten
        x_all = torch.stack([x, torch.flip(x, dims=[-1])], dim=1).flatten(0, 1).to(device)

        # tokenize
        indices_1d = encode_images(vq_model, x_all, chunk_size=args.encode_chunk_size)

        num_tokens = indices_1d.numel() // bs

        if not printed_tokens:
            printed_tokens = True
            grid = int(round(num_tokens ** 0.5))
            print(f"tokens_per_image={num_tokens}, grid={grid}x{grid}")

        codes = indices_1d.view(bs, num_tokens).view(bs, 1, num_tokens)

        payload = {
            "codes": codes.detach().cpu().numpy().astype(np.int64, copy=False),
            "labels": y.detach().cpu().numpy().astype(np.int64, copy=False),
            "indices": index.detach().cpu().numpy().astype(np.int64, copy=False),
        }

        _save_payload(payload, out_dir)

        pbar.update(1)

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/vq-vae.yaml")

    parser.add_argument("--tokenizer-name", type=str, default="vq-vae-512")
    parser.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-path", type=str, default="data/cifar10")
    parser.add_argument("--latents-path", type=str, default="data/latents_cifar_10")
    parser.add_argument("--debug", action="store_true")

    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[32, 128, 256], default=32)
    parser.add_argument("--encode-chunk-size", type=int, default=None)
    parser.add_argument("--global-seed", type=int, default=0)

    args = parser.parse_args()
    main(args)