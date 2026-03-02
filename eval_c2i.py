import argparse
import inspect
import math
import random
from typing import Optional, Tuple

from torchvision import transforms
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm
from omegaconf import OmegaConf
from torchmetrics.image.fid import FrechetInceptionDistance

from RandAR.util import instantiate_from_config
from RandAR.dataset.builder import build_dataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_uint8_0_255(x01: torch.Tensor) -> torch.Tensor:
    # x01 in [0,1], float -> uint8 [0,255]
    return (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)


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


def is_perfect_square(n: int) -> bool:
    r = int(math.isqrt(n))
    return r * r == n


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


def load_state_dict_flex(model: torch.nn.Module, ckpt_path: str):
    """
    Supports:
      - train_state.pt with key 'model'
      - raw state_dict
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        model.load_state_dict(sd["model"], strict=True)
    elif isinstance(sd, dict):
        # raw state dict
        model.load_state_dict(sd, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint format from {ckpt_path}")


@torch.no_grad()
def eval_nll_ppl_acc(
    model,
    test_loader,
    device: torch.device,
    order_mode: str,
) -> Tuple[float, float, float]:
    """
      - NLL/token computed by cross_entropy(logits, targets_pred_order), summed then normalized
      - Perplexity = exp(NLL/token)
      - Accuracy computed in prediction order aligned with the same targets (correct for raster/random)
    """
    model.eval()

    # forward signature: token_order kw is supported in your model :contentReference[oaicite:3]{index=3}
    fwd_sig = inspect.signature(model.forward)
    accepts_token_order = ("token_order" in fwd_sig.parameters)

    total_ce = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in tqdm(test_loader, desc="Evaluating NLL/PPL/Acc"):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        tokens = extract_latent_tokens(x)  # (B,T)
        cond = y.reshape(-1).long()

        B, T = tokens.shape

        # dataset tokens must match model.block_size
        if not hasattr(model, "block_size"):
            raise ValueError("Model has no attribute 'block_size'. Cannot validate token length.")
        if T != int(model.block_size):
            raise ValueError(
                f"Latent token length mismatch: dataset T={T} but model.block_size={int(model.block_size)}. "
                "This must match for correct evaluation. Fix your latent .npy generation or your config."
            )

        token_order = make_token_order(B, T, device, order_mode) if accepts_token_order else None

        logits, _, used_order = model(tokens, cond, token_order=token_order, targets=tokens)

        if logits.dim() != 3:
            raise ValueError(f"Expected logits shape (B,T,V) but got {tuple(logits.shape)}")

        B2, T2, V = logits.shape
        if B2 != B or T2 != T:
            raise ValueError(
                f"Model output shape mismatch: logits {tuple(logits.shape)} but expected (B={B},T={T},V)."
            )

        # Align targets to prediction order (the order used inside the model)
        if used_order is None:
            raise ValueError("Model did not return token_order; cannot guarantee correct alignment.")
        if used_order.shape != (B, T):
            raise ValueError(f"Returned token_order has shape {tuple(used_order.shape)} but expected {(B,T)}.")

        targets_pred_order = torch.gather(tokens, dim=1, index=used_order)

        ce = F.cross_entropy(
            logits.reshape(-1, V),
            targets_pred_order.reshape(-1),
            reduction="sum",
        )
        total_ce += float(ce.item())
        total_tokens += int(B * T)

        pred = torch.argmax(logits, dim=-1)
        total_correct += int((pred == targets_pred_order).sum().item())

    nll = total_ce / max(total_tokens, 1)
    ppl = math.exp(min(nll, 50.0))
    acc = total_correct / max(total_tokens, 1)
    return nll, ppl, acc


@torch.no_grad()
def eval_fid(
    model,
    tokenizer,
    test_loader,
    device: torch.device,
    image_size: int,
    num_samples: int,
    order_mode_for_gen: str,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
) -> float:
    """
    FIDp:
      - decode GT codes (tokens) to images using tokenizer.decode_codes_to_img
      - generate codes using model.generate(), then decode
      - feed uint8 images to torchmetrics FID
    """
    model.eval()
    tokenizer.eval()

    if not hasattr(model, "block_size"):
        raise ValueError("Model has no attribute 'block_size'.")
    T = int(model.block_size)
    if not is_perfect_square(T):
        raise ValueError(
            f"model.block_size={T} is not a perfect square."
        )

    fid = FrechetInceptionDistance(feature=2048).to(device)
    resize = Resize((299, 299))

    # generate() signature includes token_order in your model :contentReference[oaicite:6]{index=6}
    gen_sig = inspect.signature(model.generate)
    gen_accepts_order = ("token_order" in gen_sig.parameters)

    seen = 0
    for batch in tqdm(test_loader, desc="Evaluating FID"):
        if seen >= num_samples:
            break

        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        tokens = extract_latent_tokens(x)  # (B,T)
        cond = y.reshape(-1).long()
        B, T_data = tokens.shape

        if T_data != T:
            raise ValueError(
                f"Latent token length mismatch during FID: dataset T={T_data}, model.block_size={T}."
            )

        # Real images from GT codes
        real_imgs = tokenizer.decode_codes_to_img(tokens, image_size)  # float [0,1]
        real_imgs = resize(real_imgs)
        real_u8 = to_uint8_0_255(real_imgs)

        # Generation token_order: either model default ("config") or force raster/random
        token_order = make_token_order(B, T, device, order_mode_for_gen) if gen_accepts_order else None

        gen_tokens = model.generate(
            cond=cond,
            token_order=token_order,                       # can be None; model will create its own
            cfg_scales=(cfg_scale, cfg_scale),
            num_inference_steps=-1,                        # fully autoregressive; avoids ambiguity
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if hasattr(model, "remove_caches"):
            model.remove_caches()

        if gen_tokens.dim() != 2 or gen_tokens.shape != (B, T):
            raise ValueError(
                f"Generated tokens have shape {tuple(gen_tokens.shape)} but expected {(B,T)}."
            )

        fake_imgs = tokenizer.decode_codes_to_img(gen_tokens, image_size)  # float [0,1]
        fake_imgs = resize(fake_imgs)
        fake_u8 = to_uint8_0_255(fake_imgs)

        fid.update(real_u8, real=True)
        fid.update(fake_u8, real=False)

        seen += int(B)

    return float(fid.compute().item())


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    p.add_argument("--ar-ckpt", type=str, default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00024000/train_state.pt")
    p.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    # RandAR dataset args
    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument("--data-path", type=str, default="data/latents_cifar_10_test/cifar10-vq-vae-512-32_codes")
    p.add_argument("--num-workers", type=int, default=0)

    # eval options
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--order", type=str, choices=["config", "raster", "random"], default="raster",
                   help="Teacher-forced order control. 'config' means model.position_order is used.")
    p.add_argument("--seed", type=int, default=0)

    # FID
    p.add_argument("--fid-num-samples", type=int, default=5000)
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument("--fid-order", type=str, choices=["config", "raster", "random"], default="config",
                   help="Generation order control.")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    # Load config
    config = OmegaConf.load(args.config)

    # Build test dataset (latent)
    # IMPORTANT: is_train=False gives test split in RandAR builder.
    test_dataset = build_dataset(is_train=False, args=args, transform=transforms.ToTensor())

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # Instantiate + load AR model
    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict_flex(model, args.ar_ckpt)
    model.eval()

    # Instantiate + load tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    tok_sd = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tok_sd, strict=True)
    tokenizer.eval()
    for p_ in tokenizer.parameters():
        p_.requires_grad_(False)

    # Sanity: model.block_size must be square for both model and tokenizer decoding
    if not hasattr(model, "block_size"):
        raise ValueError("AR model missing 'block_size'.")
    if not is_perfect_square(int(model.block_size)):
        raise ValueError(f"AR model block_size={int(model.block_size)} is not a perfect square.")

    # Evaluate NLL/PPL/Acc
    nll, ppl, acc = eval_nll_ppl_acc(
        model=model,
        test_loader=test_loader,
        device=device,
        order_mode=args.order,
    )
    print(f"[TEST] order={args.order}")
    print(f"[TEST] NLL/token      : {nll:.6f}")
    print(f"[TEST] Perplexity     : {ppl:.4f}")
    print(f"[TEST] Token accuracy : {acc*100:.2f}%")

    # Evaluate FID (strict)
    fid_loader = DataLoader(
        test_dataset,
        batch_size=args.fid_batch_size,
        shuffle=True,  # random subset OK
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    fid_value = eval_fid(
        model=model,
        tokenizer=tokenizer,
        test_loader=fid_loader,
        device=device,
        image_size=args.image_size,
        num_samples=args.fid_num_samples,
        order_mode_for_gen=args.fid_order,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"[TEST] FID ({args.fid_num_samples} samples): {fid_value:.3f}")


if __name__ == "__main__":
    main()