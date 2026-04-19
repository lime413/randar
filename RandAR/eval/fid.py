import inspect
import math

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import Resize
from tqdm import tqdm

from RandAR.utils.latents import (
    extract_latent_tokens,
    make_token_order,
    resolve_order_mode,
    resolve_shuffle_ratio,
)


def is_perfect_square(n: int) -> bool:
    r = int(math.isqrt(n))
    return r * r == n


def to_uint8_0_255(images: torch.Tensor) -> torch.Tensor:
    if images.dtype == torch.uint8:
        return images
    return (images.clamp(0, 1) * 255.0).round().to(torch.uint8)


def _iterate_raw_images(
    raw_image_loader,
    device: torch.device,
    resize: Resize,
    num_samples: int,
):
    seen = 0
    for batch in tqdm(raw_image_loader, desc="Collecting raw images", leave=False):
        if seen >= num_samples:
            break

        images = batch[0] if isinstance(batch, (list, tuple)) else batch
        images = images.to(device, non_blocking=True)

        remaining = num_samples - seen
        if images.shape[0] > remaining:
            images = images[:remaining]

        yield to_uint8_0_255(resize(images))
        seen += int(images.shape[0])


def _compute_fid_from_image_streams(
    real_stream,
    fake_stream,
    device: torch.device,
) -> float:
    fid = FrechetInceptionDistance(feature=2048).to(device)

    for real_images in real_stream:
        fid.update(real_images, real=True)

    for fake_images in fake_stream:
        fid.update(fake_images, real=False)

    return float(fid.compute().item())


@torch.no_grad()
def eval_rfid(
    tokenizer,
    latent_loader,
    raw_image_loader,
    device: torch.device,
    image_size: int,
    num_samples: int,
) -> float:
    tokenizer.eval()
    resize = Resize((299, 299))

    def fake_stream():
        seen = 0
        for batch in tqdm(latent_loader, desc="Computing rFID", leave=False):
            if seen >= num_samples:
                break

            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, non_blocking=True)
            tokens = extract_latent_tokens(x)

            remaining = num_samples - seen
            if tokens.shape[0] > remaining:
                tokens = tokens[:remaining]

            recon_images = tokenizer.decode_codes_to_img(tokens, image_size)
            yield to_uint8_0_255(resize(recon_images))
            seen += int(tokens.shape[0])

    return _compute_fid_from_image_streams(
        real_stream=_iterate_raw_images(raw_image_loader, device, resize, num_samples),
        fake_stream=fake_stream(),
        device=device,
    )


@torch.no_grad()
def eval_generation_fid(
    model,
    tokenizer,
    latent_loader,
    raw_image_loader,
    device: torch.device,
    image_size: int,
    num_samples: int,
    order_mode_for_gen: str,
    cfg_scale: float,
    temperature: float,
    top_k: int,
    top_p: float,
    config_max_shuffle_ratio: float | None = None,
    explicit_shuffle_ratio: float | None = None,
) -> float:
    model.eval()
    tokenizer.eval()

    if not hasattr(model, "block_size"):
        raise ValueError("Model has no attribute 'block_size'.")
    T = int(model.block_size)
    if not is_perfect_square(T):
        raise ValueError(f"model.block_size={T} is not a perfect square.")

    resize = Resize((299, 299))
    gen_sig = inspect.signature(model.generate)
    gen_accepts_order = "token_order" in gen_sig.parameters
    gen_accepts_shuffle_ratio = "shuffle_ratio" in gen_sig.parameters

    resolved_order_mode = resolve_order_mode(model, order_mode_for_gen)
    resolved_shuffle_ratio = resolve_shuffle_ratio(
        resolved_order_mode,
        config_max_shuffle_ratio=config_max_shuffle_ratio,
        explicit_shuffle_ratio=explicit_shuffle_ratio,
    )

    def fake_stream():
        seen = 0
        for batch in tqdm(latent_loader, desc=f"Computing FID (cfg={cfg_scale:.3f})", leave=False):
            if seen >= num_samples:
                break

            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            tokens = extract_latent_tokens(x)
            cond = y.reshape(-1).long()

            remaining = num_samples - seen
            if tokens.shape[0] > remaining:
                tokens = tokens[:remaining]
                cond = cond[:remaining]

            token_order = None
            if gen_accepts_order:
                token_order = make_token_order(
                    cond.shape[0],
                    T,
                    device,
                    resolved_order_mode,
                    shuffle_ratio=resolved_shuffle_ratio,
                )

            gen_kwargs = {
                "cond": cond,
                "cfg_scales": (1.0, cfg_scale),
                "num_inference_steps": -1,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            }
            if gen_accepts_order:
                gen_kwargs["token_order"] = token_order
            if gen_accepts_shuffle_ratio:
                gen_kwargs["shuffle_ratio"] = resolved_shuffle_ratio

            gen_tokens = model.generate(**gen_kwargs)

            if hasattr(model, "remove_caches"):
                model.remove_caches()

            if gen_tokens.dim() != 2 or gen_tokens.shape != (cond.shape[0], T):
                raise ValueError(
                    f"Generated tokens have shape {tuple(gen_tokens.shape)} "
                    f"but expected {(cond.shape[0], T)}."
                )

            fake_images = tokenizer.decode_codes_to_img(gen_tokens, image_size)
            yield to_uint8_0_255(resize(fake_images))
            seen += int(cond.shape[0])

    return _compute_fid_from_image_streams(
        real_stream=_iterate_raw_images(raw_image_loader, device, resize, num_samples),
        fake_stream=fake_stream(),
        device=device,
    )


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
    raw_image_loader=None,
    config_max_shuffle_ratio: float | None = None,
    explicit_shuffle_ratio: float | None = None,
) -> float:
    if raw_image_loader is None:
        raise ValueError(
            "eval_fid now expects raw_image_loader so FID is measured against raw images. "
            "Use eval_rfid for reconstruction FID."
        )

    return eval_generation_fid(
        model=model,
        tokenizer=tokenizer,
        latent_loader=test_loader,
        raw_image_loader=raw_image_loader,
        device=device,
        image_size=image_size,
        num_samples=num_samples,
        order_mode_for_gen=order_mode_for_gen,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        config_max_shuffle_ratio=config_max_shuffle_ratio,
        explicit_shuffle_ratio=explicit_shuffle_ratio,
    )
