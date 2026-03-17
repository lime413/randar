import inspect
import math
import torch
from torchvision.transforms import Resize
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from RandAR.utils.latents import extract_latent_tokens, make_token_order

def is_perfect_square(n: int) -> bool:
    r = int(math.isqrt(n))
    return r * r == n

def to_uint8_0_255(x01: torch.Tensor) -> torch.Tensor:
    # x01 in [0,1], float -> uint8 [0,255]
    return (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)

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
            cfg_scales=(1.0, cfg_scale),
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