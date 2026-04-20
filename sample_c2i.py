import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont

from RandAR.utils.instantiation import (
    instantiate_from_config,
    load_safetensors,
    load_state_dict,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate one CIFAR-10 sample per class and save a labeled 2x5 PNG grid."
    )
    parser.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    parser.add_argument("--gpt-ckpt", type=str, default="checkpoints/raster_50k.pt")
    parser.add_argument("--vq-ckpt", type=str, default="checkpoints/vq_ds16_c2i.pt")
    parser.add_argument("--output", type=str, default="results/cifar10_class_grid.png")
    parser.add_argument("--image-size", type=int, default=32, choices=[32, 128, 256, 384, 512])
    parser.add_argument("--cfg-scales", type=str, default="4.0,4.0")
    parser.add_argument("--num-inference-steps", type=int, default=-1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--cell-size", type=int, default=176)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda:0")
    return torch.device(device_arg)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_checkpoint(model: torch.nn.Module, ckpt_path: str):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(ckpt_path))
        model.load_state_dict(state_dict, strict=True)
    else:
        load_state_dict(model, str(ckpt_path))


def parse_cfg_scales(cfg_scales: str):
    return [float(value.strip()) for value in cfg_scales.split(",") if value.strip()]


def build_labeled_grid(images, labels, cell_size: int) -> Image.Image:
    columns = 2
    rows = 5
    if len(images) != columns * rows:
        raise ValueError(f"Expected exactly {columns * rows} images, got {len(images)}")

    tile_padding = 14
    title_height = 28
    outer_padding = 24
    background_color = (248, 249, 251)
    panel_color = (255, 255, 255)
    border_color = (215, 220, 228)
    text_color = (24, 28, 35)

    grid_width = columns * cell_size + (columns - 1) * tile_padding + outer_padding * 2
    grid_height = rows * (cell_size + title_height) + (rows - 1) * tile_padding + outer_padding * 2
    canvas = Image.new("RGB", (grid_width, grid_height), background_color)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (image_array, label) in enumerate(zip(images, labels)):
        row = idx // columns
        col = idx % columns
        x0 = outer_padding + col * (cell_size + tile_padding)
        y0 = outer_padding + row * (cell_size + title_height + tile_padding)
        x1 = x0 + cell_size
        y1 = y0 + title_height + cell_size

        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=panel_color, outline=border_color, width=1)
        draw.text((x0 + 10, y0 + 7), label, fill=text_color, font=font)

        tile = Image.fromarray(image_array.astype(np.uint8), mode="RGB")
        tile = tile.resize((cell_size - 20, cell_size - 20), Image.Resampling.NEAREST)
        image_x = x0 + 10
        image_y = y0 + title_height + 10
        canvas.paste(tile, (image_x, image_y))

    return canvas


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    config = OmegaConf.load(args.config)

    tokenizer = instantiate_from_config(config.tokenizer).to(device).eval()
    load_checkpoint(tokenizer, args.vq_ckpt)

    model = instantiate_from_config(config.ar_model).to(device).eval()
    load_checkpoint(model, args.gpt_ckpt)

    cond = torch.arange(len(CIFAR10_CLASS_NAMES), device=device, dtype=torch.long)
    cfg_scales = parse_cfg_scales(args.cfg_scales)

    indices = model.generate(
        cond=cond,
        token_order=None,
        cfg_scales=cfg_scales,
        num_inference_steps=args.num_inference_steps,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        shuffle_ratio=0.0 if getattr(model, "position_order", None) == "adaptive" else None,
    )
    images = tokenizer.decode_codes_to_img(indices, args.image_size)
    model.remove_caches()

    if torch.is_tensor(images):
        images = images.detach().cpu().numpy()
    images = np.asarray(images)
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)

    grid = build_labeled_grid(images=images, labels=CIFAR10_CLASS_NAMES, cell_size=args.cell_size)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)

    print(f"Saved labeled class grid to {output_path.resolve()}")


if __name__ == "__main__":
    main()
