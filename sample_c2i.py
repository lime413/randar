import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torchvision import datasets

from RandAR.utils.instantiation import instantiate_from_config, load_safetensors, load_state_dict

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
    parser.add_argument("--config", type=str, default="configs/randar_cifar10_random_order.yaml")
    parser.add_argument("--gpt-ckpt", type=str, default="checkpoints/random_50k.pt")
    parser.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")
    parser.add_argument("--output", type=str, default="results/cifar10_class_grid_random.png")
    parser.add_argument("--image-size", type=int, default=32, choices=[32, 128, 256, 384, 512])
    parser.add_argument("--cfg-scales", type=str, default="5.75")
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


def load_gpt_checkpoint(model: torch.nn.Module, ckpt_path: str):
    ckpt_path = Path(ckpt_path)
    if ckpt_path.suffix == ".safetensors":
        state_dict = load_safetensors(str(ckpt_path))
        model.load_state_dict(state_dict, strict=True)
    else:
        load_state_dict(model, str(ckpt_path))


def parse_cfg_scales(cfg_scales: str):
    return [float(value.strip()) for value in cfg_scales.split(",") if value.strip()]


def find_default_cifar10_root() -> str:
    candidates = [
        Path("data/cifar10-all/cifar10"),
        Path("data/cifar10"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError("Could not find a local CIFAR-10 root in data/.")


def load_reference_images(cifar10_root: str):
    dataset = datasets.CIFAR10(root=cifar10_root, train=False, download=False)
    refs = {}
    for image, label in dataset:
        if label not in refs:
            refs[label] = np.asarray(image, dtype=np.uint8)
        if len(refs) == len(CIFAR10_CLASS_NAMES):
            break
    if len(refs) != len(CIFAR10_CLASS_NAMES):
        raise RuntimeError("Could not collect one reference CIFAR-10 image for each class.")
    return [refs[class_idx] for class_idx in range(len(CIFAR10_CLASS_NAMES))]


def build_labeled_grid(generated_images, reference_images, labels, cell_size: int) -> Image.Image:
    columns = 2
    rows = 5
    if len(generated_images) != columns * rows:
        raise ValueError(f"Expected exactly {columns * rows} generated images, got {len(generated_images)}")
    if len(reference_images) != columns * rows:
        raise ValueError(f"Expected exactly {columns * rows} reference images, got {len(reference_images)}")

    tile_padding = 14
    title_height = 44
    outer_padding = 24
    background_color = (248, 249, 251)
    panel_color = (255, 255, 255)
    border_color = (215, 220, 228)
    text_color = (24, 28, 35)
    caption_color = (90, 99, 110)
    inner_gap = 10
    inner_tile_size = (cell_size - 30) // 2

    grid_width = columns * cell_size + (columns - 1) * tile_padding + outer_padding * 2
    grid_height = rows * (cell_size + title_height) + (rows - 1) * tile_padding + outer_padding * 2
    canvas = Image.new("RGB", (grid_width, grid_height), background_color)
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for idx, (generated_array, reference_array, label) in enumerate(zip(generated_images, reference_images, labels)):
        row = idx // columns
        col = idx % columns
        x0 = outer_padding + col * (cell_size + tile_padding)
        y0 = outer_padding + row * (cell_size + title_height + tile_padding)
        x1 = x0 + cell_size
        y1 = y0 + title_height + cell_size

        draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=panel_color, outline=border_color, width=1)
        draw.text((x0 + 10, y0 + 7), label, fill=text_color, font=font)
        draw.text((x0 + 10, y0 + 22), "Real            Generated", fill=caption_color, font=font)

        reference_tile = Image.fromarray(reference_array.astype(np.uint8), mode="RGB")
        reference_tile = reference_tile.resize((inner_tile_size, inner_tile_size), Image.Resampling.NEAREST)
        generated_tile = Image.fromarray(generated_array.astype(np.uint8), mode="RGB")
        generated_tile = generated_tile.resize((inner_tile_size, inner_tile_size), Image.Resampling.NEAREST)

        image_y = y0 + title_height + 10
        reference_x = x0 + 10
        generated_x = reference_x + inner_tile_size + inner_gap
        canvas.paste(reference_tile, (reference_x, image_y))
        canvas.paste(generated_tile, (generated_x, image_y))

    return canvas


@torch.no_grad()
def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    config = OmegaConf.load(args.config)
    cifar10_root = find_default_cifar10_root()

    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    load_state_dict(tokenizer, args.vq_ckpt)
    tokenizer.eval()

    model = instantiate_from_config(config.ar_model).to(device).eval()
    load_gpt_checkpoint(model, args.gpt_ckpt)

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
        images = (images.detach().cpu().clamp(0.0, 1.0) * 255).to(torch.uint8)
        images = images.permute(0, 2, 3, 1).contiguous().numpy()
    images = np.asarray(images)
    if images.dtype != np.uint8:
        images = np.clip(images, 0, 255).astype(np.uint8)

    reference_images = load_reference_images(cifar10_root)
    grid = build_labeled_grid(
        generated_images=images,
        reference_images=reference_images,
        labels=CIFAR10_CLASS_NAMES,
        cell_size=args.cell_size,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)

    print(f"Saved labeled class grid to {output_path.resolve()}")


if __name__ == "__main__":
    main()
