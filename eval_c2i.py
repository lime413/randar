import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from RandAR.dataset.builder import build_dataset
from RandAR.eval.fid import eval_fid, is_perfect_square
from RandAR.eval.nll_ppl_acc import eval_nll_ppl_acc
from RandAR.utils.instantiation import instantiate_from_config, load_state_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_cfg_scales(text: str) -> list[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("cfg-scales must contain at least one float, for example: '1.0,4.0'")
    return values


def build_loaders(args):
    test_dataset = build_dataset(
        is_train=False,
        args=args,
        transform=transforms.ToTensor(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    fid_loader = DataLoader(
        test_dataset,
        batch_size=args.fid_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    return test_dataset, test_loader, fid_loader


def load_models(config, args, device: torch.device):
    # AR model
    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    # Tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    tok_sd = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)

    # support both plain state_dict and {"model": state_dict}
    if isinstance(tok_sd, dict) and "model" in tok_sd and isinstance(tok_sd["model"], dict):
        tok_sd = tok_sd["model"]

    tokenizer.load_state_dict(tok_sd, strict=True)
    tokenizer.eval()

    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "block_size"):
        raise ValueError("AR model is missing attribute 'block_size'.")
    if not is_perfect_square(int(model.block_size)):
        raise ValueError(f"AR model block_size={int(model.block_size)} is not a perfect square.")

    return model, tokenizer


def evaluate_all(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    config = OmegaConf.load(args.config)
    _, test_loader, fid_loader = build_loaders(args)
    model, tokenizer = load_models(config, args, device)

    results = {
        "setup": {
            "teacher_forced_order": args.order,
            "fid_generation_order": args.fid_order,
            "fid_num_samples": args.fid_num_samples,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        }
    }

    # Teacher-forced metrics: no CFG here
    nll, ppl, acc = eval_nll_ppl_acc(
        model=model,
        test_loader=test_loader,
        device=device,
        order_mode=args.order,
    )
    results["teacher_forced"] = {
        "nll_per_token": float(nll),
        "perplexity": float(ppl),
        "token_accuracy": float(acc),
    }

    print(f"[TEST] teacher-forced order={args.order}")
    print(f"[TEST] NLL/token      : {nll:.6f}")
    print(f"[TEST] Perplexity     : {ppl:.4f}")
    print(f"[TEST] Token accuracy : {acc * 100:.2f}%")

    # FID for each requested CFG
    results["fid"] = {}
    for cfg_scale in args.cfg_scales:
        fid_value = eval_fid(
            model=model,
            tokenizer=tokenizer,
            test_loader=fid_loader,
            device=device,
            image_size=args.image_size,
            num_samples=args.fid_num_samples,
            order_mode_for_gen=args.fid_order,
            cfg_scale=cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        key = f"cfg_{cfg_scale:.3f}"
        results["fid"][key] = float(fid_value)

        mode_name = "no guidance" if abs(cfg_scale - 1.0) < 1e-12 else "with CFG"
        print(f"[TEST] FID ({args.fid_num_samples} samples, cfg={cfg_scale:.3f}, {mode_name}): {fid_value:.3f}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[TEST] Saved results to: {out_path}")

    return results


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    p.add_argument("--ar_ckpt", type=str, default="results/raster_order/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00027000/train_state.pt")
    p.add_argument("--vq_ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    # dataset
    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument("--data-path", type=str, default="data/latents/cifar10c/severity_1/cifar10c-vq-vae-512-32_codes")
    p.add_argument("--num-workers", type=int, default=0)

    # teacher-forced metrics
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--order",
        type=str,
        choices=["config", "raster", "random"],
        default="config",
        help="Teacher-forced order. 'config' means use model default behavior.",
    )

    # FID
    p.add_argument("--fid-num-samples", type=int, default=10000)
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument(
        "--fid-order",
        type=str,
        choices=["config", "raster", "random"],
        default="config",
        help="Generation order for FID.",
    )
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument(
        "--cfg-scales",
        type=parse_cfg_scales,
        default=parse_cfg_scales("1.0"),
        help="Comma-separated CFG scales to evaluate, for example: '1.0,4.0'",
    )
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    # misc
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=str, default="results/random_order/basic_eval_metrics.json")

    args = p.parse_args()
    evaluate_all(args)


if __name__ == "__main__":
    main()