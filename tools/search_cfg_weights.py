import os
import json
import argparse
from typing import List

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import sys
sys.path.append("./")

from RandAR.utils.instantiation import instantiate_from_config
from RandAR.eval.fid import eval_fid

from RandAR.dataset.builder import build_dataset
from RandAR.utils.instantiation import load_state_dict

def save_results(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main(args):
    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = OmegaConf.load(args.config)

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

    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(args.vq_ckpt, map_location=device))
    tokenizer.eval()

    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    cfg_scales_search = [float(cfg_scale) for cfg_scale in args.cfg_scales_search.split(",")]
    cfg_scales = np.arange(cfg_scales_search[0], cfg_scales_search[1] + 1e-4, float(args.cfg_scales_interval))
    print(f"CFG scales to evaluate: {cfg_scales}")

    results = {
        "config": args.config,
        "ar_ckpt": args.ar_ckpt,
        "vq_ckpt": args.vq_ckpt,
        "order_mode_for_gen": args.order_mode_for_gen,
        "num_fid_samples": args.num_fid_samples,
        "image_size": args.image_size,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "search": [],
        "best": None,
    }

    best_cfg = None
    best_fid = float("inf")

    for cfg_scale in cfg_scales:
        print("=" * 80)
        print(f"Evaluating cfg_scale = {cfg_scale}")

        fid_value = eval_fid(
            model=model,
            tokenizer=tokenizer,
            test_loader=test_loader,
            device=device,
            image_size=args.image_size,
            num_samples=args.num_fid_samples,
            order_mode_for_gen=args.order_mode_for_gen,
            cfg_scale=cfg_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        fid_value = float(fid_value)

        entry = {
            "cfg_scale": cfg_scale,
            "fid": fid_value,
        }
        results["search"].append(entry)

        print(f"cfg_scale={cfg_scale:.4f} -> FID={fid_value:.6f}")

        if fid_value < best_fid:
            best_fid = fid_value
            best_cfg = cfg_scale

        results["best"] = {
            "cfg_scale": best_cfg,
            "fid": best_fid,
        }

        if args.results_path:
            save_results(results, args.results_path)

    print("=" * 80)
    print("Search finished.")
    print(f"Best cfg_scale: {best_cfg}")
    print(f"Best FID: {best_fid:.6f}")

    if args.results_path:
        save_results(results, args.results_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # required
    parser.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    parser.add_argument("--ar_ckpt", type=str, default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00024000/train_state.pt")
    parser.add_argument("--dataset", type=str, default="latent")
    parser.add_argument("--data-path", type=str, default="data/latents_cifar_10_test/cifar10-vq-vae-512-32_codes")

    # optional tokenizer checkpoint
    parser.add_argument("--vq_ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    # generation / eval setup
    parser.add_argument("--order_mode_for_gen", type=str, default="raster", choices=["raster", "random", "config"])
    parser.add_argument("--num_fid_samples", type=int, default=2000)
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=1.0)

    # search setup
    parser.add_argument("--cfg-scales-search", type=str, default="1.0, 4.0")
    parser.add_argument("--cfg-scales-interval", type=float, default=0.5)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)

    # misc
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--results_path", type=str, default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/search_cfg.json")

    args = parser.parse_args()
    main(args)