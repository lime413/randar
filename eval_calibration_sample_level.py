import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from robustbench import load_model as load_robustbench_model

from RandAR.dataset.builder import build_dataset
from RandAR.eval.fid import is_perfect_square
from RandAR.eval.calibration import evaluate_sample_calibration
from RandAR.utils.instantiation import instantiate_from_config, load_state_dict


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_cfg_scales(text: str) -> Tuple[float, float]:
    parts = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) == 1:
        return (parts[0], parts[0])
    if len(parts) == 2:
        return (parts[0], parts[1])
    raise ValueError("cfg-scales must be one float or two comma-separated floats, e.g. '1.0' or '1.0,4.0'")


def load_models(config, args, device: torch.device):
    # AR model
    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    # Tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    tok_sd = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    if isinstance(tok_sd, dict) and "model" in tok_sd and isinstance(tok_sd["model"], dict):
        tok_sd = tok_sd["model"]
    tokenizer.load_state_dict(tok_sd, strict=True)
    tokenizer.eval()

    for p in tokenizer.parameters():
        p.requires_grad_(False)

    # RobustBench classifier
    clf = load_robustbench_model(
        model_name=args.classifier_model,
        dataset="cifar10",
        threat_model="Linf",
    ).to(device)
    clf.eval()

    for p in clf.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "block_size"):
        raise ValueError("AR model is missing attribute 'block_size'.")
    if not is_perfect_square(int(model.block_size)):
        raise ValueError(f"AR model block_size={int(model.block_size)} is not a perfect square.")

    return model, tokenizer, clf


def save_plots(output_dir: Path, confidences: np.ndarray, qualities: np.ndarray, bins: dict, rc: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

    # confidence-binned quality
    plt.figure(figsize=(6, 4.5))
    x = np.array(bins["bin_centers"], dtype=np.float64)
    y = np.array(bins["bin_mean_quality"], dtype=np.float64)
    valid = ~np.isnan(y)
    plt.plot(x[valid], y[valid], marker="o")
    plt.xlabel("Mean confidence in bin")
    plt.ylabel("Mean quality in bin")
    plt.title("Confidence-binned Mean Quality")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_binned_mean_quality.png", dpi=200)
    plt.close()

    # risk-coverage
    plt.figure(figsize=(6, 4.5))
    plt.plot(rc["coverage"], rc["risk"])
    plt.xlabel("Coverage")
    plt.ylabel("Risk = 1 - mean quality")
    plt.title("Risk-Coverage Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_coverage_curve.png", dpi=200)
    plt.close()

    # scatter
    plt.figure(figsize=(6, 4.5))
    plt.scatter(confidences, qualities, s=8, alpha=0.35)
    plt.xlabel("Sample confidence")
    plt.ylabel("Sample quality")
    plt.title("Confidence vs Quality")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_vs_quality.png", dpi=200)
    plt.close()


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    p.add_argument("--ar_ckpt", type=str, default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00027000/train_state.pt")
    p.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument("--data-path", type=str, default="data/latents/cifar10/test/cifar10-vq-vae-512-32_codes")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument("--generation-order", type=str, choices=["config", "raster", "random"], default="config")
    p.add_argument("--cfg-scales", type=parse_cfg_scales, default=(1.0, 1.0))
    p.add_argument("--num-inference-steps", type=int, default=-1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    p.add_argument("--classifier-model", type=str, default="Bai2023Improving_edm")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--num-bins", type=int, default=15)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="results/random_order/sample_calibration")

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    config = OmegaConf.load(args.config)
    dataset = build_dataset(
        is_train=False,
        args=args,
        transform=transforms.ToTensor(),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    model, tokenizer, classifier = load_models(config, args, device)

    # resolve generation order
    if args.generation_order == "config":
        args.generation_order = getattr(model, "position_order", "raster")

    results = evaluate_sample_calibration(
        model=model,
        tokenizer=tokenizer,
        classifier=classifier,
        loader=loader,
        device=device,
        args=args,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    save_plots(
        output_dir=out_dir,
        confidences=np.array(results["raw"]["confidence"], dtype=np.float64),
        qualities=np.array(results["raw"]["quality"], dtype=np.float64),
        bins=results["bins"],
        rc=results["risk_coverage"],
    )

    with (out_dir / "sample_calibration_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[TEST] samples              : {results['num_samples']}")
    print(f"[TEST] classifier           : {results['classifier_model']}")
    print(f"[TEST] mean confidence      : {results['mean_confidence']:.6f}")
    print(f"[TEST] mean quality         : {results['mean_quality']:.6f}")
    print(f"[TEST] spearman rho         : {results['spearman_rho']:.6f}")
    print(f"[TEST] spearman p-value     : {results['spearman_pvalue']:.6e}")
    print(f"[TEST] AURC                : {results['risk_coverage']['aurc']:.6f}")
    print(f"[TEST] saved to             : {out_dir}")


if __name__ == "__main__":
    main()