import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

from robustbench import load_model as load_robustbench_model

from RandAR.dataset.builder import build_dataset
from RandAR.eval.calibration import evaluate_sample_calibration
from RandAR.eval.fid import is_perfect_square
from RandAR.utils.instantiation import instantiate_from_config, load_state_dict
from RandAR.utils.latents import resolve_order_mode, resolve_shuffle_ratio


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one seed.")
    return values


def load_models(config, args, device: torch.device):
    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    load_state_dict(tokenizer, args.vq_ckpt)
    tokenizer.eval()

    clf = load_robustbench_model(
        model_name=args.classifier_model,
        dataset="cifar10",
        threat_model="Linf",
    ).to(device)
    clf.eval()

    for module in (tokenizer, clf):
        for p in module.parameters():
            p.requires_grad_(False)

    if not hasattr(model, "block_size"):
        raise ValueError("AR model is missing attribute 'block_size'.")
    if not is_perfect_square(int(model.block_size)):
        raise ValueError(f"AR model block_size={int(model.block_size)} is not a perfect square.")

    return model, tokenizer, clf


def save_plots(output_dir: Path, confidences: np.ndarray, qualities: np.ndarray, bins: dict, rc: dict):
    output_dir.mkdir(parents=True, exist_ok=True)

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

    plt.figure(figsize=(6, 4.5))
    plt.plot(rc["coverage"], rc["risk"])
    plt.xlabel("Coverage")
    plt.ylabel("Risk = 1 - mean quality")
    plt.title("Risk-Coverage Curve")
    plt.tight_layout()
    plt.savefig(output_dir / "risk_coverage_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4.5))
    plt.scatter(confidences, qualities, s=8, alpha=0.35)
    plt.xlabel("Sample confidence")
    plt.ylabel("Sample quality")
    plt.title("Confidence vs Quality")
    plt.tight_layout()
    plt.savefig(output_dir / "confidence_vs_quality.png", dpi=200)
    plt.close()


def summarize_metrics(per_seed_results: list[dict]) -> dict:
    keys = [
        "mean_confidence",
        "mean_quality",
        "spearman_rho",
        "spearman_pvalue",
        ("risk_coverage", "aurc"),
    ]

    summary = {}
    for key in keys:
        if isinstance(key, tuple):
            values = [float(result[key[0]][key[1]]) for result in per_seed_results]
            name = "_".join(key)
        else:
            values = [float(result[key]) for result in per_seed_results]
            name = key

        arr = np.array(values, dtype=np.float64)
        ddof = 1 if arr.size > 1 else 0
        summary[name] = {
            "values": [float(v) for v in arr.tolist()],
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=ddof)),
            "num_seeds": int(arr.size),
        }
    return summary


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    p.add_argument(
        "--ar_ckpt",
        type=str,
        default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00027000/train_state.pt",
    )
    p.add_argument("--vq-ckpt", dest="vq_ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument(
        "--data-path",
        type=str,
        default="data/cifar10-all/latents/cifar10/test/cifar10-vq-vae-512-32_codes",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument("--num-samples", type=int, default=10000)
    p.add_argument(
        "--generation-order",
        type=str,
        choices=["config", "raster", "random", "adaptive"],
        default="config",
    )
    p.add_argument(
        "--eval-shuffle-ratio",
        type=float,
        default=None,
        help="Optional override for adaptive evaluation. Defaults to config.max_shuffle_ratio.",
    )
    p.add_argument("--num-inference-steps", type=int, default=-1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    p.add_argument("--classifier-model", type=str, default="Bai2023Improving_edm")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--num-bins", type=int, default=15)

    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2"))
    p.add_argument("--output-dir", type=str, default="results/random_order/sample_calibration")

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = OmegaConf.load(args.config)
    config_max_shuffle_ratio = float(getattr(config, "max_shuffle_ratio", 0.0))

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

    resolved_generation_order = resolve_order_mode(model, args.generation_order)
    resolved_shuffle_ratio = resolve_shuffle_ratio(
        resolved_generation_order,
        config_max_shuffle_ratio=config_max_shuffle_ratio,
        explicit_shuffle_ratio=args.eval_shuffle_ratio,
    )

    args.generation_order = resolved_generation_order
    args.shuffle_ratio = resolved_shuffle_ratio
    args.cfg_scales = (1.0, 1.0)

    per_seed_results = []
    for seed in args.seeds:
        set_seed(seed)
        result = evaluate_sample_calibration(
            model=model,
            tokenizer=tokenizer,
            classifier=classifier,
            loader=loader,
            device=device,
            args=args,
        )
        result["seed"] = seed
        per_seed_results.append(result)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reference_result = per_seed_results[0]
    save_plots(
        output_dir=out_dir,
        confidences=np.array(reference_result["raw"]["confidence"], dtype=np.float64),
        qualities=np.array(reference_result["raw"]["quality"], dtype=np.float64),
        bins=reference_result["bins"],
        rc=reference_result["risk_coverage"],
    )

    payload = {
        "setup": {
            "config": args.config,
            "ar_ckpt": args.ar_ckpt,
            "vq_ckpt": args.vq_ckpt,
            "generation_order": resolved_generation_order,
            "shuffle_ratio": resolved_shuffle_ratio,
            "cfg_scales": [1.0, 1.0],
            "num_inference_steps": int(args.num_inference_steps),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "seeds": args.seeds,
        },
        "aggregate": summarize_metrics(per_seed_results),
        "reference_seed": int(reference_result["seed"]),
        "per_seed": per_seed_results,
    }

    with (out_dir / "sample_calibration_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    aggregate = payload["aggregate"]
    print(f"[TEST] seeds                 : {args.seeds}")
    print(f"[TEST] generation_order      : {resolved_generation_order}")
    print(f"[TEST] shuffle_ratio         : {resolved_shuffle_ratio}")
    print(f"[TEST] CFG                   : off")
    print(f"[TEST] mean confidence       : {aggregate['mean_confidence']['mean']:.6f} +/- {aggregate['mean_confidence']['std']:.6f}")
    print(f"[TEST] mean quality          : {aggregate['mean_quality']['mean']:.6f} +/- {aggregate['mean_quality']['std']:.6f}")
    print(f"[TEST] spearman rho          : {aggregate['spearman_rho']['mean']:.6f} +/- {aggregate['spearman_rho']['std']:.6f}")
    print(f"[TEST] AURC                 : {aggregate['risk_coverage_aurc']['mean']:.6f} +/- {aggregate['risk_coverage_aurc']['std']:.6f}")
    print(f"[TEST] saved to              : {out_dir}")


if __name__ == "__main__":
    main()
