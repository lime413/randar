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
from RandAR.eval.calibration import eval_token_calibration, plot_reliability_diagram
from RandAR.utils.instantiation import instantiate_from_config, load_state_dict
from RandAR.utils.latents import resolve_order_mode, resolve_shuffle_ratio


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_int_list(text: str) -> list[int]:
    values = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one seed.")
    return values


def save_calibration_outputs(
    metrics: dict,
    output_dir: str,
    diagram_name: str = "reliability_diagram.png",
    metrics_name: str = "calibration_metrics.json",
) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / metrics_name
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    diagram_path = out_dir / diagram_name

    ref_metrics = metrics["reference_seed_metrics"]
    bin_counts = np.array(ref_metrics["bin_counts"], dtype=np.int64)
    bin_conf_sums = np.array(ref_metrics["bin_confidence"], dtype=np.float64) * np.maximum(bin_counts, 1)
    bin_correct_sums = np.array(ref_metrics["bin_accuracy"], dtype=np.float64) * np.maximum(bin_counts, 1)

    plot_reliability_diagram(
        bin_counts=bin_counts,
        bin_conf_sums=bin_conf_sums,
        bin_correct_sums=bin_correct_sums,
        output_path=diagram_path,
    )

    return metrics_path, diagram_path


def summarize_metrics(per_seed_metrics: list[dict]) -> dict:
    keys = [
        "nll_per_token",
        "perplexity",
        "token_accuracy",
        "mean_confidence",
        "overconfidence_gap",
        "ece",
    ]
    summary = {}
    for key in keys:
        arr = np.array([float(item[key]) for item in per_seed_metrics], dtype=np.float64)
        ddof = 1 if arr.size > 1 else 0
        summary[key] = {
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
        "--ar-ckpt",
        dest="ar_ckpt",
        type=str,
        default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00027000/train_state.pt",
    )

    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument(
        "--data-path",
        type=str,
        default="data/cifar10-all/latents/cifar10/test/cifar10-vq-vae-512-32_codes",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument(
        "--order",
        type=str,
        choices=["config", "raster", "random", "adaptive"],
        default="config",
        help="Teacher-forced order control.",
    )
    p.add_argument(
        "--eval-shuffle-ratio",
        type=float,
        default=None,
        help="Optional override for adaptive evaluation. Defaults to config.max_shuffle_ratio.",
    )
    p.add_argument("--num-bins", type=int, default=15)
    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2"))
    p.add_argument("--output-dir", type=str, default="results/random_order/token_calibration")

    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = OmegaConf.load(args.config)
    config_max_shuffle_ratio = float(getattr(config, "max_shuffle_ratio", 0.0))

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

    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    resolved_order = resolve_order_mode(model, args.order)
    resolved_shuffle_ratio = resolve_shuffle_ratio(
        resolved_order,
        config_max_shuffle_ratio=config_max_shuffle_ratio,
        explicit_shuffle_ratio=args.eval_shuffle_ratio,
    )

    per_seed_metrics = []
    for seed in args.seeds:
        set_seed(seed)
        metrics = eval_token_calibration(
            model=model,
            test_loader=test_loader,
            device=device,
            order_mode=resolved_order,
            num_bins=args.num_bins,
            shuffle_ratio=resolved_shuffle_ratio,
        )
        metrics["seed"] = seed
        per_seed_metrics.append(metrics)

    payload = {
        "setup": {
            "config": args.config,
            "ar_ckpt": args.ar_ckpt,
            "order": resolved_order,
            "shuffle_ratio": resolved_shuffle_ratio,
            "num_bins": int(args.num_bins),
            "seeds": args.seeds,
        },
        "aggregate": summarize_metrics(per_seed_metrics),
        "reference_seed": int(per_seed_metrics[0]["seed"]),
        "reference_seed_metrics": per_seed_metrics[0],
        "per_seed": per_seed_metrics,
    }

    metrics_path, diagram_path = save_calibration_outputs(
        metrics=payload,
        output_dir=args.output_dir,
    )

    aggregate = payload["aggregate"]
    print(f"[TEST] order={resolved_order}")
    print(f"[TEST] shuffle_ratio       : {resolved_shuffle_ratio}")
    print(f"[TEST] seeds               : {args.seeds}")
    print(f"[TEST] NLL/token           : {aggregate['nll_per_token']['mean']:.6f} +/- {aggregate['nll_per_token']['std']:.6f}")
    print(f"[TEST] Perplexity          : {aggregate['perplexity']['mean']:.6f} +/- {aggregate['perplexity']['std']:.6f}")
    print(f"[TEST] Token accuracy      : {aggregate['token_accuracy']['mean'] * 100:.2f}% +/- {aggregate['token_accuracy']['std'] * 100:.2f}%")
    print(f"[TEST] Mean confidence     : {aggregate['mean_confidence']['mean']:.6f} +/- {aggregate['mean_confidence']['std']:.6f}")
    print(f"[TEST] Overconfidence gap  : {aggregate['overconfidence_gap']['mean']:.6f} +/- {aggregate['overconfidence_gap']['std']:.6f}")
    print(f"[TEST] ECE                 : {aggregate['ece']['mean']:.6f} +/- {aggregate['ece']['std']:.6f}")
    print(f"[TEST] Saved metrics to    : {metrics_path}")
    print(f"[TEST] Saved diagram to    : {diagram_path}")


if __name__ == "__main__":
    main()
