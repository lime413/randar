import argparse
import random

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
import json
from pathlib import Path
from typing import Dict, Tuple

from RandAR.dataset.builder import build_dataset
from RandAR.eval.calibration import eval_token_calibration, plot_reliability_diagram
from RandAR.utils.instantiation import instantiate_from_config, load_state_dict


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def save_calibration_outputs(
    metrics: Dict[str, float],
    output_dir: str,
    diagram_name: str = "reliability_diagram.png",
    metrics_name: str = "calibration_metrics.json",
) -> Tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / metrics_name
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    diagram_path = out_dir / diagram_name

    bin_counts = np.array(metrics["bin_counts"], dtype=np.int64)
    bin_conf_sums = np.array(metrics["bin_confidence"], dtype=np.float64) * np.maximum(bin_counts, 1)
    bin_correct_sums = np.array(metrics["bin_accuracy"], dtype=np.float64) * np.maximum(bin_counts, 1)

    plot_reliability_diagram(
        bin_counts=bin_counts,
        bin_conf_sums=bin_conf_sums,
        bin_correct_sums=bin_correct_sums,
        output_path=diagram_path
    )

    return metrics_path, diagram_path


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    p.add_argument(
        "--ar-ckpt",
        type=str,
        default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00027000/train_state.pt",
    )

    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument(
        "--data-path",
        type=str,
        default="data/latents_cifar_10_test/cifar10-vq-vae-512-32_codes",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)

    p.add_argument(
        "--order",
        type=str,
        choices=["config", "raster", "random"],
        default="raster",
        help="Teacher-forced order control. 'config' means model.position_order is used.",
    )
    p.add_argument("--num-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="results/calibration_clean")

    args = p.parse_args()

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

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

    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    metrics = eval_token_calibration(
        model=model,
        test_loader=test_loader,
        device=device,
        order_mode=args.order,
        num_bins=args.num_bins,
    )

    metrics_path, diagram_path = save_calibration_outputs(
        metrics=metrics,
        output_dir=args.output_dir,
    )

    print(f"[TEST] order={args.order}")
    print(f"[TEST] NLL/token          : {metrics['nll_per_token']:.6f}")
    print(f"[TEST] Perplexity         : {metrics['perplexity']:.6f}")
    print(f"[TEST] Token accuracy     : {metrics['token_accuracy'] * 100:.2f}%")
    print(f"[TEST] Mean confidence    : {metrics['mean_confidence']:.6f}")
    print(f"[TEST] Overconfidence gap : {metrics['overconfidence_gap']:.6f}")
    print(f"[TEST] ECE                : {metrics['ece']:.6f}")
    print(f"[TEST] Saved metrics to   : {metrics_path}")
    print(f"[TEST] Saved diagram to   : {diagram_path}")


if __name__ == "__main__":
    main()