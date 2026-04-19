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
from RandAR.eval.fid import eval_generation_fid, eval_rfid, is_perfect_square
from RandAR.eval.nll_ppl_acc import eval_nll_ppl_acc
from RandAR.utils.instantiation import instantiate_from_config, load_state_dict
from RandAR.utils.latents import resolve_order_mode, resolve_shuffle_ratio


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def build_loader(
    *,
    dataset_name: str,
    data_path: str,
    batch_size: int,
    num_workers: int,
    is_train: bool,
    transform,
    cifar10c_target_total_size: int = 10000,
    cifar10c_seed: int = 42,
):
    dataset_args = argparse.Namespace(
        dataset=dataset_name,
        data_path=data_path,
        cifar10c_target_total_size=cifar10c_target_total_size,
        cifar10c_seed=cifar10c_seed,
    )
    dataset = build_dataset(
        is_train=is_train,
        args=dataset_args,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(num_workers > 0),
    )
    return dataset, loader


def load_models(config, args, device: torch.device):
    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict(model, args.ar_ckpt)
    model.eval()

    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    load_state_dict(tokenizer, args.vq_ckpt)
    tokenizer.eval()

    for p in tokenizer.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "block_size"):
        raise ValueError("AR model is missing attribute 'block_size'.")
    if not is_perfect_square(int(model.block_size)):
        raise ValueError(f"AR model block_size={int(model.block_size)} is not a perfect square.")

    return model, tokenizer


def summarize_seed_values(values: list[float]) -> dict:
    arr = np.array(values, dtype=np.float64)
    ddof = 1 if arr.size > 1 else 0
    return {
        "values": [float(v) for v in arr.tolist()],
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=ddof)),
        "num_seeds": int(arr.size),
    }


def evaluate_split(
    *,
    split_name: str,
    latent_dataset_name: str,
    latent_data_path: str,
    raw_dataset_name: str,
    raw_data_path: str,
    model,
    tokenizer,
    device: torch.device,
    args,
    config_max_shuffle_ratio: float | None,
) -> dict:
    _, latent_loader = build_loader(
        dataset_name=latent_dataset_name,
        data_path=latent_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
        transform=transforms.ToTensor(),
        cifar10c_target_total_size=args.cifar10c_target_total_size,
        cifar10c_seed=args.cifar10c_seed,
    )
    _, raw_loader = build_loader(
        dataset_name=raw_dataset_name,
        data_path=raw_data_path,
        batch_size=args.fid_batch_size,
        num_workers=args.num_workers,
        is_train=False,
        transform=transforms.ToTensor(),
        cifar10c_target_total_size=args.cifar10c_target_total_size,
        cifar10c_seed=args.cifar10c_seed,
    )

    teacher_forced_order = resolve_order_mode(model, args.order)
    teacher_forced_shuffle_ratio = resolve_shuffle_ratio(
        teacher_forced_order,
        config_max_shuffle_ratio=config_max_shuffle_ratio,
        explicit_shuffle_ratio=args.eval_shuffle_ratio,
    )
    generation_order = resolve_order_mode(model, args.fid_order)
    generation_shuffle_ratio = resolve_shuffle_ratio(
        generation_order,
        config_max_shuffle_ratio=config_max_shuffle_ratio,
        explicit_shuffle_ratio=args.eval_shuffle_ratio,
    )

    nll, ppl, acc = eval_nll_ppl_acc(
        model=model,
        test_loader=latent_loader,
        device=device,
        order_mode=teacher_forced_order,
        shuffle_ratio=teacher_forced_shuffle_ratio,
    )

    rfid_value = eval_rfid(
        tokenizer=tokenizer,
        latent_loader=latent_loader,
        raw_image_loader=raw_loader,
        device=device,
        image_size=args.image_size,
        num_samples=args.fid_num_samples,
    )

    cfg_off_values = []
    cfg_on_values = []
    per_seed = []
    for seed in args.seeds:
        set_seed(seed)

        fid_cfg_off = eval_generation_fid(
            model=model,
            tokenizer=tokenizer,
            latent_loader=latent_loader,
            raw_image_loader=raw_loader,
            device=device,
            image_size=args.image_size,
            num_samples=args.fid_num_samples,
            order_mode_for_gen=generation_order,
            cfg_scale=1.0,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            config_max_shuffle_ratio=config_max_shuffle_ratio,
            explicit_shuffle_ratio=generation_shuffle_ratio,
        )
        fid_cfg_on = eval_generation_fid(
            model=model,
            tokenizer=tokenizer,
            latent_loader=latent_loader,
            raw_image_loader=raw_loader,
            device=device,
            image_size=args.image_size,
            num_samples=args.fid_num_samples,
            order_mode_for_gen=generation_order,
            cfg_scale=args.cfg_on_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            config_max_shuffle_ratio=config_max_shuffle_ratio,
            explicit_shuffle_ratio=generation_shuffle_ratio,
        )

        cfg_off_values.append(fid_cfg_off)
        cfg_on_values.append(fid_cfg_on)
        per_seed.append(
            {
                "seed": seed,
                "FID_generated_cfg_off": float(fid_cfg_off),
                "FID_generated_cfg_on": float(fid_cfg_on),
            }
        )

    fid_cfg_off_summary = summarize_seed_values(cfg_off_values)
    fid_cfg_on_summary = summarize_seed_values(cfg_on_values)

    return {
        "split": split_name,
        "paths": {
            "latent_data_path": latent_data_path,
            "raw_data_path": raw_data_path,
        },
        "setup": {
            "teacher_forced_order": teacher_forced_order,
            "teacher_forced_shuffle_ratio": teacher_forced_shuffle_ratio,
            "generation_order": generation_order,
            "generation_shuffle_ratio": generation_shuffle_ratio,
            "fid_num_samples": args.fid_num_samples,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "cfg_on_scale": args.cfg_on_scale,
        },
        "teacher_forced": {
            "nll_per_token": float(nll),
            "perplexity": float(ppl),
            "token_accuracy": float(acc),
        },
        "rFID": float(rfid_value),
        "FID_raw": dict(fid_cfg_off_summary),
        "FID_generated_cfg_off": fid_cfg_off_summary,
        "FID_generated_cfg_on": fid_cfg_on_summary,
        "per_seed": per_seed,
    }


def build_robustness_latent_path(root: str, severity: int, subdir: str) -> str:
    return str(Path(root) / f"severity_{severity}" / subdir)


def build_robustness_raw_path(root: str, severity: int) -> str:
    return str(Path(root) / f"severity_{severity}")


def evaluate_all(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation script.")

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    config = OmegaConf.load(args.config)
    model, tokenizer = load_models(config, args, device)
    config_max_shuffle_ratio = float(getattr(config, "max_shuffle_ratio", 0.0))

    results = {
        "setup": {
            "config": args.config,
            "ar_ckpt": args.ar_ckpt,
            "vq_ckpt": args.vq_ckpt,
            "seeds": args.seeds,
            "config_max_shuffle_ratio": config_max_shuffle_ratio,
        },
        "clean": evaluate_split(
            split_name="clean",
            latent_dataset_name=args.dataset,
            latent_data_path=args.data_path,
            raw_dataset_name=args.raw_dataset,
            raw_data_path=args.raw_data_path,
            model=model,
            tokenizer=tokenizer,
            device=device,
            args=args,
            config_max_shuffle_ratio=config_max_shuffle_ratio,
        ),
    }

    if args.run_robustness:
        robustness = {}
        for severity in args.robustness_severities:
            robustness[f"severity_{severity}"] = evaluate_split(
                split_name=f"severity_{severity}",
                latent_dataset_name="latent",
                latent_data_path=build_robustness_latent_path(
                    args.robustness_latent_root,
                    severity,
                    args.robustness_latent_subdir,
                ),
                raw_dataset_name="cifar10c",
                raw_data_path=build_robustness_raw_path(args.robustness_raw_root, severity),
                model=model,
                tokenizer=tokenizer,
                device=device,
                args=args,
                config_max_shuffle_ratio=config_max_shuffle_ratio,
            )
        results["robustness"] = robustness

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
    p.add_argument(
        "--ar_ckpt",
        type=str,
        default="results/raster_order/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00027000/train_state.pt",
    )
    p.add_argument("--vq_ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument(
        "--data-path",
        type=str,
        default="data/cifar10-all/latents/cifar10/test/cifar10-vq-vae-512-32_codes",
    )
    p.add_argument("--raw-dataset", type=str, default="cifar10", choices=["cifar10", "cifar10c"])
    p.add_argument("--raw-data-path", type=str, default="data/cifar10-all/cifar10")
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--order",
        type=str,
        choices=["config", "raster", "random", "adaptive"],
        default="config",
    )

    p.add_argument("--fid-num-samples", type=int, default=10000)
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument(
        "--fid-order",
        type=str,
        choices=["config", "raster", "random", "adaptive"],
        default="config",
    )
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--cfg-on-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument(
        "--eval-shuffle-ratio",
        type=float,
        default=None,
        help="Optional override for adaptive evaluation. Defaults to config.max_shuffle_ratio.",
    )

    p.add_argument("--seeds", type=parse_int_list, default=parse_int_list("0,1,2"))

    p.add_argument("--run-robustness", action="store_true")
    p.add_argument(
        "--robustness-severities",
        type=parse_int_list,
        default=parse_int_list("1,2,3,4,5"),
    )
    p.add_argument("--robustness-latent-root", type=str, default="data/cifar10-all/latents/cifar10c")
    p.add_argument(
        "--robustness-latent-subdir",
        type=str,
        default="cifar10c-vq-vae-512-32_codes",
    )
    p.add_argument("--robustness-raw-root", type=str, default="data/cifar10-all/cifar10c-by-severity")
    p.add_argument("--cifar10c-target-total-size", type=int, default=10000)
    p.add_argument("--cifar10c-seed", type=int, default=42)

    p.add_argument("--output-json", type=str, default="results/random_order/eval_metrics.json")

    args = p.parse_args()
    evaluate_all(args)


if __name__ == "__main__":
    main()
