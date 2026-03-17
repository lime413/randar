import argparse
import random

from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from RandAR.util import instantiate_from_config
from RandAR.dataset.builder import build_dataset
from RandAR.eval.fid import eval_fid, is_perfect_square
from RandAR.eval.nll_ppl_acc import eval_nll_ppl_acc

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_state_dict_flex(model: torch.nn.Module, ckpt_path: str):
    """
    Supports:
      - train_state.pt with key 'model'
      - raw state_dict
    """
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        model.load_state_dict(sd["model"], strict=True)
    elif isinstance(sd, dict):
        # raw state dict
        model.load_state_dict(sd, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint format from {ckpt_path}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    p.add_argument("--ar-ckpt", type=str, default="results/2026-02-28_19-19-19_bs_512_lr_0.0004/checkpoints/iters_00024000/train_state.pt")
    p.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    # RandAR dataset args
    p.add_argument("--dataset", type=str, default="latent")
    p.add_argument("--data-path", type=str, default="data/latents_cifar_10_test/cifar10-vq-vae-512-32_codes")
    p.add_argument("--num-workers", type=int, default=0)

    # eval options
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--order", type=str, choices=["config", "raster", "random"], default="raster",
                   help="Teacher-forced order control. 'config' means model.position_order is used.")
    p.add_argument("--seed", type=int, default=0)

    # FID
    p.add_argument("--fid-num-samples", type=int, default=5000)
    p.add_argument("--fid-batch-size", type=int, default=128)
    p.add_argument("--fid-order", type=str, choices=["config", "raster", "random"], default="config",
                   help="Generation order control.")
    p.add_argument("--image-size", type=int, default=32)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)

    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    # Load config
    config = OmegaConf.load(args.config)

    # Build test dataset (latent)
    # IMPORTANT: is_train=False gives test split in RandAR builder.
    test_dataset = build_dataset(is_train=False, args=args, transform=transforms.ToTensor())

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # Instantiate + load AR model
    model = instantiate_from_config(config.ar_model).to(device)
    load_state_dict_flex(model, args.ar_ckpt)
    model.eval()

    # Instantiate + load tokenizer
    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    tok_sd = torch.load(args.vq_ckpt, map_location="cpu", weights_only=False)
    tokenizer.load_state_dict(tok_sd, strict=True)
    tokenizer.eval()
    for p_ in tokenizer.parameters():
        p_.requires_grad_(False)

    # Sanity: model.block_size must be square for both model and tokenizer decoding
    if not hasattr(model, "block_size"):
        raise ValueError("AR model missing 'block_size'.")
    if not is_perfect_square(int(model.block_size)):
        raise ValueError(f"AR model block_size={int(model.block_size)} is not a perfect square.")

    # Evaluate NLL/PPL/Acc
    nll, ppl, acc = eval_nll_ppl_acc(
        model=model,
        test_loader=test_loader,
        device=device,
        order_mode=args.order,
    )
    print(f"[TEST] order={args.order}")
    print(f"[TEST] NLL/token      : {nll:.6f}")
    print(f"[TEST] Perplexity     : {ppl:.4f}")
    print(f"[TEST] Token accuracy : {acc*100:.2f}%")

    # Evaluate FID (strict)
    fid_loader = DataLoader(
        test_dataset,
        batch_size=args.fid_batch_size,
        shuffle=True,  # random subset OK
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    fid_value = eval_fid(
        model=model,
        tokenizer=tokenizer,
        test_loader=fid_loader,
        device=device,
        image_size=args.image_size,
        num_samples=args.fid_num_samples,
        order_mode_for_gen=args.fid_order,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    print(f"[TEST] FID ({args.fid_num_samples} samples): {fid_value:.3f}")


if __name__ == "__main__":
    main()