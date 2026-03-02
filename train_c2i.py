import os
import time
import shutil
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from omegaconf import OmegaConf
from clearml import Task, OutputModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from RandAR.util import instantiate_from_config
from RandAR.dataset.builder import build_dataset
from RandAR.utils.visualization import make_grid
from RandAR.utils.logger import create_logger
from RandAR.utils.lr_scheduler import get_scheduler

from torchmetrics.image.fid import FrechetInceptionDistance
import math


def cycle(dl: DataLoader):
    """Loop over the dataloader indefinitely (same as original)."""
    while True:
        for data in dl:
            yield data


def set_seed(seed: int):
    """Single-process seed helper."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    assert torch.cuda.is_available(), "Requires at least one CUDA GPU."
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # -------------------------
    # Load config
    # -------------------------
    config = OmegaConf.load(args.config)

    # Allow overriding via CLI (keeps your original args semantics)
    if args.max_iters is not None:
        config.max_iters = args.max_iters
    if args.global_seed is not None:
        config.global_seed = args.global_seed

    set_seed(config.global_seed)

    # -------------------------
    # Experiment directory
    # -------------------------
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = timestamp + f"_bs_{config.global_batch_size}_lr_{config.optimizer.lr}"
    else:
        exp_name = args.exp_name
    experiment_dir = os.path.join(args.results_dir, exp_name)
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")

    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {config.global_seed}")

    # -------------------------
    # ClearML (optional)
    # -------------------------

    if args.clearml:
        project_name = "RandAR"
        task_name = f"RandAR-{exp_name}"

        task = Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=[
                exp_name,
                "train",
                "tokenizer-" + str(config.ar_model.params.vocab_size),
                "dataset-" + config.dataset.name,
                "debug" if args.debug else "full",
            ],
        )

        task.connect(OmegaConf.to_container(config, resolve=True), name="config")
        task.connect(vars(args), name="args")

        cml_logger = task.get_logger()
        output_model = OutputModel(task=task, framework="PyTorch")

    # -------------------------
    # Dataset / Dataloader
    # -------------------------
    is_train = not args.debug
    dataset = build_dataset(is_train=is_train, args=args, transform=transforms.ToTensor())

    # Single GPU => per_gpu_batch_size is just global_batch_size / grad_accum
    grad_accum = int(config.accelerator.gradient_accumulation_steps)
    assert grad_accum >= 1
    per_gpu_batch_size = int(config.global_batch_size // grad_accum)

    data_loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    data_loader = cycle(data_loader)

    logger.info(f"Dataset contains {len(dataset)} samples.")
    logger.info(f"Per-step batch size (on cuda:0): {per_gpu_batch_size}")
    logger.info(f"Grad accumulation steps: {grad_accum}")
    logger.info(f"Effective global batch size: {per_gpu_batch_size * grad_accum}")

    # -------------------------
    # Model
    # -------------------------
    model = instantiate_from_config(config.ar_model).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.train()

    # -------------------------
    # Tokenizer (CIFAR VQ-VAE)
    # -------------------------
    tokenizer = instantiate_from_config(config.tokenizer).to(device)
    tokenizer.load_state_dict(torch.load(args.vq_ckpt, map_location=device))
    tokenizer.eval()
    for p in tokenizer.parameters():
        p.requires_grad_(False)

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    # -------------------------
    # Optimizer / LR scheduler
    # -------------------------
    optimizer = model.configure_optimizer(**config.optimizer)

    lr_scheduler = get_scheduler(
        name=config.lr_scheduler.type,
        optimizer=optimizer,
        num_warmup_steps=config.lr_scheduler.warm_up_iters * grad_accum,
        num_training_steps=config.max_iters * grad_accum,
        min_lr_ratio=config.lr_scheduler.min_lr_ratio,
        num_cycles=config.lr_scheduler.num_cycles,
    )

    # -------------------------
    # Resume training
    # -------------------------
    # It will resume from the latest "iters_XXXXXXXX" folder if found.
    train_steps = 0
    saved_ckpt_dirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("iters_")]
    if len(saved_ckpt_dirs) > 0:
        saved_ckpt_dirs = sorted(saved_ckpt_dirs)
        last_dir = saved_ckpt_dirs[-1]
        ckpt_dir = os.path.join(checkpoint_dir, last_dir)

        ckpt_file = os.path.join(ckpt_dir, "train_state.pt")
        if os.path.exists(ckpt_file):
            logger.info(f"Resuming from {ckpt_file}")
            state = torch.load(ckpt_file, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            lr_scheduler.load_state_dict(state["lr_scheduler"])
            train_steps = int(state["train_steps"])
        else:
            logger.info(f"Found {ckpt_dir} but no train_state.pt; starting from scratch.")

    # -------------------------
    # Training loop
    # -------------------------
    total_iters = int(config.max_iters)
    logger.info(f"Starting training from iteration {train_steps} to {total_iters}")

    log_every = int(args.log_every)
    ckpt_every = int(args.ckpt_every)
    visualize_every = int(args.visualize_every)

    running_loss = 0.0
    running_grad_norm = 0.0
    start_time = time.time()

    scaler = None
    use_amp = (args.mixed_precision in ["fp16", "bf16"])
    amp_dtype = torch.float16 if args.mixed_precision == "fp16" else (torch.bfloat16 if args.mixed_precision == "bf16" else None)
    if args.mixed_precision == "fp16":
        scaler = torch.amp.GradScaler()

    # local helper to compute grad norm
    def compute_grad_norm_l2(m: torch.nn.Module) -> float:
        total = 0.0
        for p in m.parameters():
            if p.grad is None:
                continue
            total += p.grad.data.norm(2).item()
        return total
    
    # local helper to compute FID score
    def compute_fid(model, tokenizer, dataset, device, num_samples, batch_size, image_size):
        model.eval()
        fid_metric.reset()

        # Build a loader that yields latent codes (x) and labels (y)
        fid_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False,
            pin_memory=True,
        )

        seen = 0
        with torch.no_grad():
            for x, y, _ in fid_loader:
                if seen >= num_samples:
                    break

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                real_codes = x.reshape(x.shape[0], -1)
                cond = y.reshape(-1)

                # Real images = VQ recon of dataset codes
                real_imgs = tokenizer.decode_codes_to_img(real_codes, image_size)
                # Expect uint8 [0..255] for torchmetrics FID
                if real_imgs.dtype != torch.uint8:
                    real_imgs = (real_imgs.clamp(0, 1) * 255).to(torch.uint8)

                # Fake images = model samples
                gen_codes = model.generate(
                    cond=cond,
                    token_order=None,
                    cfg_scales=[4.0, 4.0],
                    num_inference_steps=-1,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                )
                model.remove_caches()
                fake_imgs = tokenizer.decode_codes_to_img(gen_codes, image_size)
                if fake_imgs.dtype != torch.uint8:
                    fake_imgs = (fake_imgs.clamp(0, 1) * 255).to(torch.uint8)

                # Torchmetrics expects (N, 3, H, W), uint8
                fid_metric.update(real_imgs, real=True)
                fid_metric.update(fake_imgs, real=False)

                seen += x.shape[0]

        fid = float(fid_metric.compute().item())
        model.train()
        return fid

    while train_steps < total_iters:
        model.train()

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)

        loss_this_step = 0.0
        grad_norm = 0.0

        for micro in range(grad_accum):
            x, y, _ = next(data_loader)

            # x typically has shape (B, 1, T)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            image_tokens = x.reshape(x.shape[0], -1)
            cond = y.reshape(-1)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                logits, loss, token_order = model(image_tokens, cond, targets=image_tokens)

            if scaler is not None:
                scaler.scale(loss / grad_accum).backward()
            else:
                (loss / grad_accum).backward()

            loss_this_step += float(loss.detach().item())

        # gradient clipping + step
        if config.optimizer.max_grad_norm != 0.0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.max_grad_norm)

        grad_norm = compute_grad_norm_l2(model)

        if grad_norm < config.optimizer.skip_grad_norm or train_steps < config.optimizer.skip_grad_iter:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        lr_scheduler.step()

        # bookkeeping
        running_loss += (loss_this_step / grad_accum)
        running_grad_norm += grad_norm

        train_steps += 1

        # -------------------------
        # Logging
        # -------------------------
        if train_steps % log_every == 0:
            avg_loss = running_loss / log_every
            avg_grad = running_grad_norm / log_every
            avg_ppl = math.exp(min(avg_loss, 20.0))

            end_time = time.time()
            avg_time = (end_time - start_time) / log_every
            start_time = time.time()

            lr = lr_scheduler.get_last_lr()[0]
            logger.info(
                f"Step {train_steps:08d} | Loss {avg_loss:.4f} | Time left {avg_time* (total_iters - train_steps):.0f}s | "
                f"Grad Norm {avg_grad:.4f} | LR {lr:.6f}"
            )

            if args.clearml:
                cml_logger.report_scalar("train", "loss_nll", iteration=train_steps, value=avg_loss)
                cml_logger.report_scalar("train", "ppl", iteration=train_steps, value=avg_ppl)
                cml_logger.report_scalar("train", "time_sec", iteration=train_steps, value=avg_time)
                cml_logger.report_scalar("train", "grad_norm", iteration=train_steps, value=avg_grad)
                cml_logger.report_scalar("train", "lr", iteration=train_steps, value=lr)

            running_loss = 0.0
            running_grad_norm = 0.0

        # -------------------------
        # FID evaluation
        # -------------------------
        if args.fid_every > 0 and (train_steps % args.fid_every == 0):
            fid_value = compute_fid(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset,
                device=device,
                num_samples=args.fid_num_samples,
                batch_size=args.fid_batch,
                image_size=args.image_size,
            )
            logger.info(f"Step {train_steps:08d} | FID {fid_value:.3f}")
            if args.clearml:
                cml_logger.report_scalar("eval", "FID", iteration=train_steps, value=fid_value)

        # -------------------------
        # Visualization
        # -------------------------
        if visualize_every > 0 and (train_steps % visualize_every == 0):
            model.eval()
            with torch.no_grad():
                visualize_num = int(args.visualize_num)

                visualize_logits = logits[:visualize_num]
                visualize_cond = cond[:visualize_num]
                visualize_token_order = token_order[:visualize_num]
                visualize_gt_indices = image_tokens[:visualize_num]

                orig_token_order = torch.argsort(visualize_token_order)
                img_token_num = visualize_logits.shape[1]

                # teacher forcing reconstruction
                pred_recon_indices = torch.zeros(
                    visualize_num, img_token_num, device=device, dtype=torch.long
                )
                for i in range(img_token_num):
                    pred_recon_indices[:, i : i + 1] = torch.argmax(
                        visualize_logits[:, i : i + 1], dim=-1
                    )

                pred_recon_indices = torch.gather(
                    pred_recon_indices.unsqueeze(-1),
                    dim=1,
                    index=orig_token_order.unsqueeze(-1),
                ).squeeze(-1)

                pred_recon_imgs = tokenizer.decode_codes_to_img(pred_recon_indices, args.image_size)

                # VQ reconstruction from ground truth codes
                gt_recon_imgs = tokenizer.decode_codes_to_img(visualize_gt_indices, args.image_size)

                # generation
                gen_indices = model.generate(
                    cond=visualize_cond,
                    token_order=None,
                    cfg_scales=[4.0, 4.0],
                    num_inference_steps=-1,
                    temperature=1.0,
                    top_k=0,
                    top_p=1.0,
                )
                model.remove_caches()
                gen_imgs = tokenizer.decode_codes_to_img(gen_indices, args.image_size)

                pred_recon_grid = make_grid(pred_recon_imgs)
                gt_recon_grid = make_grid(gt_recon_imgs)
                gen_grid = make_grid(gen_imgs)

                if args.clearml:
                    cml_logger.report_image("viz", "pred_recon", iteration=train_steps, image=pred_recon_grid)
                    cml_logger.report_image("viz", "gt_recon", iteration=train_steps, image=gt_recon_grid)
                    cml_logger.report_image("viz", "gen", iteration=train_steps, image=gen_grid)
            model.train()

        # -------------------------
        # Checkpointing
        # -------------------------
        if ckpt_every > 0 and (train_steps % ckpt_every == 0):
            ckpt_path = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}")
            os.makedirs(ckpt_path, exist_ok=True)

            state = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "train_steps": train_steps,
                "config": dict(config),
            }

            weights_file = os.path.join(ckpt_path, "train_state.pt")
            torch.save(state, weights_file)
            logger.info(f"Saved Iter {train_steps} checkpoint to {ckpt_path}")

            if args.clearml:
                # Track latest checkpoint as a ClearML "model" snapshot
                output_model.update_weights(
                    weights_filename=weights_file,
                    iteration=train_steps,
                    update_comment="training checkpoint",
                )

            # remove older checkpoints
            for ckpt_dir in os.listdir(checkpoint_dir):
                if ckpt_dir.startswith("iters_") and ckpt_dir != f"iters_{train_steps:08d}":
                    save_iter = int(ckpt_dir.split("_")[-1])
                    if save_iter < train_steps - args.keep_last_k * ckpt_every:
                        if save_iter not in [50000, 100000, 200000, 300000]:
                            shutil.rmtree(os.path.join(checkpoint_dir, ckpt_dir), ignore_errors=True)

    # final save
    final_ckpt_dir = os.path.join(checkpoint_dir, f"iters_{train_steps:08d}_final")
    os.makedirs(final_ckpt_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "train_steps": train_steps,
        "config": dict(config),
    }
    torch.save(state, os.path.join(final_ckpt_dir, "train_state.pt"))
    logger.info(f"Saved Final Iter {train_steps} checkpoint to {final_ckpt_dir}")

    if args.clearml:
        task.close()
    logger.info("Training Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/randar_cifar10.yaml")
    parser.add_argument("--results-dir", type=str, default="results")

    parser.add_argument("--image-size", type=int, choices=[32, 128, 256], default=32)
    parser.add_argument("--num-classes", type=int, default=10)

    # Training
    parser.add_argument("--max-iters", type=int, default=40000)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--ckpt-every", type=int, default=1000)
    parser.add_argument("--keep-last-k", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default="bf16", choices=["none", "fp16", "bf16"])

    parser.add_argument("--exp_name", type=str, default=None)

    # Tokenizer ckpt
    parser.add_argument("--vq-ckpt", type=str, default="tokenizer_vq/vqvae_cifar10.pth")

    # Data
    parser.add_argument("--dataset", type=str, default="latent")
    parser.add_argument("--data-path", type=str, default="data/latents_cifar_10/cifar10-vq-vae-512-32_codes")
    parser.add_argument("--debug", action="store_true")

    # Visualization
    parser.add_argument("--visualize-every", type=int, default=1000)
    parser.add_argument("--visualize-num", type=int, default=16)
    parser.add_argument("--fid-every", type=int, default=1000)         # 10000
    parser.add_argument("--fid-num-samples", type=int, default=5000)    # 5000
    parser.add_argument("--fid-batch", type=int, default=128)

    # ClearML
    parser.add_argument("--clearml", type=bool, default=True)

    args = parser.parse_args()

    main(args)