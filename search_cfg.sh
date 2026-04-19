#!/usr/bin/env bash
set -euo pipefail

PYTHON="./.venv/Scripts/python.exe"
VQ_CKPT="tokenizer_vq/vqvae_cifar10.pth"

CLEAN_LATENTS="data/cifar10-all/latents/cifar10/test/cifar10-vq-vae-512-32_codes"
CLEAN_RAW="data/cifar10-all/cifar10"

RASTER_CONFIG="configs/randar_cifar10.yaml"
RANDOM_CONFIG="configs/randar_cifar10_random_order.yaml"
ADAPTIVE_CONFIG="configs/randar_cifar10_adaptive.yaml"

RASTER_CKPT="checkpoints/raster_50k.pt"
RANDOM_CKPT="checkpoints/random_50k.pt"
ADAPTIVE_CKPT="checkpoints/adaptive_50k.pt"

"$PYTHON" tools/search_cfg_weights.py \
  --config "$RASTER_CONFIG" \
  --ar_ckpt "$RASTER_CKPT" \
  --vq_ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --raw-dataset cifar10 \
  --raw-data-path "$CLEAN_RAW" \
  --order_mode_for_gen config \
  --num_fid_samples 10000 \
  --cfg-scales-search "1.0,6.0" \
  --cfg-scales-interval 0.25 \
  --results_path "results/raster_order/search_cfg.json"

"$PYTHON" tools/search_cfg_weights.py \
  --config "$RANDOM_CONFIG" \
  --ar_ckpt "$RANDOM_CKPT" \
  --vq_ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --raw-dataset cifar10 \
  --raw-data-path "$CLEAN_RAW" \
  --order_mode_for_gen config \
  --num_fid_samples 10000 \
  --cfg-scales-search "1.0,6.0" \
  --cfg-scales-interval 0.25 \
  --results_path "results/random_order/search_cfg.json"

"$PYTHON" tools/search_cfg_weights.py \
  --config "$ADAPTIVE_CONFIG" \
  --ar_ckpt "$ADAPTIVE_CKPT" \
  --vq_ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --raw-dataset cifar10 \
  --raw-data-path "$CLEAN_RAW" \
  --order_mode_for_gen config \
  --num_fid_samples 10000 \
  --cfg-scales-search "1.0,6.0" \
  --cfg-scales-interval 0.25 \
  --results_path "results/adaptive_order/search_cfg.json"
