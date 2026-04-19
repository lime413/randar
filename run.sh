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

RASTER_CFG_JSON="results/raster_order/search_cfg.json"
RANDOM_CFG_JSON="results/random_order/search_cfg.json"
ADAPTIVE_CFG_JSON="results/adaptive_order/search_cfg.json"

get_best_cfg_scale() {
  local json_path="$1"
  if [[ ! -f "$json_path" ]]; then
    echo "CFG search results not found: $json_path. Run ./search_cfg.sh first." >&2
    exit 1
  fi

  "$PYTHON" -c "import json, sys; print(json.load(open(sys.argv[1], 'r', encoding='utf-8'))['best']['cfg_scale'])" "$json_path"
}

RASTER_CFG_ON="$(get_best_cfg_scale "$RASTER_CFG_JSON")"
RANDOM_CFG_ON="$(get_best_cfg_scale "$RANDOM_CFG_JSON")"
ADAPTIVE_CFG_ON="$(get_best_cfg_scale "$ADAPTIVE_CFG_JSON")"

"$PYTHON" eval_c2i.py \
  --config "$RASTER_CONFIG" \
  --ar_ckpt "$RASTER_CKPT" \
  --vq_ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --raw-dataset cifar10 \
  --raw-data-path "$CLEAN_RAW" \
  --cfg-on-scale "$RASTER_CFG_ON" \
  --output-json "results/raster_order/eval_metrics.json" \
  --run-robustness

"$PYTHON" eval_calibration_sample_level.py \
  --config "$RASTER_CONFIG" \
  --ar_ckpt "$RASTER_CKPT" \
  --vq-ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --output-dir "results/raster_order/sample_calibration"

"$PYTHON" eval_calibration_token_level.py \
  --config "$RASTER_CONFIG" \
  --ar-ckpt "$RASTER_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --output-dir "results/raster_order/token_calibration"

"$PYTHON" eval_c2i.py \
  --config "$RANDOM_CONFIG" \
  --ar_ckpt "$RANDOM_CKPT" \
  --vq_ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --raw-dataset cifar10 \
  --raw-data-path "$CLEAN_RAW" \
  --cfg-on-scale "$RANDOM_CFG_ON" \
  --output-json "results/random_order/eval_metrics.json" \
  --run-robustness

"$PYTHON" eval_calibration_sample_level.py \
  --config "$RANDOM_CONFIG" \
  --ar_ckpt "$RANDOM_CKPT" \
  --vq-ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --output-dir "results/random_order/sample_calibration"

"$PYTHON" eval_calibration_token_level.py \
  --config "$RANDOM_CONFIG" \
  --ar-ckpt "$RANDOM_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --output-dir "results/random_order/token_calibration"

"$PYTHON" eval_c2i.py \
  --config "$ADAPTIVE_CONFIG" \
  --ar_ckpt "$ADAPTIVE_CKPT" \
  --vq_ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --raw-dataset cifar10 \
  --raw-data-path "$CLEAN_RAW" \
  --cfg-on-scale "$ADAPTIVE_CFG_ON" \
  --output-json "results/adaptive_order/eval_metrics.json" \
  --run-robustness

"$PYTHON" eval_calibration_sample_level.py \
  --config "$ADAPTIVE_CONFIG" \
  --ar_ckpt "$ADAPTIVE_CKPT" \
  --vq-ckpt "$VQ_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --output-dir "results/adaptive_order/sample_calibration"

"$PYTHON" eval_calibration_token_level.py \
  --config "$ADAPTIVE_CONFIG" \
  --ar-ckpt "$ADAPTIVE_CKPT" \
  --dataset latent \
  --data-path "$CLEAN_LATENTS" \
  --output-dir "results/adaptive_order/token_calibration"
