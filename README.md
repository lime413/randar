# Random-Order AR Image Generation

## Baseline Autoregressive Image Generation on CIFAR-10 with Raster Order

This repository contains the implementation of **Random-Order Autoregressive (AR) Image Generation** on CIFAR-10. 

The project establishes a **baseline** using standard raster-scan order generation and serves as a foundation for experimenting with alternative token generation orders (e.g., random permutations) to improve model calibration and robustness.

### Project Overview

The pipeline follows a two-stage approach:
1.  **VQ-VAE Tokenizer**: Compresses $32 \times 32$ images into an $8 \times 8$ discrete latent grid.
2.  **Autoregressive Transformer**: Models the distribution of discrete tokens to generate new images.

While the baseline uses a fixed **raster scan order** (row-by-row), this codebase is designed to support research into **randomized generation orders**.

---

## Pretrained Weights

Pretrained weights for the **VQ-VAE tokenizer** and the **baseline RandAR model (raster order)** are available at:

**Google Drive:**
[https://drive.google.com/drive/folders/1B528vJu1Icn1PtIwJVfmd39WPNqIEEtg?usp=sharing](https://drive.google.com/drive/folders/1B528vJu1Icn1PtIwJVfmd39WPNqIEEtg?usp=sharing)

This allows reproducing reported results without retraining the models.

---

##  Reproduction Guide

To reproduce the baseline experiment, please follow the steps below. The project uses [`uv`](https://docs.astral.sh/uv/) for fast and reliable dependency management.

### 1. Prerequisites

*   **Python**: Version 3.12 or higher.
*   **GPU**: An NVIDIA GPU is recommended for training (tested on NVIDIA RTX GPUs).
*   **uv**: Ensure `uv` is installed on your system.

  ```bash
  # Install uv (Linux/macOS)
  curl -LsSf https://astral.sh/uv/install.sh | sh
  
  # Install uv (Windows PowerShell)
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Environment Setup

All dependencies are managed via `pyproject.toml`.

**Step 2.1: Create environment and install dependencies**
Run the following command in the project root. This will create a virtual environment (`.venv`) and install PyTorch, Transformers, and other required libraries.

```bash
uv sync
```

## Experimental Workflow

The experiment consists of five sequential steps. Run each script/notebook in the order listed below.

### Step 1: Data Preparation

Download and prepare the CIFAR-10 dataset.

```bash
data/load_CIFAR10.py
```

### Step 2: Train VQ-VAE Tokenizer

Train the vector-quantized autoencoder to learn the discrete codebook.
Input: Raw images from data/
Output: Trained weights (tokenizer_vq/vqvae_cifar10.pth)

Open and run all cells in the Jupyter Notebook
```bash
uv run jupyter notebook tokenizer_vq/vq-vae.ipynb
```

### Step 3: Extract Latent Codes

Encode the entire CIFAR-10 dataset into discrete token sequences using the trained VQ-VAE.

```bash
tools/extract_latent_codes.py
```

### Step 4: Train Autoregressive Model

Train the decoder-only Transformer on the extracted token sequences.

```bash
train_c2i.py
```

### Step 5: Evaluation

Evaluate the trained AR model by generating samples and computing metrics

```bash
eval_c2i.py
```

## Original repo of RandAR
https://github.com/ziqipang/RandAR
