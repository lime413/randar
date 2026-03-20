import inspect
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

from RandAR.utils.latents import extract_latent_tokens, make_token_order


def plot_reliability_diagram(
    bin_counts: np.ndarray,
    bin_conf_sums: np.ndarray,
    bin_correct_sums: np.ndarray,
    output_path: Path
) -> None:
    n_bins = len(bin_counts)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = (1.0 / n_bins) * 0.9

    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)

    nonzero = bin_counts > 0
    bin_acc[nonzero] = bin_correct_sums[nonzero] / bin_counts[nonzero]
    bin_conf[nonzero] = bin_conf_sums[nonzero] / bin_counts[nonzero]

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
    plt.bar(centers, bin_acc, width=width, alpha=0.7, label="Accuracy")
    plt.plot(centers, bin_conf, marker="o", linewidth=1.5, label="Confidence")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _compute_ece_from_bins(
    bin_counts: np.ndarray,
    bin_conf_sums: np.ndarray,
    bin_correct_sums: np.ndarray,
) -> float:
    total = int(bin_counts.sum())
    if total == 0:
        return 0.0

    ece = 0.0
    for i in range(len(bin_counts)):
        if bin_counts[i] == 0:
            continue
        acc_i = bin_correct_sums[i] / bin_counts[i]
        conf_i = bin_conf_sums[i] / bin_counts[i]
        ece += (bin_counts[i] / total) * abs(acc_i - conf_i)
    return float(ece)


@torch.no_grad()
def eval_token_calibration(
    model,
    test_loader,
    device: torch.device,
    order_mode: str,
    num_bins: int = 15,
) -> Dict[str, float]:
    """
    Token-level calibration for clean CIFAR-10 latent codes.

    Metrics:
    - NLL/token
    - Perplexity
    - Token accuracy
    - Mean confidence
    - Overconfidence gap = mean_confidence - accuracy
    - ECE
    - Reliability diagram stats
    """
    model.eval()

    fwd_sig = inspect.signature(model.forward)
    accepts_token_order = ("token_order" in fwd_sig.parameters)

    total_ce = 0.0
    total_tokens = 0
    total_correct = 0
    total_confidence = 0.0

    bin_counts = np.zeros(num_bins, dtype=np.int64)
    bin_conf_sums = np.zeros(num_bins, dtype=np.float64)
    bin_correct_sums = np.zeros(num_bins, dtype=np.float64)

    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=device)

    for batch in tqdm(test_loader, desc="Evaluating calibration"):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        tokens = extract_latent_tokens(x)
        cond = y.long().view(-1)

        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens shape (B,T), got {tuple(tokens.shape)}")

        B, T = tokens.shape

        token_order = make_token_order(B, T, device, order_mode) if accepts_token_order else None

        logits, _, used_order = model(tokens, cond, token_order=token_order, targets=tokens)

        if logits.dim() != 3:
            raise ValueError(f"Expected logits shape (B,T,V), got {tuple(logits.shape)}")

        B2, T2, V = logits.shape
        if B2 != B or T2 != T:
            raise ValueError(
                f"Model output shape mismatch: logits {tuple(logits.shape)} "
                f"but expected (B={B}, T={T}, V)."
            )

        if used_order is None:
            raise ValueError("Model did not return token_order; cannot align targets.")

        if used_order.shape != (B, T):
            raise ValueError(
                f"Returned token_order has shape {tuple(used_order.shape)} but expected {(B, T)}."
            )

        targets_pred_order = torch.gather(tokens, dim=1, index=used_order)

        logits_flat = logits.reshape(-1, V)
        targets_flat = targets_pred_order.reshape(-1)

        ce = F.cross_entropy(logits_flat, targets_flat, reduction="sum")
        total_ce += float(ce.item())
        total_tokens += int(targets_flat.numel())

        probs_flat = F.softmax(logits_flat, dim=-1)
        conf_flat, pred_flat = probs_flat.max(dim=-1)
        correct_flat = (pred_flat == targets_flat).float()

        total_correct += int(correct_flat.sum().item())
        total_confidence += float(conf_flat.sum().item())

        # put each prediction into confidence bin
        # use interior edges so 1.0 goes into the last bin
        bin_ids = torch.bucketize(conf_flat, bin_edges[1:-1], right=False)

        for i in range(num_bins):
            mask = (bin_ids == i)
            cnt = int(mask.sum().item())
            if cnt == 0:
                continue

            bin_counts[i] += cnt
            bin_conf_sums[i] += float(conf_flat[mask].sum().item())
            bin_correct_sums[i] += float(correct_flat[mask].sum().item())

    nll = total_ce / max(total_tokens, 1)
    ppl = math.exp(min(nll, 50.0))
    acc = total_correct / max(total_tokens, 1)
    mean_conf = total_confidence / max(total_tokens, 1)
    overconfidence_gap = mean_conf - acc
    ece = _compute_ece_from_bins(bin_counts, bin_conf_sums, bin_correct_sums)

    bin_acc = []
    bin_conf = []
    for i in range(num_bins):
        if bin_counts[i] == 0:
            bin_acc.append(0.0)
            bin_conf.append(0.0)
        else:
            bin_acc.append(float(bin_correct_sums[i] / bin_counts[i]))
            bin_conf.append(float(bin_conf_sums[i] / bin_counts[i]))

    return {
        "nll_per_token": float(nll),
        "perplexity": float(ppl),
        "token_accuracy": float(acc),
        "mean_confidence": float(mean_conf),
        "overconfidence_gap": float(overconfidence_gap),
        "ece": float(ece),
        "num_bins": int(num_bins),
        "bin_counts": bin_counts.tolist(),
        "bin_accuracy": bin_acc,
        "bin_confidence": bin_conf,
    }


