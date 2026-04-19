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

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm
from RandAR.utils.latents import make_token_order
from RandAR.model.generate import sample as randar_sample
from RandAR.model.utils import calculate_num_query_tokens_for_parallel_decoding


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
    shuffle_ratio: float | None = None,
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

        token_order = (
            make_token_order(B, T, device, order_mode, shuffle_ratio=shuffle_ratio)
            if accepts_token_order
            else None
        )

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


def make_confidence_bins(confidences: np.ndarray, qualities: np.ndarray, n_bins: int):
    edges = np.linspace(confidences.min(), confidences.max(), n_bins + 1)
    # handle degenerate case
    if np.allclose(edges[0], edges[-1]):
        edges = np.linspace(confidences.min() - 1e-6, confidences.max() + 1e-6, n_bins + 1)

    bin_centers = []
    bin_mean_quality = []
    bin_counts = []

    for i in range(n_bins):
        left = edges[i]
        right = edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences >= left) & (confidences < right)

        if mask.sum() == 0:
            bin_centers.append(0.5 * (left + right))
            bin_mean_quality.append(float("nan"))
            bin_counts.append(0)
        else:
            bin_centers.append(float(confidences[mask].mean()))
            bin_mean_quality.append(float(qualities[mask].mean()))
            bin_counts.append(int(mask.sum()))

    return {
        "edges": edges.tolist(),
        "bin_centers": bin_centers,
        "bin_mean_quality": bin_mean_quality,
        "bin_counts": bin_counts,
    }


@torch.no_grad()
def generate_with_logprobs(
    model,
    cond: torch.Tensor,
    token_order: torch.Tensor,
    cfg_scales: Tuple[float, float],
    num_inference_steps: int,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        tokens_raster:   [B, T]
        token_logprobs:  [B, T]
    """
    if token_order is None:
        raise ValueError("token_order must be provided explicitly in this helper.")
    
    device = cond.device

    bs = cond.shape[0]
    T = model.block_size

    # Step 1: token order / outputs
    if token_order.shape != (bs, T):
        raise ValueError(f"token_order must have shape {(bs, T)}, got {tuple(token_order.shape)}")

    result_indices = torch.zeros((bs, T), dtype=torch.long, device=cond.device)
    result_logprobs = torch.zeros((bs, T), dtype=torch.float32, device=cond.device)

    # Step 2: rotary / position tokens
    position_instruction_tokens = model.get_position_instruction_tokens(token_order).to(device)
    img_token_freq_cis = model.freqs_cis[model.cls_token_num:].to(device)[token_order]

    # Step 3: CFG
    use_cfg = cfg_scales[-1] > 1.0
    if use_cfg:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null], dim=0)
        img_token_freq_cis = torch.cat([img_token_freq_cis, img_token_freq_cis], dim=0)
        position_instruction_tokens = torch.cat([position_instruction_tokens, position_instruction_tokens], dim=0)
        bs_cfg = bs * 2
    else:
        cond_combined = cond
        bs_cfg = bs

    cond_combined_tokens = model.cls_embedding(cond_combined, train=False)

    # Step 4: cache
    max_seq_len = cond_combined_tokens.shape[1] + T * 2
    model.setup_caches(
        max_batch_size=bs_cfg,
        max_seq_length=max_seq_len,
        dtype=model.tok_embeddings.weight.dtype,
    )
    model.causal_mask = model.causal_mask.to(device)

    if hasattr(model, "causal_mask") and model.causal_mask is not None:
        model.causal_mask = model.causal_mask.to(device)

    if hasattr(model, "freqs_cis") and model.freqs_cis is not None:
        model.freqs_cis = model.freqs_cis.to(device)

    for layer in model.layers:
        if hasattr(layer, "attention") and hasattr(layer.attention, "kv_cache"):
            kv_cache = layer.attention.kv_cache
            if kv_cache is not None:
                if hasattr(kv_cache, "k_cache") and kv_cache.k_cache is not None:
                    kv_cache.k_cache = kv_cache.k_cache.to(device)
                if hasattr(kv_cache, "v_cache") and kv_cache.v_cache is not None:
                    kv_cache.v_cache = kv_cache.v_cache.to(device)

    # Step 5: decode
    if num_inference_steps == -1:
        num_inference_steps = T

    cur_inference_step = 0
    num_query_token_cur_step = 1
    query_token_idx_cur_step = 0

    x = torch.cat(
        [
            cond_combined_tokens,
            position_instruction_tokens[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step],
        ],
        dim=1,
    )
    prefix_freqs_cis = model.freqs_cis[:model.cls_token_num].to(device)
    prefix_freqs_cis = prefix_freqs_cis.unsqueeze(0).repeat(bs_cfg, 1, 1, 1)

    cur_freqs_cis = torch.cat(
        [
            prefix_freqs_cis,
            img_token_freq_cis[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step],
        ],
        dim=1,
    )
    input_pos = torch.arange(0, x.shape[1], device=cond.device)

    while (
        query_token_idx_cur_step <= T - num_query_token_cur_step
        and query_token_idx_cur_step <= T - 1
    ):
        logits = model.forward_inference(x, cur_freqs_cis, input_pos)

        if use_cfg:
            cur_cfg_scale = cfg_scales[0] + (cfg_scales[-1] - cfg_scales[0]) * query_token_idx_cur_step / T
            cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
            logits = uncond_logits + cur_cfg_scale * (cond_logits - uncond_logits)

        logits = logits[:, -num_query_token_cur_step:]  # [B, q, V]
        indices = torch.zeros(bs, num_query_token_cur_step, dtype=torch.long, device=cond.device)
        logps = torch.zeros(bs, num_query_token_cur_step, dtype=torch.float32, device=cond.device)

        for i in range(num_query_token_cur_step):
            idx_i, probs_i = randar_sample(
                logits[:, i:i+1],
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            idx_i = idx_i.squeeze(1)                      # [B]
            probs_i = probs_i                             # [B, V]
            chosen_prob = probs_i.gather(1, idx_i.unsqueeze(1)).squeeze(1).clamp_min(1e-12)

            indices[:, i] = idx_i
            logps[:, i] = chosen_prob.log()

        result_indices[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step] = indices
        result_logprobs[:, query_token_idx_cur_step: query_token_idx_cur_step + num_query_token_cur_step] = logps

        img_tokens = model.tok_embeddings(indices)
        if use_cfg:
            img_tokens = torch.cat([img_tokens, img_tokens], dim=0)

        cur_inference_step += 1
        num_query_token_next_step = calculate_num_query_tokens_for_parallel_decoding(
            cur_inference_step,
            num_inference_steps,
            T,
            query_token_idx_cur_step,
            num_query_token_cur_step,
        )

        x = torch.zeros(
            bs_cfg,
            2 * num_query_token_cur_step - 1 + num_query_token_next_step,
            model.dim,
            dtype=img_tokens.dtype,
            device=cond.device,
        )
        x[:, :1] = img_tokens[:, :1]

        cur_query_position_instruction_tokens = position_instruction_tokens[
            :, query_token_idx_cur_step + 1: query_token_idx_cur_step + num_query_token_cur_step
        ]
        x[:, 1: 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_position_instruction_tokens
        x[:, 1: 2 * num_query_token_cur_step - 1][:, 1::2] = img_tokens[:, 1:num_query_token_cur_step]

        query_token_idx_next_step = query_token_idx_cur_step + num_query_token_cur_step
        next_position_instruction_tokens = position_instruction_tokens[
            :, query_token_idx_next_step: query_token_idx_next_step + num_query_token_next_step
        ]
        x[:, 2 * num_query_token_cur_step - 1:] = next_position_instruction_tokens

        cur_freqs_cis = torch.zeros(
            (bs_cfg, 2 * num_query_token_cur_step - 1 + num_query_token_next_step, *model.freqs_cis.shape[-2:]),
            dtype=img_token_freq_cis.dtype,
            device=cond.device,
        )
        cur_freqs_cis[:, :1] = img_token_freq_cis[:, query_token_idx_cur_step: query_token_idx_cur_step + 1]

        cur_query_freq_cis = img_token_freq_cis[
            :, query_token_idx_cur_step + 1: query_token_idx_cur_step + num_query_token_cur_step
        ]
        cur_freqs_cis[:, 1: 2 * num_query_token_cur_step - 1][:, ::2] = cur_query_freq_cis
        cur_freqs_cis[:, 1: 2 * num_query_token_cur_step - 1][:, 1::2] = cur_query_freq_cis

        next_freq_cis = img_token_freq_cis[
            :, query_token_idx_next_step: query_token_idx_next_step + num_query_token_next_step
        ]
        cur_freqs_cis[:, 2 * num_query_token_cur_step - 1:] = next_freq_cis

        query_token_idx_cur_step = query_token_idx_next_step
        if query_token_idx_cur_step > T:
            break

        last_input_pos = input_pos[input_pos.shape[0] - num_query_token_cur_step]
        input_pos = (
            torch.arange(
                2 * num_query_token_cur_step - 1 + num_query_token_next_step,
                device=cond.device,
                dtype=torch.long,
            )
            + last_input_pos
            + 1
        )
        num_query_token_cur_step = num_query_token_next_step

    # Back to raster order
    reverse_perm = torch.argsort(token_order, dim=-1)
    tokens_raster = torch.gather(result_indices, dim=1, index=reverse_perm)
    logps_raster = torch.gather(result_logprobs, dim=1, index=reverse_perm)

    model.remove_caches()
    return tokens_raster, logps_raster


def compute_risk_coverage(confidences: np.ndarray, qualities: np.ndarray):
    """
    quality in [0,1]
    risk = 1 - mean_quality over retained top-confidence samples
    """
    order = np.argsort(-confidences)
    q_sorted = qualities[order]

    coverages = []
    risks = []

    for k in range(1, len(q_sorted) + 1):
        kept_q = q_sorted[:k]
        coverage = k / len(q_sorted)
        risk = 1.0 - float(np.mean(kept_q))
        coverages.append(coverage)
        risks.append(risk)

    aurc = float(np.trapezoid(risks, coverages))
    return {
        "coverage": coverages,
        "risk": risks,
        "aurc": aurc,
    }


@torch.no_grad()
def evaluate_sample_calibration(
    model,
    tokenizer,
    classifier,
    loader,
    device: torch.device,
    args,
):
    model.eval()
    tokenizer.eval()
    classifier.eval()

    T = int(model.block_size)
    all_confidences: List[float] = []
    all_qualities: List[float] = []
    all_targets: List[int] = []

    processed = 0
    pbar = tqdm(loader, desc="Evaluating sample-level calibration")

    for batch in pbar:
        if processed >= args.num_samples:
            break

        if len(batch) == 3:
            _, y, _ = batch
        else:
            _, y = batch

        y = y.to(device, non_blocking=True).long().view(-1)
        remaining = args.num_samples - processed
        if y.shape[0] > remaining:
            y = y[:remaining]

        B = y.shape[0]
        token_order = make_token_order(
            B,
            T,
            device,
            args.generation_order,
            shuffle_ratio=getattr(args, "shuffle_ratio", None),
        )

        gen_tokens, token_logps = generate_with_logprobs(
            model=model,
            cond=y,
            token_order=token_order,
            cfg_scales=args.cfg_scales,
            num_inference_steps=args.num_inference_steps,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

        imgs = tokenizer.decode_codes_to_img(gen_tokens, args.image_size).clamp(0, 1)  # [B,3,H,W]

        logits_clf = classifier(imgs)
        probs_clf = torch.softmax(logits_clf, dim=1)
        quality = probs_clf.gather(1, y.unsqueeze(1)).squeeze(1)  # target-class probability

        sample_conf = token_logps.mean(dim=1)

        all_confidences.extend(sample_conf.detach().cpu().tolist())
        all_qualities.extend(quality.detach().cpu().tolist())
        all_targets.extend(y.detach().cpu().tolist())

        processed += B
        pbar.set_postfix(
            mean_conf=f"{float(sample_conf.mean()):.4f}",
            mean_quality=f"{float(quality.mean()):.4f}",
        )

    confidences = np.array(all_confidences, dtype=np.float64)
    qualities = np.array(all_qualities, dtype=np.float64)

    spear = spearmanr(confidences, qualities)
    bins = make_confidence_bins(confidences, qualities, args.num_bins)
    rc = compute_risk_coverage(confidences, qualities)

    return {
        "num_samples": int(len(confidences)),
        "classifier_model": args.classifier_model,
        "confidence_definition": "mean_token_logprob",
        "quality_definition": "target_class_probability",
        "cfg_scales": list(args.cfg_scales),
        "num_inference_steps": int(args.num_inference_steps),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "mean_confidence": float(confidences.mean()),
        "mean_quality": float(qualities.mean()),
        "spearman_rho": float(spear.statistic),
        "spearman_pvalue": float(spear.pvalue),
        "bins": bins,
        "risk_coverage": rc,
        "raw": {
            "confidence": confidences.tolist(),
            "quality": qualities.tolist(),
            "targets": all_targets,
        },
    }
