"""
Teacher-forced conditional evaluation.

Computes:
  - NLL/token
  - Perplexity
  - Token accuracy

CFG is not used here, because this function does not sample with model.generate().
It evaluates next-token prediction under teacher forcing.
"""

import inspect
import math
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from tqdm import tqdm
from RandAR.utils.latents import extract_latent_tokens, make_token_order

@torch.no_grad()
def eval_nll_ppl_acc(
    model,
    test_loader,
    device: torch.device,
    order_mode: str,
    shuffle_ratio: Optional[float] = None,
) -> Tuple[float, float, float]:
    """
      - NLL/token computed by cross_entropy(logits, targets_pred_order), summed then normalized
      - Perplexity = exp(NLL/token)
      - Accuracy computed in prediction order aligned with the same targets (correct for raster/random)
    """
    model.eval()

    # forward signature: token_order kw is supported in your model :contentReference[oaicite:3]{index=3}
    fwd_sig = inspect.signature(model.forward)
    accepts_token_order = ("token_order" in fwd_sig.parameters)

    total_ce = 0.0
    total_tokens = 0
    total_correct = 0

    for batch in tqdm(test_loader, desc="Evaluating NLL/PPL/Acc"):
        if len(batch) == 3:
            x, y, _ = batch
        else:
            x, y = batch

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        tokens = extract_latent_tokens(x)  # (B,T)
        cond = y.reshape(-1).long()

        B, T = tokens.shape

        # dataset tokens must match model.block_size
        if not hasattr(model, "block_size"):
            raise ValueError("Model has no attribute 'block_size'. Cannot validate token length.")
        if T != int(model.block_size):
            raise ValueError(
                f"Latent token length mismatch: dataset T={T} but model.block_size={int(model.block_size)}. "
                "This must match for correct evaluation. Fix your latent .npy generation or your config."
            )

        token_order = (
            make_token_order(B, T, device, order_mode, shuffle_ratio=shuffle_ratio)
            if accepts_token_order
            else None
        )

        logits, _, used_order = model(tokens, cond, token_order=token_order, targets=tokens)

        if logits.dim() != 3:
            raise ValueError(f"Expected logits shape (B,T,V) but got {tuple(logits.shape)}")

        B2, T2, V = logits.shape
        if B2 != B or T2 != T:
            raise ValueError(
                f"Model output shape mismatch: logits {tuple(logits.shape)} but expected (B={B},T={T},V)."
            )

        # Align targets to prediction order (the order used inside the model)
        if used_order is None:
            raise ValueError("Model did not return token_order; cannot guarantee correct alignment.")
        if used_order.shape != (B, T):
            raise ValueError(f"Returned token_order has shape {tuple(used_order.shape)} but expected {(B,T)}.")

        targets_pred_order = torch.gather(tokens, dim=1, index=used_order)

        ce = F.cross_entropy(
            logits.reshape(-1, V),
            targets_pred_order.reshape(-1),
            reduction="sum",
        )
        total_ce += float(ce.item())
        total_tokens += int(B * T)

        pred = torch.argmax(logits, dim=-1)
        total_correct += int((pred == targets_pred_order).sum().item())

    nll = total_ce / max(total_tokens, 1)
    ppl = math.exp(min(nll, 50.0))
    acc = total_correct / max(total_tokens, 1)
    return nll, ppl, acc
