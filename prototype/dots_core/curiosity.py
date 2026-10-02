"""Curiosity (active-inference surrogate): read what is surprising but learnable.

Expected information gain is approximated by current surprise, bounded on both sides:
  too low  -> already known (no epistemic value)
  too high -> unlearnable noise (surprise that will not reduce)
"""
import torch

from .data import encode_with_mask
from .learner import token_nll


@torch.no_grad()
def doc_surprise(model, docs: dict) -> dict:
    """Mean per-byte NLL (nats) for each named document (list of lines)."""
    out = {}
    for name, lines in docs.items():
        x, mask = encode_with_mask(lines, model.cfg.max_len)
        nll = token_nll(model(x), x, mask)
        out[name] = float(nll.sum() / mask[:, 1:].sum())
    return out


def goldilocks_select(scores: dict, low: float, high: float, budget: int) -> list:
    band = [(s, n) for n, s in scores.items() if low <= s <= high]
    return [n for _, n in sorted(band, reverse=True)[:budget]]
