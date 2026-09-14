"""Reference solutions for topic 03."""

import torch
from torch import Tensor


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    n = max(len(a), len(b))
    for i in range(1, n + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(f"incompatible shapes {a} and {b} on axis -{i}: {da} vs {db}")
        out.append(max(da, db))
    return tuple(reversed(out))


def explain_trap() -> tuple[int, ...]:
    return (3, 3)


def add_bias(x: Tensor, bias: Tensor) -> Tensor:
    return x + bias


def scale_per_head(x: Tensor, scales: Tensor) -> Tensor:
    return x * scales.view(1, -1, 1, 1)


def normalize_last_dim(x: Tensor) -> Tensor:
    return x - x.mean(dim=-1, keepdim=True)


def weighted_sum(weights: Tensor, values: Tensor) -> Tensor:
    return (weights.unsqueeze(-1) * values).sum(dim=1)


def outer_product(a: Tensor, b: Tensor) -> Tensor:
    return a[:, None] * b[None, :]


def pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    return (a[:, None, :] - b[None, :, :]).pow(2).sum(dim=-1)


def one_hot_by_comparison(ids: Tensor, vocab_size: int) -> Tensor:
    return (ids[..., None] == torch.arange(vocab_size, device=ids.device)).float()


def causal_mask(seq_len: int) -> Tensor:
    idx = torch.arange(seq_len)
    return idx[:, None] >= idx[None, :]


def relative_positions(q_len: int, k_len: int) -> Tensor:
    return torch.arange(q_len)[:, None] - torch.arange(k_len)[None, :]


def alibi_bias(slopes: Tensor, seq_len: int) -> Tensor:
    rel = relative_positions(seq_len, seq_len).to(slopes.dtype)
    return -slopes[:, None, None] * rel[None, :, :]


def apply_additive_mask(scores: Tensor, mask: Tensor) -> Tensor:
    return scores.masked_fill(~mask, float("-inf"))


def key_padding_mask_apply(scores: Tensor, pad_mask: Tensor) -> Tensor:
    return scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))
