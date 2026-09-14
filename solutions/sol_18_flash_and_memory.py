"""Эталонные решения к топику 18."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def online_softmax_update(m_prev: Tensor, l_prev: Tensor, scores_block: Tensor) -> tuple[Tensor, Tensor]:
    block_max = scores_block.max(dim=-1, keepdim=True).values
    m_new = torch.maximum(m_prev, block_max)
    l_new = l_prev * (m_prev - m_new).exp() + (scores_block - m_new).exp().sum(dim=-1, keepdim=True)
    return m_new, l_new


def chunked_softmax(x: Tensor, chunk: int) -> Tensor:
    m = torch.full((*x.shape[:-1], 1), float("-inf"), dtype=x.dtype, device=x.device)
    l = torch.zeros_like(m)
    for start in range(0, x.shape[-1], chunk):
        m, l = online_softmax_update(m, l, x[..., start : start + chunk])
    return (x - m).exp() / l


def chunked_logsumexp(x: Tensor, chunk: int) -> Tensor:
    m = torch.full((*x.shape[:-1], 1), float("-inf"), dtype=x.dtype, device=x.device)
    l = torch.zeros_like(m)
    for start in range(0, x.shape[-1], chunk):
        m, l = online_softmax_update(m, l, x[..., start : start + chunk])
    return (m + l.log()).squeeze(-1)


def flash_attention_forward(q: Tensor, k: Tensor, v: Tensor, block_size: int = 32, causal: bool = False) -> Tensor:
    B, H, T, Dh = q.shape
    S = k.shape[-2]
    scale = 1.0 / math.sqrt(Dh)
    out = torch.empty_like(q)

    for i in range(0, T, block_size):
        qb = q[..., i : i + block_size, :]
        rows = qb.shape[-2]
        q_pos = torch.arange(i, i + rows, device=q.device)[:, None]

        m = torch.full((B, H, rows, 1), float("-inf"), dtype=q.dtype, device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros_like(qb)

        for j in range(0, S, block_size):
            if causal and j > i + rows - 1:
                break  # весь блок ключей лежит в будущем
            kb = k[..., j : j + block_size, :]
            vb = v[..., j : j + block_size, :]
            scores = qb @ kb.transpose(-2, -1) * scale
            if causal:
                k_pos = torch.arange(j, j + kb.shape[-2], device=q.device)[None, :]
                scores = scores.masked_fill(q_pos < k_pos, float("-inf"))

            m_new, l = online_softmax_update(m, l, scores)
            correction = (m - m_new).exp()
            p = (scores - m_new).exp()
            acc = acc * correction + p @ vb
            m = m_new

        out[..., i : i + block_size, :] = acc / l
    return out


def chunked_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor, chunk: int) -> Tensor:
    total = hidden.new_zeros(())
    n = hidden.shape[0]
    for start in range(0, n, chunk):
        h = hidden[start : start + chunk]
        t = targets[start : start + chunk]
        logits = h @ weight.T
        total = total + F.cross_entropy(logits, t, reduction="sum")
    return total / n


def checkpointed_sequential(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool) -> Tensor:
    for block in blocks:
        x = checkpoint(block, x, use_reentrant=False) if use_checkpoint else block(x)
    return x


def count_saved_tensors(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool) -> int:
    count = 0

    def pack(t: Tensor) -> Tensor:
        nonlocal count
        count += 1
        return t

    def unpack(t: Tensor) -> Tensor:
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        checkpointed_sequential(blocks, x, use_checkpoint)
    return count


def scale_and_check(grads: list[Tensor], scale: float) -> tuple[list[Tensor], bool]:
    unscaled = [g / scale for g in grads]
    found_inf = any(not bool(torch.isfinite(g).all()) for g in unscaled)
    return unscaled, found_inf


def next_scale(scale: float, found_inf: bool, growth_interval_reached: bool) -> float:
    if found_inf:
        return scale / 2
    return scale * 2 if growth_interval_reached else scale


def safe_softmax_fp16(logits: Tensor) -> Tensor:
    x = logits.to(torch.float32)
    return (x - x.max(dim=-1, keepdim=True).values).softmax(dim=-1).to(logits.dtype)
