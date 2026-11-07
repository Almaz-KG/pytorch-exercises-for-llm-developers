"""Reference solutions for topic 13."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def scaled_token_embedding(table: Tensor, ids: Tensor) -> Tensor:
    return table[ids] * math.sqrt(table.shape[-1])


def learned_positional(embed: Tensor, pos_table: Tensor, offset: int = 0) -> Tensor:
    T = embed.shape[-2]
    return embed + pos_table[offset : offset + T]


def sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    inv_freq = torch.pow(10000.0, -i / d_model)
    angles = pos * inv_freq[None, :]
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles.cos()
    return pe


def rope_inv_freq(d_head: int, theta: float = 10000.0) -> Tensor:
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    return torch.pow(theta, -i / d_head)


def rope_cos_sin(seq_len: int, d_head: int, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)[:, None]
    angles = pos * rope_inv_freq(d_head, theta)[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1)
    return out.flatten(start_dim=-2)


def apply_rope_to_qk(q: Tensor, k: Tensor, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    cos, sin = rope_cos_sin(q.shape[-2], q.shape[-1], theta, offset)
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def rope_dot_product(q: Tensor, k: Tensor, pos_q: int, pos_k: int, theta: float = 10000.0) -> Tensor:
    d = q.shape[-1]
    cq, sq = rope_cos_sin(1, d, theta, offset=pos_q)
    ck, sk = rope_cos_sin(1, d, theta, offset=pos_k)
    qr = apply_rope(q[None, :], cq, sq)[0]
    kr = apply_rope(k[None, :], ck, sk)[0]
    return (qr * kr).sum()


def rope_norm_change(x: Tensor, pos: int, theta: float = 10000.0) -> Tensor:
    cos, sin = rope_cos_sin(1, x.shape[-1], theta, offset=pos)
    rotated = apply_rope(x[None, :], cos, sin)[0]
    return rotated.norm() - x.norm()


class RotaryEmbedding(nn.Module):
    cos_cached: Tensor
    sin_cached: Tensor

    def __init__(self, d_head: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        cos, sin = rope_cos_sin(max_len, d_head, theta)
        self.register_buffer("cos_cached", cos, persistent=True)
        self.register_buffer("sin_cached", sin, persistent=True)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos_cached[offset : offset + T].to(x.dtype)
        sin = self.sin_cached[offset : offset + T].to(x.dtype)
        return apply_rope(x, cos, sin)
