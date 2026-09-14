"""Reference solutions for topic 14."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    B, T, D = x.shape
    return x.view(B, T, num_heads, D // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * Dh)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    return attention_weights(q, k, mask) @ v


def attention_weights(q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return scores.softmax(dim=-1)


def _causal_mask(t: int, s: int, device) -> Tensor:
    q_idx = torch.arange(t, device=device)[:, None]
    k_idx = torch.arange(s, device=device)[None, :]
    return q_idx >= k_idx


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    mask = _causal_mask(q.shape[-2], k.shape[-2], q.device)
    return scaled_dot_product_attention(q, k, v, mask)


def attention_with_padding(q: Tensor, k: Tensor, v: Tensor, pad_mask: Tensor) -> Tensor:
    causal = _causal_mask(q.shape[-2], k.shape[-2], q.device)
    mask = causal[None, None, :, :] & pad_mask[:, None, None, :]
    return scaled_dot_product_attention(q, k, v, mask)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = split_heads(self.q_proj(x), self.num_heads)
        k = split_heads(self.k_proj(x), self.num_heads)
        v = split_heads(self.v_proj(x), self.num_heads)
        return self.output_proj(merge_heads(causal_attention(q, k, v)))


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        groups = self.num_heads // self.num_kv_heads
        q = split_heads(self.q_proj(x), self.num_heads)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)
        return self.output_proj(merge_heads(causal_attention(q, k, v)))


def attention_matrix_bytes(batch: int, heads: int, seq_len: int, dtype: torch.dtype) -> int:
    return batch * heads * seq_len * seq_len * torch.empty(0, dtype=dtype).element_size()


def attention_flops(batch: int, heads: int, seq_len: int, d_head: int) -> int:
    qk = 2 * seq_len * seq_len * d_head
    av = 2 * seq_len * d_head * seq_len
    return batch * heads * (qk + av)
