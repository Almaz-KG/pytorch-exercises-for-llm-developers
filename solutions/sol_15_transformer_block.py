"""Эталонные решения к топику 15."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def _heads(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        return x.view(B, T, self.num_heads, D // self.num_heads).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        B, T, D = x.shape
        q, k, v = self._heads(self.q_proj(x)), self._heads(self.k_proj(x)), self._heads(self.v_proj(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        idx = torch.arange(T, device=x.device)
        scores = scores.masked_fill(idx[:, None] < idx[None, :], float("-inf"))
        out = scores.softmax(dim=-1) @ v
        return self.output_proj(out.transpose(1, 2).reshape(B, T, D))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.token_embeddings(ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))


def count_lm_params(vocab_size: int, d_model: int, d_ff: int, num_layers: int, tied_embeddings: bool) -> int:
    embed = vocab_size * d_model
    per_layer = 4 * d_model * d_model + 3 * d_model * d_ff
    head = 0 if tied_embeddings else vocab_size * d_model
    return embed + num_layers * per_layer + head


def activation_memory_bytes(batch: int, seq_len: int, d_model: int, num_layers: int, dtype: torch.dtype) -> int:
    return batch * seq_len * d_model * num_layers * torch.empty(0, dtype=dtype).element_size()
