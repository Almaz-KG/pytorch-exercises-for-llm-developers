"""Reference solutions for topic 17."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    if temperature == 0.0:
        out = torch.full_like(logits, float("-inf"))
        return out.scatter(-1, logits.argmax(dim=-1, keepdim=True), 0.0)
    return logits / temperature


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    if k <= 0:
        return logits
    k = min(k, logits.shape[-1])
    values, idx = torch.topk(logits, k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, idx, values)


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cum_before = probs.cumsum(dim=-1) - probs
    remove_sorted = cum_before >= p
    remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def repetition_penalty(logits: Tensor, generated: Tensor, penalty: float) -> Tensor:
    out = logits.clone()
    picked = out.gather(-1, generated)
    penalised = torch.where(picked > 0, picked / penalty, picked * penalty)
    return out.scatter(-1, generated, penalised)


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    filtered = top_p_filter(top_k_filter(logits / temperature, top_k), top_p)
    probs = filtered.softmax(dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


class KVCache:
    def __init__(self):
        self.k: Tensor | None = None
        self.v: Tensor | None = None

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=-2)
            self.v = torch.cat([self.v, v], dim=-2)
        return self.k, self.v

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.shape[-2]


class CachedSelfAttention(nn.Module):
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

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        B, T, D = x.shape
        past = len(cache) if cache is not None else 0
        q, k, v = self._heads(self.q_proj(x)), self._heads(self.k_proj(x)), self._heads(self.v_proj(x))
        if cache is not None:
            k, v = cache.append(k, v)

        scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        S = k.shape[-2]
        q_pos = torch.arange(past, past + T, device=x.device)[:, None]
        k_pos = torch.arange(S, device=x.device)[None, :]
        scores = scores.masked_fill(q_pos < k_pos, float("-inf"))

        out = scores.softmax(dim=-1) @ v
        return self.output_proj(out.transpose(1, 2).reshape(B, T, D))


def cache_memory_bytes(
    batch: int, num_layers: int, num_kv_heads: int, seq_len: int, head_dim: int, dtype: torch.dtype
) -> int:
    element = torch.empty(0, dtype=dtype).element_size()
    return 2 * batch * num_layers * num_kv_heads * seq_len * head_dim * element


def greedy_generate(step_fn, prompt: Tensor, max_new_tokens: int) -> Tensor:
    ids = prompt
    for _ in range(max_new_tokens):
        logits = step_fn(ids)
        nxt = logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids
