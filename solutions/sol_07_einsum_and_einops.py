"""Reference solutions for topic 07."""

import torch
from einops import rearrange, reduce, repeat
from torch import Tensor


def es_trace(x: Tensor) -> Tensor:
    return torch.einsum("ii->", x)


def es_diagonal(x: Tensor) -> Tensor:
    return torch.einsum("ii->i", x)


def es_transpose(x: Tensor) -> Tensor:
    return torch.einsum("ij->ji", x)


def es_matmul(a: Tensor, b: Tensor) -> Tensor:
    return torch.einsum("mk,kn->mn", a, b)


def es_outer(a: Tensor, b: Tensor) -> Tensor:
    return torch.einsum("n,m->nm", a, b)


def es_row_sums(x: Tensor) -> Tensor:
    return torch.einsum("ij->i", x)


def es_attention_scores(q: Tensor, k: Tensor) -> Tensor:
    return torch.einsum("bhtd,bhsd->bhts", q, k)


def es_attention_output(weights: Tensor, v: Tensor) -> Tensor:
    return torch.einsum("bhts,bhsd->bhtd", weights, v)


def es_per_head_projection(x: Tensor, w: Tensor) -> Tensor:
    return torch.einsum("btd,hde->bhte", x, w)


def es_weight_grad(x: Tensor, grad_out: Tensor) -> Tensor:
    return torch.einsum("ni,no->oi", x, grad_out)


def es_batched_bilinear(x: Tensor, W: Tensor, y: Tensor) -> Tensor:
    return torch.einsum("bi,ij,bj->b", x, W, y)


def eo_split_heads(x: Tensor, num_heads: int) -> Tensor:
    return rearrange(x, "b t (h d) -> b h t d", h=num_heads)


def eo_merge_heads(x: Tensor) -> Tensor:
    return rearrange(x, "b h t d -> b t (h d)")


def eo_flatten_tokens(x: Tensor) -> Tensor:
    return rearrange(x, "b t d -> (b t) d")


def eo_split_qkv(qkv: Tensor, num_heads: int) -> Tensor:
    return rearrange(qkv, "b t (three h d) -> three b h t d", three=3, h=num_heads)


def eo_group_tokens(x: Tensor, group: int) -> Tensor:
    return rearrange(x, "b (n g) d -> b n g d", g=group)


def eo_mean_pool(x: Tensor) -> Tensor:
    return reduce(x, "b t d -> b d", "mean")


def eo_max_pool_heads(x: Tensor) -> Tensor:
    return reduce(x, "b h t d -> b t d", "max")


def eo_repeat_kv(kv: Tensor, num_groups: int) -> Tensor:
    return repeat(kv, "b h t d -> b (h g) t d", g=num_groups)


def eo_broadcast_mask(mask: Tensor, num_heads: int) -> Tensor:
    return repeat(mask, "b t s -> b h t s", h=num_heads)
