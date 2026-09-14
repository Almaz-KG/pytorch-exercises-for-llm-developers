"""Reference solutions for topic 02."""

import torch
import torch.nn.functional as F
from torch import Tensor


def flatten_tokens(x: Tensor) -> Tensor:
    B, T, D = x.shape
    return x.reshape(B * T, D)


def unflatten_tokens(x: Tensor, batch_size: int) -> Tensor:
    return x.reshape(batch_size, -1, x.shape[-1])


def flatten_all_but_last(x: Tensor) -> Tensor:
    return x.reshape(-1, x.shape[-1])


def add_batch_dim(x: Tensor) -> Tensor:
    return x.unsqueeze(0)


def drop_singleton_dims(x: Tensor) -> Tensor:
    return x.squeeze()


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    B, T, D = x.shape
    return x.view(B, T, num_heads, D // num_heads).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).reshape(B, T, H * Dh)


def transpose_last_two(x: Tensor) -> Tensor:
    return x.transpose(-2, -1)


def is_contiguous_after_transpose(x: Tensor) -> tuple[bool, bool]:
    return x.is_contiguous(), x.transpose(0, 1).is_contiguous()


def strides_of(x: Tensor) -> tuple[int, ...]:
    return tuple(x.stride())


def concat_sequences(a: Tensor, b: Tensor) -> Tensor:
    return torch.cat([a, b], dim=1)


def stack_batch(items: list[Tensor]) -> Tensor:
    return torch.stack(items, dim=0)


def split_qkv(qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    q, k, v = qkv.chunk(3, dim=-1)
    return q, k, v


def broadcast_kv_heads(kv: Tensor, num_groups: int) -> Tensor:
    B, H_kv, T, Dh = kv.shape
    return kv.unsqueeze(2).expand(B, H_kv, num_groups, T, Dh).reshape(B, H_kv * num_groups, T, Dh)


def expand_row(x: Tensor, n: int) -> Tensor:
    return x.expand(n, -1)


def repeat_row(x: Tensor, n: int) -> Tensor:
    return x.repeat(n, 1)


def sliding_windows(x: Tensor, window: int) -> Tensor:
    return x.unfold(0, window, 1)


def pad_to_length(x: Tensor, target_len: int, pad_value: int) -> Tensor:
    return F.pad(x, (0, target_len - x.shape[-1]), value=pad_value)


def shift_right(x: Tensor, pad_value: int) -> Tensor:
    return F.pad(x[:, :-1], (1, 0), value=pad_value)
