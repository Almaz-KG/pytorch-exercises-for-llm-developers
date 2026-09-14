"""Эталонные решения к топику 01."""

import torch
from torch import Tensor


def make_arange_matrix(n: int, m: int) -> Tensor:
    return torch.arange(n * m).reshape(n, m)


def make_constant(shape: tuple[int, ...], value: float, dtype: torch.dtype) -> Tensor:
    return torch.full(shape, value, dtype=dtype)


def make_identity_like(x: Tensor) -> Tensor:
    n = x.shape[0]
    return torch.eye(n, dtype=x.dtype, device=x.device)


def describe(x: Tensor) -> dict:
    return {
        "shape": tuple(x.shape),
        "ndim": x.ndim,
        "numel": x.numel(),
        "dtype": x.dtype,
        "bytes": x.numel() * x.element_size(),
    }


def promoted_dtype(a: Tensor, b: Tensor) -> torch.dtype:
    return torch.promote_types(a.dtype, b.dtype)


def linear_combination(a: Tensor, b: Tensor, alpha: float, beta: float) -> Tensor:
    return alpha * a + beta * b


def to_float32(x: Tensor) -> Tensor:
    return x.to(torch.float32)  # .to() возвращает self, если dtype уже совпадает


def mean_of_ints(x: Tensor) -> Tensor:
    return x.to(torch.float32).mean()


def independent_copy(x: Tensor) -> Tensor:
    return x.clone()


def shares_memory(a: Tensor, b: Tensor) -> bool:
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def add_inplace(x: Tensor, value: float) -> Tensor:
    x.add_(value)
    return x


def zero_row_(x: Tensor, i: int) -> Tensor:
    x[i].zero_()
    return x


def count_greater(x: Tensor, threshold: float) -> int:
    return int((x > threshold).sum().item())


def clamp_and_scale(x: Tensor, lo: float, hi: float) -> Tensor:
    return (x.clamp(lo, hi) - lo) / (hi - lo)


def model_memory_bytes(num_params: int, dtype: torch.dtype) -> int:
    return num_params * torch.empty(0, dtype=dtype).element_size()
