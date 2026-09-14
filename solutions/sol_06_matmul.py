"""Эталонные решения к топику 06."""

import torch
from torch import Tensor


def matvec(A: Tensor, x: Tensor) -> Tensor:
    return A @ x


def matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    if a[-1] != b[-2]:
        raise ValueError(f"внутренние размерности не совпадают: {a[-1]} vs {b[-2]}")
    lead_a, lead_b = a[:-2], b[:-2]
    lead: list[int] = []
    for i in range(1, max(len(lead_a), len(lead_b)) + 1):
        da = lead_a[-i] if i <= len(lead_a) else 1
        db = lead_b[-i] if i <= len(lead_b) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(f"ведущие оси не broadcast-ятся: {a} vs {b}")
        lead.append(max(da, db))
    return tuple(reversed(lead)) + (a[-2], b[-1])


def batched_matmul(a: Tensor, b: Tensor) -> Tensor:
    return a @ b


def gram_matrix(x: Tensor) -> Tensor:
    return x @ x.T


def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    out = x @ weight.T
    return out if bias is None else out + bias


def linear_backward(grad_out: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    grad_x = grad_out @ weight
    grad_weight = grad_out.T @ x
    grad_bias = grad_out.sum(dim=0)
    return grad_x, grad_weight, grad_bias


def lora_delta(x: Tensor, A: Tensor, B: Tensor, alpha: float, rank: int) -> Tensor:
    # (x @ A) @ B: промежуточный тензор (N, r) вместо матрицы (in, out)
    return (x @ A) @ B * (alpha / rank)


def attention_scores(q: Tensor, k: Tensor) -> Tensor:
    return q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)


def attention_output(weights: Tensor, v: Tensor) -> Tensor:
    return weights @ v


def matmul_flops(m: int, k: int, n: int) -> int:
    return 2 * m * n * k


def cheaper_order(m: int, k: int, n: int, p: int) -> str:
    left = matmul_flops(m, k, n) + matmul_flops(m, n, p)
    right = matmul_flops(k, n, p) + matmul_flops(m, k, p)
    if left < right:
        return "left"
    if right < left:
        return "right"
    return "equal"


def transformer_block_params(d_model: int, d_ff: int, num_heads: int) -> int:
    return 4 * d_model * d_model + 2 * d_model * d_ff


def matmul_accumulate_fp32(a: Tensor, b: Tensor) -> Tensor:
    return a.to(torch.float32) @ b.to(torch.float32)
