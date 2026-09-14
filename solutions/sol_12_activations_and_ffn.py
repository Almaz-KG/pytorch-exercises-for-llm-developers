"""Reference solutions for topic 12."""

import math

import torch
import torch.nn as nn
from torch import Tensor


def relu(x: Tensor) -> Tensor:
    return torch.maximum(x, torch.zeros_like(x))


def gelu_exact(x: Tensor) -> Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_tanh(x: Tensor) -> Tensor:
    inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))
    return 0.5 * x * (1.0 + torch.tanh(inner))


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def silu_grad(x: Tensor) -> Tensor:
    s = torch.sigmoid(x)
    return s * (1 + x * (1 - s))


def ffn_relu(x: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    return relu(x @ w1.T) @ w2.T


def swiglu(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    return (silu(x @ w1.T) * (x @ w3.T)) @ w2.T


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


def swiglu_d_ff(d_model: int, multiple_of: int = 64) -> int:
    raw = 8 * d_model / 3
    return int(math.ceil(raw / multiple_of) * multiple_of)


def ffn_params(d_model: int, d_ff: int, gated: bool) -> int:
    return (3 if gated else 2) * d_model * d_ff


def ffn_flops(batch_tokens: int, d_model: int, d_ff: int, gated: bool) -> int:
    per_matmul = 2 * batch_tokens * d_model * d_ff
    return (3 if gated else 2) * per_matmul
