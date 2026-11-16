"""Reference solutions for topic 10."""

import torch
import torch.nn as nn
from torch import Tensor


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * weight


def rms_norm_upcast(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    in_dtype = x.dtype
    out = rms_norm(x.to(torch.float32), weight.to(torch.float32), eps)
    return out.to(in_dtype)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)


class LayerNormModule(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        return layer_norm(x, self.weight, self.bias, self.eps)


def post_norm_residual(x: Tensor, sublayer_out: Tensor, norm: nn.Module) -> Tensor:
    return norm(x + sublayer_out)


def pre_norm_residual(x: Tensor, sublayer: nn.Module, norm: nn.Module) -> Tensor:
    return x + sublayer(norm(x))


def scale_invariance_gap(x: Tensor, c: float) -> Tensor:
    w = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
    return (rms_norm(x, w) - rms_norm(c * x, w)).abs().max()


def stats_after_layernorm(x: Tensor) -> tuple[Tensor, Tensor]:
    d = x.shape[-1]
    out = layer_norm(x, torch.ones(d), torch.zeros(d))
    return out.mean(dim=-1), out.std(dim=-1, unbiased=False)
