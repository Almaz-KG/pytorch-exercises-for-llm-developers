"""Reference solutions for topic 15."""

from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor


def grad_of_sum_of_squares(x: Tensor) -> Tensor:
    x = x.detach().clone().requires_grad_(True)
    (x**2).sum().backward()
    assert x.grad is not None
    return x.grad


def grad_functional(x: Tensor) -> Tensor:
    x = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad((x**2).sum(), x)
    return grad


def grads_accumulate(x: Tensor) -> tuple[Tensor, Tensor]:
    y = (x**2).sum()
    y.backward(retain_graph=True)
    assert x.grad is not None
    first = x.grad.clone()
    y.backward()
    return first, x.grad.clone()


def stop_gradient(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor | None]:
    (a * b.detach()).sum().backward()
    assert a.grad is not None
    return a.grad, b.grad


def forward_without_graph(x: Tensor, w: Tensor) -> Tensor:
    with torch.no_grad():
        return x @ w


def second_derivative(x: Tensor) -> Tensor:
    x = x.detach().clone().requires_grad_(True)
    (first,) = torch.autograd.grad((x**3).sum(), x, create_graph=True)
    (second,) = torch.autograd.grad(first.sum(), x)
    return second


def vjp(f: Callable[[Tensor], Tensor], x: Tensor, v: Tensor) -> Tensor:
    x = x.detach().clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(f(x), x, grad_outputs=v)
    return grad


def numeric_grad(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-4) -> Tensor:
    grad = torch.zeros_like(x)
    flat_x, flat_g = x.reshape(-1), grad.reshape(-1)
    for i in range(flat_x.numel()):
        original = flat_x[i].item()
        flat_x[i] = original + eps
        plus = f(x).item()
        flat_x[i] = original - eps
        minus = f(x).item()
        flat_x[i] = original
        flat_g[i] = (plus - minus) / (2 * eps)
    return grad


def softmax_vjp(probs: Tensor, grad_out: Tensor) -> Tensor:
    return probs * (grad_out - (grad_out * probs).sum(dim=-1, keepdim=True))


class SquareFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        ctx.save_for_backward(x)
        return x * x

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        (x,) = ctx.saved_tensors
        return 2 * x * grad_output


class STEQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        return grad_output


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
        r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        ctx.save_for_backward(x, weight, r)
        return x * r * weight

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        x, weight, r = ctx.saved_tensors
        d = x.shape[-1]
        a = grad_out * weight
        grad_x = a * r - x * r.pow(3) * (a * x).sum(dim=-1, keepdim=True) / d
        grad_w = (grad_out * x * r).reshape(-1, d).sum(dim=0)
        return grad_x, grad_w, None


def embedding_grad(table_size: tuple[int, int], ids: Tensor) -> Tensor:
    table = torch.randn(*table_size, requires_grad=True)
    table[ids].sum().backward()
    assert table.grad is not None
    return table.grad


def freeze_all_but_last(model: nn.Sequential) -> int:
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model[-1].parameters():
        p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def grad_norm_after_backward(model: nn.Module, x: Tensor, target: Tensor) -> float:
    loss = torch.nn.functional.mse_loss(model(x), target)
    model.zero_grad(set_to_none=True)
    loss.backward()
    total = torch.stack([p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None]).sum()
    return float(total.sqrt().item())
