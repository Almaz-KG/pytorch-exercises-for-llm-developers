"""Reference solutions for topic 16."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def sgd_step(params: list[Tensor], lr: float) -> None:
    with torch.no_grad():
        for p in params:
            if p.grad is not None:
                p -= lr * p.grad


def adamw_step(
    p: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    t: int,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    b1, b2 = betas
    m = b1 * m + (1 - b1) * grad
    v = b2 * v + (1 - b2) * grad.pow(2)
    m_hat = m / (1 - b1**t)
    v_hat = v / (1 - b2**t)
    p = p - lr * weight_decay * p
    p = p - lr * m_hat / (v_hat.sqrt() + eps)
    return p, m, v


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                state["step"] += 1
                t = state["step"]
                m, v = state["m"], state["v"]
                g = p.grad
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                m_hat = m / (1 - b1**t)
                v_hat = v / (1 - b2**t)
                p.mul_(1 - lr * wd)
                p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)
        return loss


def cosine_lr(step: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int) -> float:
    if step < warmup_steps:
        return lr_max * step / warmup_steps
    if step <= cosine_steps:
        progress = (step - warmup_steps) / max(cosine_steps - warmup_steps, 1)
        return lr_min + 0.5 * (1 + math.cos(math.pi * progress)) * (lr_max - lr_min)
    return lr_min


def clip_grad_norm(params: list[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    grads = [p.grad for p in params if p.grad is not None]
    total = torch.sqrt(sum(g.pow(2).sum() for g in grads))
    if total.item() > max_norm:
        scale = max_norm / (total + eps)
        for g in grads:
            g.mul_(scale)
    return float(total.item())


def accumulate_gradients(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    model.zero_grad(set_to_none=True)
    total = 0.0
    for x, y in batches:
        loss = F.mse_loss(model(x), y) / len(batches)
        loss.backward()
        total += loss.item()
    return total


def train_steps(model: nn.Module, x: Tensor, y: Tensor, steps: int, lr: float) -> list[float]:
    opt = AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        loss = F.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_norm(list(model.parameters()), 1.0)
        opt.step()
        losses.append(loss.item())
    return losses


def training_flops(num_params: int, num_tokens: int) -> int:
    return 6 * num_params * num_tokens


def chinchilla_tokens(num_params: int) -> int:
    return 20 * num_params


def tokens_per_step(batch_size: int, seq_len: int, grad_accum: int) -> int:
    return batch_size * seq_len * grad_accum
