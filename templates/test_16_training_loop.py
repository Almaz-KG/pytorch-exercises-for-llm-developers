"""Topic 16 - the training loop: AdamW, the lr schedule, clipping, gradient accumulation.

This is where the "tensor exercises" end and the engineering of training begins.
Everything below is what you will be writing by hand in assignment 1 of CS336.

AdamW step by step (worth being able to reproduce from memory):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2
    m_hat = m / (1 - b1^t)
    v_hat = v / (1 - b2^t)
    p = p - lr * m_hat / (sqrt(v_hat) + eps)
    p = p - lr * wd * p      <- decoupled weight decay, not through the gradient
"""

import itertools
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 16.1 Optimization steps by hand
# ==========================================================================


def sgd_step(params: list[Tensor], lr: float) -> None:
    """Update the parameters in place: p -= lr * p.grad.

    Necessarily under torch.no_grad(), otherwise the update lands in the graph.
    """
    raise NotImplementedError


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
    """One AdamW step for a single tensor. Return (the new p, the new m, the new v).

    t is the step number, starting from 1 (needed for the bias correction).
    Change nothing in place: return new tensors, that way it is easier to check.
    """
    raise NotImplementedError


class AdamW(torch.optim.Optimizer):
    """A full optimizer in the PyTorch interface.

    The requirements:
      - __init__(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01);
      - the state (m, v, the number of steps) is kept in self.state[p];
      - step() updates the parameters and returns the loss if a closure is passed;
      - the result is obliged to match torch.optim.AdamW numerically.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        raise NotImplementedError

    def step(self, closure=None):
        raise NotImplementedError


# ==========================================================================
# 16.2 The learning rate schedule
# ==========================================================================


def cosine_lr(step: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int) -> float:
    """The schedule from the CS336 assignment:

      step < warmup_steps:                    a linear warmup from 0 to lr_max
      warmup_steps <= step <= cosine_steps:   a cosine decay from lr_max to lr_min
      step > cosine_steps:                    the constant lr_min

    The cosine: lr_min + 0.5 * (1 + cos(pi * (step - warmup) / (cosine_steps - warmup))) * (lr_max - lr_min)
    """
    raise NotImplementedError


# ==========================================================================
# 16.3 Gradient clipping
# ==========================================================================


def clip_grad_norm(params: list[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Clip the gradients by their global L2 norm and return the norm BEFORE the clipping (a python float).

    If the norm is greater than max_norm, all the gradients are multiplied by max_norm / (norm + eps).
    Implement it yourself, without torch.nn.utils.clip_grad_norm_.
    """
    raise NotImplementedError


# ==========================================================================
# 16.4 Gradient accumulation
# ==========================================================================


def accumulate_gradients(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    """Run several microbatches, accumulating the gradient so that the result matches
    one big batch of the same total length. Return the total (averaged) loss.

    The key detail: the loss of every microbatch has to be divided by the number of microbatches,
    otherwise the gradient comes out N times too large. Everyone stumbles on this.

    The loss is the MSE between model(x) and y. Zero the gradients at the start.
    """
    raise NotImplementedError


def train_steps(model: nn.Module, x: Tensor, y: Tensor, steps: int, lr: float) -> list[float]:
    """The simplest training loop on your own AdamW: return the list of losses per step.

    The order of the operations inside a step is part of the muscle memory too:
    forward -> loss -> zero_grad -> backward -> clip -> step.
    """
    raise NotImplementedError


# ==========================================================================
# 16.5 The arithmetic of training
# ==========================================================================


def training_flops(num_params: int, num_tokens: int) -> int:
    """The 6ND rule: the forward pass is roughly 2ND, the backward pass roughly 4ND."""
    raise NotImplementedError


def chinchilla_tokens(num_params: int) -> int:
    """The compute-optimal number of tokens by Chinchilla: roughly 20 tokens per parameter."""
    raise NotImplementedError


def tokens_per_step(batch_size: int, seq_len: int, grad_accum: int) -> int:
    """How many tokens the model actually sees per optimizer step."""
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_sgd_step():
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([0.5, -0.5])
    sgd_step([p], lr=0.1)
    torch.testing.assert_close(p.detach(), torch.tensor([0.95, 2.05]))
    assert p.requires_grad and p.is_leaf, "the parameter must stay a leaf of the graph"


def test_adamw_step_matches_torch():
    torch.manual_seed(0)
    p_ref = torch.randn(5, requires_grad=True)
    p_mine = p_ref.detach().clone()
    grads = [torch.randn(5) for _ in range(4)]

    opt = torch.optim.AdamW([p_ref], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    m = torch.zeros(5)
    v = torch.zeros(5)
    for t, g in enumerate(grads, start=1):
        p_ref.grad = g.clone()
        opt.step()
        p_mine, m, v = adamw_step(p_mine, g, m, v, t, lr=0.1)
        torch.testing.assert_close(p_mine, p_ref.detach(), atol=1e-6, rtol=1e-5)


def test_adamw_weight_decay_is_decoupled():
    """Even with a zero gradient the weight is obliged to shrink - that is what decoupled means."""
    p = torch.ones(3)
    out, _, _ = adamw_step(p, torch.zeros(3), torch.zeros(3), torch.zeros(3), t=1, lr=0.1, weight_decay=0.1)
    torch.testing.assert_close(out, torch.full((3,), 1.0 - 0.1 * 0.1), atol=1e-6, rtol=0)


def test_adamw_optimizer_matches_torch():
    torch.manual_seed(0)
    mine = nn.Linear(4, 3)
    ref = nn.Linear(4, 3)
    ref.load_state_dict(mine.state_dict())

    opt_mine = AdamW(mine.parameters(), lr=0.05, weight_decay=0.05)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=0.05, weight_decay=0.05)

    x = torch.randn(16, 4)
    y = torch.randn(16, 3)
    for _ in range(8):
        for model, opt in ((mine, opt_mine), (ref, opt_ref)):
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.testing.assert_close(mine.weight, ref.weight, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(mine.bias, ref.bias, atol=1e-6, rtol=1e-5)


def test_cosine_lr_schedule():
    lr_max, lr_min, warmup, cosine = 1.0, 0.1, 10, 110
    assert cosine_lr(0, lr_max, lr_min, warmup, cosine) == 0.0
    assert abs(cosine_lr(5, lr_max, lr_min, warmup, cosine) - 0.5) < 1e-9
    assert abs(cosine_lr(10, lr_max, lr_min, warmup, cosine) - lr_max) < 1e-9
    # the middle of the cosine
    assert abs(cosine_lr(60, lr_max, lr_min, warmup, cosine) - (lr_min + 0.5 * (lr_max - lr_min))) < 1e-9
    assert abs(cosine_lr(110, lr_max, lr_min, warmup, cosine) - lr_min) < 1e-9
    assert cosine_lr(500, lr_max, lr_min, warmup, cosine) == lr_min
    # a monotone decrease on the cosine stretch
    values = [cosine_lr(s, lr_max, lr_min, warmup, cosine) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(values))


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    mine = [torch.randn(4, 3), torch.randn(3)]
    for p in mine:
        p.grad = torch.randn_like(p) * 10
    ref = [p.detach().clone() for p in mine]
    for p, q in zip(ref, mine, strict=True):
        p.grad = q.grad.clone()

    returned = clip_grad_norm(mine, 1.0)
    expected = torch.nn.utils.clip_grad_norm_(ref, 1.0).item()
    assert abs(returned - expected) < 1e-4
    for p, q in zip(mine, ref, strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)

    total = torch.sqrt(sum(p.grad.pow(2).sum() for p in mine)).item()
    assert total <= 1.0 + 1e-4


def test_clip_grad_norm_leaves_small_gradients_alone():
    p = torch.zeros(3)
    p.grad = torch.tensor([0.1, 0.0, 0.0])
    before = p.grad.clone()
    norm = clip_grad_norm([p], 1.0)
    assert abs(norm - 0.1) < 1e-6
    torch.testing.assert_close(p.grad, before)


def test_gradient_accumulation_equals_big_batch():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    loss = F.mse_loss(model(x), y)
    model.zero_grad()
    loss.backward()
    full_grads = [p.grad.clone() for p in model.parameters()]

    micro = [(x[i : i + 4], y[i : i + 4]) for i in range(0, 16, 4)]
    accum_loss = accumulate_gradients(model, micro)
    assert abs(accum_loss - loss.item()) < 1e-5
    for p, g in zip(model.parameters(), full_grads, strict=True):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-5)


def test_train_steps_reduces_loss():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    x = torch.randn(32, 4)
    y = x @ torch.randn(4, 1) + 0.1
    losses = train_steps(model, x, y, steps=50, lr=0.05)
    assert len(losses) == 50
    assert losses[-1] < 0.2 * losses[0], f"the loss barely fell: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_training_flops():
    # 7B parameters, 1.4T tokens - roughly LLaMA-7B
    assert training_flops(7e9, 1.4e12) == 6 * 7e9 * 1.4e12
    assert training_flops(1, 1) == 6


def test_chinchilla_tokens():
    assert chinchilla_tokens(7_000_000_000) == 140_000_000_000
    assert chinchilla_tokens(1_000_000) == 20_000_000


def test_tokens_per_step():
    assert tokens_per_step(8, 1024, 4) == 32768
    assert tokens_per_step(8, 1024, 1) == 8192


@pytest.mark.parametrize("warmup,cosine", [(0, 100), (10, 100), (50, 50)])
def test_cosine_lr_bounds(warmup, cosine):
    for step in range(cosine + 20):
        lr = cosine_lr(step, 1.0, 0.1, warmup, cosine)
        assert 0.0 <= lr <= 1.0 + 1e-12
        assert not math.isnan(lr)
