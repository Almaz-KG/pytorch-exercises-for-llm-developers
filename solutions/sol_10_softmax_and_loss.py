"""Reference solutions for topic 10."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def softmax_manual(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    e = shifted.exp()
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax_manual(x: Tensor, dim: int) -> Tensor:
    return x - torch.logsumexp(x, dim=dim, keepdim=True)


def temperature_softmax(logits: Tensor, temperature: float) -> Tensor:
    return softmax_manual(logits / temperature, dim=-1)


def cross_entropy_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - target_logits).mean()


def sequence_cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    safe_targets = flat_targets.clamp(min=0)
    per_token = torch.logsumexp(flat_logits, dim=-1) - flat_logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (per_token * valid).sum() / valid.sum()


def label_smoothing_loss(logits: Tensor, targets: Tensor, smoothing: float) -> Tensor:
    logp = log_softmax_manual(logits, dim=-1)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()


def perplexity(loss: Tensor) -> Tensor:
    return loss.exp()


def kl_divergence(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    log_p = log_softmax_manual(teacher_logits, dim=-1)
    log_q = log_softmax_manual(student_logits, dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean()


def cross_entropy_grad(logits: Tensor, targets: Tensor) -> Tensor:
    probs = softmax_manual(logits, dim=-1)
    onehot = F.one_hot(targets, logits.shape[-1]).to(probs.dtype)
    return (probs - onehot) / logits.shape[0]


def shift_invariance_gap(x: Tensor, shift: float) -> Tensor:
    return (softmax_manual(x, -1) - softmax_manual(x + shift, -1)).abs().max()


def uniform_loss(vocab_size: int) -> float:
    return math.log(vocab_size)
