"""Reference solutions for topic 04."""

import torch
from torch import Tensor


def sum_over_time(x: Tensor) -> Tensor:
    return x.sum(dim=1)


def mean_over_batch_and_time(x: Tensor) -> Tensor:
    return x.mean(dim=(0, 1))


def center_and_keepdim(x: Tensor) -> tuple[Tensor, Tensor]:
    mean = x.mean(dim=-1, keepdim=True)
    return mean, x - mean


def biased_variance(x: Tensor) -> Tensor:
    return x.var(dim=-1, keepdim=True, unbiased=False)


def rms(x: Tensor) -> Tensor:
    return x.pow(2).mean(dim=-1, keepdim=True).sqrt()


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    m = mask.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


def count_tokens(mask: Tensor) -> tuple[Tensor, int]:
    lengths = mask.sum(dim=-1).to(torch.int64)
    return lengths, int(lengths.sum().item())


def masked_max(x: Tensor, mask: Tensor) -> Tensor:
    return x.masked_fill(~mask, float("-inf")).max(dim=-1).values


def greedy_tokens(logits: Tensor) -> Tensor:
    return logits.argmax(dim=-1)


def top_k_tokens(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    values, idx = torch.topk(logits, k, dim=-1)
    return values, idx


def argmax_2d(x: Tensor) -> tuple[int, int]:
    flat = int(x.argmax().item())
    row, col = divmod(flat, x.shape[1])
    return row, col


def accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> float:
    preds = logits.argmax(dim=-1)
    valid = targets != ignore_index
    correct = (preds == targets) & valid
    return float(correct.sum().item() / max(int(valid.sum().item()), 1))


def logsumexp_manual(x: Tensor, dim: int) -> Tensor:
    m = x.max(dim=dim, keepdim=True).values
    m = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    return (m + (x - m).exp().sum(dim=dim, keepdim=True).log()).squeeze(dim)


def entropy(probs: Tensor) -> Tensor:
    logp = torch.where(probs > 0, probs.log(), torch.zeros_like(probs))
    return -(probs * logp).sum(dim=-1)


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp(min=eps)


def global_grad_norm(tensors: list[Tensor]) -> Tensor:
    return torch.sqrt(sum(t.pow(2).sum() for t in tensors))


def token_frequencies(ids: Tensor, vocab_size: int) -> Tensor:
    return torch.bincount(ids.flatten(), minlength=vocab_size)


def cumulative_mean(x: Tensor) -> Tensor:
    counts = torch.arange(1, x.numel() + 1, dtype=x.dtype, device=x.device)
    return x.cumsum(dim=0) / counts
