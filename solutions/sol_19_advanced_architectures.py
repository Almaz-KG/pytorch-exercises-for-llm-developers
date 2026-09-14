"""Reference solutions for topic 19."""

import torch
import torch.nn.functional as F
from torch import Tensor


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
        r = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
        ctx.save_for_backward(x, weight, r)
        return x * r * weight

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, weight, r = ctx.saved_tensors
        d = x.shape[-1]
        a = grad_out * weight
        grad_x = a * r - x * r.pow(3) * (a * x).sum(dim=-1, keepdim=True) / d
        grad_w = (grad_out * x * r).reshape(-1, d).sum(dim=0)
        return grad_x, grad_w, None


def sliding_window_mask(seq_len: int, window: int) -> Tensor:
    i = torch.arange(seq_len)[:, None]
    j = torch.arange(seq_len)[None, :]
    return (j <= i) & (j > i - window)


def document_mask(doc_ids: Tensor) -> Tensor:
    t = doc_ids.shape[0]
    idx = torch.arange(t, device=doc_ids.device)
    causal = idx[:, None] >= idx[None, :]
    same_doc = doc_ids[:, None] == doc_ids[None, :]
    return causal & same_doc


def block_diagonal_lengths_to_doc_ids(lengths: list[int]) -> Tensor:
    return torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))


def top2_routing(router_logits: Tensor) -> tuple[Tensor, Tensor]:
    values, idx = torch.topk(router_logits, 2, dim=-1)
    return idx, values.softmax(dim=-1)


def capacity_mask(expert_idx: Tensor, num_experts: int, capacity: int) -> Tensor:
    n, k = expert_idx.shape
    # the order in which slots are taken: all of slot 0 across every token first, then slot 1
    order = expert_idx.T.reshape(-1)  # (K*N,)
    onehot = F.one_hot(order, num_experts)
    position = onehot.cumsum(dim=0) - onehot  # how many tokens this expert already has
    rank = (position * onehot).sum(dim=-1)
    keep = rank < capacity
    return keep.reshape(k, n).T


def load_balancing_loss(router_probs: Tensor, top1_idx: Tensor, num_experts: int) -> Tensor:
    f = torch.bincount(top1_idx, minlength=num_experts).to(router_probs.dtype) / top1_idx.numel()
    p = router_probs.mean(dim=0)
    return num_experts * (f * p).sum()


def moe_forward(x: Tensor, router: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    idx, weights = top2_routing(x @ router.T)
    out = torch.zeros_like(x)
    for e in range(router.shape[0]):
        hit = idx == e  # (N, K)
        token_idx, slot_idx = hit.nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        xe = x[token_idx]
        he = F.relu(xe @ w1[e].T) @ w2[e].T
        out.index_add_(0, token_idx, he * weights[token_idx, slot_idx].unsqueeze(-1))
    return out


def speculative_accept(
    draft_probs: Tensor, target_probs: Tensor, draft_tokens: Tensor, u: Tensor
) -> tuple[int, Tensor]:
    k = draft_tokens.shape[0]
    for i in range(k):
        token = int(draft_tokens[i])
        q = draft_probs[i, token]
        p = target_probs[i, token]
        if u[i] > (p / q).clamp(max=1.0):
            residual = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            return i, residual / residual.sum()
    return k, target_probs[k]


def expected_speedup(acceptance_rate: float, k: int) -> float:
    if acceptance_rate == 1.0:
        return float(k + 1)
    return (1 - acceptance_rate ** (k + 1)) / (1 - acceptance_rate)


def beam_search_step(beam_scores: Tensor, next_log_probs: Tensor, beam_size: int) -> tuple[Tensor, Tensor, Tensor]:
    vocab = next_log_probs.shape[-1]
    total = (beam_scores[:, None] + next_log_probs).reshape(-1)
    scores, flat_idx = torch.topk(total, beam_size)
    return scores, flat_idx // vocab, flat_idx % vocab
