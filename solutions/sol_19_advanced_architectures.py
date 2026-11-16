"""Reference solutions for topic 19."""

import math

import torch
import torch.nn.functional as F
from einops import einsum
from torch import Tensor


def mla_expand_kv(c_kv: Tensor, w_uk: Tensor, w_uv: Tensor) -> tuple[Tensor, Tensor]:
    k = einsum(c_kv, w_uk, "b s c, h d c -> b h s d")
    v = einsum(c_kv, w_uv, "b s c, h d c -> b h s d")
    return k, v


def mla_absorbed_scores(q: Tensor, c_kv: Tensor, w_uk: Tensor) -> Tensor:
    q_latent = einsum(q, w_uk, "b h t d, h d c -> b h t c")
    return einsum(q_latent, c_kv, "b h t c, b s c -> b h t s") / math.sqrt(q.shape[-1])


def mla_absorbed_values(probs: Tensor, c_kv: Tensor, w_uv: Tensor) -> Tensor:
    ctx_latent = einsum(probs, c_kv, "b h t s, b s c -> b h t c")
    return einsum(ctx_latent, w_uv, "b h t c, h d c -> b h t d")


def gated_delta_step(
    state: Tensor, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    alpha = alpha[..., None, None]
    beta = beta[..., None, None]
    k_state = einsum(k, state, "b h dk, b h dk dv -> b h dv")
    erased = state - beta * k[..., :, None] * k_state[..., None, :]
    new_state = alpha * erased + beta * k[..., :, None] * v[..., None, :]
    out = einsum(new_state, q, "b h dk dv, b h dk -> b h dv")
    return new_state, out


def gated_delta_recurrent(
    q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor, initial_state: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    state = initial_state if initial_state is not None else k.new_zeros(b, h, dk, dv)
    outs = []
    for i in range(t):
        state, o = gated_delta_step(state, q[:, :, i], k[:, :, i], v[:, :, i], alpha[:, :, i], beta[:, :, i])
        outs.append(o)
    return torch.stack(outs, dim=2), state


def gated_linear_attention_parallel(q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor) -> Tensor:
    t = q.shape[-2]
    cum = log_alpha.cumsum(dim=-1)  # (B, H, T)
    log_decay = cum[..., :, None] - cum[..., None, :]  # [t, s] = sum of log_alpha over (s, t]
    causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    decay = log_decay.masked_fill(~causal, float("-inf")).exp()
    return (q @ k.transpose(-2, -1) * decay) @ v


def sigmoid_output_gate(y: Tensor, x: Tensor, w_gate: Tensor) -> Tensor:
    gate = torch.sigmoid(x @ w_gate.T)
    return y * gate.reshape(y.shape)


def kv_cache_elems(
    layer_types: list[str],
    seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    window: int,
    latent_dim: int,
    rope_dim: int,
) -> int:
    per_type = {
        "full": 2 * seq_len * n_kv_heads * head_dim,
        "sliding": 2 * min(seq_len, window) * n_kv_heads * head_dim,
        "linear": n_kv_heads * head_dim * head_dim,
        "mla": seq_len * (latent_dim + rope_dim),
    }
    return sum(per_type[kind] for kind in layer_types)


def topk_sigmoid_routing(router_logits: Tensor, bias: Tensor, k: int) -> tuple[Tensor, Tensor]:
    scores = torch.sigmoid(router_logits)
    idx = torch.topk(scores + bias, k, dim=-1).indices
    chosen = scores.gather(-1, idx)
    return idx, chosen / chosen.sum(dim=-1, keepdim=True)


def capacity_mask(expert_idx: Tensor, num_experts: int, capacity: int) -> Tensor:
    n, k = expert_idx.shape
    # the order in which slots are taken: all of slot 0 across every token first, then slot 1
    order = expert_idx.T.reshape(-1)  # (K*N,)
    onehot = F.one_hot(order, num_experts)
    position = onehot.cumsum(dim=0) - onehot  # how many tokens this expert already has
    rank = (position * onehot).sum(dim=-1)
    keep = rank < capacity
    return keep.reshape(k, n).T


def load_balancing_loss(router_probs: Tensor, expert_idx: Tensor, num_experts: int) -> Tensor:
    f = torch.bincount(expert_idx.reshape(-1), minlength=num_experts).to(router_probs.dtype) / expert_idx.numel()
    p = router_probs.mean(dim=0)
    return num_experts * (f * p).sum()


def update_routing_bias(bias: Tensor, expert_idx: Tensor, num_experts: int, rate: float) -> Tensor:
    load = torch.bincount(expert_idx.reshape(-1), minlength=num_experts).to(bias.dtype)
    return bias + rate * torch.sign(load.mean() - load)


def moe_forward(
    x: Tensor,
    router: Tensor,
    bias: Tensor,
    w1: Tensor,
    w2: Tensor,
    shared_w1: Tensor,
    shared_w2: Tensor,
    k: int,
) -> Tensor:
    idx, weights = topk_sigmoid_routing(x @ router.T, bias, k)
    out = F.relu(x @ shared_w1.T) @ shared_w2.T
    for e in range(router.shape[0]):
        token_idx, slot_idx = torch.where(idx == e)
        he = F.relu(x[token_idx] @ w1[e].T) @ w2[e].T
        out.index_add_(0, token_idx, he * weights[token_idx, slot_idx].unsqueeze(-1))
    return out


def ngram_hash_ids(tokens: Tensor, multipliers: Tensor, table_sizes: Tensor) -> Tensor:
    n = multipliers.shape[1]
    h = torch.zeros(*tokens.shape, multipliers.shape[0], dtype=torch.int64, device=tokens.device)
    for j in range(n):
        shifted = F.pad(tokens, (j, 0))[..., : tokens.shape[-1]]  # shifted[..., t] = tokens[..., t - j]
        h = torch.bitwise_xor(h, shifted[..., None] * multipliers[:, j])
    return h % table_sizes


def _rms(z: Tensor, eps: float) -> Tensor:
    return z * torch.rsqrt(z.pow(2).mean(dim=-1, keepdim=True) + eps)


def ngram_memory(
    h: Tensor, ids: Tensor, table: Tensor, offsets: Tensor, w_k: Tensor, w_v: Tensor, eps: float = 1e-6
) -> Tensor:
    e = table[ids + offsets].flatten(-2)  # (B, T, Hn * De)
    key = e @ w_k.T
    value = e @ w_v.T
    gate = torch.sigmoid((_rms(h, eps) * _rms(key, eps)).sum(dim=-1, keepdim=True) / math.sqrt(h.shape[-1]))
    return h + gate * value


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
