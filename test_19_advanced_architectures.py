"""Topic 19 - the advanced mechanisms of modern LLMs.

Five storylines, each one a paper that became part of the open frontier models of 2025-2026:
  1. Multi-head Latent Attention: a compressed KV cache and the weight absorption trick (DeepSeek, Kimi);
  2. linear attention: Gated DeltaNet, its recurrent and parallel forms, and hybrid stacks (Qwen3.5, Kimi Linear);
  3. Mixture-of-Experts: sigmoid top-k routing, a shared expert, capacity, balancing with and without a loss;
  4. n-gram memory: hashed n-gram embeddings read from a table and gated into the residual stream;
  5. speculative decoding: how to get exactly the same distribution, but faster.

If this file passes in full, you hold the whole conceptual apparatus
of modern LLM inference and training.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor

# ==========================================================================
# 19.1 Multi-head Latent Attention
# ==========================================================================


def mla_expand_kv(c_kv: Tensor, w_uk: Tensor, w_uv: Tensor) -> tuple[Tensor, Tensor]:
    """The naive way: rebuild per-head keys and values from the cached latent.

    c_kv: (B, S, Dc) - one latent vector per position, the only thing MLA caches;
    w_uk, w_uv: (H, Dh, Dc) - the up-projections, w_uk[h] maps a latent to the key of head h.

    Return (k, v), each (B, H, S, Dh): k[b, h, s] = w_uk[h] @ c_kv[b, s], the same for v.
    """
    raise NotImplementedError


def mla_absorbed_scores(q: Tensor, c_kv: Tensor, w_uk: Tensor) -> Tensor:
    """The attention scores q k^T / sqrt(Dh), computed without ever building k.

    q: (B, H, T, Dh), c_kv: (B, S, Dc), w_uk: (H, Dh, Dc). Return (B, H, T, S).

    Absorb w_uk into the query instead: q . (w_uk c) = (w_uk^T q) . c, so every query is moved
    into the latent space once, and is then compared with the cached latents directly.
    The scale stays 1 / sqrt(Dh), the head width, not the latent width.
    """
    raise NotImplementedError


def mla_absorbed_values(probs: Tensor, c_kv: Tensor, w_uv: Tensor) -> Tensor:
    """The attention output probs @ v, computed without ever building v.

    probs: (B, H, T, S) - the attention weights, c_kv: (B, S, Dc), w_uv: (H, Dh, Dc).
    Return (B, H, T, Dh).

    Mix the latents first, then up-project the mixture once per query: the up-projection is linear,
    so it commutes with the weighted sum over positions.
    """
    raise NotImplementedError


# ==========================================================================
# 19.2 Linear attention: Gated DeltaNet and hybrid stacks
# ==========================================================================


def gated_delta_step(
    state: Tensor, q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor
) -> tuple[Tensor, Tensor]:
    """One step of the gated delta rule (Gated DeltaNet, Qwen3-Next and Qwen3.5).

    state: (B, H, Dk, Dv) - the memory, a matrix per head;
    q, k: (B, H, Dk) - k is L2-normalized by the caller; v: (B, H, Dv);
    alpha, beta: (B, H) - the decay gate and the write strength, both in (0, 1).

        S_new = alpha * (I - beta k k^T) S + beta k v^T
        out   = S_new^T q

    Return (S_new, out (B, H, Dv)). Do not build the (Dk, Dk) matrix I - beta k k^T:
    first read what the memory holds at k, then erase and write.
    """
    raise NotImplementedError


def gated_delta_recurrent(
    q: Tensor, k: Tensor, v: Tensor, alpha: Tensor, beta: Tensor, initial_state: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """The gated delta rule over a whole sequence.

    q, k: (B, H, T, Dk), v: (B, H, T, Dv), alpha, beta: (B, H, T),
    initial_state: (B, H, Dk, Dv) or None for a zero memory.

    Return (out (B, H, T, Dv), the final state (B, H, Dk, Dv)). A Python loop over T is allowed here,
    and it is the point: this is the recurrent form that generation runs one token at a time.
    """
    raise NotImplementedError


def gated_linear_attention_parallel(q: Tensor, k: Tensor, v: Tensor, log_alpha: Tensor) -> Tensor:
    """The parallel form of gated linear attention, the delta rule's simpler cousin.

    The recurrence  S_t = alpha_t S_{t-1} + k_t v_t^T,  out_t = S_t^T q_t  (zero initial state)
    unrolls into attention with a decay instead of a softmax:

        out_t = sum over s <= t of (prod of alpha_r for r in (s, t]) * (q_t . k_s) * v_s

    q, k: (B, H, T, Dk), v: (B, H, T, Dv), log_alpha: (B, H, T), all entries <= 0.
    Return (B, H, T, Dv), without a loop over T: build the (T, T) decay table from a cumsum of
    log_alpha. Keep it finite for strong decay too; exp of a large positive number is inf.
    """
    raise NotImplementedError


def sigmoid_output_gate(y: Tensor, x: Tensor, w_gate: Tensor) -> Tensor:
    """The output gate of gated attention and of Gated DeltaNet (Qwen3-Next, Qwen3.5).

    y: (B, T, H, Dh) - the per-head outputs of the token mixer, x: (B, T, D) - the layer input,
    w_gate: (H * Dh, D). Return y * sigmoid(x @ w_gate.T), the gate laid out as (B, T, H, Dh).
    Every element of every head gets its own gate, computed from the token itself.
    """
    raise NotImplementedError


def kv_cache_elems(
    layer_types: list[str],
    seq_len: int,
    n_kv_heads: int,
    head_dim: int,
    window: int,
    latent_dim: int,
    rope_dim: int,
) -> int:
    """How many numbers one sequence keeps in its cache across a hybrid stack of layers.

    layer_types holds one entry per layer, each of them one of
        "full"    - global attention with GQA: keys and values for every position;
        "sliding" - the same, but only for the last `window` positions;
        "linear"  - a linear-attention layer: one (head_dim, head_dim) state per kv head, whatever seq_len is;
        "mla"     - Multi-head Latent Attention: one latent of latent_dim plus one shared rope key of rope_dim
                    per position, for all the heads together.
    Return a Python int. Topic 17's cache_memory_bytes is the "full" case alone; this one is the whole stack.
    """
    raise NotImplementedError


# ==========================================================================
# 19.3 Mixture of Experts
# ==========================================================================


def topk_sigmoid_routing(router_logits: Tensor, bias: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """router_logits: (N, E), bias: (E,) -> (the expert indices (N, k) int64, the weights (N, k)).

    The DeepSeek-V3 rule: every expert gets an independent score sigmoid(logit).
    The bias decides WHICH experts are chosen: the top k of score + bias, sorted descending.
    It does not decide HOW MUCH they weigh: the weights are the chosen experts' unbiased scores,
    normalized to sum to 1 for every token.
    """
    raise NotImplementedError


def capacity_mask(expert_idx: Tensor, num_experts: int, capacity: int) -> Tensor:
    """expert_idx: (N, K) - which experts every token was sent to.
    Return (N, K) bool: True if the token fits into the capacity of the expert.

    The rule: every expert has no more than `capacity` slots, and tokens occupy them
    in order of increasing token index; slot number 0 first (the most confident choice),
    then slot 1, and so on. The surplus tokens are "dropped" - their contribution is simply lost.

    Without loops over the tokens: a cumsum over the one-hot gives the position of a token in the expert's queue.
    """
    raise NotImplementedError


def load_balancing_loss(router_probs: Tensor, expert_idx: Tensor, num_experts: int) -> Tensor:
    """The auxiliary balancing loss, Switch Transformer's, generalized to top-k:
        loss = E * sum_e f_e * P_e,
    where f_e is the fraction of the N * K assignments in expert_idx (N, K) that went to expert e,
        P_e is the mean of router_probs (N, E) over all the tokens.

    Under perfectly uniform routing this is exactly 1.0, and it grows as routing concentrates.
    Return a 0-dim float tensor.
    """
    raise NotImplementedError


def update_routing_bias(bias: Tensor, expert_idx: Tensor, num_experts: int, rate: float) -> Tensor:
    """Auxiliary-loss-free balancing (DeepSeek-V3): one step of the bias update after a batch.

    bias: (E,), expert_idx: (N, K) - the batch's routing decisions.
    Count the load of every expert; an expert above the mean load gets its bias lowered by `rate`,
    one below the mean gets it raised by `rate`, one exactly at the mean keeps it.

    Return the new bias as a new tensor; the argument stays untouched.
    """
    raise NotImplementedError


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
    """A sparse MoE layer with one shared expert (DeepSeek-V3, Kimi K2, Llama 4).

    x: (N, D), router: (E, D), bias: (E,), w1: (E, D_ff, D), w2: (E, D, D_ff),
    shared_w1: (D_ff, D), shared_w2: (D, D_ff).
    Every token goes to k routed experts by the topk_sigmoid_routing rule, and their outputs are added
    with the routing weights. The shared expert sees every token, with weight 1.
    An expert: out = w2_e @ relu(w1_e @ x).

    The idea of the implementation: do not push everyone through everyone. For every expert pick the tokens
    that landed on it (a mask and indexing), compute only those, and add them back
    through index_add_. This is exactly what real implementations do.
    """
    raise NotImplementedError


# ==========================================================================
# 19.4 N-gram memory
# ==========================================================================


def ngram_hash_ids(tokens: Tensor, multipliers: Tensor, table_sizes: Tensor) -> Tensor:
    """Hash the n-gram that ends at every position into a row of an embedding table.

    tokens: (B, T) int64; multipliers: (Hn, n) int64 - one row per hash head, n is the n-gram order;
    table_sizes: (Hn,) int64 - the number of rows of every head's table.

    For position t the n-gram is tokens[t - n + 1 .. t]; positions before the start count as token 0.
    Head h hashes it as
        (tokens[t] * m[h, 0])  XOR  (tokens[t - 1] * m[h, 1])  XOR ...  XOR  (tokens[t - n + 1] * m[h, n - 1]),
    and then takes the result modulo table_sizes[h].

    Return (B, T, Hn) int64. A Python loop over the n-gram order is fine; a loop over positions is not.
    """
    raise NotImplementedError


def ngram_memory(
    h: Tensor, ids: Tensor, table: Tensor, offsets: Tensor, w_k: Tensor, w_v: Tensor, eps: float = 1e-6
) -> Tensor:
    """Read the n-gram memory and gate it into the residual stream (Engram, Qwen3.8-Flash-Next).

    h: (B, T, D) - the hidden states; ids: (B, T, Hn) - what ngram_hash_ids returned;
    table: (R, De) - the tables of all heads stacked; head j owns the rows starting at offsets[j];
    offsets: (Hn,) int64; w_k, w_v: (D, Hn * De).

    1. look up one row per head and concatenate the heads: e is (B, T, Hn * De);
    2. key = e @ w_k.T and value = e @ w_v.T, both (B, T, D);
    3. gate = sigmoid( sum(rms(h) * rms(key)) / sqrt(D) ), one number per position,
       where rms(z) = z / sqrt(mean(z^2) + eps) over the last axis, with no weight;
    4. return h + gate * value.
    """
    raise NotImplementedError


# ==========================================================================
# 19.5 Speculative decoding
# ==========================================================================


def speculative_accept(
    draft_probs: Tensor, target_probs: Tensor, draft_tokens: Tensor, u: Tensor
) -> tuple[int, Tensor]:
    """The acceptance rule from the speculative decoding paper.

    draft_probs: (K, V) - the distributions of the draft model at the K proposed positions;
    target_probs: (K+1, V) - the distributions of the big model at the same positions plus the bonus one;
    draft_tokens: (K,) - what the draft model proposed;
    u: (K,) - uniform random numbers from [0, 1), passed in from outside for the sake of determinism.

    At position i we accept the token if u[i] <= p_target(token) / p_draft(token).
    On the first rejection, at position j, we return (j, the distribution norm(relu(p_target - p_draft))
    at that position). If all K are accepted, we return (K, target_probs[K]) - the bonus token
    comes for free.

    The key property of the scheme: the resulting distribution of tokens matches the distribution
    of the big model EXACTLY. The speedup comes from the big model
    checking K tokens in a single pass.
    """
    raise NotImplementedError


def expected_speedup(acceptance_rate: float, k: int) -> float:
    """The expected number of tokens per verification cycle at an acceptance probability alpha:
        (1 - alpha^(k+1)) / (1 - alpha),
    and at alpha == 1 it is simply k + 1.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================
#
# The tests build their inputs and expectations with the plain-torch references below rather than
# with your own stubs, so the stubs can be written in any order. Where two of your pieces have to
# agree with each other, the test carries depends_on and turns on once both exist.


def _ref_topk_sigmoid(router_logits: Tensor, bias: Tensor, k: int) -> tuple[Tensor, Tensor]:
    scores = torch.sigmoid(router_logits)
    idx = (scores + bias).topk(k, dim=-1).indices
    chosen = scores.gather(-1, idx)
    return idx, chosen / chosen.sum(-1, keepdim=True)


def _ref_dense_moe(x, idx, weights, w1, w2, shared_w1, shared_w2):
    out = torch.zeros_like(x)
    for n in range(x.shape[0]):
        out[n] = shared_w2 @ F.relu(shared_w1 @ x[n])
        for slot in range(idx.shape[1]):
            e = int(idx[n, slot])
            out[n] += weights[n, slot] * (w2[e] @ F.relu(w1[e] @ x[n]))
    return out


def _ref_delta_rule(q, k, v, alpha, beta, state=None):
    """The gated delta rule written literally, one (b, h, t) at a time, with the (Dk, Dk) matrix."""
    B, H, T, Dk = k.shape
    Dv = v.shape[-1]
    state = torch.zeros(B, H, Dk, Dv) if state is None else state.clone()
    out = torch.zeros(B, H, T, Dv)
    eye = torch.eye(Dk)
    for b in range(B):
        for h in range(H):
            S = state[b, h]
            for t in range(T):
                kt = k[b, h, t][:, None]
                S = alpha[b, h, t] * (eye - beta[b, h, t] * kt @ kt.T) @ S + beta[b, h, t] * kt @ v[b, h, t][None, :]
                out[b, h, t] = S.T @ q[b, h, t]
            state[b, h] = S
    return out, state


def _ref_gla_recurrent(q, k, v, log_alpha):
    B, H, T, Dk = k.shape
    S = torch.zeros(B, H, Dk, v.shape[-1], dtype=q.dtype)
    out = []
    for t in range(T):
        S = log_alpha[:, :, t, None, None].exp() * S + k[:, :, t, :, None] * v[:, :, t, None, :]
        out.append((S * q[:, :, t, :, None]).sum(-2))
    return torch.stack(out, dim=2)


def _delta_inputs(B=2, H=3, T=7, Dk=4, Dv=5):
    q = torch.randn(B, H, T, Dk)
    k = F.normalize(torch.randn(B, H, T, Dk), dim=-1)
    v = torch.randn(B, H, T, Dv)
    alpha = torch.sigmoid(torch.randn(B, H, T) + 2)
    beta = torch.sigmoid(torch.randn(B, H, T))
    return q, k, v, alpha, beta


def test_mla_expand_kv():
    B, S, Dc, H, Dh = 2, 6, 8, 3, 4
    c_kv = torch.randn(B, S, Dc)
    w_uk, w_uv = torch.randn(H, Dh, Dc), torch.randn(H, Dh, Dc)
    k, v = mla_expand_kv(c_kv, w_uk, w_uv)
    assert k.shape == v.shape == (B, H, S, Dh)
    torch.testing.assert_close(k[1, 2, 5], w_uk[2] @ c_kv[1, 5])
    torch.testing.assert_close(k, c_kv[:, None] @ w_uk.transpose(-2, -1), atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(v, c_kv[:, None] @ w_uv.transpose(-2, -1), atol=1e-5, rtol=1e-4)


def test_mla_absorbed_scores():
    B, T, S, Dc, H, Dh = 2, 3, 6, 8, 4, 5
    q = torch.randn(B, H, T, Dh)
    c_kv = torch.randn(B, S, Dc)
    w_uk = torch.randn(H, Dh, Dc)
    k = c_kv[:, None] @ w_uk.transpose(-2, -1)  # (B, H, S, Dh), the keys absorption avoids
    expected = q @ k.transpose(-2, -1) / math.sqrt(Dh)
    torch.testing.assert_close(mla_absorbed_scores(q, c_kv, w_uk), expected, atol=1e-4, rtol=1e-4)


def test_mla_absorbed_values():
    B, T, S, Dc, H, Dh = 2, 3, 6, 8, 4, 5
    probs = torch.randn(B, H, T, S).softmax(-1)
    c_kv = torch.randn(B, S, Dc)
    w_uv = torch.randn(H, Dh, Dc)
    v = c_kv[:, None] @ w_uv.transpose(-2, -1)
    torch.testing.assert_close(mla_absorbed_values(probs, c_kv, w_uv), probs @ v, atol=1e-5, rtol=1e-4)


@pytest.mark.depends_on("mla_expand_kv", "mla_absorbed_scores", "mla_absorbed_values")
def test_mla_absorbed_attention_equals_naive():
    """The absorbed path, scores then softmax then values, is the same attention as the naive one."""
    B, T, Dc, H, Dh = 2, 6, 8, 4, 5
    q = torch.randn(B, H, T, Dh)
    c_kv = torch.randn(B, T, Dc)
    w_uk, w_uv = torch.randn(H, Dh, Dc), torch.randn(H, Dh, Dc)
    mask = torch.ones(T, T, dtype=torch.bool).tril()

    k, v = mla_expand_kv(c_kv, w_uk, w_uv)
    naive = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    probs = mla_absorbed_scores(q, c_kv, w_uk).masked_fill(~mask, float("-inf")).softmax(-1)
    torch.testing.assert_close(mla_absorbed_values(probs, c_kv, w_uv), naive, atol=1e-4, rtol=1e-4)


def test_gated_delta_step_matches_the_formula():
    q, k, v, alpha, beta = _delta_inputs(T=1)
    state = torch.randn(2, 3, 4, 5)
    expected_out, expected_state = _ref_delta_rule(q, k, v, alpha, beta, state)
    new_state, out = gated_delta_step(state, q[:, :, 0], k[:, :, 0], v[:, :, 0], alpha[:, :, 0], beta[:, :, 0])
    torch.testing.assert_close(new_state, expected_state, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(out, expected_out[:, :, 0], atol=1e-5, rtol=1e-4)


def test_gated_delta_step_overwrites_the_value_at_a_key():
    """With alpha = beta = 1 the delta rule replaces what the memory held at k with v, whatever it held."""
    B, H, Dk, Dv = 2, 2, 6, 3
    state = torch.randn(B, H, Dk, Dv)
    k = F.normalize(torch.randn(B, H, Dk), dim=-1)
    v = torch.randn(B, H, Dv)
    ones = torch.ones(B, H)
    _, read_back = gated_delta_step(state, k, k, v, ones, ones)
    torch.testing.assert_close(read_back, v, atol=1e-5, rtol=1e-4)


def test_gated_delta_recurrent_matches_reference():
    q, k, v, alpha, beta = _delta_inputs()
    expected_out, expected_state = _ref_delta_rule(q, k, v, alpha, beta)
    out, state = gated_delta_recurrent(q, k, v, alpha, beta)
    torch.testing.assert_close(out, expected_out, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(state, expected_state, atol=1e-5, rtol=1e-4)


def test_gated_delta_recurrent_carries_state_across_calls():
    """Two halves, the first one's final state handed to the second, give the same result as one call."""
    q, k, v, alpha, beta = _delta_inputs(T=8)
    full_out, full_state = gated_delta_recurrent(q, k, v, alpha, beta)
    first, mid = gated_delta_recurrent(q[:, :, :3], k[:, :, :3], v[:, :, :3], alpha[:, :, :3], beta[:, :, :3])
    second, last = gated_delta_recurrent(
        q[:, :, 3:], k[:, :, 3:], v[:, :, 3:], alpha[:, :, 3:], beta[:, :, 3:], initial_state=mid
    )
    torch.testing.assert_close(torch.cat([first, second], dim=2), full_out, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(last, full_state, atol=1e-5, rtol=1e-4)


def test_gated_linear_attention_parallel_matches_recurrence():
    B, H, T, Dk, Dv = 2, 3, 9, 4, 5
    q, k, v = torch.randn(B, H, T, Dk), torch.randn(B, H, T, Dk), torch.randn(B, H, T, Dv)
    log_alpha = F.logsigmoid(torch.randn(B, H, T) + 1)
    expected = _ref_gla_recurrent(q, k, v, log_alpha)
    torch.testing.assert_close(gated_linear_attention_parallel(q, k, v, log_alpha), expected, atol=1e-4, rtol=1e-4)


def test_gated_linear_attention_parallel_survives_strong_decay():
    B, H, T, D = 1, 2, 32, 4
    q, k, v = torch.randn(B, H, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D)
    log_alpha = torch.full((B, H, T), -30.0)  # a cumsum of -930 at the end: exp of the wrong half overflows
    out = gated_linear_attention_parallel(q, k, v, log_alpha)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, _ref_gla_recurrent(q, k, v, log_alpha), atol=1e-4, rtol=1e-4)


def test_sigmoid_output_gate():
    B, T, H, Dh, D = 2, 3, 4, 5, 8
    y = torch.randn(B, T, H, Dh)
    x = torch.randn(B, T, D)
    w_gate = torch.randn(H * Dh, D)
    out = sigmoid_output_gate(y, x, w_gate)
    assert out.shape == (B, T, H, Dh)
    gate_h2 = torch.sigmoid(x @ w_gate[2 * Dh : 3 * Dh].T)  # the rows of w_gate that belong to head 2
    torch.testing.assert_close(out[:, :, 2], y[:, :, 2] * gate_h2)
    assert (out.abs() <= y.abs()).all()


def test_kv_cache_elems_attention_layers():
    common = {"n_kv_heads": 8, "head_dim": 128, "window": 1024, "latent_dim": 512, "rope_dim": 64}
    assert kv_cache_elems(["full"] * 4, seq_len=4096, **common) == 4 * 2 * 4096 * 8 * 128
    assert kv_cache_elems(["sliding"], seq_len=4096, **common) == 2 * 1024 * 8 * 128
    assert kv_cache_elems(["sliding"], seq_len=100, **common) == 2 * 100 * 8 * 128
    assert kv_cache_elems(["mla"] * 3, seq_len=4096, **common) == 3 * 4096 * (512 + 64)


def test_kv_cache_elems_hybrid_stacks():
    common = {"n_kv_heads": 4, "head_dim": 64, "window": 512, "latent_dim": 256, "rope_dim": 32}
    linear = kv_cache_elems(["linear"], seq_len=10, **common)
    assert linear == 4 * 64 * 64
    assert kv_cache_elems(["linear"], seq_len=100_000, **common) == linear
    pattern = ["sliding"] * 5 + ["full"]
    assert kv_cache_elems(pattern, seq_len=8192, **common) == 5 * 2 * 512 * 4 * 64 + 2 * 8192 * 4 * 64
    assert isinstance(kv_cache_elems(pattern, seq_len=8192, **common), int)


def test_topk_sigmoid_routing():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0], [5.0, 0.0, 0.0, 4.0]])
    idx, weights = topk_sigmoid_routing(logits, torch.zeros(4), k=2)
    torch.testing.assert_close(idx, torch.tensor([[1, 2], [0, 3]]))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)
    s = torch.sigmoid(torch.tensor([3.0, 2.0]))
    torch.testing.assert_close(weights[0], s / s.sum())
    assert (weights[:, 0] >= weights[:, 1]).all()


def test_topk_sigmoid_routing_bias_selects_but_does_not_weigh():
    logits = torch.tensor([[2.0, 1.0, 0.0, -1.0]])
    bias = torch.tensor([0.0, 0.0, 0.0, 5.0])
    idx, weights = topk_sigmoid_routing(logits, bias, k=2)
    torch.testing.assert_close(idx, torch.tensor([[3, 0]]))
    s = torch.sigmoid(torch.tensor([-1.0, 2.0]))
    torch.testing.assert_close(weights[0], s / s.sum())


def test_capacity_mask():
    # four tokens, all of them going to expert 0 as their first choice, capacity 2
    expert_idx = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    m = capacity_mask(expert_idx, num_experts=2, capacity=2)
    torch.testing.assert_close(m[:, 0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(m[:, 1], torch.tensor([True, True, False, False]))

    # with enough capacity nothing gets dropped
    big = capacity_mask(expert_idx, num_experts=2, capacity=10)
    assert big.all()


def test_capacity_mask_slot_order():
    """Slot 0 (the token's main expert) takes up capacity before slot 1."""
    expert_idx = torch.tensor([[1, 0], [0, 1]])
    m = capacity_mask(expert_idx, num_experts=2, capacity=1)
    # expert 1: token 0 arrives as slot 0, token 1 as slot 1 -> token 0 wins
    assert bool(m[0, 0]) and not bool(m[1, 1])
    # expert 0: token 1 arrives as slot 0, token 0 as slot 1 -> token 1 wins
    assert bool(m[1, 0]) and not bool(m[0, 1])


def test_load_balancing_loss():
    E = 4
    uniform_probs = torch.full((8, E), 1 / E)
    uniform_idx = torch.arange(16).reshape(8, 2) % E
    torch.testing.assert_close(load_balancing_loss(uniform_probs, uniform_idx, E), torch.tensor(1.0), atol=1e-6, rtol=0)

    # top-1 collapse onto one expert reads E
    collapsed = torch.zeros(8, E)
    collapsed[:, 0] = 1.0
    collapsed_idx = torch.zeros(8, 1, dtype=torch.int64)
    torch.testing.assert_close(
        load_balancing_loss(collapsed, collapsed_idx, E), torch.tensor(float(E)), atol=1e-6, rtol=0
    )

    # top-2 collapse onto two experts out of four: f = P = (1/2, 1/2, 0, 0)
    pair = torch.zeros(8, E)
    pair[:, :2] = 0.5
    pair_idx = torch.tensor([[0, 1]] * 8)
    torch.testing.assert_close(load_balancing_loss(pair, pair_idx, E), torch.tensor(2.0), atol=1e-6, rtol=0)


def test_update_routing_bias():
    expert_idx = torch.tensor([[0, 1], [0, 2], [0, 1], [0, 3]])  # loads (4, 2, 1, 1), mean 2
    bias = torch.tensor([0.5, 0.0, 0.0, -0.5])
    new = update_routing_bias(bias, expert_idx, num_experts=4, rate=0.1)
    torch.testing.assert_close(new, torch.tensor([0.4, 0.0, 0.1, -0.4]))
    torch.testing.assert_close(bias, torch.tensor([0.5, 0.0, 0.0, -0.5]))  # the argument is untouched


def test_update_routing_bias_balances_a_skewed_router():
    """Run the update for a few hundred batches: the skew the logits carry gets absorbed into the bias."""
    N, E, k = 512, 8, 2
    logits = torch.randn(N, E) + torch.linspace(2, -2, E)
    bias = torch.zeros(E)

    def max_load(b):
        idx, _ = _ref_topk_sigmoid(logits, b, k)
        return int(torch.bincount(idx.flatten(), minlength=E).max())

    mean_load = N * k / E
    assert max_load(bias) > 2.5 * mean_load
    for _ in range(300):
        idx, _ = _ref_topk_sigmoid(logits, bias, k)
        bias = update_routing_bias(bias, idx, E, rate=0.01)
    assert max_load(bias) < 1.25 * mean_load


def _moe_params(N=12, D=8, E=6, D_ff=16):
    x = torch.randn(N, D)
    router = torch.randn(E, D)
    bias = torch.randn(E) * 0.5
    w1 = torch.randn(E, D_ff, D) * 0.1
    w2 = torch.randn(E, D, D_ff) * 0.1
    shared_w1 = torch.randn(D_ff, D) * 0.1
    shared_w2 = torch.randn(D, D_ff) * 0.1
    return x, router, bias, w1, w2, shared_w1, shared_w2


def test_moe_forward_matches_dense_reference():
    x, router, bias, w1, w2, shared_w1, shared_w2 = _moe_params()
    idx, weights = _ref_topk_sigmoid(x @ router.T, bias, 3)
    expected = _ref_dense_moe(x, idx, weights, w1, w2, shared_w1, shared_w2)
    out = moe_forward(x, router, bias, w1, w2, shared_w1, shared_w2, k=3)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_moe_uses_only_the_chosen_experts():
    """If every expert no token chose is zeroed out, the output must not change."""
    x, router, bias, w1, w2, shared_w1, shared_w2 = _moe_params(N=4, E=8)
    idx, _ = _ref_topk_sigmoid(x @ router.T, bias, 2)
    out = moe_forward(x, router, bias, w1, w2, shared_w1, shared_w2, k=2)

    unused = sorted(set(range(8)) - set(idx.flatten().tolist()))
    assert unused, "the fixture is meant to leave some experts unused"
    w1_mod, w2_mod = w1.clone(), w2.clone()
    w1_mod[unused] = 0
    w2_mod[unused] = 0
    torch.testing.assert_close(moe_forward(x, router, bias, w1_mod, w2_mod, shared_w1, shared_w2, k=2), out)


@pytest.mark.depends_on("topk_sigmoid_routing", "moe_forward")
def test_moe_routing_agrees_with_topk_sigmoid_routing():
    """The layer has to route by the same rule the routing function implements."""
    x, router, bias, w1, w2, shared_w1, shared_w2 = _moe_params()
    idx, weights = topk_sigmoid_routing(x @ router.T, bias, 2)
    expected = _ref_dense_moe(x, idx, weights, w1, w2, shared_w1, shared_w2)
    out = moe_forward(x, router, bias, w1, w2, shared_w1, shared_w2, k=2)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_ngram_hash_ids_by_hand():
    tokens = torch.tensor([[5, 7, 9]])
    multipliers = torch.tensor([[3, 11], [2, 1]])
    sizes = torch.tensor([13, 17])
    expected = torch.tensor(
        [
            [
                [(5 * 3 ^ 0 * 11) % 13, (5 * 2 ^ 0 * 1) % 17],
                [(7 * 3 ^ 5 * 11) % 13, (7 * 2 ^ 5 * 1) % 17],
                [(9 * 3 ^ 7 * 11) % 13, (9 * 2 ^ 7 * 1) % 17],
            ]
        ]
    )
    torch.testing.assert_close(ngram_hash_ids(tokens, multipliers, sizes), expected)


def test_ngram_hash_ids_depend_only_on_the_ngram():
    """The same trigram gets the same ids wherever it sits, and every id fits its own table."""
    tokens = torch.randint(1, 50_000, (2, 20))
    tokens[0, 4:7] = torch.tensor([101, 202, 303])
    tokens[1, 15:18] = torch.tensor([101, 202, 303])
    multipliers = torch.randint(1, 2**31, (4, 3))
    sizes = torch.tensor([1009, 2003, 3001, 4001])

    ids = ngram_hash_ids(tokens, multipliers, sizes)
    assert ids.shape == (2, 20, 4) and ids.dtype == torch.int64
    torch.testing.assert_close(ids[0, 6], ids[1, 17])
    assert (ids >= 0).all() and (ids < sizes).all()


def test_ngram_memory_matches_reference():
    B, T, D, Hn, De = 2, 5, 16, 3, 4
    sizes = [7, 11, 13]
    offsets = torch.tensor([0, 7, 18])
    h = torch.randn(B, T, D)
    ids = torch.stack([torch.randint(0, s, (B, T)) for s in sizes], dim=-1)
    table = torch.randn(sum(sizes), De)
    w_k, w_v = torch.randn(D, Hn * De), torch.randn(D, Hn * De)

    expected = torch.zeros(B, T, D)
    for b in range(B):
        for t in range(T):
            e = torch.cat([table[offsets[j] + ids[b, t, j]] for j in range(Hn)])
            key, value = w_k @ e, w_v @ e
            hn = h[b, t] / torch.sqrt(h[b, t].pow(2).mean() + 1e-6)
            kn = key / torch.sqrt(key.pow(2).mean() + 1e-6)
            gate = torch.sigmoid((hn * kn).sum() / math.sqrt(D))
            expected[b, t] = h[b, t] + gate * value

    out = ngram_memory(h, ids, table, offsets, w_k, w_v)
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-4)


def test_speculative_accept_all_accepted():
    draft = torch.full((3, 4), 0.25)
    target = torch.full((4, 4), 0.25)
    tokens = torch.tensor([0, 1, 2])
    n, dist = speculative_accept(draft, target, tokens, u=torch.zeros(3))
    assert n == 3
    torch.testing.assert_close(dist, target[3])


def test_speculative_accept_rejection():
    draft = torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    tokens = torch.tensor([0, 0])
    # the first position: p/q = 1, we accept at any u; the second: p/q = 0.111, u = 0.5 -> a rejection
    n, dist = speculative_accept(draft, target, tokens, u=torch.tensor([0.5, 0.5]))
    assert n == 1
    residual = torch.clamp(target[1] - draft[1], min=0)
    torch.testing.assert_close(dist, residual / residual.sum(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(dist.sum(), torch.tensor(1.0))


def test_speculative_accept_is_a_distribution():
    torch.manual_seed(0)
    V, K = 8, 4
    draft = torch.rand(K, V).softmax(-1)
    target = torch.rand(K + 1, V).softmax(-1)
    tokens = torch.randint(0, V, (K,))
    for u_val in (0.0, 0.3, 0.7, 0.99):
        n, dist = speculative_accept(draft, target, tokens, u=torch.full((K,), u_val))
        assert 0 <= n <= K
        assert torch.isfinite(dist).all()
        torch.testing.assert_close(dist.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_expected_speedup():
    assert abs(expected_speedup(1.0, 4) - 5.0) < 1e-9
    assert abs(expected_speedup(0.0, 4) - 1.0) < 1e-9
    assert abs(expected_speedup(0.8, 4) - (1 - 0.8**5) / 0.2) < 1e-9
    assert expected_speedup(0.9, 8) > expected_speedup(0.5, 8)
