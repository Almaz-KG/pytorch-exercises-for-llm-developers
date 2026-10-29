"""Topic 14 - attention.

The formula in full:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(Dh) + mask) V

What is worth understanding here as a whole rather than syllable by syllable:
  - dividing by sqrt(Dh) keeps the variance of the logits at ~1, otherwise softmax saturates and the gradient dies;
  - the mask is additive (-inf), not multiplicative: it has to be added BEFORE the softmax;
  - causality is not a "property of the model" but precisely this mask;
  - the attention matrix weighs B*H*T*T - quadratic in the length, hence all the fuss about flash-attention.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor

# ==========================================================================
# 14.0 Helpers (a repetition of topic 02, needed further down)
# ==========================================================================


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh)."""
    return rearrange(x, "b t (h d) -> b h t d", h=num_heads)


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D)."""
    return rearrange(x, "b h t d -> b t (h d)")


# ==========================================================================
# 14.1 The core of attention
# ==========================================================================


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """q: (..., T, Dh), k, v: (..., S, Dh), mask: bool broadcastable to (..., T, S),
    where True = allowed to look.

    Return (..., T, Dh). The softmax is taken along the key axis.
    """
    # Attention(Q, K, V) = softmax(Q K^T / sqrt(Dh) + mask) V
    print(q.shape)
    print(k.shape)
    print(v.shape)

    qk = einsum(q, k, "... t d, ... s d -> ... t s") / math.sqrt(q.shape[-1])
    if mask is not None:
        masked = torch.masked_fill(qk, ~mask, float("-inf"))
        sm = torch.softmax(masked, dim=-1)
    else:
        sm = torch.softmax(qk, dim=-1)
    return einsum(sm, v, "... t s, ... s d -> ... t d")


def attention_weights(q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
    """The same computation, but return the attention matrix (..., T, S) itself after the softmax.

    Useful for debugging and for the pictures in reports.
    """
    qk = einsum(q, k, "... t d, ... s d -> ... t s") / math.sqrt(q.shape[-1])
    if mask is not None:
        return torch.softmax(torch.masked_fill(qk, ~mask, float("-inf")), dim=-1)
    return torch.softmax(qk, dim=-1)


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Attention with a causal mask: position i sees only the keys j <= i.

    Build the mask inside, from the lengths of q and k (assume they are equal).
    """
    # print(q.shape)
    # print(k.shape)
    # print(v.shape)

    T, D = q.shape[-2], q.shape[-1]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))

    qk = einsum(q, k, "... t d, ... s d -> ... t s") / math.sqrt(D)
    # print(qk.shape)
    # print(mask.shape)

    masked = qk.masked_fill(~mask, float("-inf"))
    sm = masked.softmax(dim=-1)
    return einsum(sm, v, "... t s, ... s d -> ... t d")


def attention_with_padding(q: Tensor, k: Tensor, v: Tensor, pad_mask: Tensor) -> Tensor:
    """pad_mask: (B, S) bool, True = a real token. Causality plus a ban on looking at padding.

    q, k, v: (B, H, T, Dh), T == S.
    """
    print(q.shape)
    print(k.shape)
    print(v.shape)
    print(pad_mask.shape)

    T, D = q.shape[-2], q.shape[-1]
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device))
    mask_with_padding = mask * pad_mask[:, None, None, :]
    print(mask.shape)
    print(mask_with_padding.shape)

    qk = einsum(q, k, "... t d, ... s d -> ... t s") / math.sqrt(D)
    masked = qk.masked_fill(~mask_with_padding, float("-inf"))
    sm = masked.softmax(dim=-1)
    return einsum(sm, v, "... t s, ... s d -> ... t d")


# ==========================================================================
# 14.2 Multi-head attention as a module
# ==========================================================================


class MultiHeadSelfAttention(nn.Module):
    """Multi-head causal self-attention.

    The requirements (the names matter - they are the contract of the CS336 tests):
      - the submodules q_proj, k_proj, v_proj, output_proj: nn.Linear(d_model, d_model, bias=False);
      - forward(x: (B, T, D)) -> (B, T, D);
      - the attention is causal.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        q = split_heads(self.q_proj(x), self.num_heads)
        k = split_heads(self.k_proj(x), self.num_heads)
        v = split_heads(self.v_proj(x), self.num_heads)
        return self.output_proj(merge_heads(causal_attention(q, k, v)))


class GroupedQueryAttention(nn.Module):
    """GQA: num_heads query heads, but num_kv_heads key/value heads (num_heads is a multiple of it).

    Each kv head serves a group of num_heads / num_kv_heads query heads.
    This is the standard in modern models: the KV cache shrinks several times over with almost no loss of quality.

    The projections: q_proj: d_model -> num_heads * Dh, k_proj and v_proj: d_model -> num_kv_heads * Dh,
    output_proj: num_heads * Dh -> d_model, all without a bias, Dh = d_model // num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        groups = self.num_heads // self.num_kv_heads

        q = split_heads(self.q_proj(x), self.num_heads)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        k = k.repeat_interleave(groups, dim=1)
        v = v.repeat_interleave(groups, dim=1)

        return self.output_proj(merge_heads(causal_attention(q, k, v)))


# ==========================================================================
# 14.3 Cost
# ==========================================================================


def attention_matrix_bytes(batch: int, heads: int, seq_len: int, dtype: torch.dtype) -> int:
    """How many bytes the materialized attention matrix takes.

    Work out for yourself what a 32k context turns into - it will become clear what flash-attention is for.
    """
    return batch * heads * (seq_len**2) * dtype.itemsize


def attention_flops(batch: int, heads: int, seq_len: int, d_head: int) -> int:
    """The FLOPs of attention without the projections: Q K^T plus weights V.

    Both multiplications: (T, Dh) x (Dh, T) and (T, T) x (T, Dh), for every head and every example.
    """
    return 4 * batch * heads * (seq_len**2) * d_head


# ==========================================================================
# TESTS
# ==========================================================================


def test_split_heads():
    B, T, H, Dh = 2, 5, 3, 4
    x = torch.randn(B, T, H * Dh)
    out = split_heads(x, H)
    assert out.shape == (B, H, T, Dh)
    torch.testing.assert_close(out[:, 1], x[..., Dh : 2 * Dh])


def test_merge_heads():
    B, H, T, Dh = 2, 3, 5, 4
    x = torch.randn(B, H, T, Dh)
    out = merge_heads(x)
    assert out.shape == (B, T, H * Dh)
    torch.testing.assert_close(out[..., Dh : 2 * Dh], x[:, 1])


@pytest.mark.depends_on("split_heads", "merge_heads")
def test_heads_roundtrip():
    x = torch.randn(2, 5, 12)
    torch.testing.assert_close(merge_heads(split_heads(x, 3)), x)


def test_scaled_dot_product_attention():
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        atol=1e-5,
        rtol=1e-5,
    )


def test_scaled_dot_product_attention_with_mask():
    q, k, v = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    idx = torch.arange(6)
    mask = idx[:, None] >= idx[None, :]
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        atol=1e-5,
        rtol=1e-5,
    )


def test_attention_weights():
    q, k = torch.randn(2, 2, 4, 16), torch.randn(2, 2, 6, 16)
    w = attention_weights(q, k)
    assert w.shape == (2, 2, 4, 6)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-5, rtol=0)
    assert (w >= 0).all()


def test_attention_weights_with_mask():
    q, k = torch.randn(1, 2, 3, 8), torch.randn(1, 2, 5, 8)
    mask = torch.tensor([True, True, False, True, False])
    w = attention_weights(q, k, mask)
    assert (w[..., ~mask] == 0).all(), "a masked key must get exactly zero weight"
    expected = (q @ k[..., mask, :].mT / math.sqrt(8)).softmax(dim=-1)
    torch.testing.assert_close(w[..., mask], expected, atol=1e-5, rtol=1e-5)


def test_attention_weights_uniform_for_zero_queries():
    """Zero queries -> all the logits are equal -> uniform attention. A good check of the formula."""
    q = torch.zeros(1, 1, 3, 8)
    k = torch.randn(1, 1, 5, 8)
    w = attention_weights(q, k)
    torch.testing.assert_close(w, torch.full((1, 1, 3, 5), 0.2), atol=1e-6, rtol=0)


def test_attention_weights_scale_by_sqrt_dh():
    """Without the division by sqrt(Dh) the logits grow like sqrt(Dh) and the softmax saturates.
    We check that the division is really there."""
    torch.manual_seed(0)  # type: ignore
    q = torch.randn(1, 1, 1, 1024)
    k = torch.randn(1, 1, 2, 1024)
    w = attention_weights(q, k)
    logits_scale = (q[0, 0, 0] @ k[0, 0].T) / math.sqrt(1024)
    expected = logits_scale.softmax(dim=-1)
    torch.testing.assert_close(w[0, 0, 0], expected, atol=1e-5, rtol=1e-5)


def test_causal_attention():
    q, k, v = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
    out = causal_attention(q, k, v)
    assert out.shape == (1, 1, 4, 8)
    # the first output is obliged to be exactly the first value: there is nothing else to look at
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5, rtol=1e-5)


def test_attention_with_padding():
    q, k, v = torch.randn(2, 2, 4, 8), torch.randn(2, 2, 4, 8), torch.randn(2, 2, 4, 8)
    pad = torch.tensor([[True, True, False, False], [True, True, True, True]])
    out = attention_with_padding(q, k, v, pad)
    assert out.shape == (2, 2, 4, 8)
    idx = torch.arange(4)
    allowed = (idx[:, None] >= idx[None, :]) & pad[:, None, None, :]
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, attn_mask=allowed), atol=1e-5, rtol=1e-5)
    # the queries at positions 2 and 3 of the first sequence may look only at the two real keys,
    # so they are obliged to coincide with the attention computed over those two keys alone
    ref = F.scaled_dot_product_attention(q[0:1, :, 2:], k[0:1, :, :2], v[0:1, :, :2])
    torch.testing.assert_close(out[0:1, :, 2:], ref, atol=1e-5, rtol=1e-5)


def _ref_attention(x: Tensor, module: nn.Module, num_heads: int, num_kv_heads: int) -> Tensor:
    """Causal multi-head (or grouped-query) self-attention built from plain torch and the module's own weights.

    The reference the module tests compare against; it needs nothing from the stubs above.
    """
    B, T, _ = x.shape
    q = (x @ module.get_parameter("q_proj.weight").T).view(B, T, num_heads, -1).transpose(1, 2)
    k = (x @ module.get_parameter("k_proj.weight").T).view(B, T, num_kv_heads, -1).transpose(1, 2)
    v = (x @ module.get_parameter("v_proj.weight").T).view(B, T, num_kv_heads, -1).transpose(1, 2)
    # enable_gqa repeats every kv head consecutively, query head h reading kv head h // (num_heads // num_kv_heads)
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
    return out.transpose(1, 2).reshape(B, T, -1) @ module.get_parameter("output_proj.weight").T


@pytest.mark.parametrize("num_heads", [1, 2, 8])
def test_multi_head_self_attention(num_heads: int):
    attn = MultiHeadSelfAttention(16, num_heads)
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(attn(x), _ref_attention(x, attn, num_heads, num_heads), atol=1e-5, rtol=1e-5)


def test_multi_head_self_attention_names():
    attn = MultiHeadSelfAttention(32, 4)
    names = {n for n, _ in attn.named_parameters()}
    assert names == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight"}
    assert sum(p.numel() for p in attn.parameters()) == 4 * 32 * 32


def test_multi_head_self_attention_is_causal():
    """The real check of causality: the output at position t must read the inputs up to t, and nothing after."""
    x = torch.randn(1, 6, 16, requires_grad=True)
    attn = MultiHeadSelfAttention(16, 4)
    attn(x)[0, 2].sum().backward()
    assert x.grad is not None
    assert x.grad[0, 3:].abs().max().item() == 0.0, "information leaked in from the future"
    assert (x.grad[0, :2].abs().amax(dim=-1) > 0).all(), "the output does not read the earlier tokens"


@pytest.mark.depends_on("split_heads", "merge_heads", "causal_attention")
def test_multi_head_self_attention_matches_bricks():
    """We assemble the same thing out of the bricks already implemented - the result is obliged to match."""
    torch.manual_seed(0)  # type: ignore
    d_model, heads = 16, 4
    attn = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(2, 5, d_model)

    q = split_heads(x @ attn.get_parameter("q_proj.weight").T, heads)
    k = split_heads(x @ attn.get_parameter("k_proj.weight").T, heads)
    v = split_heads(x @ attn.get_parameter("v_proj.weight").T, heads)
    expected = merge_heads(causal_attention(q, k, v)) @ attn.get_parameter("output_proj.weight").T

    torch.testing.assert_close(attn(x), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_kv_heads", [1, 2, 8])
def test_grouped_query_attention(num_kv_heads: int):
    """1 kv head is multi-query attention, 8 of 8 is plain MHA, 2 is GQA proper with groups of 4 query heads.

    The kv heads are repeated consecutively: query head h uses kv head h // (num_heads // num_kv_heads).
    """
    gqa = GroupedQueryAttention(64, num_heads=8, num_kv_heads=num_kv_heads)
    x = torch.randn(2, 5, 64)
    torch.testing.assert_close(gqa(x), _ref_attention(x, gqa, 8, num_kv_heads), atol=1e-5, rtol=1e-5)


def test_grouped_query_attention_projections():
    gqa = GroupedQueryAttention(64, num_heads=8, num_kv_heads=2)
    names = {n for n, _ in gqa.named_parameters()}
    assert names == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight"}
    assert gqa.get_parameter("q_proj.weight").shape == (64, 64)
    assert gqa.get_parameter("k_proj.weight").shape == (2 * 8, 64), "the kv projection must be narrow"
    assert gqa.get_parameter("v_proj.weight").shape == (2 * 8, 64), "the kv projection must be narrow"
    assert gqa.get_parameter("output_proj.weight").shape == (64, 64)

    # with num_kv_heads == num_heads GQA degenerates into ordinary MHA
    full = GroupedQueryAttention(64, num_heads=8, num_kv_heads=8)
    assert full.get_parameter("k_proj.weight").shape == (64, 64)


def test_grouped_query_attention_is_causal():
    x = torch.randn(1, 6, 32, requires_grad=True)
    gqa = GroupedQueryAttention(32, num_heads=4, num_kv_heads=2)
    gqa(x)[0, 2].sum().backward()
    assert x.grad is not None
    assert x.grad[0, 3:].abs().max().item() == 0.0, "information leaked in from the future"
    assert (x.grad[0, :2].abs().amax(dim=-1) > 0).all(), "the output does not read the earlier tokens"


def test_attention_matrix_bytes():
    # bf16, batch 8, 32 heads, a context of 32768
    b = attention_matrix_bytes(8, 32, 32768, torch.bfloat16)
    assert b == 8 * 32 * 32768 * 32768 * 2
    assert b > 500e9  # about 550 GB for one layer - it cannot be materialized, hence flash-attention
    assert attention_matrix_bytes(1, 1, 128, torch.float32) == 128 * 128 * 4


def test_attention_flops():
    assert attention_flops(1, 1, 128, 64) == 2 * (2 * 128 * 128 * 64)
    assert attention_flops(4, 8, 128, 64) == 4 * 8 * 2 * (2 * 128 * 128 * 64)
