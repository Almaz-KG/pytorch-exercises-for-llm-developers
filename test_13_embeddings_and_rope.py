"""Topic 13 - embeddings and positional encodings, RoPE included.

Positional information reaches a transformer in one of three ways:
  1. absolute sinusoids (2017): added to the embeddings once;
  2. learned absolute positions (GPT-2): simply one more embedding table;
  3. RoPE (LLaMA, Qwen, CS336): a rotation of q and k by an angle proportional to the position.

Why RoPE won: after the rotation the dot product q_i . k_j depends ONLY
on the difference of the positions i - j. That is, the model gets relative positions for free,
without spending a single parameter, and it carries over to lengths that were never in training.

The RoPE convention in this file (the same one as in CS336): a head vector of length Dh (Dh even)
is split into ADJACENT pairs (x0, x1), (x2, x3), ... Pair number i is rotated
by the angle pos * theta^(-2i/Dh).
"""

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 13.1 Embeddings
# ==========================================================================


def scaled_token_embedding(table: Tensor, ids: Tensor) -> Tensor:
    """A lookup multiplied by sqrt(d_model) - this is what the original transformer does,
    so that the embeddings and the positional sinusoids are comparable in scale.
    """
    return table[ids] * math.sqrt(table.shape[-1])


def learned_positional(embed: Tensor, pos_table: Tensor, offset: int = 0) -> Tensor:
    """embed: (B, T, D), pos_table: (max_len, D) -> (B, T, D): add the positional vectors
    for the positions offset, offset+1, ..., offset+T-1.

    The offset is needed during generation: at a decoding step we have one token, but its position is not zero.
    """
    T = embed.shape[-2]
    return embed + pos_table[offset : offset + T]


def sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
    """The sinusoidal positional encodings (T, D) from the 2017 paper:
    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    """
    pos = torch.arange(seq_len, dtype=torch.float32)[:, None]
    i = torch.arange(0, d_model, 2, dtype=torch.float32)
    inv_freq = torch.pow(10000.0, -i / d_model)
    angles = pos * inv_freq[None, :]
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = angles.sin()
    pe[:, 1::2] = angles.cos()
    return pe


# ==========================================================================
# 13.2 RoPE: frequencies and the cache
# ==========================================================================


def rope_inv_freq(d_head: int, theta: float = 10000.0) -> Tensor:
    """(d_head // 2,): the inverse frequencies theta^(-2i/d_head) for i = 0, 1, ..., d_head/2 - 1.

    The first pair spins fast (frequency 1), the last one very slowly.
    """
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    return torch.pow(theta, -i / d_head)


def rope_cos_sin(seq_len: int, d_head: int, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    """The cache of rotations: two tensors (seq_len, d_head // 2) with the cosines and sines of the angles
    (offset + pos) * inv_freq[i].

    In a real model this is computed once and kept as a buffer.
    """
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)[:, None]
    angles = pos * rope_inv_freq(d_head, theta)[None, :]
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (..., T, Dh), cos/sin: (T, Dh // 2) -> a tensor of the same shape.

    Every adjacent pair (x[..., 2i], x[..., 2i+1]) is rotated by its own angle:
      out_even = x_even * cos - x_odd * sin
      out_odd  = x_even * sin + x_odd * cos

    The function must work both for (T, Dh) and for (B, H, T, Dh) - thanks to broadcasting.
    """
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = torch.stack([out_even, out_odd], dim=-1)
    return out.flatten(start_dim=-2)


def apply_rope_to_qk(q: Tensor, k: Tensor, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    """A convenience wrapper: q, k of shape (B, H, T, Dh) -> the rotated q, k.

    Note: RoPE is applied to q and k, but NOT to v.
    """
    cos, sin = rope_cos_sin(seq_len=q.shape[-2], d_head=q.shape[-1], theta=theta, offset=offset)
    cos, sin = cos.to(q.dtype), sin.to(q.dtype)
    return apply_rope(q, cos=cos, sin=sin), apply_rope(k, cos, sin)


# ==========================================================================
# 13.3 The properties of RoPE that make it useful
# ==========================================================================


def rope_dot_product(q: Tensor, k: Tensor, pos_q: int, pos_k: int, theta: float = 10000.0) -> Tensor:
    """q, k: (Dh,). Rotate q to the position pos_q, k to the position pos_k, and return their dot product.

    This will be needed to check the relativity.
    """
    d = q.shape[-1]
    cq, sq = rope_cos_sin(1, d, theta, offset=pos_q)
    ck, sk = rope_cos_sin(1, d, theta, offset=pos_k)
    qr = apply_rope(q[None, :], cq, sq)[0]
    kr = apply_rope(k[None, :], ck, sk)[0]
    return (qr * kr).sum()


def rope_norm_change(x: Tensor, pos: int, theta: float = 10000.0) -> Tensor:
    """By how much the rotation changes the norm of the vector. As expected - not at all: it is a pure rotation."""
    cos, sin = rope_cos_sin(1, x.shape[-1], theta, offset=pos)
    rotated = apply_rope(x[None, :], cos, sin)[0]
    return rotated.norm() - x.norm()  # type: ignore


# ==========================================================================
# 13.4 The module
# ==========================================================================


class RotaryEmbedding(nn.Module):
    """A module that keeps the cos and sin caches as BUFFERS (not parameters!) up to max_len
    and applies them to a tensor (B, H, T, Dh), taking the position offset into account.

    The buffers must be named `cos_cached` and `sin_cached`, and there must be no trainable parameters.
    """

    def __init__(self, d_head: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        cos, sin = rope_cos_sin(max_len, d_head, theta)
        self.register_buffer("cos_cached", cos, persistent=True)
        self.register_buffer("sin_cached", sin, persistent=True)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        T = x.shape[-2]
        cos = self.cos_cached[offset : offset + T].to(x.dtype)
        sin = self.sin_cached[offset : offset + T].to(x.dtype)
        return apply_rope(x, cos, sin)


# ==========================================================================
# TESTS
# ==========================================================================
#
# The rotations below are rebuilt from plain torch rather than taken from rope_cos_sin, so that
# each test rests on exactly one stub and the pieces can be written in any order. That the pieces
# then agree with each other is checked separately, in test_rope_pieces_agree.


def _ref_cos_sin(seq_len: int, d_head: int, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    inv_freq = theta ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
    pos = torch.arange(offset, offset + seq_len, dtype=torch.float32)
    angles = pos[:, None] * inv_freq[None, :]
    return angles.cos(), angles.sin()


def _ref_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = even * cos - odd * sin
    out[..., 1::2] = even * sin + odd * cos
    return out


def test_scaled_token_embedding():
    table = torch.randn(10, 16)
    ids = torch.tensor([[1, 2]])
    out = scaled_token_embedding(table, ids)
    torch.testing.assert_close(out, table[ids] * math.sqrt(16))


def test_learned_positional():
    embed = torch.zeros(2, 3, 4)
    pos = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    out = learned_positional(embed, pos)
    torch.testing.assert_close(out[0], pos[:3])
    out2 = learned_positional(embed, pos, offset=2)
    torch.testing.assert_close(out2[0], pos[2:5])


def test_sinusoidal_pe():
    pe = sinusoidal_pe(6, 8)
    assert pe.shape == (6, 8)
    torch.testing.assert_close(pe[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    # the first pair of channels spins with frequency 1
    torch.testing.assert_close(pe[1, 0], torch.sin(torch.tensor(1.0)))
    torch.testing.assert_close(pe[1, 1], torch.cos(torch.tensor(1.0)))
    # the last pair very slowly
    assert pe[1, -1].item() > 0.999


def test_rope_inv_freq():
    f = rope_inv_freq(8, theta=10000.0)
    assert f.shape == (4,)
    torch.testing.assert_close(f[0], torch.tensor(1.0))
    torch.testing.assert_close(f[3], torch.tensor(10000.0 ** (-6 / 8)))
    assert (f[1:] < f[:-1]).all(), "the frequencies must decrease"


def test_rope_cos_sin():
    cos, sin = rope_cos_sin(4, 8)
    assert cos.shape == sin.shape == (4, 4)
    torch.testing.assert_close(cos[0], torch.ones(4))
    torch.testing.assert_close(sin[0], torch.zeros(4))
    torch.testing.assert_close(cos.pow(2) + sin.pow(2), torch.ones(4, 4), atol=1e-6, rtol=0)

    cos_off, sin_off = rope_cos_sin(2, 8, offset=2)
    torch.testing.assert_close(cos_off[0], cos[2])
    torch.testing.assert_close(sin_off[1], sin[3])


def test_apply_rope_position_zero_is_identity():
    x = torch.randn(2, 3, 5, 8)
    cos, sin = _ref_cos_sin(5, 8)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    torch.testing.assert_close(out[..., 0, :], x[..., 0, :], atol=1e-6, rtol=0)


def test_apply_rope_explicit_small_case():
    """Dh = 2, one pair, inv_freq = [1]. So the rotation is by an angle equal to the position itself."""
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (T=3, Dh=2)
    cos, sin = _ref_cos_sin(3, 2)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out[1], torch.tensor([math.cos(1.0), math.sin(1.0)]))
    torch.testing.assert_close(out[2], torch.tensor([-math.sin(2.0), math.cos(2.0)]))


def test_apply_rope_preserves_norm():
    x = torch.randn(2, 4, 6, 16)
    cos, sin = _ref_cos_sin(6, 16)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rope_is_relative():
    """The key property: the dot product depends only on the difference of the positions."""
    q, k = torch.randn(16), torch.randn(16)
    base = rope_dot_product(q, k, 5, 3)
    for pq, pk in [(2, 0), (10, 8), (101, 99)]:
        torch.testing.assert_close(rope_dot_product(q, k, pq, pk), base, atol=1e-4, rtol=1e-4)
    # and at a different difference the value is obliged to differ
    assert not torch.allclose(rope_dot_product(q, k, 5, 0), base, atol=1e-3)


def test_rope_norm_change():
    x = torch.randn(32)
    assert rope_norm_change(x, 17).abs().item() < 1e-5


def test_apply_rope_to_qk():
    B, H, T, Dh = 2, 3, 4, 8
    q, k = torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh)
    qr, kr = apply_rope_to_qk(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape
    cos, sin = _ref_cos_sin(T, Dh)
    torch.testing.assert_close(qr, _ref_rope(q, cos, sin), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kr, _ref_rope(k, cos, sin), atol=1e-5, rtol=1e-5)

    # a position offset - the generation mode, one token at a time
    q1 = q[:, :, 3:4]
    q1_shifted, _ = apply_rope_to_qk(q1, q1, offset=3)
    torch.testing.assert_close(q1_shifted, qr[:, :, 3:4], atol=1e-5, rtol=1e-5)


def test_rotary_embedding_module():
    rope = RotaryEmbedding(8, max_len=16)
    assert sum(p.numel() for p in rope.parameters()) == 0, "RoPE has no trainable parameters"
    assert set(rope.state_dict().keys()) == {"cos_cached", "sin_cached"}

    x = torch.randn(2, 3, 5, 8)
    torch.testing.assert_close(rope(x), _ref_rope(x, *_ref_cos_sin(5, 8)), atol=1e-5, rtol=1e-5)

    x1 = torch.randn(2, 3, 1, 8)
    expected = _ref_rope(x1, *_ref_cos_sin(1, 8, offset=4))
    torch.testing.assert_close(rope(x1, offset=4), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.depends_on("rope_cos_sin", "apply_rope", "apply_rope_to_qk")
def test_rope_pieces_agree():
    """The wrapper must be exactly apply_rope over rope_cos_sin - the same rotation, assembled."""
    B, H, T, Dh = 2, 3, 4, 8
    q, k = torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh)
    cos, sin = rope_cos_sin(T, Dh)
    qr, kr = apply_rope_to_qk(q, k)
    torch.testing.assert_close(qr, apply_rope(q, cos, sin), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(kr, apply_rope(k, cos, sin), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("d_head", [2, 8, 64])
def test_rope_relative_property_across_dims(d_head):
    q, k = torch.randn(d_head), torch.randn(d_head)
    torch.testing.assert_close(
        rope_dot_product(q, k, 7, 4),
        rope_dot_product(q, k, 3, 0),
        atol=1e-4,
        rtol=1e-4,
    )
