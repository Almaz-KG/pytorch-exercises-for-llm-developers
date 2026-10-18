"""Topic 03 - broadcasting.

There is one rule, and it has to be known by heart:
shapes are aligned FROM THE RIGHT, an axis is compatible if the sizes are equal or one of them is 1,
and missing leading axes count as 1.

90% of the "magic" in transformer code is broadcasting: adding a bias, masks,
RoPE, ALiBi, per-head scaling. Once it is in the fingers, the code reads itself.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 3.1 The broadcasting rule by hand
# ==========================================================================


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Return the shape of the result of a op b under the broadcasting rules.
    If the shapes are incompatible - raise ValueError.

    Implement the rule manually (without torch.broadcast_shapes) - this is an exercise in understanding.
    """
    result: list[int] = []

    n = max(len(a), len(b))

    for i in range(1, n + 1):
        ai = a[-i] if i <= len(a) else 1
        bi = b[-i] if i <= len(b) else 1

        if ai != bi and ai != 1 and bi != 1:
            raise ValueError("Ooops")
        result.append(max(ai, bi))

    return tuple(reversed(result))


def explain_trap() -> tuple[int, ...]:
    """The classic trap: a (3, 1) tensor plus a (3,) tensor.
    Return the shape of the result. Many expect (3,) or (3, 1) and end up with a bug in the loss.
    """
    return 3, 3


# ==========================================================================
# 3.2 Typical uses in a transformer
# ==========================================================================


def add_bias(x: Tensor, bias: Tensor) -> Tensor:
    """x: (B, T, D), bias: (D,) -> (B, T, D)."""
    return x + bias[None, None, :]


def scale_per_head(x: Tensor, scales: Tensor) -> Tensor:
    """x: (B, H, T, Dh), scales: (H,). Multiply each head by its own scalar.

    Hint: fit the shape of scales so that it aligns from the right: (1, H, 1, 1).
    """
    return x * scales[None, :, None, None]


def normalize_last_dim(x: Tensor) -> Tensor:
    """Subtract its own mean from every vector along the last axis. The shape does not change.

    This is where keepdim=True lives: without it the shapes will not align.
    """
    return x - x.mean(dim=-1, keepdim=True)


def weighted_sum(weights: Tensor, values: Tensor) -> Tensor:
    """weights: (B, T), values: (B, T, D) -> (B, D): the sum of the values weighted over time.

    This is attention in miniature: weights on the tokens, a weighted sum of their representations.
    """
    return (values * weights[:, :, None]).sum(dim=1)


# ==========================================================================
# 3.3 Matrices out of vectors
# ==========================================================================


def outer_product(a: Tensor, b: Tensor) -> Tensor:
    """a: (N,), b: (M,) -> (N, M), out[i, j] = a[i] * b[j]. Broadcasting only, no torch.outer."""
    return a[:, None] * b[None, :]


def pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """a: (N, D), b: (M, D) -> (N, M) of squared euclidean distances.

    Through broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D), then a sum over D.
    """
    return (a[:, None, :] - b[None, :, :]).pow(2).sum(dim=-1)


def one_hot_by_comparison(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (...,) int64 -> (..., vocab_size) float32 one-hot.

    Through a comparison against arange by broadcasting, without F.one_hot.
    """
    return (ids[..., None] == torch.arange(vocab_size, device=ids.device)).float()


# ==========================================================================
# 3.4 Masks and relative positions
# ==========================================================================


def causal_mask(seq_len: int) -> Tensor:
    """(T, T) bool: True where query position i MAY look at key position j (j <= i).

    Build it by comparing two arange tensors of different shape, without torch.tril.
    """
    return torch.arange(seq_len)[None, :] <= torch.arange(seq_len)[:, None]


def relative_positions(q_len: int, k_len: int) -> Tensor:
    """(q_len, k_len) int64, out[i, j] = i - j. The basis of relative position bias and ALiBi."""
    q = torch.arange(q_len)[:, None]
    k = torch.arange(k_len)[None, :]

    return q - k


def alibi_bias(slopes: Tensor, seq_len: int) -> Tensor:
    """slopes: (H,) -> (H, T, T) additive ALiBi bias.

    bias[h, i, j] = -slopes[h] * (i - j) for j <= i, and 0 (does not matter) above the diagonal;
    here, for simplicity, compute the formula for all j, without zeroing anything out.
    """
    i = torch.arange(seq_len)[:, None]
    j = torch.arange(seq_len)[None, :]
    return -slopes[:, None, None] * (i - j).float()


def apply_additive_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """scores: (B, H, T, T), mask: (T, T) bool (True = allowed to look).
    Return scores where the forbidden positions are -inf.

    No loops: the mask has to spread across the batch and the heads on its own.
    """
    return scores.masked_fill(mask=~mask, value=float("-inf"))


def key_padding_mask_apply(scores: Tensor, pad_mask: Tensor) -> Tensor:
    """scores: (B, H, T, T), pad_mask: (B, T) bool (True = a real token, False = padding).
    Forbid EVERY query from looking at padding keys: the corresponding columns -> -inf.

    Careful with the axes: pad_mask refers to the keys, that is to the LAST axis of scores.
    """
    return scores.masked_fill(mask=~pad_mask[:, None, None, :], value=float("-inf"))


# ==========================================================================
# TESTS
# ==========================================================================


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((3, 1), (3,), (3, 3)),
        ((5, 1, 4), (3, 4), (5, 3, 4)),
        ((2, 3, 4), (4,), (2, 3, 4)),
        ((1,), (7, 5), (7, 5)),
        ((), (2, 2), (2, 2)),
        ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),
    ],
)
def test_broadcast_shape_ok(a, b, expected):  # type: ignore
    assert broadcast_shape(a, b) == expected  # type: ignore
    # cross-check against torch itself
    assert tuple(torch.broadcast_shapes(a, b)) == expected  # type: ignore


@pytest.mark.parametrize("a,b", [((3,), (4,)), ((2, 3), (3, 2)), ((5, 2), (5, 3))])
def test_broadcast_shape_error(a, b):  # type: ignore
    with pytest.raises(ValueError):
        broadcast_shape(a, b)  # type: ignore


def test_explain_trap():
    assert explain_trap() == (3, 3)


def test_add_bias():
    x = torch.randn(2, 3, 4)
    b = torch.randn(4)
    out = add_bias(x, b)
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out[1, 2], x[1, 2] + b)


def test_scale_per_head():
    x = torch.randn(2, 3, 5, 4)
    s = torch.tensor([1.0, 2.0, 3.0])
    out = scale_per_head(x, s)
    assert out.shape == x.shape
    for h in range(3):
        torch.testing.assert_close(out[:, h], x[:, h] * s[h])


def test_normalize_last_dim():
    x = torch.randn(2, 3, 8)
    out = normalize_last_dim(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)


def test_weighted_sum():
    w = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    out = weighted_sum(w, v)
    assert out.shape == (2, 2)
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0], [5.0, 6.0]]))


def test_outer_product():
    a, b = torch.randn(4), torch.randn(6)
    out = outer_product(a, b)
    assert out.shape == (4, 6)
    torch.testing.assert_close(out, torch.outer(a, b))


def test_pairwise_sq_dist():
    a, b = torch.randn(5, 3), torch.randn(7, 3)
    out = pairwise_sq_dist(a, b)
    assert out.shape == (5, 7)
    torch.testing.assert_close(out, torch.cdist(a, b).pow(2), atol=1e-4, rtol=1e-4)


def test_one_hot_by_comparison():
    ids = torch.tensor([[0, 3], [2, 1]])
    out = one_hot_by_comparison(ids, 4)
    assert out.shape == (2, 2, 4)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.nn.functional.one_hot(ids, 4).float())


def test_causal_mask():
    m = causal_mask(4)
    assert m.dtype == torch.bool
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(m, expected)


def test_relative_positions():
    r = relative_positions(3, 4)
    assert r.shape == (3, 4)
    torch.testing.assert_close(r, torch.tensor([[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1]]))


def test_alibi_bias():
    slopes = torch.tensor([0.5, 0.25])
    bias = alibi_bias(slopes, 3)
    assert bias.shape == (2, 3, 3)
    # the further back a query looks, the stronger the penalty
    torch.testing.assert_close(bias[0, 2, 0], torch.tensor(-1.0))
    torch.testing.assert_close(bias[0, 2, 1], torch.tensor(-0.5))
    torch.testing.assert_close(bias[1, 2, 0], torch.tensor(-0.5))
    torch.testing.assert_close(bias[0, 1, 1], torch.tensor(0.0))


def test_apply_additive_mask():
    scores = torch.randn(2, 3, 4, 4)
    idx = torch.arange(4)
    mask = idx[:, None] >= idx[None, :]  # built here so this test needs causal_mask from nobody
    out = apply_additive_mask(scores, mask)
    assert out.shape == scores.shape
    assert torch.isinf(out[:, :, 0, 1]).all() and (out[:, :, 0, 1] < 0).all()
    torch.testing.assert_close(out[:, :, 3, 0], scores[:, :, 3, 0])
    assert torch.isfinite(out[mask.expand_as(out)]).all()


def test_key_padding_mask_apply():
    scores = torch.zeros(2, 2, 3, 3)
    pad = torch.tensor([[True, True, False], [True, False, False]])
    out = key_padding_mask_apply(scores, pad)
    assert torch.isinf(out[0, :, :, 2]).all()
    assert torch.isfinite(out[0, :, :, :2]).all()
    assert torch.isinf(out[1, :, :, 1:]).all()
    assert torch.isfinite(out[1, :, :, 0]).all()
