"""Topic 07 - einsum and einops.

einsum reads like this: "list the axes of the inputs, list the axes of the output;
everything that did not make it into the output is summed over".

einops is the same thing, but by names: rearrange (permuting/merging axes),
reduce (a reduction), repeat (replication). In CS336 einops is the main tool,
so the hands must type "b t (h d) -> b h t d" on their own.

The assignment: in this file you may NOT use .transpose/.permute/.view/.reshape
where einsum or einops is asked for. The whole point is to train exactly those.
"""

import pytest
import torch
from einops import rearrange, reduce, repeat
from torch import Tensor

# ==========================================================================
# 7.1 einsum: warm-up
# ==========================================================================


def es_trace(x: Tensor) -> Tensor:
    """The trace of a square matrix through einsum."""
    raise NotImplementedError


def es_diagonal(x: Tensor) -> Tensor:
    """The diagonal of a square matrix through einsum (without summing)."""
    raise NotImplementedError


def es_transpose(x: Tensor) -> Tensor:
    """Transposing a matrix through einsum."""
    raise NotImplementedError


def es_matmul(a: Tensor, b: Tensor) -> Tensor:
    """(M, K) @ (K, N) through einsum."""
    raise NotImplementedError


def es_outer(a: Tensor, b: Tensor) -> Tensor:
    """The outer product of (N,) and (M,) -> (N, M) through einsum."""
    raise NotImplementedError


def es_row_sums(x: Tensor) -> Tensor:
    """The row sums of a matrix through einsum."""
    raise NotImplementedError


# ==========================================================================
# 7.2 einsum: what it is actually there for
# ==========================================================================


def es_attention_scores(q: Tensor, k: Tensor) -> Tensor:
    """q: (B, H, T, Dh), k: (B, H, S, Dh) -> (B, H, T, S) without dividing by sqrt(Dh).

    One einsum line instead of transpose + matmul.
    """
    raise NotImplementedError


def es_attention_output(weights: Tensor, v: Tensor) -> Tensor:
    """weights: (B, H, T, S), v: (B, H, S, Dh) -> (B, H, T, Dh)."""
    raise NotImplementedError


def es_per_head_projection(x: Tensor, w: Tensor) -> Tensor:
    """x: (B, T, D), w: (H, D, Dh) -> (B, H, T, Dh).

    Every head has its own projection matrix - that is exactly the convenient way to think about MHA.
    """
    raise NotImplementedError


def es_weight_grad(x: Tensor, grad_out: Tensor) -> Tensor:
    """x: (N, in), grad_out: (N, out) -> (out, in): the weight gradient of a linear layer through einsum.

    The batch axis is summed over, which is why it is not in the output.
    """
    raise NotImplementedError


def es_batched_bilinear(x: Tensor, W: Tensor, y: Tensor) -> Tensor:
    """x: (B, D1), W: (D1, D2), y: (B, D2) -> (B,): x_i^T W y_i for every example of the batch.

    Careful: the batch is NOT summed over, everything else is.
    """
    raise NotImplementedError


# ==========================================================================
# 7.3 einops: rearrange
# ==========================================================================


def eo_split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, T, (H Dh)) -> (B, H, T, Dh) in a single rearrange."""
    raise NotImplementedError


def eo_merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, (H Dh)) in a single rearrange."""
    raise NotImplementedError


def eo_flatten_tokens(x: Tensor) -> Tensor:
    """(B, T, D) -> ((B T), D) in a single rearrange."""
    raise NotImplementedError


def eo_split_qkv(qkv: Tensor, num_heads: int) -> Tensor:
    """(B, T, (3 H Dh)) -> (3, B, H, T, Dh) in a single rearrange.

    The order inside the parentheses matters: first q/k/v, inside that the heads, inside that the channel.
    """
    raise NotImplementedError


def eo_group_tokens(x: Tensor, group: int) -> Tensor:
    """(B, T, D) -> (B, T/group, group, D): cut the sequence into non-overlapping blocks.

    This is how chunked attention and flash-attention-like schemes are done.
    """
    raise NotImplementedError


# ==========================================================================
# 7.4 einops: reduce and repeat
# ==========================================================================


def eo_mean_pool(x: Tensor) -> Tensor:
    """(B, T, D) -> (B, D): the mean over time through reduce."""
    raise NotImplementedError


def eo_max_pool_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, Dh): the maximum over the heads through reduce."""
    raise NotImplementedError


def eo_repeat_kv(kv: Tensor, num_groups: int) -> Tensor:
    """GQA through repeat: (B, H_kv, T, Dh) -> (B, (H_kv num_groups), T, Dh),
    where every kv head is repeated num_groups times in a row.
    """
    raise NotImplementedError


def eo_broadcast_mask(mask: Tensor, num_heads: int) -> Tensor:
    """(B, T, S) -> (B, H, T, S) through repeat."""
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_es_basics():
    x = torch.randn(5, 5)
    torch.testing.assert_close(es_trace(x), x.trace())
    torch.testing.assert_close(es_diagonal(x), x.diagonal())
    torch.testing.assert_close(es_transpose(x), x.T)

    a, b = torch.randn(3, 4), torch.randn(4, 6)
    torch.testing.assert_close(es_matmul(a, b), a @ b)

    u, v = torch.randn(3), torch.randn(5)
    torch.testing.assert_close(es_outer(u, v), torch.outer(u, v))
    torch.testing.assert_close(es_row_sums(x), x.sum(dim=1))


def test_es_attention():
    B, H, T, S, Dh = 2, 3, 4, 5, 6
    q, k, v = torch.randn(B, H, T, Dh), torch.randn(B, H, S, Dh), torch.randn(B, H, S, Dh)
    scores = es_attention_scores(q, k)
    assert scores.shape == (B, H, T, S)
    torch.testing.assert_close(scores, q @ k.transpose(-2, -1), atol=1e-5, rtol=1e-5)

    w = scores.softmax(dim=-1)
    torch.testing.assert_close(es_attention_output(w, v), w @ v, atol=1e-5, rtol=1e-5)


def test_es_per_head_projection():
    B, T, D, H, Dh = 2, 4, 8, 3, 5
    x = torch.randn(B, T, D)
    w = torch.randn(H, D, Dh)
    out = es_per_head_projection(x, w)
    assert out.shape == (B, H, T, Dh)
    torch.testing.assert_close(out[:, 1], x @ w[1], atol=1e-5, rtol=1e-5)


def test_es_weight_grad():
    x = torch.randn(6, 4, requires_grad=True)
    w = torch.randn(7, 4, requires_grad=True)
    out = torch.nn.functional.linear(x, w)
    g = torch.randn_like(out)
    out.backward(g)
    torch.testing.assert_close(es_weight_grad(x.detach(), g), w.grad, atol=1e-5, rtol=1e-5)


def test_es_batched_bilinear():
    x, W, y = torch.randn(4, 3), torch.randn(3, 5), torch.randn(4, 5)
    out = es_batched_bilinear(x, W, y)
    assert out.shape == (4,)
    expected = torch.stack([x[i] @ W @ y[i] for i in range(4)])
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_eo_heads_roundtrip():
    x = torch.randn(2, 5, 12)
    heads = eo_split_heads(x, 3)
    assert heads.shape == (2, 3, 5, 4)
    torch.testing.assert_close(heads[:, 1], x[..., 4:8])
    torch.testing.assert_close(eo_merge_heads(heads), x)


def test_eo_flatten_tokens():
    x = torch.randn(2, 3, 4)
    out = eo_flatten_tokens(x)
    assert out.shape == (6, 4)
    torch.testing.assert_close(out[4], x[1, 1])


def test_eo_split_qkv():
    B, T, H, Dh = 2, 4, 3, 5
    qkv = torch.randn(B, T, 3 * H * Dh)
    out = eo_split_qkv(qkv, H)
    assert out.shape == (3, B, H, T, Dh)
    q, v = out[0], out[2]
    torch.testing.assert_close(q, rearrange(qkv[..., : H * Dh], "b t (h d) -> b h t d", h=H))
    torch.testing.assert_close(v, rearrange(qkv[..., 2 * H * Dh :], "b t (h d) -> b h t d", h=H))


def test_eo_group_tokens():
    x = torch.randn(2, 8, 4)
    out = eo_group_tokens(x, 4)
    assert out.shape == (2, 2, 4, 4)
    torch.testing.assert_close(out[:, 0], x[:, :4])
    torch.testing.assert_close(out[:, 1], x[:, 4:])


def test_eo_reduce_and_repeat():
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(eo_mean_pool(x), x.mean(dim=1))

    h = torch.randn(2, 3, 4, 5)
    torch.testing.assert_close(eo_max_pool_heads(h), h.max(dim=1).values)

    kv = torch.randn(2, 2, 3, 4)
    out = eo_repeat_kv(kv, 3)
    assert out.shape == (2, 6, 3, 4)
    for h_i in range(6):
        torch.testing.assert_close(out[:, h_i], kv[:, h_i // 3])

    mask = torch.randn(2, 4, 4)
    m = eo_broadcast_mask(mask, 3)
    assert m.shape == (2, 3, 4, 4)
    torch.testing.assert_close(m[:, 2], mask)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_eo_heads_property(num_heads):
    x = torch.randn(3, 6, 8)
    torch.testing.assert_close(eo_merge_heads(eo_split_heads(x, num_heads)), x)
