"""Topic 02 - shapes, views, axis permutations.

What we drill: reshape/view, permute/transpose, contiguous, squeeze/unsqueeze,
cat/stack/split/chunk, expand vs repeat, unfold (sliding window), pad.

This is the most frequent source of bugs in LLM code: "why do I have (B, H, T, D) when I need (B, T, H, D)".
The axis naming convention used throughout the course:
  B - batch, T - sequence length (time), D - d_model, H - number of heads, Dh - head size.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 2.1 reshape / view / flatten
# ==========================================================================


def flatten_tokens(x: Tensor) -> Tensor:
    """(B, T, D) -> (B*T, D). This is exactly how the batch is collapsed before lm_head or the loss."""
    return x.view((-1, x.shape[-1]))


def unflatten_tokens(x: Tensor, batch_size: int) -> Tensor:
    """(B*T, D) -> (B, T, D). The inverse operation."""
    return x.view((batch_size, -1, x.shape[-1]))


def flatten_all_but_last(x: Tensor) -> Tensor:
    """(..., D) -> (N, D), where N is the product of all the leading axes.

    Hint: -1 in reshape is computed automatically.
    """
    return x.view((-1, x.shape[-1]))


def add_batch_dim(x: Tensor) -> Tensor:
    """(T, D) -> (1, T, D) without copying the data."""
    return x.unsqueeze(dim=0)


def drop_singleton_dims(x: Tensor) -> Tensor:
    """Remove ALL axes of size 1."""
    return x.squeeze()


# ==========================================================================
# 2.2 Permuting axes
# ==========================================================================


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh), where D = H * Dh.

    The order matters: first split the last axis into (H, Dh), then permute the axes.
    Do it the other way round and the heads get mixed up, which the test will catch.
    """
    B, T, D = x.shape
    return x.view((B, T, num_heads, D // num_heads)).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D). Exactly the inverse of split_heads.

    Careful: after permute the tensor is not contiguous and .view() will fail. Think about the fix.
    """
    B, H, T, D = x.shape
    return x.transpose(1, 2).reshape((B, T, H * D))


def transpose_last_two(x: Tensor) -> Tensor:
    """Swap the last two axes of a tensor of any rank.

    This is how K^T is done in attention: (B, H, T, Dh) -> (B, H, Dh, T).
    """
    return x.transpose(-2, -1)


def is_contiguous_after_transpose(x: Tensor) -> tuple[bool, bool]:
    """Return (contiguous before transposing, contiguous after .transpose(0, 1)).

    A question of understanding: transposing is only a change of strides, the data does not move.
    """
    assert x.is_contiguous() != x.transpose(0, 1).is_contiguous()
    return x.is_contiguous(), x.transpose(0, 1).is_contiguous()


def strides_of(x: Tensor) -> tuple[int, ...]:
    """The steps (in elements) along each axis. For a contiguous (2,3,4) float32 that is (12, 4, 1)."""
    return x.stride()


# ==========================================================================
# 2.3 Joining and cutting
# ==========================================================================


def concat_sequences(a: Tensor, b: Tensor) -> Tensor:
    """Glue two tensors (B, T1, D) and (B, T2, D) along time -> (B, T1+T2, D).

    This is literally what the KV cache does when a new token is appended.
    """
    return torch.concat((a, b), dim=1)


def stack_batch(items: list[Tensor]) -> Tensor:
    """Collect a list of N tensors (T, D) into a batch (N, T, D)."""
    return torch.stack(items)


def split_qkv(qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(B, T, 3*D) -> three tensors (B, T, D): q, k, v.

    This is how a fused qkv projector works: one matrix instead of three, then a cut.
    """
    q, k, v = torch.split(qkv, qkv.shape[-1] // 3, dim=-1)
    return q, k, v


# ==========================================================================
# 2.4 expand vs repeat - the difference in memory
# ==========================================================================


def broadcast_kv_heads(kv: Tensor, num_groups: int) -> Tensor:
    """GQA: (B, H_kv, T, Dh) -> (B, H_kv * num_groups, T, Dh), WITHOUT copying the data
    wherever that is possible, and with the right head order:
    kv head number i must repeat num_groups times in a row.

    Hint: unsqueeze -> expand -> reshape. The reshape at the end will copy,
    but the expand along the way must not allocate a new buffer.
    """
    B, H, T, D = kv.shape
    return kv.unsqueeze(dim=2).expand([-1, -1, num_groups, -1, -1]).reshape((B, H * num_groups, T, D))


def expand_row(x: Tensor, n: int) -> Tensor:
    """(1, D) -> (n, D) WITHOUT copying (a view with zero stride)."""
    return x.expand([n, -1])


def repeat_row(x: Tensor, n: int) -> Tensor:
    """(1, D) -> (n, D) WITH an actual copy of the data."""
    return x.repeat(n, 1)


# ==========================================================================
# 2.5 Sliding window and padding
# ==========================================================================


def sliding_windows(x: Tensor, window: int) -> Tensor:
    """A one-dimensional tensor of length T -> every window of length `window` with step 1: (T-window+1, window).

    This is how training examples are cut out of a continuous stream of tokens.
    Hint: Tensor.unfold(dim, size, step).
    """
    return x.unfold(dimension=-1, size=window, step=1)


def pad_to_length(x: Tensor, target_len: int, pad_value: int) -> Tensor:
    """(B, T) -> (B, target_len), filling up on the right with pad_value. Assume T <= target_len.

    Hint: F.pad and the order of the pairs in its argument - from the last axis to the first.
    """
    return F.pad(x, pad=[0, target_len - x.shape[-1]], value=pad_value)


def shift_right(x: Tensor, pad_value: int) -> Tensor:
    """(B, T) -> (B, T): a shift right by one position, pad_value is inserted on the left,
    the last token falls off. Classic teacher forcing: turning the input into the decoder input.
    """
    return F.pad(x[:, :-1], [1, 0], value=pad_value)


# ==========================================================================
# TESTS
# ==========================================================================


def test_flatten_tokens():
    x = torch.randn(2, 3, 4)
    flat = flatten_tokens(x)
    assert flat.shape == (6, 4)
    torch.testing.assert_close(flat[0], x[0, 0])
    torch.testing.assert_close(flat[3], x[1, 0])


def test_unflatten_tokens():
    flat = torch.randn(6, 4)
    out = unflatten_tokens(flat, 2)
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out[0, 0], flat[0])
    torch.testing.assert_close(out[1, 0], flat[3])


@pytest.mark.depends_on("flatten_tokens", "unflatten_tokens")
def test_flatten_unflatten_roundtrip():
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(unflatten_tokens(flatten_tokens(x), 2), x)


def test_flatten_all_but_last():
    assert flatten_all_but_last(torch.randn(2, 3, 4, 5)).shape == (24, 5)
    assert flatten_all_but_last(torch.randn(7, 5)).shape == (7, 5)
    x = torch.randn(2, 3, 5)
    torch.testing.assert_close(flatten_all_but_last(x)[4], x[1, 1])


def test_add_batch_dim():
    x = torch.randn(3, 4)
    y = add_batch_dim(x)
    assert y.shape == (1, 3, 4)
    assert y.data_ptr() == x.data_ptr(), "unsqueeze must not copy"


def test_drop_singleton_dims():
    assert drop_singleton_dims(torch.randn(1, 3, 1, 4, 1)).shape == (3, 4)
    assert drop_singleton_dims(torch.randn(2, 3)).shape == (2, 3)


def test_split_heads():
    B, T, H, Dh = 2, 5, 4, 8
    x = torch.randn(B, T, H * Dh)
    out = split_heads(x, H)
    assert out.shape == (B, H, T, Dh)
    # head h, position t is the slice [h*Dh : (h+1)*Dh] of the original vector
    for h in range(H):
        torch.testing.assert_close(out[:, h, :, :], x[..., h * Dh : (h + 1) * Dh])


def test_merge_heads():
    B, H, T, Dh = 2, 4, 5, 8
    x = torch.randn(B, H, T, Dh)
    out = merge_heads(x)
    assert out.shape == (B, T, H * Dh)
    # position t of the merged vector is the heads laid end to end
    torch.testing.assert_close(out[1, 2], x[1, :, 2, :].reshape(H * Dh))


@pytest.mark.depends_on("split_heads", "merge_heads")
def test_merge_heads_roundtrip():
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(merge_heads(split_heads(x, 4)), x)


def test_transpose_last_two():
    x = torch.randn(2, 4, 5, 8)
    out = transpose_last_two(x)
    assert out.shape == (2, 4, 8, 5)
    torch.testing.assert_close(out[1, 2, 3, 4], x[1, 2, 4, 3])
    assert transpose_last_two(torch.randn(3, 7)).shape == (7, 3)


def test_is_contiguous_after_transpose():
    before, after = is_contiguous_after_transpose(torch.randn(3, 4))
    assert before is True
    assert after is False


def test_strides_of():
    assert strides_of(torch.zeros(2, 3, 4)) == (12, 4, 1)
    assert strides_of(torch.zeros(2, 3, 4).transpose(0, 1)) == (4, 12, 1)


def test_concat_sequences():
    a, b = torch.randn(2, 3, 8), torch.randn(2, 1, 8)
    out = concat_sequences(a, b)
    assert out.shape == (2, 4, 8)
    torch.testing.assert_close(out[:, :3], a)
    torch.testing.assert_close(out[:, 3:], b)


def test_stack_batch():
    items = [torch.randn(3, 4) for _ in range(5)]
    out = stack_batch(items)
    assert out.shape == (5, 3, 4)
    torch.testing.assert_close(out[2], items[2])


def test_split_qkv():
    B, T, D = 2, 3, 8
    qkv = torch.randn(B, T, 3 * D)
    q, k, v = split_qkv(qkv)
    for t in (q, k, v):
        assert t.shape == (B, T, D)
    torch.testing.assert_close(q, qkv[..., :D])
    torch.testing.assert_close(k, qkv[..., D : 2 * D])
    torch.testing.assert_close(v, qkv[..., 2 * D :])


def test_broadcast_kv_heads():
    B, H_kv, T, Dh = 2, 2, 3, 4
    kv = torch.randn(B, H_kv, T, Dh)
    out = broadcast_kv_heads(kv, 3)
    assert out.shape == (B, 6, T, Dh)
    # heads 0,1,2 are copies of kv head 0; heads 3,4,5 are copies of kv head 1
    for h in range(6):
        torch.testing.assert_close(out[:, h], kv[:, h // 3])


def test_expand_row():
    x = torch.randn(1, 5)
    e = expand_row(x, 4)
    assert e.shape == (4, 5)
    torch.testing.assert_close(e[3], x[0])
    assert e.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(), "expand does not copy"
    assert e.stride(0) == 0, "for expand the stride along the broadcast axis is zero"


def test_repeat_row():
    x = torch.randn(1, 5)
    r = repeat_row(x, 4)
    assert r.shape == (4, 5)
    torch.testing.assert_close(r[3], x[0])
    assert r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr(), "repeat does copy"
    r[0, 0] = 100.0
    assert x[0, 0].item() != 100.0, "a real copy must not write back into the source"


@pytest.mark.depends_on("expand_row", "repeat_row")
def test_expand_and_repeat_hold_the_same_values():
    """Same numbers, different memory: that is the whole distinction."""
    x = torch.randn(1, 5)
    torch.testing.assert_close(expand_row(x, 4), repeat_row(x, 4))


def test_sliding_windows():
    x = torch.arange(6)
    out = sliding_windows(x, 3)
    assert out.shape == (4, 3)
    torch.testing.assert_close(out[0], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(out[3], torch.tensor([3, 4, 5]))


def test_pad_to_length():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = pad_to_length(x, 5, pad_value=0)
    assert out.shape == (2, 5)
    torch.testing.assert_close(out, torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]))


def test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = shift_right(x, pad_value=-1)
    torch.testing.assert_close(out, torch.tensor([[-1, 1, 2], [-1, 4, 5]]))
    assert out.dtype == x.dtype


@pytest.mark.depends_on("split_heads", "merge_heads")
@pytest.mark.parametrize("shape", [(2, 3, 8), (1, 1, 8), (4, 7, 16)])
def test_heads_roundtrip_property(shape):  # type: ignore
    x = torch.randn(*shape)  # type: ignore
    torch.testing.assert_close(merge_heads(split_heads(x, 2)), x)
