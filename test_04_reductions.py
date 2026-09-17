"""Topic 04 - reductions: sum, mean, max, topk, logsumexp, norms.

The key things to bring to the point of automatism:
  - dim=... is the axis THAT DISAPPEARS (or becomes 1 with keepdim=True);
  - keepdim=True is always needed when the result is broadcast back afterwards;
  - logsumexp is not "one more function" but the way not to overflow exp;
  - in an LLM almost every averaging comes with a mask over padding tokens.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 4.1 Basic reductions and axes
# ==========================================================================


def sum_over_time(x: Tensor) -> Tensor:
    """(B, T, D) -> (B, D): the sum over the time axis."""
    raise NotImplementedError


def mean_over_batch_and_time(x: Tensor) -> Tensor:
    """(B, T, D) -> (D,): the mean over two axes at once."""
    raise NotImplementedError


def center_and_keepdim(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return (the mean over the last axis with keepdim=True, x minus that mean).

    The first element must have shape (..., 1) - precisely because of keepdim.
    """
    raise NotImplementedError


def biased_variance(x: Tensor) -> Tensor:
    """The variance over the last axis with denominator N (not N-1), keepdim=True.

    LayerNorm uses exactly the biased estimate - it is easy to slip on the unbiased default.
    """
    raise NotImplementedError


def rms(x: Tensor) -> Tensor:
    """sqrt(mean(x^2)) over the last axis, keepdim=True. The heart of RMSNorm."""
    raise NotImplementedError


# ==========================================================================
# 4.2 Reductions with a mask (padding)
# ==========================================================================


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """x: (B, T, D), mask: (B, T) bool (True = a real token) -> (B, D).

    The average must be taken over the real tokens only. Divide by mask.sum(), not by T.
    """
    raise NotImplementedError


def count_tokens(mask: Tensor) -> tuple[Tensor, int]:
    """mask: (B, T) bool -> (the sequence lengths (B,) int64, the total number of tokens as a python int)."""
    raise NotImplementedError


def masked_max(x: Tensor, mask: Tensor) -> Tensor:
    """x: (B, T), mask: (B, T) bool -> (B,): the maximum over the allowed positions only.

    Hint: fill the forbidden positions with -inf before the max.
    """
    raise NotImplementedError


# ==========================================================================
# 4.3 max / argmax / topk - decoding
# ==========================================================================


def greedy_tokens(logits: Tensor) -> Tensor:
    """logits: (B, T, V) -> (B, T) int64: the index of the most probable token at each position."""
    raise NotImplementedError


def top_k_tokens(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """logits: (B, V) -> (values (B, k), indices (B, k)), sorted in descending order."""
    raise NotImplementedError


def argmax_2d(x: Tensor) -> tuple[int, int]:
    """Find the (row, column) of the maximum of a matrix. Through a flat argmax and divmod, without loops."""
    raise NotImplementedError


def accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> float:
    """logits: (B, T, V), targets: (B, T) -> the fraction of correct predictions,
    with the positions where targets == ignore_index not counted at all.

    Return a python float.
    """
    raise NotImplementedError


# ==========================================================================
# 4.4 logsumexp and entropy
# ==========================================================================


def logsumexp_manual(x: Tensor, dim: int) -> Tensor:
    """A numerically stable log(sum(exp(x))) over the axis dim, WITHOUT torch.logsumexp.

    The scheme: m = max(x, dim, keepdim=True); m + log(sum(exp(x - m))).
    It must work on inputs like 1000.0, where a naive exp gives inf.
    """
    raise NotImplementedError


def entropy(probs: Tensor) -> Tensor:
    """probs: (..., V), summing to 1 along the last axis -> (...,) the entropy in nats.

    Careful with log(0): zero probabilities must contribute 0, not nan.
    """
    raise NotImplementedError


# ==========================================================================
# 4.5 Norms
# ==========================================================================


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Divide every vector along the last axis by its L2 norm."""
    raise NotImplementedError


def global_grad_norm(tensors: list[Tensor]) -> Tensor:
    """The total L2 norm of a list of tensors: sqrt(the sum of squares of all elements of all tensors).

    This is exactly the quantity that stands in gradient clipping.
    """
    raise NotImplementedError


def token_frequencies(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (...,) int64 -> (vocab_size,) int64 counter of the occurrences of every id.

    Hint: torch.bincount, do not forget minlength and flatten.
    """
    raise NotImplementedError


def cumulative_mean(x: Tensor) -> Tensor:
    """(T,) -> (T,): out[i] = the mean of x[0..i]. Through cumsum, without loops."""
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_sum_over_time():
    x = torch.randn(2, 5, 3)
    out = sum_over_time(x)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, x.sum(dim=1))


def test_mean_over_batch_and_time():
    x = torch.randn(2, 5, 3)
    out = mean_over_batch_and_time(x)
    assert out.shape == (3,)
    torch.testing.assert_close(out, x.reshape(-1, 3).mean(dim=0))


def test_center_and_keepdim():
    x = torch.randn(2, 3, 4)
    mean, centered = center_and_keepdim(x)
    assert mean.shape == (2, 3, 1), "keepdim was lost"
    torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)


def test_biased_variance():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    v = biased_variance(x)
    assert v.shape == (1, 1)
    torch.testing.assert_close(v, torch.tensor([[2.0 / 3.0]]))  # not 1.0, the way unbiased would give


def test_rms():
    x = torch.tensor([[3.0, 4.0]])
    torch.testing.assert_close(rms(x), torch.tensor([[(12.5) ** 0.5]]))


def test_masked_mean():
    x = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [100.0, 100.0]]])
    mask = torch.tensor([[True, True, False]])
    out = masked_mean(x, mask)
    assert out.shape == (1, 2)
    torch.testing.assert_close(out, torch.tensor([[2.0, 2.0]]))


def test_count_tokens():
    mask = torch.tensor([[True, True, False], [True, False, False]])
    lengths, total = count_tokens(mask)
    assert lengths.dtype == torch.int64
    torch.testing.assert_close(lengths, torch.tensor([2, 1]))
    assert total == 3 and isinstance(total, int)


def test_masked_max():
    x = torch.tensor([[1.0, 5.0, 2.0], [7.0, 0.0, 3.0]])
    mask = torch.tensor([[True, False, True], [False, True, True]])
    torch.testing.assert_close(masked_max(x, mask), torch.tensor([2.0, 3.0]))


def test_greedy_tokens():
    logits = torch.tensor([[[0.1, 0.9, 0.0], [5.0, 1.0, 2.0]]])
    out = greedy_tokens(logits)
    assert out.shape == (1, 2)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([[1, 0]]))


def test_top_k_tokens():
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
    values, idx = top_k_tokens(logits, 2)
    torch.testing.assert_close(values, torch.tensor([[3.0, 2.0]]))
    torch.testing.assert_close(idx, torch.tensor([[1, 3]]))


def test_argmax_2d():
    x = torch.zeros(3, 4)
    x[2, 1] = 9.0
    assert argmax_2d(x) == (2, 1)


def test_accuracy():
    logits = torch.zeros(1, 4, 3)
    logits[0, 0, 1] = 1.0  # predicted 1
    logits[0, 1, 0] = 1.0  # predicted 0
    logits[0, 2, 2] = 1.0  # predicted 2
    logits[0, 3, 0] = 1.0  # predicted 0, but the position is ignored
    targets = torch.tensor([[1, 2, 2, -100]])
    acc = accuracy(logits, targets)
    assert isinstance(acc, float)
    assert abs(acc - 2 / 3) < 1e-6


def test_logsumexp_manual_matches_torch():
    x = torch.randn(4, 5, 6)
    for dim in (0, 1, 2, -1):
        torch.testing.assert_close(logsumexp_manual(x, dim), torch.logsumexp(x, dim=dim))


def test_logsumexp_manual_is_stable():
    x = torch.tensor([1000.0, 1000.0, 1000.0])
    out = logsumexp_manual(x, dim=0)
    assert torch.isfinite(out).all(), "the naive exp overflowed"
    torch.testing.assert_close(out, torch.tensor(1000.0 + torch.log(torch.tensor(3.0))))

    # masked positions arrive as -inf and must not break the result
    y = torch.tensor([0.0, float("-inf"), 0.0])
    torch.testing.assert_close(logsumexp_manual(y, dim=0), torch.logsumexp(y, dim=0))


def test_entropy():
    uniform = torch.full((1, 4), 0.25)
    torch.testing.assert_close(entropy(uniform), torch.log(torch.tensor([4.0])))

    deterministic = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = entropy(deterministic)
    assert torch.isfinite(out).all(), "log(0) produced nan - handle the zero probabilities"
    torch.testing.assert_close(out, torch.zeros(1), atol=1e-6, rtol=0)


def test_l2_normalize():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize(x)
    torch.testing.assert_close(out[0], torch.tensor([0.6, 0.8]))
    assert torch.isfinite(out).all(), "a zero vector must not give nan - that is what eps is for"


def test_global_grad_norm():
    a = torch.tensor([3.0, 4.0])
    b = torch.tensor([[12.0]])
    out = global_grad_norm([a, b])
    assert out.ndim == 0
    torch.testing.assert_close(out, torch.tensor(13.0))


def test_token_frequencies():
    ids = torch.tensor([[0, 2, 2], [1, 2, 0]])
    counts = token_frequencies(ids, vocab_size=5)
    assert counts.shape == (5,)
    torch.testing.assert_close(counts, torch.tensor([2, 1, 3, 0, 0]))


def test_cumulative_mean():
    x = torch.tensor([2.0, 4.0, 6.0])
    torch.testing.assert_close(cumulative_mean(x), torch.tensor([2.0, 3.0, 4.0]))


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4)])
def test_l2_normalize_unit_norm(shape):
    x = torch.randn(*shape) + 1.0
    out = l2_normalize(x)
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(shape[:-1]), atol=1e-5, rtol=0)
