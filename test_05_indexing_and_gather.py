"""Topic 05 - indexing, gather/scatter, masks.

Here live the operations without which neither a loss nor sampling can be written:
  - an embedding lookup is just indexing a matrix;
  - "the probability of the correct token" is a gather;
  - "keep only the top-k logits" is topk + scatter;
  - "take the last non-empty token" is a gather over the lengths.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 5.1 Indexing as an embedding lookup
# ==========================================================================


def embedding_lookup(table: Tensor, ids: Tensor) -> Tensor:
    """table: (V, D), ids: (B, T) int64 -> (B, T, D). With ordinary indexing, without nn.Embedding."""
    return table[ids]


def select_rows(x: Tensor, rows: list[int]) -> Tensor:
    """Pick the given rows of a matrix in the given order (repeats allowed)."""
    return x[rows]


def fancy_index_pairs(x: Tensor, batch_idx: Tensor, pos_idx: Tensor) -> Tensor:
    """x: (B, T, D), batch_idx: (N,), pos_idx: (N,) -> (N, D),
    where out[n] = x[batch_idx[n], pos_idx[n]].

    This is "advanced indexing with two tensors" - the index axes go in pairs.
    """
    return x[batch_idx, pos_idx]


# ==========================================================================
# 5.2 gather - pulling out values by index
# ==========================================================================


def gather_target_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """logits: (B, T, V), targets: (B, T) -> (B, T): the logit of the correct token.

    The most frequent gather construction in all of LM training.
    Hint: torch.gather requires index to have as many axes as input.
    """
    return torch.gather(logits, dim=-1, index=targets.unsqueeze(dim=-1)).squeeze()


def last_valid_hidden(h: Tensor, lengths: Tensor) -> Tensor:
    """h: (B, T, D), lengths: (B,) int64 (the number of real tokens) -> (B, D):
    the hidden state of the last real token of each sequence.

    This is how pooling for classification is done on a decoder-only model.
    """
    return h[torch.arange(h.shape[0]), lengths - 1]


def gather_beams(x: Tensor, beam_idx: Tensor) -> Tensor:
    """x: (B, D), beam_idx: (B,) -> (B, D): reorder the rows by the indices.
    This is how beam search reshuffles the KV cache after every step.
    """
    return x[beam_idx]


# ==========================================================================
# 5.3 scatter - writing by index
# ==========================================================================


def scatter_one_hot(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (B, T) -> (B, T, V) float one-hot, assembled through scatter_ (not F.one_hot)."""
    B, T = ids.shape
    zeros = torch.zeros((B, T, vocab_size), dtype=torch.float32)

    return zeros.scatter_(dim=-1, index=ids.unsqueeze(dim=-1), value=1)


def bag_of_words(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (B, T) -> (B, V) float: how many times each token occurred in the row.

    Hint: scatter_add_ with a source of ones.
    """
    zeros = torch.zeros((ids.shape[0], vocab_size), dtype=torch.float32)

    return zeros.scatter_add_(dim=-1, index=ids, src=torch.ones_like(ids, dtype=torch.float32))


def keep_top_k(logits: Tensor, k: int) -> Tensor:
    """logits: (B, V) -> (B, V), where everything except the top-k values in the row is replaced by -inf.

    This exact trick sits inside top-k sampling.
    Hint: topk gives the indices; make a tensor of -inf and scatter the found values into it.
    """
    vals, idx = torch.topk(logits, k=k)
    inf = torch.full(logits.shape, fill_value=float("-inf"))

    return inf.scatter_(dim=1, index=idx, src=vals)


# ==========================================================================
# 5.4 Boolean masks
# ==========================================================================


def select_valid(x: Tensor, mask: Tensor) -> Tensor:
    """x: (B, T, D), mask: (B, T) bool -> (N, D): only the real tokens, "straightened out" into one list.

    This is how packing before the loss is done, so that no gradients are computed over padding.
    """
    return x[mask]


def replace_where(x: Tensor, cond: Tensor, value: float) -> Tensor:
    """Return a copy of x where value stands wherever cond holds (leave in-place alone)."""
    x = x.clone()
    x[cond] = value
    return x


def find_positions(ids: Tensor, token_id: int) -> Tensor:
    """ids: (T,) -> (N,) int64: the positions where token_id stands.

    Useful for finding <|endoftext|> in a stream of tokens.
    """
    return torch.nonzero(ids == token_id).squeeze()


def mask_after_eos(ids: Tensor, eos_id: int) -> Tensor:
    """ids: (B, T) -> (B, T) bool: True for the positions UP TO the first eos inclusive,
    False for everything after it. If there is no eos - the whole row is True.

    Hint: a cumsum over the boolean tensor "there was an eos here" gives the number of eos seen;
    compare it against 1 carefully, so that the eos itself stays True. Without loops.
    """
    zeros = torch.zeros(ids.shape)
    zeros[ids == eos_id] = 1
    return (zeros.cumsum(dim=-1) - zeros) < 1


# ==========================================================================
# 5.5 Preparing training pairs
# ==========================================================================


def make_lm_pairs(ids: Tensor) -> tuple[Tensor, Tensor]:
    """ids: (B, T+1) -> (inputs (B, T), targets (B, T)), a shift by one token.

    At position i the model predicts token i+1 - that is the whole of next token prediction.
    """
    inpt = ids[:, :-1]
    target = ids[:, 1:]

    return inpt, target


def expand_for_beams(x: Tensor, num_beams: int) -> Tensor:
    """x: (B, T) -> (B * num_beams, T), where the rows of ONE example go one after another.

    Hint: repeat_interleave, not repeat - the order differs, and that is the trap.
    """
    return x.repeat_interleave(repeats=num_beams, dim=0)


# ==========================================================================
# TESTS
# ==========================================================================


def test_embedding_lookup():
    table = torch.randn(10, 4)
    ids = torch.tensor([[0, 3], [9, 3]])
    out = embedding_lookup(table, ids)
    assert out.shape == (2, 2, 4)
    torch.testing.assert_close(out[0, 1], table[3])
    torch.testing.assert_close(out[1, 0], table[9])
    torch.testing.assert_close(out, torch.nn.functional.embedding(ids, table))


def test_select_rows():
    x = torch.arange(12).reshape(4, 3)
    out = select_rows(x, [2, 0, 2])
    assert out.shape == (3, 3)
    torch.testing.assert_close(out, torch.stack([x[2], x[0], x[2]]))


def test_fancy_index_pairs():
    x = torch.randn(3, 5, 2)
    b = torch.tensor([0, 2, 2])
    p = torch.tensor([4, 0, 1])
    out = fancy_index_pairs(x, b, p)
    assert out.shape == (3, 2)
    torch.testing.assert_close(out[0], x[0, 4])
    torch.testing.assert_close(out[2], x[2, 1])


def test_gather_target_logits():
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    targets = torch.tensor([[0, 1, 2], [3, 3, 0]])
    out = gather_target_logits(logits, targets)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, torch.tensor([[0.0, 5.0, 10.0], [15.0, 19.0, 20.0]]))


def test_last_valid_hidden():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    lengths = torch.tensor([2, 4])
    out = last_valid_hidden(h, lengths)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out[0], h[0, 1])
    torch.testing.assert_close(out[1], h[1, 3])


def test_gather_beams():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    idx = torch.tensor([3, 3, 0, 1])
    out = gather_beams(x, idx)
    torch.testing.assert_close(out, x[idx])


def test_scatter_one_hot():
    ids = torch.tensor([[0, 2], [1, 1]])
    out = scatter_one_hot(ids, 3)
    assert out.shape == (2, 2, 3)
    assert out.dtype in (torch.float32, torch.float64)
    torch.testing.assert_close(out, torch.nn.functional.one_hot(ids, 3).to(out.dtype))


def test_bag_of_words():
    ids = torch.tensor([[0, 2, 2], [1, 1, 1]])
    out = bag_of_words(ids, 4)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]]))


def test_keep_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    out = keep_top_k(logits, 2)
    assert out.shape == logits.shape
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0
    assert torch.isinf(out[0, 3]) and out[0, 3] < 0
    torch.testing.assert_close(out[0, 1], torch.tensor(5.0))
    torch.testing.assert_close(out[0, 2], torch.tensor(3.0))
    # a softmax over such a tensor must give exactly k non-zero probabilities
    probs = out.softmax(dim=-1)
    assert int((probs > 0).sum().item()) == 2
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_select_valid():
    x = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    out = select_valid(x, mask)
    assert out.shape == (4, 2)
    torch.testing.assert_close(out[0], x[0, 0])
    torch.testing.assert_close(out[1], x[0, 2])
    torch.testing.assert_close(out[2], x[1, 1])


def test_replace_where():
    x = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    original = x.clone()
    out = replace_where(x, x < 0, 0.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    torch.testing.assert_close(x, original, msg="the input must not be modified")


def test_find_positions():
    ids = torch.tensor([5, 1, 5, 2])
    out = find_positions(ids, 5)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([0, 2]))


def test_mask_after_eos():
    ids = torch.tensor([[1, 2, 9, 3, 4], [1, 2, 3, 4, 5], [9, 1, 2, 9, 3]])
    out = mask_after_eos(ids, eos_id=9)
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False],
        ]
    )
    assert torch.equal(out, expected)


def test_make_lm_pairs():
    ids = torch.tensor([[1, 2, 3, 4]])
    inputs, targets = make_lm_pairs(ids)
    torch.testing.assert_close(inputs, torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(targets, torch.tensor([[2, 3, 4]]))


def test_expand_for_beams():
    x = torch.tensor([[1, 2], [3, 4]])
    out = expand_for_beams(x, 3)
    assert out.shape == (6, 2)
    torch.testing.assert_close(out, torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]))


@pytest.mark.parametrize("k", [1, 3, 5])
def test_keep_top_k_counts(k):
    logits = torch.randn(4, 7)
    out = keep_top_k(logits, k)
    assert (torch.isfinite(out).sum(dim=-1) == k).all()
