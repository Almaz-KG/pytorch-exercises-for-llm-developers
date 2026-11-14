"""Topic 20 - data, batches and checkpoints.

The practical tail, without which training will not start. These exact functions
are the ones asked for in CS336 assignment 1 (get_batch, save_checkpoint, load_checkpoint).

The main idea of storing data for an LM: one long array of uint16/int32 tokens on disk,
out of which windows of length context_length are cut at random. No "examples" and no "dataset" -
only a stream of tokens and a memmap, so that the whole thing is never read into memory.
"""

import io

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 20.1 Cutting batches out of a stream of tokens
# ==========================================================================


def get_batch(
    data: np.ndarray, batch_size: int, context_length: int, generator: np.random.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Cut batch_size random windows of length context_length out of a one-dimensional array of tokens.

    Return (inputs, targets), both (batch_size, context_length) int64:
    targets are inputs shifted one token forward.

    The start position i must allow reading data[i : i + context_length + 1],
    that is, i is chosen from [0, len(data) - context_length - 1].
    """
    raise NotImplementedError


def max_start_index(num_tokens: int, context_length: int) -> int:
    """The largest admissible start index of a window (inclusive). The classic source of an off-by-one."""
    raise NotImplementedError


def steps_per_epoch(num_tokens: int, batch_size: int, context_length: int) -> int:
    """How many optimizer steps it takes for the model to see all the tokens exactly once
    (rounding down - an incomplete last batch is not taken).
    """
    raise NotImplementedError


def sequential_windows(data: np.ndarray, context_length: int) -> Tensor:
    """Cut the stream into NON-OVERLAPPING windows of length context_length, discarding the tail.
    Return (num_windows, context_length) int64. This is how validation is done: deterministically and without repeats.
    """
    raise NotImplementedError


# ==========================================================================
# 20.2 Padding and masks
# ==========================================================================


def collate_with_padding(sequences: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """A list of sequences of different lengths -> (ids (B, T_max) int64, mask (B, T_max) bool),
    where T_max is the length of the longest one and mask=True on the real tokens.
    """
    raise NotImplementedError


def make_labels(ids: Tensor, mask: Tensor, ignore_index: int = -100) -> Tensor:
    """Out of (B, T) ids and (B, T) mask make the labels for the loss: a shift one token to the left,
    the padding positions replaced by ignore_index, and the last position of every row also ignore_index
    (there is no next token for it).
    """
    raise NotImplementedError


# ==========================================================================
# 20.3 Checkpoints
# ==========================================================================


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: io.BytesIO) -> None:
    """Save into the buffer a dict with the state of the model, of the optimizer, and the iteration number.

    Saving the optimizer state is mandatory: without the Adam moments the training
    sags after a restart.
    """
    raise NotImplementedError


def load_checkpoint(src: io.BytesIO, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Load the state into the model and the optimizer in place, and return the iteration number."""
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_get_batch_shapes_and_shift():
    data = np.arange(100, dtype=np.int64)
    x, y = get_batch(data, batch_size=8, context_length=10)
    assert x.shape == (8, 10) and y.shape == (8, 10)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    torch.testing.assert_close(y, x + 1)  # the data is an arange, so the shift is visible directly


def test_get_batch_stays_in_bounds():
    data = np.arange(50, dtype=np.int64)
    for _ in range(50):
        x, y = get_batch(data, batch_size=4, context_length=16)
        assert int(x.min()) >= 0
        assert int(y.max()) <= 49


def test_get_batch_is_reproducible_with_generator():
    data = np.arange(200, dtype=np.int64)
    a = get_batch(data, 4, 8, generator=np.random.default_rng(0))
    b = get_batch(data, 4, 8, generator=np.random.default_rng(0))
    torch.testing.assert_close(a[0], b[0])
    c = get_batch(data, 4, 8, generator=np.random.default_rng(1))
    assert not torch.equal(a[0], c[0])


def test_get_batch_works_with_memmap(tmp_path):
    """The real scenario: the tokens lie on disk and are never read into memory."""
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    data = np.memmap(path, dtype=np.uint16, mode="r")
    x, y = get_batch(data, batch_size=4, context_length=32)
    assert x.shape == (4, 32)
    assert x.dtype == torch.int64, "uint16 has to be cast to int64, otherwise embedding will not take it"
    torch.testing.assert_close(y[:, :-1], x[:, 1:])


class _MaxDrawGenerator:
    """Stands in for np.random.Generator: integers() always returns the largest value its bounds allow."""

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if high is None:
            low, high = 0, low
        top = high if endpoint else high - 1
        return np.dtype(dtype).type(top) if size is None else np.full(size, top, dtype=dtype)


def test_get_batch_last_window_past_2_24():
    """A real corpus runs past 2**24 tokens, where float32 no longer holds every integer exactly.

    The last window must be read exactly, not one token off.
    """
    context_length = 8
    max_start = 2**24 + 1  # odd, so a round trip through float32 lands on 2**24
    data = np.zeros(max_start + context_length + 1, dtype=np.uint16)  # 32 MB
    data[-(context_length + 3) :] = np.arange(1, context_length + 4)
    x, y = get_batch(data, batch_size=2, context_length=context_length, generator=_MaxDrawGenerator())  # type: ignore
    window = torch.from_numpy(data[max_start:].astype(np.int64))
    torch.testing.assert_close(x, window[:-1].expand(2, -1))
    torch.testing.assert_close(y, window[1:].expand(2, -1))


@pytest.mark.parametrize("context_length", [1, 8, 64])
def test_get_batch_context_lengths(context_length):
    data = np.arange(500, dtype=np.int64)
    x, y = get_batch(data, batch_size=3, context_length=context_length)
    assert x.shape == (3, context_length)
    torch.testing.assert_close(y, x + 1)


def test_max_start_index():
    assert max_start_index(100, 10) == 89
    assert max_start_index(11, 10) == 0


def test_steps_per_epoch():
    assert steps_per_epoch(1_000_000, batch_size=8, context_length=1024) == 122
    assert steps_per_epoch(100, 4, 10) == 2


def test_sequential_windows():
    data = np.arange(11, dtype=np.int64)
    out = sequential_windows(data, 4)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out[0], torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(out[1], torch.tensor([4, 5, 6, 7]))


def test_collate_with_padding():
    ids, mask = collate_with_padding([[1, 2, 3], [4], [5, 6]], pad_id=0)
    assert ids.shape == (3, 3)
    torch.testing.assert_close(ids, torch.tensor([[1, 2, 3], [4, 0, 0], [5, 6, 0]]))
    torch.testing.assert_close(mask, torch.tensor([[True, True, True], [True, False, False], [True, True, False]]))


def test_make_labels():
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    labels = make_labels(ids, mask)
    torch.testing.assert_close(labels, torch.tensor([[2, 3, -100], [5, -100, -100]]))


def test_save_checkpoint():
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    buf = io.BytesIO()
    save_checkpoint(model, opt, iteration=7, out=buf)
    buf.seek(0)
    blob = torch.load(buf, weights_only=False)

    assert isinstance(blob, dict)
    values = list(blob.values())
    assert 7 in values, "the iteration number has to be in there"
    assert any(isinstance(v, dict) and "weight" in v for v in values), "the model state is missing"
    assert any(isinstance(v, dict) and "param_groups" in v for v in values), "the optimizer state is missing"


def test_load_checkpoint():
    """Built with torch.save directly, so loading can be written before saving is."""
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    buf = io.BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "iteration": 11}, buf)
    buf.seek(0)

    fresh_model = nn.Linear(4, 3)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    assert load_checkpoint(buf, fresh_model, fresh_opt) == 11
    torch.testing.assert_close(fresh_model.weight, model.weight)
    assert fresh_opt.state_dict()["state"], "the optimizer state was lost"


@pytest.mark.depends_on("save_checkpoint", "load_checkpoint")
def test_checkpoint_roundtrip():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)

    # take a step, so that the optimizer acquires some state
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    buf = io.BytesIO()
    save_checkpoint(model, opt, iteration=42, out=buf)
    buf.seek(0)

    fresh_model = nn.Linear(4, 3)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    it = load_checkpoint(buf, fresh_model, fresh_opt)

    assert it == 42
    torch.testing.assert_close(fresh_model.weight, model.weight)
    assert fresh_opt.state_dict()["state"], "the optimizer state was lost"
    torch.testing.assert_close(fresh_opt.state_dict()["state"][0]["exp_avg"], opt.state_dict()["state"][0]["exp_avg"])
