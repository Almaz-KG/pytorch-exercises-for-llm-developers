"""Topic 01 - tensor basics.

What we drill: creating tensors, dtype, shape, elementary arithmetic,
the difference between a copy and a view, in-place operations, type promotion.

How to work:
  1. Go top to bottom, replacing `raise NotImplementedError` with an implementation.
  2. `uv run pytest test_01_tensor_basics.py -x` until green.
  3. Stuck - the hint is in the docstring, the answer in `solutions/sol_01_tensor_basics.py`.

Rule of the game: no `for` loops over elements unless the task says otherwise.
Everything is done vectorized - that is exactly the muscle memory we are building.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 1.1 Creating tensors
# ==========================================================================


def make_arange_matrix(n: int, m: int) -> Tensor:
    """An (n, m) matrix filled row by row with 0, 1, ..., n*m-1, dtype=int64.

    Hint: torch.arange + .reshape / .view
    """
    return torch.arange(0, n * m, dtype=torch.int64).view(n, m)


def make_constant(shape: tuple[int, ...], value: float, dtype: torch.dtype) -> Tensor:
    """A tensor of the given shape, filled entirely with `value`, with the given dtype.

    Hint: torch.full takes fill_value and dtype.
    """
    return torch.full(shape, fill_value=value, dtype=dtype)


def make_identity_like(x: Tensor) -> Tensor:
    """An identity matrix of the same size and dtype as the square matrix `x`.

    Hint: torch.eye(n, dtype=..., device=...); torch.eye_like does not exist, assemble it by hand.
    """
    return torch.eye(x.shape[-1], dtype=x.dtype, device=x.device)


# ==========================================================================
# 1.2 Tensor metadata
# ==========================================================================


def describe(x: Tensor) -> dict:  # type: ignore
    """Return a dict with the keys: shape (tuple[int, ...]), ndim (int),
    numel (int), dtype (torch.dtype), bytes (int - how much memory the data takes).

    Hint: x.element_size() gives the size of one element in bytes.
    """
    return {"shape": x.shape, "ndim": x.ndim, "numel": x.numel(), "dtype": x.dtype, "bytes": x.numel() * x.itemsize}  # type: ignore


def promoted_dtype(a: Tensor, b: Tensor) -> torch.dtype:
    """Which dtype `a + b` ends up with (the type promotion rules), without computing the sum.

    Hint: torch.promote_types / torch.result_type.
    """
    return torch.promote_types(a.dtype, b.dtype)


# ==========================================================================
# 1.3 Arithmetic and type casting
# ==========================================================================


def linear_combination(a: Tensor, b: Tensor, alpha: float, beta: float) -> Tensor:
    """alpha * a + beta * b, a new tensor, leaving the inputs alone."""
    return alpha * a + beta * b


def to_float32(x: Tensor) -> Tensor:
    """Cast the tensor to float32. If it already is float32 - return it as is (the same object)."""
    return x.to(torch.float32)


def mean_of_ints(x: Tensor) -> Tensor:
    """The mean over an integer tensor. torch.mean on int64 fails - think about where the cast belongs.

    Result: a scalar float32 tensor.
    """
    return x.mean(dtype=torch.float32)


# ==========================================================================
# 1.4 Copies, views and in-place
# ==========================================================================


def independent_copy(x: Tensor) -> Tensor:
    """A copy that does NOT share memory with `x`."""
    return x.clone()


def shares_memory(a: Tensor, b: Tensor) -> bool:
    """True if the two tensors look into one and the same storage.

    Hint: a.untyped_storage().data_ptr()
    """
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def add_inplace(x: Tensor, value: float) -> Tensor:
    """Add `value` to every element DIRECTLY IN THE MEMORY of `x` and return `x` itself.

    What the test checks: the object id and the data_ptr do not change.
    """
    x.add_(value)
    return x


def zero_row_(x: Tensor, i: int) -> Tensor:
    """Zero out row `i` of the matrix `x` in-place, return `x`."""
    x[i] = 0
    return x


# ==========================================================================
# 1.5 Small useful things that LLM code needs constantly
# ==========================================================================


def count_greater(x: Tensor, threshold: float) -> int:
    """How many elements are strictly greater than threshold. Return a python int, not a tensor."""
    return int((x > threshold).sum().item())


def clamp_and_scale(x: Tensor, lo: float, hi: float) -> Tensor:
    """Clip the values into [lo, hi], then map [lo, hi] linearly onto [0, 1]."""
    return (x.clamp(min=lo, max=hi) - lo) / (hi - lo)


def model_memory_bytes(num_params: int, dtype: torch.dtype) -> int:
    """How many bytes `num_params` parameters take in the given dtype.

    This is exactly the arithmetic behind "will the model fit into the GPU".
    Hint: torch.finfo/iinfo know bits, or create an empty tensor and ask element_size().
    """
    return num_params * dtype.itemsize


# ==========================================================================
# TESTS
# ==========================================================================


def test_make_arange_matrix():
    x = make_arange_matrix(3, 4)
    assert x.shape == (3, 4)
    assert x.dtype == torch.int64
    torch.testing.assert_close(x, torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]))


def test_make_constant():
    x = make_constant((2, 3), 7.5, torch.float64)
    assert x.shape == (2, 3)
    assert x.dtype == torch.float64
    assert (x == 7.5).all()

    y = make_constant((4,), -1, torch.int32)
    assert y.dtype == torch.int32
    assert y.tolist() == [-1, -1, -1, -1]  # type: ignore


def test_make_identity_like():
    x = torch.randn(5, 5, dtype=torch.float64)
    eye = make_identity_like(x)
    assert eye.shape == x.shape
    assert eye.dtype == x.dtype
    torch.testing.assert_close(eye @ x, x)


def test_describe():
    x = torch.zeros(2, 3, 4, dtype=torch.float32)
    d = describe(x)  # type: ignore
    assert d["shape"] == (2, 3, 4)
    assert d["ndim"] == 3
    assert d["numel"] == 24
    assert d["dtype"] == torch.float32
    assert d["bytes"] == 24 * 4


def test_promoted_dtype():
    assert promoted_dtype(torch.zeros(3, dtype=torch.float16), torch.zeros(3, dtype=torch.float32)) == torch.float32
    assert promoted_dtype(torch.zeros(3, dtype=torch.int32), torch.zeros(3, dtype=torch.int64)) == torch.int64
    assert promoted_dtype(torch.zeros(3, dtype=torch.int64), torch.zeros(3, dtype=torch.float16)) == torch.float16
    assert promoted_dtype(torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.uint8)) == torch.uint8


def test_linear_combination():
    a = torch.randn(4, 5)
    b = torch.randn(4, 5)
    a_backup, b_backup = a.clone(), b.clone()
    out = linear_combination(a, b, 2.0, -3.0)
    torch.testing.assert_close(out, 2.0 * a - 3.0 * b)
    torch.testing.assert_close(a, a_backup)
    torch.testing.assert_close(b, b_backup)


def test_to_float32():
    x = torch.randn(3, dtype=torch.float64)
    y = to_float32(x)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, x.float())

    z = torch.randn(3)
    assert to_float32(z) is z, "if it already is float32 - no need to make an extra copy"


def test_mean_of_ints():
    x = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    m = mean_of_ints(x)
    assert m.dtype == torch.float32
    assert m.ndim == 0
    torch.testing.assert_close(m, torch.tensor(2.5))


def test_independent_copy():
    x = torch.arange(6, dtype=torch.float32)
    y = independent_copy(x)
    torch.testing.assert_close(y, x)
    y[0] = 100.0
    assert x[0].item() == 0.0, "the copy turned out to be a view - the data drifted"


def test_shares_memory():
    x = torch.arange(12, dtype=torch.float32)
    assert shares_memory(x, x.view(3, 4)) is True
    assert shares_memory(x, x[2:7]) is True
    assert shares_memory(x, x.clone()) is False
    assert shares_memory(x, torch.arange(12, dtype=torch.float32)) is False


def test_add_inplace():
    x = torch.zeros(3, 3)
    ptr_before = x.data_ptr()
    out = add_inplace(x, 5.0)
    assert out is x, "you have to return the same object"
    assert x.data_ptr() == ptr_before, "the operation must be in-place, without a new allocation"
    assert (x == 5.0).all()


def test_zero_row():
    x = torch.ones(3, 4)
    out = zero_row_(x, 1)
    assert out is x
    torch.testing.assert_close(x[0], torch.ones(4))
    torch.testing.assert_close(x[1], torch.zeros(4))
    torch.testing.assert_close(x[2], torch.ones(4))


def test_count_greater():
    x = torch.tensor([[-1.0, 0.5], [2.0, 3.0]])
    n = count_greater(x, 0.0)
    assert isinstance(n, int), "a python int is needed, not a tensor"
    assert n == 3


def test_clamp_and_scale():
    x = torch.tensor([-5.0, 0.0, 0.5, 1.0, 5.0])
    out = clamp_and_scale(x, 0.0, 1.0)
    torch.testing.assert_close(out, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))

    y = torch.tensor([-2.0, 0.0, 2.0])
    torch.testing.assert_close(clamp_and_scale(y, -1.0, 1.0), torch.tensor([0.0, 0.5, 1.0]))


@pytest.mark.parametrize(
    "dtype,expected_per_param",
    [(torch.float32, 4), (torch.float16, 2), (torch.bfloat16, 2), (torch.int8, 1), (torch.float64, 8)],
)
def test_model_memory_bytes(dtype, expected_per_param):  # type: ignore
    # 7B parameters - the classic size you have to be able to compute in your head
    assert model_memory_bytes(7_000_000_000, dtype) == 7_000_000_000 * expected_per_param  # type: ignore
