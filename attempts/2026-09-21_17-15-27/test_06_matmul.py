"""Topic 06 - matrix multiplication.

What has to be kept in mind:
  - (M, K) @ (K, N) -> (M, N): the inner dimension is the one that must match;
  - in torch.matmul the leading axes broadcast and the last two get multiplied;
  - nn.Linear stores the weight as (out_features, in_features) and computes x @ W.T;
  - the FLOPs of one matmul (M,K)x(K,N) = 2*M*N*K (a multiply plus an add).

This is the very layer where 99% of LLM training time is born.
"""

import pytest
import torch
import math
from torch import Tensor

# ==========================================================================
# 6.1 Basic shapes
# ==========================================================================


def matvec(A: Tensor, x: Tensor) -> Tensor:
    """A: (M, K), x: (K,) -> (M,)."""
    return A @ x


def matmul_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Predict the shape of torch.matmul(a, b) from the shapes (both with >= 2 axes),
    taking the broadcasting of the leading axes into account. Incompatibility - ValueError.
    """
    # a = [4 5 6 2]
    # b =      [ 2 3]
    # r = [4 5 6 x 3]

    # a = (2, 1, 3, 4),
    # b =    (5, 4, 6),  expected = (2, 5, 3, 6)
    # r = (2, 1, 3, 5, 6)

    # a = (8, 2, 3),
    # b = (8, 3, 3), expected = (8, 2, 3)
    # r = (8, 2, 3)

    if a[-1] != b[-2]:
        raise ValueError("Shapes are not compatiable")

    lead_a, lead_b = a[:-2], b[:-2]
    lead: list[int] = []
    for i in range(1, max(len(lead_a), len(lead_b)) + 1):
        da = lead_a[-i] if i <= len(lead_a) else 1
        db = lead_b[-i] if i <= len(lead_b) else 1
        if da != db and da != 1 and db != 1:
            raise ValueError(f"the leading axes do not broadcast: {a} vs {b}")
        lead.append(max(da, db))
    return tuple(reversed(lead)) + (a[-2], b[-1])


def batched_matmul(a: Tensor, b: Tensor) -> Tensor:
    """a: (B, H, T, Dh), b: (B, H, Dh, S) -> (B, H, T, S). In one call, without loops over the heads."""
    return a @ b


def gram_matrix(x: Tensor) -> Tensor:
    """x: (N, D) -> (N, N) the matrix of pairwise dot products."""
    return x @ x.T


# ==========================================================================
# 6.2 A linear layer by hand
# ==========================================================================


def linear_forward(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """The exact semantics of nn.Linear: x: (..., in), weight: (out, in), bias: (out,) or None -> (..., out).

    No reshaping of the batch is needed - matmul sorts the leading axes out by itself.
    """

    out = x @ weight.T
    return out if bias is None else out + bias


def linear_backward(grad_out: Tensor, x: Tensor, weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """The gradients of a linear layer by hand (without autograd).

    Input: grad_out (N, out), x (N, in), weight (out, in).
    Return (grad_x (N, in), grad_weight (out, in), grad_bias (out,)).

    The formulas worth deriving on paper once and remembering forever:
      grad_x = grad_out @ W
      grad_W = grad_out^T @ x
      grad_b = the sum of grad_out over the batch
    """
    grad_x = grad_out @ weight
    grad_W = grad_out.T @ x
    grad_b = grad_out.sum(dim=0)
    return grad_x, grad_W, grad_b


def lora_delta(x: Tensor, A: Tensor, B: Tensor, alpha: float, rank: int) -> Tensor:
    """The LoRA correction: x: (N, in), A: (in, r), B: (r, out) -> (N, out), multiplied by alpha / rank.

    And a question of understanding along the way: why is it cheaper to compute (x @ A) @ B than x @ (A @ B).
    """
    return ((x @ A) @ B) * alpha / rank


# ==========================================================================
# 6.3 Attention without einsum
# ==========================================================================


def attention_scores(q: Tensor, k: Tensor) -> Tensor:
    """q: (B, H, T, Dh), k: (B, H, S, Dh) -> (B, H, T, S), divided by sqrt(Dh).

    Only matmul and transpose - no einsum, this is an exercise in shapes.
    """
    return (q @ k.transpose(-2, -1)) / math.sqrt(k.shape[-1])


def attention_output(weights: Tensor, v: Tensor) -> Tensor:
    """weights: (B, H, T, S), v: (B, H, S, Dh) -> (B, H, T, Dh)."""
    return weights @ v


# ==========================================================================
# 6.4 The arithmetic of cost
# ==========================================================================


def matmul_flops(m: int, k: int, n: int) -> int:
    """The FLOPs for (m, k) @ (k, n). Count a multiply and an add as one flop each: 2*m*n*k."""
    return 2 * m * k * n


def cheaper_order(m: int, k: int, n: int, p: int) -> str:
    """Given A: (m, k), B: (k, n), C: (n, p).
    Return "left" if (A@B)@C is cheaper, "right" if A@(B@C), "equal" if they are the same.

    This is not an abstraction: it is exactly how one decides whether to compute (x@A)@B or x@(A@B) in LoRA,
    and in which order to contract attention on a very long context.
    """
    left = 2 * (m * k * n + m * n * p)
    right = 2 * (k * n * p + m * k * p)
    # left = matmul_flops(m, k, n) + matmul_flops(m, n, p)
    # right = matmul_flops(k, n, p) + matmul_flops(m, k, p)
    if left == right:
        return "equal"
    return "left" if left < right else "right"


def transformer_block_params(d_model: int, d_ff: int, num_heads: int) -> int:
    """The number of parameters of one transformer block WITHOUT biases and without normalizations:
    four attention matrices (q, k, v, out) of size d_model x d_model
    plus an FFN with two matrices d_model x d_ff.

    num_heads does not affect the number of parameters - that too is part of the understanding check.
    """
    return 4 * d_model * d_model + 2 * d_model * d_ff


# ==========================================================================
# 6.5 Numerics
# ==========================================================================


def matmul_accumulate_fp32(a: Tensor, b: Tensor) -> Tensor:
    """a, b are in bfloat16. Multiply them with accumulation in float32 and return float32.

    This is what a real GPU kernel does: low precision inputs, a high precision accumulator.
    """
    return a.to(torch.float32) @ b.to(torch.float32)


# ==========================================================================
# TESTS
# ==========================================================================


def test_matvec():
    A = torch.randn(3, 4)
    x = torch.randn(4)
    out = matvec(A, x)
    assert out.shape == (3,)
    torch.testing.assert_close(out, A @ x)


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((3, 4), (4, 5), (3, 5)),
        ((2, 3, 4), (4, 5), (2, 3, 5)),
        ((2, 1, 3, 4), (5, 4, 6), (2, 5, 3, 6)),
        ((8, 2, 3), (8, 3, 3), (8, 2, 3)),
    ],
)
def test_matmul_shape(a: tuple[int], b: tuple[int], expected: tuple[int]):
    assert matmul_shape(a, b) == expected
    assert tuple(torch.matmul(torch.zeros(a), torch.zeros(b)).shape) == expected


def test_matmul_shape_error():
    with pytest.raises(ValueError):
        matmul_shape((3, 4), (5, 6))
    with pytest.raises(ValueError):
        matmul_shape((2, 3, 4), (5, 4, 6))


def test_batched_matmul():
    a = torch.randn(2, 3, 5, 4)
    b = torch.randn(2, 3, 4, 6)
    out = batched_matmul(a, b)
    assert out.shape == (2, 3, 5, 6)
    torch.testing.assert_close(out[1, 2], a[1, 2] @ b[1, 2])


def test_gram_matrix():
    x = torch.randn(5, 3)
    out = gram_matrix(x)
    assert out.shape == (5, 5)
    torch.testing.assert_close(out, out.T, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out[1, 2], (x[1] * x[2]).sum())


def test_linear_forward_matches_nn():
    layer = torch.nn.Linear(4, 7)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(linear_forward(x, layer.weight, layer.bias), layer(x))
    torch.testing.assert_close(linear_forward(x, layer.weight, None), x @ layer.weight.T)


def test_linear_backward_matches_autograd():
    x = torch.randn(6, 4, requires_grad=True)
    w = torch.randn(7, 4, requires_grad=True)
    b = torch.zeros(7, requires_grad=True)
    out = torch.nn.functional.linear(x, w, b)
    g = torch.randn_like(out)
    out.backward(g)  # type: ignore

    gx, gw, gb = linear_backward(g, x.detach(), w.detach())
    torch.testing.assert_close(gx, x.grad)
    torch.testing.assert_close(gw, w.grad)
    torch.testing.assert_close(gb, b.grad)


def test_lora_delta():
    x = torch.randn(5, 8)
    A = torch.randn(8, 2)
    B = torch.randn(2, 8)
    out = lora_delta(x, A, B, alpha=4.0, rank=2)
    assert out.shape == (5, 8)
    torch.testing.assert_close(out, (x @ A @ B) * 2.0, atol=1e-5, rtol=1e-5)


def test_attention_scores():
    B, H, T, S, Dh = 2, 3, 4, 5, 8
    q, k = torch.randn(B, H, T, Dh), torch.randn(B, H, S, Dh)
    scores = attention_scores(q, k)
    assert scores.shape == (B, H, T, S)
    torch.testing.assert_close(scores[0, 1, 2, 3], (q[0, 1, 2] * k[0, 1, 3]).sum() / (Dh**0.5))
    torch.testing.assert_close(scores, q @ k.transpose(-2, -1) / (Dh**0.5), atol=1e-5, rtol=1e-5)


def test_attention_output():
    B, H, T, S, Dh = 2, 3, 4, 5, 8
    w = torch.randn(B, H, T, S).softmax(dim=-1)
    v = torch.randn(B, H, S, Dh)
    out = attention_output(w, v)
    assert out.shape == (B, H, T, Dh)
    torch.testing.assert_close(out[0, 1, 2], (w[0, 1, 2, :, None] * v[0, 1]).sum(dim=0), atol=1e-5, rtol=1e-5)


def test_matmul_flops():
    assert matmul_flops(2, 3, 4) == 48
    # the forward pass of one layer (batch 1024, d_model 4096, d_ff 16384)
    assert matmul_flops(1024, 4096, 16384) == 2 * 1024 * 4096 * 16384


def test_cheaper_order():
    # the bottleneck in the middle (n=2): cheaper to do A@B first, the intermediate matrix is tiny
    assert cheaper_order(1000, 1000, 2, 1000) == "left"
    # a narrow output (p=2): cheaper to do B@C first - exactly the logic of LoRA and lm_head
    assert cheaper_order(1000, 1000, 1000, 2) == "right"
    assert cheaper_order(10, 10, 10, 10) == "equal"


def test_transformer_block_params():
    d, dff = 512, 2048
    assert transformer_block_params(d, dff, num_heads=8) == 4 * d * d + 2 * d * dff
    assert transformer_block_params(d, dff, num_heads=8) == transformer_block_params(d, dff, num_heads=16)


def test_matmul_accumulate_fp32():
    a = torch.randn(64, 128).to(torch.bfloat16)
    b = torch.randn(128, 64).to(torch.bfloat16)
    out = matmul_accumulate_fp32(a, b)
    assert out.dtype == torch.float32
    reference = a.float() @ b.float()
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)
