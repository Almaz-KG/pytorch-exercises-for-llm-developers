"""Topic 12 - activations and the FFN block.

The evolution of the FFN in transformers:
  2017: FFN(x) = W2 @ ReLU(W1 @ x),                     two matrices, d_ff = 4 * d_model
  2020: GELU instead of ReLU (GPT-2/3)
  2022+: SwiGLU (PaLM, LLaMA, the CS336 assignment):
         FFN(x) = W2 @ (SiLU(W1 @ x) * W3 @ x),         three matrices, d_ff = 8/3 * d_model

The factor 8/3 is no accident: it is chosen so that the three matrices of SwiGLU weigh as much
as the two matrices of an ordinary FFN with d_ff = 4 * d_model.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 12.1 Activations by hand
# ==========================================================================


def relu(x: Tensor) -> Tensor:
    """max(x, 0) without F.relu."""
    raise NotImplementedError


def gelu_exact(x: Tensor) -> Tensor:
    """The exact GELU: x * 0.5 * (1 + erf(x / sqrt(2))). Without F.gelu."""
    raise NotImplementedError


def gelu_tanh(x: Tensor) -> Tensor:
    """The tanh approximation of GELU (the one that sits in GPT-2):
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    raise NotImplementedError


def silu(x: Tensor) -> Tensor:
    """SiLU (also known as swish): x * sigmoid(x). Without F.silu."""
    raise NotImplementedError


def silu_grad(x: Tensor) -> Tensor:
    """The derivative of SiLU, derived by hand: sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
    raise NotImplementedError


# ==========================================================================
# 12.2 FFN blocks
# ==========================================================================


def ffn_relu(x: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    """The classic FFN: x: (..., d_model), w1: (d_ff, d_model), w2: (d_model, d_ff).

    The weights are given in the nn.Linear convention (out, in), so they have to be multiplied transposed.
    """
    raise NotImplementedError


def swiglu(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    """The SwiGLU FFN in the CS336 convention:
    w1: (d_ff, d_model), w2: (d_model, d_ff), w3: (d_ff, d_model)
    out = (SiLU(x @ w1.T) * (x @ w3.T)) @ w2.T
    """
    raise NotImplementedError


class SwiGLUFFN(nn.Module):
    """The SwiGLU module with the parameters w1, w2, w3 (that is exactly how they are named in the CS336 tests).

    There are no biases.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 12.3 The arithmetic of sizes
# ==========================================================================


def swiglu_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """Pick d_ff for SwiGLU: 8/3 * d_model, rounded UP to a multiple of multiple_of.

    The multiple is needed for the efficiency of the GPU tensor cores.
    """
    raise NotImplementedError


def ffn_params(d_model: int, d_ff: int, gated: bool) -> int:
    """The number of FFN parameters without biases: two matrices for the ordinary one, three for a gated one (SwiGLU)."""
    raise NotImplementedError


def ffn_flops(batch_tokens: int, d_model: int, d_ff: int, gated: bool) -> int:
    """The FLOPs of the FFN forward pass: the sum of 2*m*n*k over all the matrix multiplications."""
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_relu():
    x = torch.randn(1000) * 3
    torch.testing.assert_close(relu(x), F.relu(x))
    torch.testing.assert_close(relu(torch.tensor([-3.0, 0.0, 3.0])), torch.tensor([0.0, 0.0, 3.0]))


def test_gelu_exact():
    x = torch.randn(1000) * 3
    torch.testing.assert_close(gelu_exact(x), F.gelu(x, approximate="none"))
    # unlike ReLU, GELU does not zero slightly negative values out completely
    assert gelu_exact(torch.tensor([-1.0])).item() < 0


def test_gelu_tanh():
    x = torch.randn(1000) * 3
    torch.testing.assert_close(gelu_tanh(x), F.gelu(x, approximate="tanh"), atol=1e-6, rtol=1e-5)


def test_silu():
    x = torch.randn(1000) * 3
    torch.testing.assert_close(silu(x), F.silu(x))
    assert silu(torch.tensor([-1.0])).item() < 0
    torch.testing.assert_close(silu(torch.tensor([0.0])), torch.tensor([0.0]))


@pytest.mark.depends_on("gelu_exact", "gelu_tanh")
def test_gelu_approximations_are_close():
    """The two forms have to agree: that is what makes the cheap one a drop-in for the exact one."""
    x = torch.linspace(-5, 5, 200)
    assert (gelu_exact(x) - gelu_tanh(x)).abs().max().item() < 1e-3


def test_silu_grad_matches_autograd():
    x = torch.randn(100, requires_grad=True)
    F.silu(x).sum().backward()
    torch.testing.assert_close(silu_grad(x.detach()), x.grad, atol=1e-6, rtol=1e-5)


def test_ffn_relu():
    x = torch.randn(2, 3, 8)
    w1, w2 = torch.randn(32, 8), torch.randn(8, 32)
    out = ffn_relu(x, w1, w2)
    assert out.shape == (2, 3, 8)
    torch.testing.assert_close(out, F.relu(x @ w1.T) @ w2.T, atol=1e-5, rtol=1e-5)


def test_swiglu():
    x = torch.randn(2, 3, 8)
    w1, w2, w3 = torch.randn(21, 8), torch.randn(8, 21), torch.randn(21, 8)
    out = swiglu(x, w1, w2, w3)
    assert out.shape == (2, 3, 8)
    expected = (F.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_swiglu_module():
    ffn = SwiGLUFFN(8, 21)
    names = set(ffn.state_dict().keys())
    assert names == {"w1.weight", "w2.weight", "w3.weight"} or names == {"w1", "w2", "w3"}
    x = torch.randn(4, 8)
    assert ffn(x).shape == (4, 8)
    assert sum(p.numel() for p in ffn.parameters()) == 3 * 8 * 21


def test_swiglu_d_ff():
    assert swiglu_d_ff(512) == 1408  # 8/3*512 = 1365.3 -> up to a multiple of 64
    assert swiglu_d_ff(768) == 2048
    assert swiglu_d_ff(4096) == 10944
    assert swiglu_d_ff(4096, multiple_of=256) == 11008  # exactly the d_ff of LLaMA-7B
    assert swiglu_d_ff(64, multiple_of=1) == 171


@pytest.mark.depends_on("swiglu_d_ff")
def test_ffn_params_are_comparable():
    d = 512
    plain = ffn_params(d, 4 * d, gated=False)
    gated = ffn_params(d, swiglu_d_ff(d), gated=True)
    assert plain == 2 * d * 4 * d
    # three matrices with d_ff = 8/3 d weigh roughly as much as two with 4d
    assert abs(gated - plain) / plain < 0.1


def test_ffn_flops():
    n, d, dff = 1024, 512, 2048
    assert ffn_flops(n, d, dff, gated=False) == 2 * (2 * n * d * dff)
    assert ffn_flops(n, d, dff, gated=True) == 3 * (2 * n * d * dff)


@pytest.mark.parametrize("d_model", [128, 512, 1024, 4096])
def test_swiglu_d_ff_is_multiple(d_model):
    d_ff = swiglu_d_ff(d_model)
    assert d_ff % 64 == 0
    assert d_ff >= 8 * d_model / 3
