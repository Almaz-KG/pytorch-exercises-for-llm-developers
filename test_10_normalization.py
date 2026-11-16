"""Topic 10 - LayerNorm and RMSNorm.

The formulas:
  LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta, with the statistics over the last axis.
  RMSNorm:   x / sqrt(mean(x^2) + eps) * gamma - without subtracting the mean and without a bias.

Why this matters: RMSNorm is cheaper (no mean, no beta) and works no worse,
which is why LLaMA, Qwen, Mistral and the CS336 assignment all use it.

A separate topic is precision: the statistics are computed in float32 even when the model is in bf16,
otherwise the variance collapses into garbage.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 10.1 The functional versions
# ==========================================================================


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm along the last axis, without F.layer_norm. The variance is biased (denominator N)."""
    raise NotImplementedError


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm along the last axis: x / sqrt(mean(x^2) + eps) * weight."""
    raise NotImplementedError


def rms_norm_upcast(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """The same, but the input may be bfloat16: compute the normalization in float32
    and return the result IN THE ORIGINAL dtype of the input.

    This is exactly how RMSNorm is written in real repositories.
    """
    raise NotImplementedError


# ==========================================================================
# 10.2 The modules
# ==========================================================================


class RMSNorm(nn.Module):
    """The RMSNorm module: a trainable `weight` parameter of shape (d_model,), initialized with ones.

    No buffers and no bias.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LayerNormModule(nn.Module):
    """The LayerNorm module with the parameters `weight` (ones) and `bias` (zeros)."""

    weight: nn.Parameter
    bias: nn.Parameter

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 10.3 Where to put the norm
# ==========================================================================


def post_norm_residual(x: Tensor, sublayer_out: Tensor, norm: nn.Module) -> Tensor:
    """The original transformer (2017): norm(x + sublayer(x))."""
    raise NotImplementedError


def pre_norm_residual(x: Tensor, sublayer: nn.Module, norm: nn.Module) -> Tensor:
    """The modern variant: x + sublayer(norm(x)).

    The difference is not cosmetic: in pre-norm the residual stream runs from the input to the output
    without a single normalization, so the gradient does not decay and the model trains without warmup crutches.
    """
    raise NotImplementedError


# ==========================================================================
# 10.4 Properties
# ==========================================================================


def scale_invariance_gap(x: Tensor, c: float) -> Tensor:
    """The maximum difference between rms_norm(x) and rms_norm(c*x) for c > 0 (weight=1).

    RMSNorm is invariant to the scale of the input - that is why it stabilizes training.
    """
    raise NotImplementedError


def stats_after_layernorm(x: Tensor) -> tuple[Tensor, Tensor]:
    """Apply layer_norm with weight=1, bias=0 and return (the mean over the last axis,
    the biased standard deviation over the last axis).

    It should come out as roughly 0 and 1 - that is the definition of LayerNorm.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_layer_norm_matches_torch():
    x = torch.randn(2, 3, 16)
    w, b = torch.randn(16), torch.randn(16)
    torch.testing.assert_close(layer_norm(x, w, b), F.layer_norm(x, (16,), w, b))


def test_layer_norm_zero_input_is_finite():
    x = torch.zeros(2, 8)
    out = layer_norm(x, torch.ones(8), torch.zeros(8))
    assert torch.isfinite(out).all(), "a zero input must not give nan - that is what eps is for"


def test_layer_norm_is_shift_invariant():
    """LayerNorm does subtract the mean, so the same shift changes nothing. That is the contrast."""
    x = torch.randn(4, 8)
    w, b = torch.ones(8), torch.zeros(8)
    torch.testing.assert_close(layer_norm(x, w, b), layer_norm(x + 5.0, w, b), atol=1e-4, rtol=1e-4)


def test_rms_norm_matches_reference():
    x = torch.randn(2, 3, 16)
    w = torch.randn(16)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(rms_norm(x, w), expected)


def test_rms_norm_is_not_shift_invariant():
    """RMSNorm does not subtract the mean, so shifting the whole vector DOES change its output."""
    x = torch.randn(4, 8)
    w = torch.ones(8)
    assert not torch.allclose(rms_norm(x, w), rms_norm(x + 5.0, w), atol=1e-3)


def test_rms_norm_gradients():
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: rms_norm(a, b), (x, w)), "the RMSNorm gradients did not converge"


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8)])
def test_rms_norm_shapes(shape):
    x = torch.randn(*shape)
    out = rms_norm(x, torch.ones(8))
    assert out.shape == shape


def test_rms_norm_upcast():
    x = torch.randn(4, 64).to(torch.bfloat16)
    w = torch.ones(64, dtype=torch.bfloat16)
    out = rms_norm_upcast(x, w)
    assert out.dtype == torch.bfloat16, "the original dtype is what has to be returned"
    xf, wf = x.float(), w.float()
    reference = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5) * wf).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)


def test_rms_norm_module():
    norm = RMSNorm(16)
    assert isinstance(norm.weight, nn.Parameter)
    torch.testing.assert_close(norm.weight, torch.ones(16))
    assert set(norm.state_dict().keys()) == {"weight"}, "RMSNorm has no bias"

    x = torch.randn(2, 5, 16)
    w = norm.weight.detach()
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(norm(x), expected)


def test_layer_norm_module():
    norm = LayerNormModule(16)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(norm(x), F.layer_norm(x, (16,), norm.weight, norm.bias))
    assert set(norm.state_dict().keys()) == {"weight", "bias"}


def test_post_norm_residual():
    """The norm module is torch's own here, so this tests the placement and nothing else."""
    torch.manual_seed(0)
    d = 8
    norm = nn.LayerNorm(d)
    sublayer = nn.Linear(d, d)
    x = torch.randn(3, d)

    out = post_norm_residual(x, sublayer(x), norm)
    torch.testing.assert_close(out, norm(x + sublayer(x)))


def test_pre_norm_residual():
    torch.manual_seed(0)
    d = 8
    norm = nn.LayerNorm(d)
    sublayer = nn.Linear(d, d)
    x = torch.randn(3, d)

    out = pre_norm_residual(x, sublayer, norm)
    torch.testing.assert_close(out, x + sublayer(norm(x)))


def test_pre_norm_keeps_identity_path():
    """If the sublayer outputs zeros, pre-norm must return the input UNCHANGED."""

    class Zero(nn.Module):
        def forward(self, t: Tensor) -> Tensor:
            return torch.zeros_like(t)

    x = torch.randn(4, 8)
    torch.testing.assert_close(pre_norm_residual(x, Zero(), nn.LayerNorm(8)), x)


def test_scale_invariance():
    x = torch.randn(4, 32)
    assert scale_invariance_gap(x, 100.0).item() < 1e-3


def test_stats_after_layernorm():
    x = torch.randn(4, 64) * 5 + 3
    mean, std = stats_after_layernorm(x)
    torch.testing.assert_close(mean, torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(std, torch.ones(4), atol=1e-3, rtol=1e-3)
