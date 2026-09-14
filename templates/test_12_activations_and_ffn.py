"""Топик 12 - активации и FFN-блок.

Эволюция FFN в трансформерах:
  2017: FFN(x) = W2 @ ReLU(W1 @ x),                     две матрицы, d_ff = 4 * d_model
  2020: GELU вместо ReLU (GPT-2/3)
  2022+: SwiGLU (PaLM, LLaMA, задание CS336):
         FFN(x) = W2 @ (SiLU(W1 @ x) * W3 @ x),         три матрицы, d_ff = 8/3 * d_model

Множитель 8/3 не случаен: он подобран так, чтобы три матрицы SwiGLU весили столько же,
сколько две матрицы обычного FFN с d_ff = 4 * d_model.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 12.1 Активации руками
# ==========================================================================


def relu(x: Tensor) -> Tensor:
    """max(x, 0) без F.relu."""
    raise NotImplementedError


def gelu_exact(x: Tensor) -> Tensor:
    """Точный GELU: x * 0.5 * (1 + erf(x / sqrt(2))). Без F.gelu."""
    raise NotImplementedError


def gelu_tanh(x: Tensor) -> Tensor:
    """Приближение GELU через tanh (то, что стоит в GPT-2):
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
    """
    raise NotImplementedError


def silu(x: Tensor) -> Tensor:
    """SiLU (он же swish): x * sigmoid(x). Без F.silu."""
    raise NotImplementedError


def silu_grad(x: Tensor) -> Tensor:
    """Производная SiLU, выведенная руками: sigmoid(x) * (1 + x * (1 - sigmoid(x)))."""
    raise NotImplementedError


# ==========================================================================
# 12.2 FFN-блоки
# ==========================================================================


def ffn_relu(x: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    """Классический FFN: x: (..., d_model), w1: (d_ff, d_model), w2: (d_model, d_ff).

    Веса заданы в конвенции nn.Linear (out, in), то есть умножать надо на транспонированные.
    """
    raise NotImplementedError


def swiglu(x: Tensor, w1: Tensor, w2: Tensor, w3: Tensor) -> Tensor:
    """SwiGLU-FFN в конвенции CS336:
    w1: (d_ff, d_model), w2: (d_model, d_ff), w3: (d_ff, d_model)
    out = (SiLU(x @ w1.T) * (x @ w3.T)) @ w2.T
    """
    raise NotImplementedError


class SwiGLUFFN(nn.Module):
    """Модуль SwiGLU с параметрами w1, w2, w3 (именно так они называются в тестах CS336).

    Смещений (bias) нет.
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 12.3 Арифметика размеров
# ==========================================================================


def swiglu_d_ff(d_model: int, multiple_of: int = 64) -> int:
    """Подобрать d_ff для SwiGLU: 8/3 * d_model, округлённое ВВЕРХ до кратного multiple_of.

    Кратность нужна ради эффективности тензорных ядер GPU.
    """
    raise NotImplementedError


def ffn_params(d_model: int, d_ff: int, gated: bool) -> int:
    """Число параметров FFN без bias: две матрицы для обычного, три для gated (SwiGLU)."""
    raise NotImplementedError


def ffn_flops(batch_tokens: int, d_model: int, d_ff: int, gated: bool) -> int:
    """FLOPs forward-прохода FFN: сумма 2*m*n*k по всем матричным умножениям."""
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_activations_match_torch():
    x = torch.randn(1000) * 3
    torch.testing.assert_close(relu(x), F.relu(x))
    torch.testing.assert_close(gelu_exact(x), F.gelu(x, approximate="none"))
    torch.testing.assert_close(gelu_tanh(x), F.gelu(x, approximate="tanh"), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(silu(x), F.silu(x))


def test_gelu_approximations_are_close():
    x = torch.linspace(-5, 5, 200)
    assert (gelu_exact(x) - gelu_tanh(x)).abs().max().item() < 1e-3


def test_activation_shapes_and_signs():
    x = torch.tensor([-3.0, 0.0, 3.0])
    torch.testing.assert_close(relu(x), torch.tensor([0.0, 0.0, 3.0]))
    # в отличие от ReLU, GELU и SiLU не обнуляют слегка отрицательные значения полностью
    assert gelu_exact(torch.tensor([-1.0])).item() < 0
    assert silu(torch.tensor([-1.0])).item() < 0
    torch.testing.assert_close(silu(torch.tensor([0.0])), torch.tensor([0.0]))


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
    assert swiglu_d_ff(512) == 1408  # 8/3*512 = 1365.3 -> вверх до кратного 64
    assert swiglu_d_ff(768) == 2048
    assert swiglu_d_ff(4096) == 10944
    assert swiglu_d_ff(4096, multiple_of=256) == 11008  # ровно d_ff LLaMA-7B
    assert swiglu_d_ff(64, multiple_of=1) == 171


def test_ffn_params_are_comparable():
    d = 512
    plain = ffn_params(d, 4 * d, gated=False)
    gated = ffn_params(d, swiglu_d_ff(d), gated=True)
    assert plain == 2 * d * 4 * d
    # три матрицы с d_ff = 8/3 d весят примерно столько же, сколько две с 4d
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
