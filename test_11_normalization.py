"""Топик 11 - LayerNorm и RMSNorm.

Формулы:
  LayerNorm: (x - mean) / sqrt(var + eps) * gamma + beta, статистики по последней оси.
  RMSNorm:   x / sqrt(mean(x^2) + eps) * gamma - без вычитания среднего и без bias.

Почему это важно: RMSNorm дешевле (нет среднего, нет beta) и работает не хуже,
поэтому LLaMA, Qwen, Mistral и задание CS336 используют именно его.

Отдельная тема - точность: статистики считают в float32, даже если модель в bf16,
иначе дисперсия схлопывается в мусор.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 11.1 Функциональные версии
# ==========================================================================


def layer_norm(x: Tensor, weight: Tensor, bias: Tensor, eps: float = 1e-5) -> Tensor:
    """LayerNorm по последней оси, без F.layer_norm. Дисперсия смещённая (знаменатель N)."""
    raise NotImplementedError


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm по последней оси: x / sqrt(mean(x^2) + eps) * weight."""
    raise NotImplementedError


def rms_norm_upcast(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """То же, но вход может быть bfloat16: посчитать нормализацию в float32
    и вернуть результат В ИСХОДНОМ dtype входа.

    Именно так написан RMSNorm в реальных репозиториях.
    """
    raise NotImplementedError


# ==========================================================================
# 11.2 Модули
# ==========================================================================


class RMSNorm(nn.Module):
    """Модуль RMSNorm: обучаемый параметр `weight` формы (d_model,), инициализированный единицами.

    Никаких buffer-ов и bias-а.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LayerNormModule(nn.Module):
    """Модуль LayerNorm с параметрами `weight` (единицы) и `bias` (нули)."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 11.3 Куда ставить норму
# ==========================================================================


def post_norm_residual(x: Tensor, sublayer_out: Tensor, norm: nn.Module) -> Tensor:
    """Оригинальный трансформер (2017): norm(x + sublayer(x))."""
    raise NotImplementedError


def pre_norm_residual(x: Tensor, sublayer: nn.Module, norm: nn.Module) -> Tensor:
    """Современный вариант: x + sublayer(norm(x)).

    Разница не косметическая: в pre-norm остаточный поток идёт от входа к выходу
    без единой нормализации, поэтому градиент не затухает и модель учится без warmup-костылей.
    """
    raise NotImplementedError


# ==========================================================================
# 11.4 Свойства
# ==========================================================================


def scale_invariance_gap(x: Tensor, c: float) -> Tensor:
    """Максимальная разница между rms_norm(x) и rms_norm(c*x) при c > 0 (weight=1).

    RMSNorm инвариантен к масштабу входа - вот почему он стабилизирует обучение.
    """
    raise NotImplementedError


def stats_after_layernorm(x: Tensor) -> tuple[Tensor, Tensor]:
    """Применить layer_norm с weight=1, bias=0 и вернуть (среднее по последней оси,
    смещённое стандартное отклонение по последней оси).

    Должно получиться примерно 0 и 1 - это определение LayerNorm.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_layer_norm_matches_torch():
    x = torch.randn(2, 3, 16)
    w, b = torch.randn(16), torch.randn(16)
    torch.testing.assert_close(layer_norm(x, w, b), F.layer_norm(x, (16,), w, b))


def test_layer_norm_zero_input_is_finite():
    x = torch.zeros(2, 8)
    out = layer_norm(x, torch.ones(8), torch.zeros(8))
    assert torch.isfinite(out).all(), "нулевой вход не должен давать nan - для этого eps"


def test_rms_norm_matches_reference():
    x = torch.randn(2, 3, 16)
    w = torch.randn(16)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(rms_norm(x, w), expected)


def test_rms_norm_no_mean_subtraction():
    """Сдвиг всего вектора на константу МЕНЯЕТ выход RMSNorm (в отличие от LayerNorm)."""
    x = torch.randn(4, 8)
    w = torch.ones(8)
    assert not torch.allclose(rms_norm(x, w), rms_norm(x + 5.0, w), atol=1e-3)
    torch.testing.assert_close(
        layer_norm(x, w, torch.zeros(8)), layer_norm(x + 5.0, w, torch.zeros(8)), atol=1e-4, rtol=1e-4
    )


def test_rms_norm_upcast():
    x = torch.randn(4, 64).to(torch.bfloat16)
    w = torch.ones(64, dtype=torch.bfloat16)
    out = rms_norm_upcast(x, w)
    assert out.dtype == torch.bfloat16, "вернуть надо исходный dtype"
    reference = rms_norm(x.float(), w.float()).to(torch.bfloat16)
    torch.testing.assert_close(out, reference, atol=1e-2, rtol=1e-2)


def test_rms_norm_module():
    norm = RMSNorm(16)
    assert isinstance(norm.weight, nn.Parameter)
    torch.testing.assert_close(norm.weight, torch.ones(16))
    assert set(norm.state_dict().keys()) == {"weight"}, "у RMSNorm нет bias"

    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(norm(x), rms_norm(x, norm.weight.detach()))


def test_layer_norm_module():
    norm = LayerNormModule(16)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(norm(x), F.layer_norm(x, (16,), norm.weight, norm.bias))
    assert set(norm.state_dict().keys()) == {"weight", "bias"}


def test_norm_placement():
    torch.manual_seed(0)
    d = 8
    norm = RMSNorm(d)
    sublayer = nn.Linear(d, d)
    x = torch.randn(3, d)

    post = post_norm_residual(x, sublayer(x), norm)
    torch.testing.assert_close(post, norm(x + sublayer(x)))

    pre = pre_norm_residual(x, sublayer, norm)
    torch.testing.assert_close(pre, x + sublayer(norm(x)))


def test_pre_norm_keeps_identity_path():
    """Если подслой выдаёт нули, pre-norm обязан вернуть вход БЕЗ изменений."""

    class Zero(nn.Module):
        def forward(self, t: Tensor) -> Tensor:
            return torch.zeros_like(t)

    x = torch.randn(4, 8)
    torch.testing.assert_close(pre_norm_residual(x, Zero(), RMSNorm(8)), x)


def test_scale_invariance():
    x = torch.randn(4, 32)
    assert scale_invariance_gap(x, 100.0).item() < 1e-3


def test_stats_after_layernorm():
    x = torch.randn(4, 64) * 5 + 3
    mean, std = stats_after_layernorm(x)
    torch.testing.assert_close(mean, torch.zeros(4), atol=1e-5, rtol=0)
    torch.testing.assert_close(std, torch.ones(4), atol=1e-3, rtol=1e-3)


def test_rms_norm_gradients():
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: rms_norm(a, b), (x, w)), "градиенты RMSNorm не сошлись"


@pytest.mark.parametrize("shape", [(8,), (3, 8), (2, 3, 8)])
def test_rms_norm_shapes(shape):
    x = torch.randn(*shape)
    out = rms_norm(x, torch.ones(8))
    assert out.shape == shape
