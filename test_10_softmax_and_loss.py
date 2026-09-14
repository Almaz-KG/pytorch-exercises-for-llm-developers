"""Топик 10 - softmax, log_softmax, кросс-энтропия.

Три вещи, которые надо знать про softmax:
  1. Он инвариантен к сдвигу: softmax(x) == softmax(x - c). Отсюда вычитание максимума.
  2. log_softmax считают НЕ как log(softmax(x)) - это теряет точность. Правильно: x - logsumexp(x).
  3. Кросс-энтропия принимает ЛОГИТЫ, а не вероятности. Если подать softmax, получишь тихо неверный лосс.

И главный практический факт: градиент кросс-энтропии по логитам равен (p - onehot) / N.
Красивее формулы в глубоком обучении, пожалуй, нет.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 10.1 softmax руками
# ==========================================================================


def softmax_manual(x: Tensor, dim: int) -> Tensor:
    """Численно устойчивый softmax по оси dim, без torch.softmax."""
    raise NotImplementedError


def log_softmax_manual(x: Tensor, dim: int) -> Tensor:
    """Численно устойчивый log_softmax: x - logsumexp(x, dim, keepdim=True). Без torch.log_softmax."""
    raise NotImplementedError


def temperature_softmax(logits: Tensor, temperature: float) -> Tensor:
    """softmax(logits / T). При T -> 0 распределение стремится к argmax, при больших T - к равномерному."""
    raise NotImplementedError


# ==========================================================================
# 10.2 Кросс-энтропия
# ==========================================================================


def cross_entropy_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """logits: (N, V), targets: (N,) int64 -> скаляр: средний по батчу -log p(target).

    Без F.cross_entropy и без F.log_softmax: только logsumexp и gather.
    Формула: loss_i = logsumexp(logits_i) - logits_i[target_i].
    """
    raise NotImplementedError


def sequence_cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """logits: (B, T, V), targets: (B, T) -> скаляр.

    Позиции с targets == ignore_index не входят ни в сумму, ни в знаменатель.
    Это и есть настоящий лосс языковой модели с padding-ом.
    """
    raise NotImplementedError


def label_smoothing_loss(logits: Tensor, targets: Tensor, smoothing: float) -> Tensor:
    """Кросс-энтропия со сглаживанием меток: целевое распределение это
    (1 - smoothing) на правильном классе плюс smoothing / V равномерно по всем классам.

    Вернуть средний по батчу лосс.
    """
    raise NotImplementedError


def perplexity(loss: Tensor) -> Tensor:
    """Перплексия из среднего лосса в натах. Проверка на понимание: это просто exp(loss)."""
    raise NotImplementedError


def kl_divergence(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """KL(teacher || student), усреднённая по батчу. Входы - логиты (N, V).

    Это лосс дистилляции. Считать надо через log_softmax обоих, без явного деления вероятностей.
    """
    raise NotImplementedError


# ==========================================================================
# 10.3 Градиенты
# ==========================================================================


def cross_entropy_grad(logits: Tensor, targets: Tensor) -> Tensor:
    """Градиент среднего CE-лосса по логитам, выведенный руками (без autograd).

    Ответ: (softmax(logits) - onehot(targets)) / N.
    """
    raise NotImplementedError


# ==========================================================================
# 10.4 Свойства, которые надо прочувствовать
# ==========================================================================


def shift_invariance_gap(x: Tensor, shift: float) -> Tensor:
    """Максимальная абсолютная разница между softmax(x) и softmax(x + shift).

    Ожидается ~0: это и есть инвариантность к сдвигу.
    """
    raise NotImplementedError


def uniform_loss(vocab_size: int) -> float:
    """Какой лосс у необученной модели, которая выдаёт равномерное распределение?

    Ответ надо знать наизусть: это log(V). Для V=50257 это ~10.8 -
    первое, что проверяют на первом шаге обучения.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_softmax_manual():
    x = torch.randn(3, 5, 7)
    for dim in (0, 1, 2, -1):
        torch.testing.assert_close(softmax_manual(x, dim), x.softmax(dim=dim))


def test_softmax_manual_stability():
    x = torch.tensor([[1000.0, 1000.0, 1000.0]])
    out = softmax_manual(x, dim=-1)
    assert torch.isfinite(out).all(), "переполнение exp - не вычел максимум"
    torch.testing.assert_close(out, torch.full((1, 3), 1 / 3))

    y = torch.tensor([[0.0, float("-inf")]])
    torch.testing.assert_close(softmax_manual(y, dim=-1), torch.tensor([[1.0, 0.0]]))


def test_log_softmax_manual():
    x = torch.randn(4, 6) * 20
    torch.testing.assert_close(log_softmax_manual(x, -1), F.log_softmax(x, dim=-1))
    torch.testing.assert_close(log_softmax_manual(x, -1).exp().sum(-1), torch.ones(4), atol=1e-5, rtol=0)


def test_temperature_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    cold = temperature_softmax(logits, 0.1)
    hot = temperature_softmax(logits, 100.0)
    assert cold.argmax(-1).item() == 2
    assert cold.max().item() > 0.99, "низкая температура должна почти схлопнуть в argmax"
    assert (hot - 1 / 3).abs().max().item() < 0.01, "высокая температура должна сгладить"


def test_cross_entropy_from_logits():
    logits = torch.randn(8, 12)
    targets = torch.randint(0, 12, (8,))
    torch.testing.assert_close(cross_entropy_from_logits(logits, targets), F.cross_entropy(logits, targets))


def test_cross_entropy_is_stable():
    logits = torch.tensor([[0.0, 1e4], [1e4, 0.0]])
    targets = torch.tensor([1, 0])
    out = cross_entropy_from_logits(logits, targets)
    assert torch.isfinite(out), "большие логиты не должны давать inf/nan"
    torch.testing.assert_close(out, torch.tensor(0.0), atol=1e-5, rtol=0)


def test_sequence_cross_entropy():
    logits = torch.randn(2, 5, 9)
    targets = torch.randint(0, 9, (2, 5))
    targets[0, 3:] = -100
    expected = F.cross_entropy(logits.reshape(-1, 9), targets.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(sequence_cross_entropy(logits, targets), expected)


def test_label_smoothing_loss():
    logits = torch.randn(6, 10)
    targets = torch.randint(0, 10, (6,))
    torch.testing.assert_close(
        label_smoothing_loss(logits, targets, 0.1),
        F.cross_entropy(logits, targets, label_smoothing=0.1),
    )
    torch.testing.assert_close(label_smoothing_loss(logits, targets, 0.0), F.cross_entropy(logits, targets))


def test_perplexity():
    torch.testing.assert_close(perplexity(torch.tensor(0.0)), torch.tensor(1.0))
    torch.testing.assert_close(perplexity(torch.log(torch.tensor(50257.0))), torch.tensor(50257.0))


def test_kl_divergence():
    s = torch.randn(4, 7)
    t = torch.randn(4, 7)
    expected = F.kl_div(F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1), log_target=True, reduction="batchmean")
    torch.testing.assert_close(kl_divergence(s, t), expected)
    torch.testing.assert_close(kl_divergence(t, t), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_cross_entropy_grad():
    logits = torch.randn(5, 9, requires_grad=True)
    targets = torch.randint(0, 9, (5,))
    F.cross_entropy(logits, targets).backward()
    torch.testing.assert_close(cross_entropy_grad(logits.detach(), targets), logits.grad, atol=1e-6, rtol=1e-5)


def test_shift_invariance():
    x = torch.randn(3, 8)
    assert shift_invariance_gap(x, 100.0).item() < 1e-6


def test_uniform_loss():
    assert abs(uniform_loss(50257) - math.log(50257)) < 1e-9
    assert abs(uniform_loss(2) - math.log(2)) < 1e-9


@pytest.mark.parametrize("vocab", [4, 100, 50257])
def test_untrained_model_loss_matches_log_vocab(vocab):
    """Настоящая проверка: у модели с нулевыми логитами лосс должен быть ровно log(V)."""
    logits = torch.zeros(16, vocab)
    targets = torch.randint(0, vocab, (16,))
    torch.testing.assert_close(
        cross_entropy_from_logits(logits, targets),
        torch.tensor(uniform_loss(vocab)),
        atol=1e-5,
        rtol=1e-5,
    )
