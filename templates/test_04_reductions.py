"""Топик 04 - редукции: sum, mean, max, topk, logsumexp, нормы.

Ключевые вещи, которые надо довести до автоматизма:
  - dim=... это ось, КОТОРАЯ ИСЧЕЗАЕТ (или становится 1 при keepdim=True);
  - keepdim=True нужен всегда, когда результат потом broadcast-ится обратно;
  - logsumexp - это не "ещё одна функция", а способ не переполнить exp;
  - в LLM почти любое усреднение идёт с маской padding-токенов.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 4.1 Базовые редукции и оси
# ==========================================================================


def sum_over_time(x: Tensor) -> Tensor:
    """(B, T, D) -> (B, D): сумма по оси времени."""
    raise NotImplementedError


def mean_over_batch_and_time(x: Tensor) -> Tensor:
    """(B, T, D) -> (D,): среднее сразу по двум осям."""
    raise NotImplementedError


def center_and_keepdim(x: Tensor) -> tuple[Tensor, Tensor]:
    """Вернуть (среднее по последней оси с keepdim=True, x минус это среднее).

    Первый элемент должен иметь форму (..., 1) - именно из-за keepdim.
    """
    raise NotImplementedError


def biased_variance(x: Tensor) -> Tensor:
    """Дисперсия по последней оси со знаменателем N (не N-1), keepdim=True.

    LayerNorm использует именно смещённую оценку - легко ошибиться на unbiased по умолчанию.
    """
    raise NotImplementedError


def rms(x: Tensor) -> Tensor:
    """sqrt(mean(x^2)) по последней оси, keepdim=True. Сердце RMSNorm."""
    raise NotImplementedError


# ==========================================================================
# 4.2 Редукции с маской (padding)
# ==========================================================================


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """x: (B, T, D), mask: (B, T) bool (True = настоящий токен) -> (B, D).

    Усреднять надо только по настоящим токенам. Делить на mask.sum(), а не на T.
    """
    raise NotImplementedError


def count_tokens(mask: Tensor) -> tuple[Tensor, int]:
    """mask: (B, T) bool -> (длины последовательностей (B,) int64, общее число токенов python int)."""
    raise NotImplementedError


def masked_max(x: Tensor, mask: Tensor) -> Tensor:
    """x: (B, T), mask: (B, T) bool -> (B,): максимум только по разрешённым позициям.

    Подсказка: заполни запрещённые позиции -inf перед max.
    """
    raise NotImplementedError


# ==========================================================================
# 4.3 max / argmax / topk - декодирование
# ==========================================================================


def greedy_tokens(logits: Tensor) -> Tensor:
    """logits: (B, T, V) -> (B, T) int64: индекс самого вероятного токена в каждой позиции."""
    raise NotImplementedError


def top_k_tokens(logits: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """logits: (B, V) -> (значения (B, k), индексы (B, k)), отсортированные по убыванию."""
    raise NotImplementedError


def argmax_2d(x: Tensor) -> tuple[int, int]:
    """Найти (строка, столбец) максимума матрицы. Через плоский argmax и divmod, без циклов."""
    raise NotImplementedError


def accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> float:
    """logits: (B, T, V), targets: (B, T) -> доля правильных предсказаний,
    позиции с targets == ignore_index не считаются вовсе.

    Вернуть python float.
    """
    raise NotImplementedError


# ==========================================================================
# 4.4 logsumexp и энтропия
# ==========================================================================


def logsumexp_manual(x: Tensor, dim: int) -> Tensor:
    """Численно устойчивый log(sum(exp(x))) по оси dim, БЕЗ torch.logsumexp.

    Схема: m = max(x, dim, keepdim=True); m + log(sum(exp(x - m))).
    Должно работать на входах вроде 1000.0, где наивный exp даёт inf.
    """
    raise NotImplementedError


def entropy(probs: Tensor) -> Tensor:
    """probs: (..., V), суммируются в 1 по последней оси -> (...,) энтропия в натах.

    Осторожно с log(0): нулевые вероятности должны давать вклад 0, а не nan.
    """
    raise NotImplementedError


# ==========================================================================
# 4.5 Нормы
# ==========================================================================


def l2_normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Поделить каждый вектор по последней оси на его L2-норму."""
    raise NotImplementedError


def global_grad_norm(tensors: list[Tensor]) -> Tensor:
    """Общая L2-норма списка тензоров: sqrt(сумма квадратов всех элементов всех тензоров).

    Ровно эта величина стоит в gradient clipping.
    """
    raise NotImplementedError


def token_frequencies(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (...,) int64 -> (vocab_size,) int64 счётчик вхождений каждого id.

    Подсказка: torch.bincount, не забудь minlength и flatten.
    """
    raise NotImplementedError


def cumulative_mean(x: Tensor) -> Tensor:
    """(T,) -> (T,): out[i] = среднее x[0..i]. Через cumsum, без циклов."""
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
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
    assert mean.shape == (2, 3, 1), "потерян keepdim"
    torch.testing.assert_close(centered.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)


def test_biased_variance():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    v = biased_variance(x)
    assert v.shape == (1, 1)
    torch.testing.assert_close(v, torch.tensor([[2.0 / 3.0]]))  # не 1.0, как у unbiased


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
    logits[0, 0, 1] = 1.0  # предсказали 1
    logits[0, 1, 0] = 1.0  # предсказали 0
    logits[0, 2, 2] = 1.0  # предсказали 2
    logits[0, 3, 0] = 1.0  # предсказали 0, но позиция игнорируется
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
    assert torch.isfinite(out).all(), "наивный exp переполнился"
    torch.testing.assert_close(out, torch.tensor(1000.0 + torch.log(torch.tensor(3.0))))

    # маскированные позиции приходят как -inf и не должны ломать результат
    y = torch.tensor([0.0, float("-inf"), 0.0])
    torch.testing.assert_close(logsumexp_manual(y, dim=0), torch.logsumexp(y, dim=0))


def test_entropy():
    uniform = torch.full((1, 4), 0.25)
    torch.testing.assert_close(entropy(uniform), torch.log(torch.tensor([4.0])))

    deterministic = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    out = entropy(deterministic)
    assert torch.isfinite(out).all(), "log(0) дал nan - обработай нулевые вероятности"
    torch.testing.assert_close(out, torch.zeros(1), atol=1e-6, rtol=0)


def test_l2_normalize():
    x = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalize(x)
    torch.testing.assert_close(out[0], torch.tensor([0.6, 0.8]))
    assert torch.isfinite(out).all(), "нулевой вектор не должен давать nan - для этого eps"


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
