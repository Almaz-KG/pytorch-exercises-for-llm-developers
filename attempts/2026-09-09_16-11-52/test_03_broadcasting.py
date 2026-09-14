"""Топик 03 - broadcasting.

Правило одно, и его надо знать наизусть:
формы выравниваются СПРАВА, ось совместима если размеры равны либо один из них 1,
недостающие ведущие оси считаются равными 1.

90% "магии" в коде трансформера - это broadcasting: добавление bias, маски,
RoPE, ALiBi, per-head масштабирование. Если он в пальцах, код читается сам.
"""

from itertools import zip_longest

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 3.1 Правило broadcasting руками
# ==========================================================================


def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """Вернуть форму результата a op b по правилам broadcasting.
    Если формы несовместимы - бросить ValueError.

    Реализуй правило вручную (без torch.broadcast_shapes) - это упражнение на понимание.
    """

    result: list[int] = []
    for l, r in zip_longest(a[::-1], b[::-1], fillvalue=1):
        if l != r and l != 1 and r != 1:
            raise ValueError(f"Incompatible shapes: {a} and {b}")
        result.append(max(l, r))
    return tuple(result[::-1])


def explain_trap() -> tuple[int, ...]:
    """Классическая ловушка: тензор (3, 1) плюс тензор (3,).
    Верни форму результата. Многие ожидают (3,) или (3, 1) и ловят баг в лоссе.
    """
    return (3, 3)


# ==========================================================================
# 3.2 Типовые применения в трансформере
# ==========================================================================


def add_bias(x: Tensor, bias: Tensor) -> Tensor:
    """x: (B, T, D), bias: (D,) -> (B, T, D)."""
    return x + bias


def scale_per_head(x: Tensor, scales: Tensor) -> Tensor:
    """x: (B, H, T, Dh), scales: (H,). Каждую голову умножить на свой скаляр.

    Подсказка: подгони форму scales так, чтобы она выровнялась справа: (1, H, 1, 1).
    """
    return x * scales[None, :, None, None]


def normalize_last_dim(x: Tensor) -> Tensor:
    """Из каждого вектора по последней оси вычесть его среднее. Форма не меняется.

    Здесь и живёт keepdim=True: без него формы не выровняются.
    """
    return x - x.mean(dim=-1, keepdim=True)


def weighted_sum(weights: Tensor, values: Tensor) -> Tensor:
    """weights: (B, T), values: (B, T, D) -> (B, D): сумма значений с весами по времени.

    Это attention в миниатюре: веса на токенах, взвешенная сумма их представлений.
    """
    return (weights.unsqueeze(-1) * values).sum(dim=1)


# ==========================================================================
# 3.3 Матрицы из векторов
# ==========================================================================


def outer_product(a: Tensor, b: Tensor) -> Tensor:
    """a: (N,), b: (M,) -> (N, M), out[i, j] = a[i] * b[j]. Только broadcasting, без torch.outer."""
    return a[:, None] * b[None, :]


def pairwise_sq_dist(a: Tensor, b: Tensor) -> Tensor:
    """a: (N, D), b: (M, D) -> (N, M) квадратов евклидовых расстояний.

    Через broadcasting: (N, 1, D) - (1, M, D) -> (N, M, D), затем сумма по D.
    """
    return torch.pow(a.unsqueeze(dim=1) - b.unsqueeze(dim=0), 2).sum(dim=-1)


def one_hot_by_comparison(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (...,) int64 -> (..., vocab_size) float32 one-hot.

    Через сравнение с arange по broadcasting, без F.one_hot.
    """
    # print(torch.arange(vocab_size, device=ids.device))
    # print(ids.unsqueeze(-1).shape)
    # print(ids.unsqueeze(-1))

    return (torch.arange(vocab_size, device=ids.device) == ids.unsqueeze(-1)).to(torch.float32)


# ==========================================================================
# 3.4 Маски и относительные позиции
# ==========================================================================


def causal_mask(seq_len: int) -> Tensor:
    """(T, T) bool: True там, где позиция-запрос i МОЖЕТ смотреть на позицию-ключ j (j <= i).

    Собери через сравнение двух arange разной формы, без torch.tril.
    """
    a = torch.arange(seq_len).unsqueeze(-1)
    b = torch.arange(seq_len).unsqueeze(0)

    return a >= b


def relative_positions(q_len: int, k_len: int) -> Tensor:
    """(q_len, k_len) int64, out[i, j] = i - j. Основа relative position bias и ALiBi."""
    a = torch.arange(q_len).unsqueeze(-1)
    b = torch.arange(k_len)
    return a - b


def alibi_bias(slopes: Tensor, seq_len: int) -> Tensor:
    """slopes: (H,) -> (H, T, T) аддитивный ALiBi-биас.

    bias[h, i, j] = -slopes[h] * (i - j) для j <= i, и 0 (не важно) выше диагонали;
    здесь для простоты считаем формулу для всех j, без обнуления.
    """
    a = torch.arange(seq_len).unsqueeze(-1)
    b = torch.arange(seq_len)

    return -slopes[:, None, None] * (a - b)


def apply_additive_mask(scores: Tensor, mask: Tensor) -> Tensor:
    """scores: (B, H, T, T), mask: (T, T) bool (True = можно смотреть).
    Вернуть scores, где запрещённые позиции равны -inf.

    Никаких циклов: mask должна разъехаться по батчу и головам сама.
    """
    return torch.masked_fill(scores, ~mask, value=-torch.inf)


def key_padding_mask_apply(scores: Tensor, pad_mask: Tensor) -> Tensor:
    """scores: (B, H, T, T), pad_mask: (B, T) bool (True = настоящий токен, False = padding).
    Запретить ВСЕМ запросам смотреть на padding-ключи: соответствующие столбцы -> -inf.

    Внимательно с осями: pad_mask относится к ключам, то есть к ПОСЛЕДНЕЙ оси scores.
    """
    return torch.masked_fill(scores, ~pad_mask[:, None, None, :], value=-torch.inf)


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


@pytest.mark.parametrize(
    "a,b,expected",
    [
        ((3, 1), (3,), (3, 3)),
        ((5, 1, 4), (3, 4), (5, 3, 4)),
        ((2, 3, 4), (4,), (2, 3, 4)),
        ((1,), (7, 5), (7, 5)),
        ((), (2, 2), (2, 2)),
        ((8, 1, 6, 1), (7, 1, 5), (8, 7, 6, 5)),
    ],
)
def test_broadcast_shape_ok(a, b, expected):
    assert broadcast_shape(a, b) == expected
    # сверяемся с самим torch
    assert tuple(torch.broadcast_shapes(a, b)) == expected


@pytest.mark.parametrize("a,b", [((3,), (4,)), ((2, 3), (3, 2)), ((5, 2), (5, 3))])
def test_broadcast_shape_error(a, b):
    with pytest.raises(ValueError):
        broadcast_shape(a, b)


def test_explain_trap():
    assert explain_trap() == (3, 3)


def test_add_bias():
    x = torch.randn(2, 3, 4)
    b = torch.randn(4)
    out = add_bias(x, b)
    assert out.shape == (2, 3, 4)
    torch.testing.assert_close(out[1, 2], x[1, 2] + b)


def test_scale_per_head():
    x = torch.randn(2, 3, 5, 4)
    s = torch.tensor([1.0, 2.0, 3.0])
    out = scale_per_head(x, s)
    assert out.shape == x.shape
    for h in range(3):
        torch.testing.assert_close(out[:, h], x[:, h] * s[h])


def test_normalize_last_dim():
    x = torch.randn(2, 3, 8)
    out = normalize_last_dim(x)
    assert out.shape == x.shape
    torch.testing.assert_close(out.mean(dim=-1), torch.zeros(2, 3), atol=1e-6, rtol=0)


def test_weighted_sum():
    w = torch.tensor([[0.5, 0.5], [1.0, 0.0]])
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    out = weighted_sum(w, v)
    assert out.shape == (2, 2)
    torch.testing.assert_close(out, torch.tensor([[2.0, 3.0], [5.0, 6.0]]))


def test_outer_product():
    a, b = torch.randn(4), torch.randn(6)
    out = outer_product(a, b)
    assert out.shape == (4, 6)
    torch.testing.assert_close(out, torch.outer(a, b))


def test_pairwise_sq_dist():
    a, b = torch.randn(5, 3), torch.randn(7, 3)
    out = pairwise_sq_dist(a, b)
    assert out.shape == (5, 7)
    torch.testing.assert_close(out, torch.cdist(a, b).pow(2), atol=1e-4, rtol=1e-4)


def test_one_hot_by_comparison():
    ids = torch.tensor([[0, 3], [2, 1]])
    out = one_hot_by_comparison(ids, 4)
    assert out.shape == (2, 2, 4)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, torch.nn.functional.one_hot(ids, 4).float())


def test_causal_mask():
    m = causal_mask(4)
    assert m.dtype == torch.bool
    expected = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, False],
            [True, True, True, True],
        ]
    )
    assert torch.equal(m, expected)


def test_relative_positions():
    r = relative_positions(3, 4)
    assert r.shape == (3, 4)
    torch.testing.assert_close(r, torch.tensor([[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1]]))


def test_alibi_bias():
    slopes = torch.tensor([0.5, 0.25])
    bias = alibi_bias(slopes, 3)
    assert bias.shape == (2, 3, 3)
    # чем дальше назад смотрит запрос, тем сильнее штраф
    torch.testing.assert_close(bias[0, 2, 0], torch.tensor(-1.0))
    torch.testing.assert_close(bias[0, 2, 1], torch.tensor(-0.5))
    torch.testing.assert_close(bias[1, 2, 0], torch.tensor(-0.5))
    torch.testing.assert_close(bias[0, 1, 1], torch.tensor(0.0))


def test_apply_additive_mask():
    scores = torch.randn(2, 3, 4, 4)
    mask = causal_mask(4)
    out = apply_additive_mask(scores, mask)
    assert out.shape == scores.shape
    assert torch.isinf(out[:, :, 0, 1]).all() and (out[:, :, 0, 1] < 0).all()
    torch.testing.assert_close(out[:, :, 3, 0], scores[:, :, 3, 0])
    assert torch.isfinite(out[mask.expand_as(out)]).all()


def test_key_padding_mask_apply():
    scores = torch.zeros(2, 2, 3, 3)
    pad = torch.tensor([[True, True, False], [True, False, False]])
    out = key_padding_mask_apply(scores, pad)
    assert torch.isinf(out[0, :, :, 2]).all()
    assert torch.isfinite(out[0, :, :, :2]).all()
    assert torch.isinf(out[1, :, :, 1:]).all()
    assert torch.isfinite(out[1, :, :, 0]).all()
