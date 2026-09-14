"""Топик 01 - основы тензоров.

Что тренируем: создание тензоров, dtype, форма, элементарная арифметика,
разница между копией и view, in-place операции, продвижение типов.

Как работать:
  1. Идёшь сверху вниз, заменяешь `raise NotImplementedError` на реализацию.
  2. `uv run pytest test_01_tensor_basics.py -x` до зелёного.
  3. Застрял - подсказка в докстринге, ответ в `solutions/sol_01_tensor_basics.py`.

Правило игры: никаких циклов `for` по элементам, если в задаче не сказано иначе.
Всё делается векторно - это и есть мышечная память, которую мы качаем.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 1.1 Создание тензоров
# ==========================================================================


def make_arange_matrix(n: int, m: int) -> Tensor:
    """Матрица (n, m), заполненная числами 0, 1, ..., n*m-1 по строкам, dtype=int64.

    Подсказка: torch.arange + .reshape / .view
    """
    return torch.arange(0, n * m, dtype=torch.int64).reshape((n, m))


def make_constant(shape: tuple[int, ...], value: float, dtype: torch.dtype) -> Tensor:
    """Тензор заданной формы, целиком заполненный `value`, с заданным dtype.

    Подсказка: torch.full принимает fill_value и dtype.
    """
    return torch.full(shape, value, dtype=dtype)


def make_identity_like(x: Tensor) -> Tensor:
    """Единичная матрица того же размера и dtype, что квадратная матрица `x`.

    Подсказка: torch.eye(n, dtype=..., device=...) либо torch.eye_like не существует - собери руками.
    """
    return torch.eye(x.shape[0], dtype=x.dtype)


# ==========================================================================
# 1.2 Метаданные тензора
# ==========================================================================


def describe(x: Tensor) -> dict:
    """Вернуть словарь с ключами: shape (tuple[int, ...]), ndim (int),
    numel (int), dtype (torch.dtype), bytes (int - сколько памяти занимают данные).

    Подсказка: x.element_size() даёт размер одного элемента в байтах.
    """
    return {
        "shape": x.shape,
        "ndim": x.ndim,
        "numel": x.numel(),
        "dtype": x.dtype,
        "bytes": x.numel() * x.element_size(),
    }


def promoted_dtype(a: Tensor, b: Tensor) -> torch.dtype:
    """Какой dtype получится у `a + b` (правила type promotion), без вычисления суммы.

    Подсказка: torch.promote_types / torch.result_type.
    """
    return torch.promote_types(a.dtype, b.dtype)


# ==========================================================================
# 1.3 Арифметика и приведение типов
# ==========================================================================


def linear_combination(a: Tensor, b: Tensor, alpha: float, beta: float) -> Tensor:
    """alpha * a + beta * b, новый тензор, входы не трогаем."""
    return alpha * a + beta * b


def to_float32(x: Tensor) -> Tensor:
    """Привести тензор к float32. Если он уже float32 - вернуть его же (тот же объект)."""
    if torch.can_cast(x.dtype, torch.float32):
        return x.to(torch.float32)
    raise ValueError(f"Cannot cast {x.dtype} to {torch.float32}")


def mean_of_ints(x: Tensor) -> Tensor:
    """Среднее по целочисленному тензору. torch.mean на int64 падает - подумай, где нужен cast.

    Результат: скалярный тензор float32.
    """
    return x.mean(dtype=torch.float32)


# ==========================================================================
# 1.4 Копии, view и in-place
# ==========================================================================


def independent_copy(x: Tensor) -> Tensor:
    """Копия, которая НЕ делит память с `x`."""
    return x.clone()


def shares_memory(a: Tensor, b: Tensor) -> bool:
    """True, если два тензора смотрят в одно и то же хранилище (storage).

    Подсказка: a.untyped_storage().data_ptr()
    """
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


def add_inplace(x: Tensor, value: float) -> Tensor:
    """Прибавить `value` к каждому элементу ПРЯМО В ПАМЯТИ `x` и вернуть сам `x`.

    Проверка в тесте: id объекта и data_ptr не меняются.
    """
    return x.add_(value)


def zero_row_(x: Tensor, i: int) -> Tensor:
    """Занулить строку `i` матрицы `x` in-place, вернуть `x`."""
    x[i] = 0
    return x


# ==========================================================================
# 1.5 Полезные мелочи, которые постоянно нужны в LLM-коде
# ==========================================================================


def count_greater(x: Tensor, threshold: float) -> int:
    """Сколько элементов строго больше threshold. Вернуть python int, не тензор."""
    return int((x > threshold).sum().item())


def clamp_and_scale(x: Tensor, lo: float, hi: float) -> Tensor:
    """Обрезать значения в [lo, hi], затем линейно отобразить [lo, hi] -> [0, 1]."""
    return (x.clamp_(lo, hi) - lo) / (hi - lo)


def model_memory_bytes(num_params: int, dtype: torch.dtype) -> int:
    """Сколько байт занимают `num_params` параметров в данном dtype.

    Это ровно та арифметика, которой считают "влезет ли модель в GPU".
    Подсказка: torch.finfo/iinfo знают bits, либо создай пустой тензор и спроси element_size().
    """
    return dtype.itemsize * num_params


# ==========================================================================
# ТЕСТЫ
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
    assert y.tolist() == [-1, -1, -1, -1]


def test_make_identity_like():
    x = torch.randn(5, 5, dtype=torch.float64)
    eye = make_identity_like(x)
    assert eye.shape == x.shape
    assert eye.dtype == x.dtype
    torch.testing.assert_close(eye @ x, x)


def test_describe():
    x = torch.zeros(2, 3, 4, dtype=torch.float32)
    d = describe(x)
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
    assert to_float32(z) is z, "если уже float32 - лишнюю копию делать не надо"


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
    assert x[0].item() == 0.0, "копия оказалась view - данные поехали"


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
    assert out is x, "надо вернуть тот же объект"
    assert x.data_ptr() == ptr_before, "операция должна быть in-place, без новой аллокации"
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
    assert isinstance(n, int), "нужен python int, а не тензор"
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
def test_model_memory_bytes(dtype, expected_per_param):
    # 7B параметров - классический размер, который надо уметь считать в уме
    assert model_memory_bytes(7_000_000_000, dtype) == 7_000_000_000 * expected_per_param
