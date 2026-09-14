"""Топик 20 - данные, батчи и чекпоинты.

Практический хвост, без которого обучение не запустится. Ровно эти функции
просят написать в ассайнменте 1 CS336 (get_batch, save_checkpoint, load_checkpoint).

Главная идея хранения данных для LM: один длинный массив токенов uint16/int32 на диске,
из которого случайно вырезают окна длины context_length. Никаких "примеров" и "датасета" -
только поток токенов и memmap, чтобы не читать всё в память.
"""

import io

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 20.1 Нарезка батчей из потока токенов
# ==========================================================================


def get_batch(
    data: np.ndarray, batch_size: int, context_length: int, generator: np.random.Generator | None = None
) -> tuple[Tensor, Tensor]:
    """Вырезать batch_size случайных окон длины context_length из одномерного массива токенов.

    Вернуть (inputs, targets), оба (batch_size, context_length) int64:
    targets - это inputs, сдвинутые на один токен вперёд.

    Стартовая позиция i должна допускать чтение data[i : i + context_length + 1],
    то есть i выбирается из [0, len(data) - context_length - 1].
    """
    raise NotImplementedError


def max_start_index(num_tokens: int, context_length: int) -> int:
    """Наибольший допустимый стартовый индекс окна (включительно). Классический источник off-by-one."""
    raise NotImplementedError


def steps_per_epoch(num_tokens: int, batch_size: int, context_length: int) -> int:
    """Сколько шагов оптимизатора нужно, чтобы модель увидела все токены ровно один раз
    (округляем вниз - неполный последний батч не берём).
    """
    raise NotImplementedError


def sequential_windows(data: np.ndarray, context_length: int) -> Tensor:
    """Разрезать поток на НЕПЕРЕСЕКАЮЩИЕСЯ окна длины context_length, хвост отбросить.
    Вернуть (num_windows, context_length) int64. Так делают валидацию: детерминированно и без повторов.
    """
    raise NotImplementedError


# ==========================================================================
# 20.2 Паддинг и маски
# ==========================================================================


def collate_with_padding(sequences: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    """Список последовательностей разной длины -> (ids (B, T_max) int64, mask (B, T_max) bool),
    где T_max - длина самой длинной, mask=True на настоящих токенах.
    """
    raise NotImplementedError


def make_labels(ids: Tensor, mask: Tensor, ignore_index: int = -100) -> Tensor:
    """Из (B, T) ids и (B, T) mask сделать метки для лосса: сдвиг на один токен влево,
    padding-позиции заменены на ignore_index, последняя позиция каждой строки - тоже ignore_index
    (для неё нет следующего токена).
    """
    raise NotImplementedError


# ==========================================================================
# 20.3 Чекпоинты
# ==========================================================================


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: io.BytesIO) -> None:
    """Сохранить в буфер словарь с состоянием модели, оптимизатора и номером итерации.

    Состояние оптимизатора сохранять обязательно: без моментов Adam обучение
    после рестарта проседает.
    """
    raise NotImplementedError


def load_checkpoint(src: io.BytesIO, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    """Загрузить состояние в модель и оптимизатор на месте, вернуть номер итерации."""
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_get_batch_shapes_and_shift():
    data = np.arange(100, dtype=np.int64)
    x, y = get_batch(data, batch_size=8, context_length=10)
    assert x.shape == (8, 10) and y.shape == (8, 10)
    assert x.dtype == torch.int64 and y.dtype == torch.int64
    torch.testing.assert_close(y[:, :-1], x[:, 1:])
    torch.testing.assert_close(y, x + 1)  # данные - это arange, значит сдвиг виден напрямую


def test_get_batch_stays_in_bounds():
    data = np.arange(50, dtype=np.int64)
    for _ in range(50):
        x, y = get_batch(data, batch_size=4, context_length=16)
        assert int(x.min()) >= 0
        assert int(y.max()) <= 49


def test_get_batch_is_reproducible_with_generator():
    data = np.arange(200, dtype=np.int64)
    a = get_batch(data, 4, 8, generator=np.random.default_rng(0))
    b = get_batch(data, 4, 8, generator=np.random.default_rng(0))
    torch.testing.assert_close(a[0], b[0])
    c = get_batch(data, 4, 8, generator=np.random.default_rng(1))
    assert not torch.equal(a[0], c[0])


def test_get_batch_works_with_memmap(tmp_path):
    """Настоящий сценарий: токены лежат на диске, в память не читаются."""
    path = tmp_path / "tokens.bin"
    np.arange(1000, dtype=np.uint16).tofile(path)
    data = np.memmap(path, dtype=np.uint16, mode="r")
    x, y = get_batch(data, batch_size=4, context_length=32)
    assert x.shape == (4, 32)
    assert x.dtype == torch.int64, "uint16 надо привести к int64, иначе embedding не примет"
    torch.testing.assert_close(y[:, :-1], x[:, 1:])


def test_max_start_index():
    assert max_start_index(100, 10) == 89
    assert max_start_index(11, 10) == 0


def test_steps_per_epoch():
    assert steps_per_epoch(1_000_000, batch_size=8, context_length=1024) == 122
    assert steps_per_epoch(100, 4, 10) == 2


def test_sequential_windows():
    data = np.arange(11, dtype=np.int64)
    out = sequential_windows(data, 4)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out[0], torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(out[1], torch.tensor([4, 5, 6, 7]))


def test_collate_with_padding():
    ids, mask = collate_with_padding([[1, 2, 3], [4], [5, 6]], pad_id=0)
    assert ids.shape == (3, 3)
    torch.testing.assert_close(ids, torch.tensor([[1, 2, 3], [4, 0, 0], [5, 6, 0]]))
    torch.testing.assert_close(mask, torch.tensor([[True, True, True], [True, False, False], [True, True, False]]))


def test_make_labels():
    ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    labels = make_labels(ids, mask)
    torch.testing.assert_close(labels, torch.tensor([[2, 3, -100], [5, -100, -100]]))


def test_checkpoint_roundtrip():
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)

    # сделаем шаг, чтобы у оптимизатора появилось состояние
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

    buf = io.BytesIO()
    save_checkpoint(model, opt, iteration=42, out=buf)
    buf.seek(0)

    fresh_model = nn.Linear(4, 3)
    fresh_opt = torch.optim.AdamW(fresh_model.parameters(), lr=0.01)
    it = load_checkpoint(buf, fresh_model, fresh_opt)

    assert it == 42
    torch.testing.assert_close(fresh_model.weight, model.weight)
    assert fresh_opt.state_dict()["state"], "состояние оптимизатора потерялось"
    torch.testing.assert_close(fresh_opt.state_dict()["state"][0]["exp_avg"], opt.state_dict()["state"][0]["exp_avg"])


@pytest.mark.parametrize("context_length", [1, 8, 64])
def test_get_batch_context_lengths(context_length):
    data = np.arange(500, dtype=np.int64)
    x, y = get_batch(data, batch_size=3, context_length=context_length)
    assert x.shape == (3, context_length)
    torch.testing.assert_close(y, x + 1)
