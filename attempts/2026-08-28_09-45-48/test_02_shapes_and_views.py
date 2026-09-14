"""Топик 02 - формы, view, перестановки осей.

Что тренируем: reshape/view, permute/transpose, contiguous, squeeze/unsqueeze,
cat/stack/split/chunk, expand vs repeat, unfold (скользящее окно), pad.

Это самый частый источник багов в LLM-коде: "почему у меня (B, H, T, D), а надо (B, T, H, D)".
Соглашение по именам осей во всём курсе:
  B - batch, T - длина последовательности (time), D - d_model, H - число голов, Dh - размер головы.
"""

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 2.1 reshape / view / flatten
# ==========================================================================


def flatten_tokens(x: Tensor) -> Tensor:
    """(B, T, D) -> (B*T, D). Именно так схлопывают батч перед lm_head или лоссом."""
    return x.flatten(0, 1)


def unflatten_tokens(x: Tensor, batch_size: int) -> Tensor:
    """(B*T, D) -> (B, T, D). Обратная операция."""
    return x.reshape(batch_size, -1, x.shape[-1])


def flatten_all_but_last(x: Tensor) -> Tensor:
    """(..., D) -> (N, D), где N - произведение всех ведущих осей.

    Подсказка: -1 в reshape вычисляется автоматически.
    """
    return x.reshape((-1, x.shape[-1]))


def add_batch_dim(x: Tensor) -> Tensor:
    """(T, D) -> (1, T, D) без копирования данных."""
    return x.unsqueeze(0)


def drop_singleton_dims(x: Tensor) -> Tensor:
    """Убрать ВСЕ оси размера 1."""
    return x.squeeze()


# ==========================================================================
# 2.2 Перестановка осей
# ==========================================================================


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh), где D = H * Dh.

    Порядок важен: сначала разложить последнюю ось на (H, Dh), потом переставить оси.
    Если сделать наоборот - головы перемешаются, и тест это поймает.
    """
    B, T, D = x.shape
    return x.reshape((B, T, num_heads, D // num_heads)).transpose(1, 2)


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D). Ровно обратна split_heads.

    Осторожно: после permute тензор не contiguous, и .view() упадёт. Подумай, чем чинить.
    """
    B, H, T, Dh = x.shape
    return x.transpose(1, 2).reshape((B, T, H * Dh))


def transpose_last_two(x: Tensor) -> Tensor:
    """Поменять местами две последние оси у тензора любой размерности.

    Так делают K^T в attention: (B, H, T, Dh) -> (B, H, Dh, T).
    """
    return x.transpose(-2, -1)


def is_contiguous_after_transpose(x: Tensor) -> tuple[bool, bool]:
    """Вернуть (contiguous до транспонирования, contiguous после .transpose(0, 1)).

    Задача на понимание: транспонирование - это только смена strides, данные не двигаются.
    """
    return (x.is_contiguous(), x.transpose(0, 1).is_contiguous())


def strides_of(x: Tensor) -> tuple[int, ...]:
    """Шаги (в элементах) по каждой оси. Для (2,3,4) float32 contiguous это (12, 4, 1)."""
    return x.stride()


# ==========================================================================
# 2.3 Склейка и разрезание
# ==========================================================================


def concat_sequences(a: Tensor, b: Tensor) -> Tensor:
    """Два тензора (B, T1, D) и (B, T2, D) склеить по времени -> (B, T1+T2, D).

    Это буквально то, что делает KV-кеш при добавлении нового токена.
    """
    return torch.concat((a, b), dim=1)


def stack_batch(items: list[Tensor]) -> Tensor:
    """Список из N тензоров (T, D) собрать в батч (N, T, D)."""
    return torch.stack(items)


def split_qkv(qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(B, T, 3*D) -> три тензора (B, T, D): q, k, v.

    Так устроен fused qkv-проектор: одна матрица вместо трёх, потом разрез.
    """
    _, _, D = qkv.shape
    q, k, v = torch.split(qkv, D // 3, dim=2)

    return (q, k, v)


# ==========================================================================
# 2.4 expand vs repeat - разница в памяти
# ==========================================================================


def broadcast_kv_heads(kv: Tensor, num_groups: int) -> Tensor:
    """GQA: (B, H_kv, T, Dh) -> (B, H_kv * num_groups, T, Dh), БЕЗ копирования данных
    там, где это возможно, и с правильным порядком голов:
    голова kv-номер i должна повториться num_groups раз подряд.

    Подсказка: unsqueeze -> expand -> reshape. reshape в конце скопирует,
    но expand по пути не должен аллоцировать новый буфер.
    """
    B, H, T, D = kv.shape

    # print(f"AA0: {kv.shape}")
    # print(f"AA1: {kv.unsqueeze(dim=2).shape}")
    # print(f"AA2: {kv.unsqueeze(dim=2).expand(B, H, num_groups, T, D).shape}")
    # print(f"AA3: {kv.unsqueeze(dim=2).expand(B, H, num_groups, T, D).reshape([B, H * num_groups, T, D]).shape}")
    return kv.unsqueeze(dim=2).expand([B, H, num_groups, T, D]).reshape([B, H * num_groups, T, D])


def expand_row(x: Tensor, n: int) -> Tensor:
    """(1, D) -> (n, D) БЕЗ копирования (view с нулевым stride)."""
    # print(f"AA0: {n}")
    # print(f"AA0: {x.shape}")
    # print(f"AA1: {x.expand([n, x.shape[1]]).shape}")

    return x.expand([n, x.shape[-1]])


def repeat_row(x: Tensor, n: int) -> Tensor:
    """(1, D) -> (n, D) С реальным копированием данных."""
    # print(f"AA0: {n}")
    # print(f"AA0: {x.shape}")
    # print(f"AA1: {x.expand([n, x.shape[1]]).shape}")
    return x.repeat([n, 1])


# ==========================================================================
# 2.5 Скользящее окно и паддинг
# ==========================================================================


def sliding_windows(x: Tensor, window: int) -> Tensor:
    """Одномерный тензор длины T -> все окна длины `window` с шагом 1: (T-window+1, window).

    Так нарезают обучающие примеры из непрерывного потока токенов.
    Подсказка: Tensor.unfold(dim, size, step).
    """
    return x.unfold(0, window, 1)


def pad_to_length(x: Tensor, target_len: int, pad_value: int) -> Tensor:
    """(B, T) -> (B, target_len), добивая справа значением pad_value. Считаем T <= target_len.

    Подсказка: F.pad и порядок пар в его аргументе - от последней оси к первой.
    """
    ln = target_len - x.shape[-1]
    return F.pad(x, [0, ln], value=pad_value)


def shift_right(x: Tensor, pad_value: int) -> Tensor:
    """(B, T) -> (B, T): сдвиг вправо на одну позицию, слева вставляется pad_value,
    последний токен выпадает. Классический teacher forcing: из input делаем decoder input.
    """
    # print(f"AA0: {pad_value}")
    # print(f"AA0: {x.shape}")
    # print(f"AA1: {x[:, :-1].shape}")
    # print(f"AA2: {F.pad(x[:, :-1], [1, 0], value=pad_value).shape}")

    return F.pad(x[:, :-1], [1, 0], value=pad_value)


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_flatten_and_unflatten_tokens():
    x = torch.randn(2, 3, 4)
    flat = flatten_tokens(x)
    assert flat.shape == (6, 4)
    torch.testing.assert_close(flat[0], x[0, 0])
    torch.testing.assert_close(flat[3], x[1, 0])
    torch.testing.assert_close(unflatten_tokens(flat, 2), x)


def test_flatten_all_but_last():
    assert flatten_all_but_last(torch.randn(2, 3, 4, 5)).shape == (24, 5)
    assert flatten_all_but_last(torch.randn(7, 5)).shape == (7, 5)
    x = torch.randn(2, 3, 5)
    torch.testing.assert_close(flatten_all_but_last(x)[4], x[1, 1])


def test_add_batch_dim():
    x = torch.randn(3, 4)
    y = add_batch_dim(x)
    assert y.shape == (1, 3, 4)
    assert y.data_ptr() == x.data_ptr(), "unsqueeze не должен копировать"


def test_drop_singleton_dims():
    assert drop_singleton_dims(torch.randn(1, 3, 1, 4, 1)).shape == (3, 4)
    assert drop_singleton_dims(torch.randn(2, 3)).shape == (2, 3)


def test_split_heads():
    B, T, H, Dh = 2, 5, 4, 8
    x = torch.randn(B, T, H * Dh)
    out = split_heads(x, H)
    assert out.shape == (B, H, T, Dh)
    # голова h, позиция t - это срез [h*Dh : (h+1)*Dh] исходного вектора
    for h in range(H):
        torch.testing.assert_close(out[:, h, :, :], x[..., h * Dh : (h + 1) * Dh])


def test_merge_heads_roundtrip():
    x = torch.randn(2, 5, 32)
    torch.testing.assert_close(merge_heads(split_heads(x, 4)), x)


def test_transpose_last_two():
    x = torch.randn(2, 4, 5, 8)
    out = transpose_last_two(x)
    assert out.shape == (2, 4, 8, 5)
    torch.testing.assert_close(out[1, 2, 3, 4], x[1, 2, 4, 3])
    assert transpose_last_two(torch.randn(3, 7)).shape == (7, 3)


def test_is_contiguous_after_transpose():
    before, after = is_contiguous_after_transpose(torch.randn(3, 4))
    assert before is True
    assert after is False


def test_strides_of():
    assert strides_of(torch.zeros(2, 3, 4)) == (12, 4, 1)
    assert strides_of(torch.zeros(2, 3, 4).transpose(0, 1)) == (4, 12, 1)


def test_concat_sequences():
    a, b = torch.randn(2, 3, 8), torch.randn(2, 1, 8)
    out = concat_sequences(a, b)
    assert out.shape == (2, 4, 8)
    torch.testing.assert_close(out[:, :3], a)
    torch.testing.assert_close(out[:, 3:], b)


def test_stack_batch():
    items = [torch.randn(3, 4) for _ in range(5)]
    out = stack_batch(items)
    assert out.shape == (5, 3, 4)
    torch.testing.assert_close(out[2], items[2])


def test_split_qkv():
    B, T, D = 2, 3, 8
    qkv = torch.randn(B, T, 3 * D)
    q, k, v = split_qkv(qkv)
    for t in (q, k, v):
        assert t.shape == (B, T, D)
    torch.testing.assert_close(q, qkv[..., :D])
    torch.testing.assert_close(k, qkv[..., D : 2 * D])
    torch.testing.assert_close(v, qkv[..., 2 * D :])


def test_broadcast_kv_heads():
    B, H_kv, T, Dh = 2, 2, 3, 4
    kv = torch.randn(B, H_kv, T, Dh)
    out = broadcast_kv_heads(kv, 3)
    assert out.shape == (B, 6, T, Dh)
    # головы 0,1,2 - копии kv-головы 0; головы 3,4,5 - копии kv-головы 1
    for h in range(6):
        torch.testing.assert_close(out[:, h], kv[:, h // 3])


def test_expand_vs_repeat():
    x = torch.randn(1, 5)
    e = expand_row(x, 4)
    r = repeat_row(x, 4)
    assert e.shape == r.shape == (4, 5)
    torch.testing.assert_close(e, r)
    assert e.untyped_storage().data_ptr() == x.untyped_storage().data_ptr(), "expand не копирует"
    assert r.untyped_storage().data_ptr() != x.untyped_storage().data_ptr(), "repeat копирует"
    assert e.stride(0) == 0, "у expand шаг по размноженной оси равен нулю"


def test_sliding_windows():
    x = torch.arange(6)
    out = sliding_windows(x, 3)
    assert out.shape == (4, 3)
    torch.testing.assert_close(out[0], torch.tensor([0, 1, 2]))
    torch.testing.assert_close(out[3], torch.tensor([3, 4, 5]))


def test_pad_to_length():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = pad_to_length(x, 5, pad_value=0)
    assert out.shape == (2, 5)
    torch.testing.assert_close(out, torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]]))


def test_shift_right():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    out = shift_right(x, pad_value=-1)
    torch.testing.assert_close(out, torch.tensor([[-1, 1, 2], [-1, 4, 5]]))
    assert out.dtype == x.dtype


@pytest.mark.parametrize("shape", [(2, 3, 8), (1, 1, 8), (4, 7, 16)])
def test_heads_roundtrip_property(shape):
    x = torch.randn(*shape)
    torch.testing.assert_close(merge_heads(split_heads(x, 2)), x)
