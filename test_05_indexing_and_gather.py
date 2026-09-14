"""Топик 05 - индексация, gather/scatter, маски.

Здесь живут операции, без которых не написать ни лосс, ни сэмплирование:
  - embedding lookup - это просто индексация матрицы;
  - "вероятность правильного токена" - это gather;
  - "оставить только top-k логитов" - это topk + scatter;
  - "взять последний непустой токен" - это gather по длинам.
"""

import pytest
import torch
from torch import Tensor

# ==========================================================================
# 5.1 Индексация как embedding lookup
# ==========================================================================


def embedding_lookup(table: Tensor, ids: Tensor) -> Tensor:
    """table: (V, D), ids: (B, T) int64 -> (B, T, D). Обычной индексацией, без nn.Embedding."""
    raise NotImplementedError


def select_rows(x: Tensor, rows: list[int]) -> Tensor:
    """Выбрать указанные строки матрицы в указанном порядке (можно с повторами)."""
    raise NotImplementedError


def fancy_index_pairs(x: Tensor, batch_idx: Tensor, pos_idx: Tensor) -> Tensor:
    """x: (B, T, D), batch_idx: (N,), pos_idx: (N,) -> (N, D),
    где out[n] = x[batch_idx[n], pos_idx[n]].

    Это "продвинутая индексация двумя тензорами" - оси индексов идут парами.
    """
    raise NotImplementedError


# ==========================================================================
# 5.2 gather - выдёргивание значений по индексам
# ==========================================================================


def gather_target_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """logits: (B, T, V), targets: (B, T) -> (B, T): логит правильного токена.

    Самая частая gather-конструкция во всём обучении LM.
    Подсказка: torch.gather требует, чтобы index имел столько же осей, сколько input.
    """
    raise NotImplementedError


def last_valid_hidden(h: Tensor, lengths: Tensor) -> Tensor:
    """h: (B, T, D), lengths: (B,) int64 (число настоящих токенов) -> (B, D):
    скрытое состояние последнего настоящего токена каждой последовательности.

    Так делают pooling для классификации на decoder-only модели.
    """
    raise NotImplementedError


def gather_beams(x: Tensor, beam_idx: Tensor) -> Tensor:
    """x: (B, D), beam_idx: (B,) -> (B, D): переупорядочить строки по индексам.
    Так beam search перетасовывает KV-кеш после каждого шага.
    """
    raise NotImplementedError


# ==========================================================================
# 5.3 scatter - запись по индексам
# ==========================================================================


def scatter_one_hot(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (B, T) -> (B, T, V) float one-hot, собранный через scatter_ (не F.one_hot)."""
    raise NotImplementedError


def bag_of_words(ids: Tensor, vocab_size: int) -> Tensor:
    """ids: (B, T) -> (B, V) float: сколько раз каждый токен встретился в строке.

    Подсказка: scatter_add_ с источником из единиц.
    """
    raise NotImplementedError


def keep_top_k(logits: Tensor, k: int) -> Tensor:
    """logits: (B, V) -> (B, V), где всё, кроме top-k значений в строке, заменено на -inf.

    Ровно этот приём стоит внутри top-k сэмплирования.
    Подсказка: topk даёт индексы; сделай тензор из -inf и scatter в него найденные значения.
    """
    raise NotImplementedError


# ==========================================================================
# 5.4 Булевы маски
# ==========================================================================


def select_valid(x: Tensor, mask: Tensor) -> Tensor:
    """x: (B, T, D), mask: (B, T) bool -> (N, D): только настоящие токены, "распрямлённые" в один список.

    Так делают packing перед лоссом, чтобы не считать градиенты по padding.
    """
    raise NotImplementedError


def replace_where(x: Tensor, cond: Tensor, value: float) -> Tensor:
    """Вернуть копию x, где по условию cond стоит value (in-place не трогаем)."""
    raise NotImplementedError


def find_positions(ids: Tensor, token_id: int) -> Tensor:
    """ids: (T,) -> (N,) int64: позиции, где стоит token_id.

    Пригодится для поиска <|endoftext|> в потоке токенов.
    """
    raise NotImplementedError


def mask_after_eos(ids: Tensor, eos_id: int) -> Tensor:
    """ids: (B, T) -> (B, T) bool: True для позиций ДО первого eos включительно,
    False для всего, что после него. Если eos нет - вся строка True.

    Подсказка: cumsum по булеву тензору "здесь был eos" даёт число встреченных eos;
    сравни его с 1 аккуратно, чтобы сам eos остался True. Без циклов.
    """
    raise NotImplementedError


# ==========================================================================
# 5.5 Подготовка обучающих пар
# ==========================================================================


def make_lm_pairs(ids: Tensor) -> tuple[Tensor, Tensor]:
    """ids: (B, T+1) -> (inputs (B, T), targets (B, T)), сдвиг на один токен.

    Модель по позиции i предсказывает токен i+1 - вот и весь next token prediction.
    """
    raise NotImplementedError


def expand_for_beams(x: Tensor, num_beams: int) -> Tensor:
    """x: (B, T) -> (B * num_beams, T), где строки ОДНОГО примера идут подряд.

    Подсказка: repeat_interleave, а не repeat - порядок разный, и это ловушка.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_embedding_lookup():
    table = torch.randn(10, 4)
    ids = torch.tensor([[0, 3], [9, 3]])
    out = embedding_lookup(table, ids)
    assert out.shape == (2, 2, 4)
    torch.testing.assert_close(out[0, 1], table[3])
    torch.testing.assert_close(out[1, 0], table[9])
    torch.testing.assert_close(out, torch.nn.functional.embedding(ids, table))


def test_select_rows():
    x = torch.arange(12).reshape(4, 3)
    out = select_rows(x, [2, 0, 2])
    assert out.shape == (3, 3)
    torch.testing.assert_close(out, torch.stack([x[2], x[0], x[2]]))


def test_fancy_index_pairs():
    x = torch.randn(3, 5, 2)
    b = torch.tensor([0, 2, 2])
    p = torch.tensor([4, 0, 1])
    out = fancy_index_pairs(x, b, p)
    assert out.shape == (3, 2)
    torch.testing.assert_close(out[0], x[0, 4])
    torch.testing.assert_close(out[2], x[2, 1])


def test_gather_target_logits():
    logits = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    targets = torch.tensor([[0, 1, 2], [3, 3, 0]])
    out = gather_target_logits(logits, targets)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out, torch.tensor([[0.0, 5.0, 10.0], [15.0, 19.0, 20.0]]))


def test_last_valid_hidden():
    h = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    lengths = torch.tensor([2, 4])
    out = last_valid_hidden(h, lengths)
    assert out.shape == (2, 3)
    torch.testing.assert_close(out[0], h[0, 1])
    torch.testing.assert_close(out[1], h[1, 3])


def test_gather_beams():
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    idx = torch.tensor([3, 3, 0, 1])
    out = gather_beams(x, idx)
    torch.testing.assert_close(out, x[idx])


def test_scatter_one_hot():
    ids = torch.tensor([[0, 2], [1, 1]])
    out = scatter_one_hot(ids, 3)
    assert out.shape == (2, 2, 3)
    assert out.dtype in (torch.float32, torch.float64)
    torch.testing.assert_close(out, torch.nn.functional.one_hot(ids, 3).to(out.dtype))


def test_bag_of_words():
    ids = torch.tensor([[0, 2, 2], [1, 1, 1]])
    out = bag_of_words(ids, 4)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0, 2.0, 0.0], [0.0, 3.0, 0.0, 0.0]]))


def test_keep_top_k():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    out = keep_top_k(logits, 2)
    assert out.shape == logits.shape
    assert torch.isinf(out[0, 0]) and out[0, 0] < 0
    assert torch.isinf(out[0, 3]) and out[0, 3] < 0
    torch.testing.assert_close(out[0, 1], torch.tensor(5.0))
    torch.testing.assert_close(out[0, 2], torch.tensor(3.0))
    # softmax по такому тензору обязан дать ровно k ненулевых вероятностей
    probs = out.softmax(dim=-1)
    assert int((probs > 0).sum().item()) == 2
    torch.testing.assert_close(probs.sum(), torch.tensor(1.0))


def test_select_valid():
    x = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    out = select_valid(x, mask)
    assert out.shape == (4, 2)
    torch.testing.assert_close(out[0], x[0, 0])
    torch.testing.assert_close(out[1], x[0, 2])
    torch.testing.assert_close(out[2], x[1, 1])


def test_replace_where():
    x = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    original = x.clone()
    out = replace_where(x, x < 0, 0.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, 0.0], [2.0, 0.0]]))
    torch.testing.assert_close(x, original, msg="вход изменять нельзя")


def test_find_positions():
    ids = torch.tensor([5, 1, 5, 2])
    out = find_positions(ids, 5)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([0, 2]))


def test_mask_after_eos():
    ids = torch.tensor([[1, 2, 9, 3, 4], [1, 2, 3, 4, 5], [9, 1, 2, 9, 3]])
    out = mask_after_eos(ids, eos_id=9)
    expected = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
            [True, False, False, False, False],
        ]
    )
    assert torch.equal(out, expected)


def test_make_lm_pairs():
    ids = torch.tensor([[1, 2, 3, 4]])
    inputs, targets = make_lm_pairs(ids)
    torch.testing.assert_close(inputs, torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(targets, torch.tensor([[2, 3, 4]]))


def test_expand_for_beams():
    x = torch.tensor([[1, 2], [3, 4]])
    out = expand_for_beams(x, 3)
    assert out.shape == (6, 2)
    torch.testing.assert_close(out, torch.tensor([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]))


@pytest.mark.parametrize("k", [1, 3, 5])
def test_keep_top_k_counts(k):
    logits = torch.randn(4, 7)
    out = keep_top_k(logits, k)
    assert (torch.isfinite(out).sum(dim=-1) == k).all()
