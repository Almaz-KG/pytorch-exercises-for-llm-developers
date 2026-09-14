"""Топик 17 - генерация: сэмплирование и KV-кеш.

Две независимые темы, которые всегда идут вместе.

Сэмплирование: из логитов надо выбрать токен. Температура растягивает или сжимает
распределение, top-k и top-p отрезают хвост, из которого модель несёт чушь.

KV-кеш: при генерации по одному токену пересчитывать внимание по всему префиксу
заново - это O(T^2) на каждый шаг. Вместо этого K и V прошлых позиций хранят и
дописывают. Главное, что надо проверить - что с кешем получается РОВНО то же самое.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 17.1 Фильтры для логитов
# ==========================================================================


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Поделить логиты на температуру. При temperature == 0 вернуть логиты,
    из которых argmax выбирается детерминированно (например, one-hot из -inf и 0):
    это режим greedy.
    """
    raise NotImplementedError


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """(B, V) -> (B, V): всё, кроме top-k, заменить на -inf. При k <= 0 вернуть как есть."""
    raise NotImplementedError


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Nucleus sampling. Оставить минимальное множество токенов, суммарная вероятность
    которых >= p (токен, который пересекает порог, ОСТАВЛЯЕМ), остальное -> -inf.

    Алгоритм: отсортировать по убыванию, взять cumsum вероятностей, отбросить те,
    у кого накопленная сумма ДО них уже >= p, вернуть маску в исходном порядке
    (пригодится scatter из топика 05).
    """
    raise NotImplementedError


def repetition_penalty(logits: Tensor, generated: Tensor, penalty: float) -> Tensor:
    """Штраф за повтор: логиты уже сгенерированных токенов делятся на penalty,
    если они положительные, и умножаются на penalty, если отрицательные.

    logits: (B, V), generated: (B, T) уже выданные токены.
    """
    raise NotImplementedError


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(B, V) -> (B,) int64. Порядок операций стандартный: температура, top-k, top-p, softmax,
    multinomial. При temperature == 0 - жадный выбор без случайности.
    """
    raise NotImplementedError


# ==========================================================================
# 17.2 KV-кеш
# ==========================================================================


class KVCache:
    """Хранилище K и V для одного слоя.

    append(k, v) дописывает новые позиции по оси времени и возвращает полные (k, v).
    len(cache) - сколько позиций уже накоплено. Форма тензоров: (B, H, T, Dh).
    """

    def __init__(self):
        raise NotImplementedError

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CachedSelfAttention(nn.Module):
    """Причинное самовнимание, умеющее работать пошагово.

    forward(x, cache=None):
      - x: (B, T, D) - при первом вызове весь префикс, дальше по одному токену;
      - если cache передан, новые k и v дописываются в него, а внимание считается
        по всей накопленной истории;
      - причинная маска нужна только когда T > 1: один новый токен и так видит всё прошлое.

    Проекции: q_proj, k_proj, v_proj, output_proj без bias.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        raise NotImplementedError


def cache_memory_bytes(
    batch: int, num_layers: int, num_kv_heads: int, seq_len: int, head_dim: int, dtype: torch.dtype
) -> int:
    """Память под KV-кеш: два тензора (K и V) на каждый слой.

    Именно эта величина - причина существования GQA и MQA.
    """
    raise NotImplementedError


# ==========================================================================
# 17.3 Генерация
# ==========================================================================


def greedy_generate(step_fn, prompt: Tensor, max_new_tokens: int) -> Tensor:
    """step_fn(ids) -> логиты (B, T, V) для всего входа. Дописывать по одному
    самому вероятному токену, возвращая (B, T_prompt + max_new_tokens).

    Это наивная версия без кеша: на каждом шаге прогоняется весь префикс.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_apply_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(apply_temperature(logits, 2.0), logits / 2.0)
    hot = apply_temperature(logits, 0.5).softmax(-1)
    warm = apply_temperature(logits, 1.0).softmax(-1)
    assert hot.max() > warm.max(), "низкая температура должна заострять распределение"

    greedy = apply_temperature(logits, 0.0)
    assert greedy.argmax(-1).item() == 2
    assert greedy.softmax(-1).max().item() > 0.999


def test_top_k_filter():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    out = top_k_filter(logits, 2)
    assert torch.isfinite(out).sum().item() == 2
    assert torch.isfinite(out[0, 1]) and torch.isfinite(out[0, 2])
    torch.testing.assert_close(top_k_filter(logits, 0), logits)


def test_top_p_filter():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    out = top_p_filter(logits, 0.8)
    assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 1])
    assert torch.isinf(out[0, 2]) and torch.isinf(out[0, 3])

    # p = 1.0 не должен отрезать ничего
    torch.testing.assert_close(top_p_filter(logits, 1.0).softmax(-1), probs, atol=1e-6, rtol=1e-5)
    # даже при крошечном p один токен обязан остаться, иначе сэмплировать не из чего
    assert torch.isfinite(top_p_filter(logits, 0.01)).sum().item() >= 1


def test_top_p_keeps_order_of_original_vocab():
    logits = torch.tensor([[0.1, 5.0, 0.2, 4.0]])
    out = top_p_filter(logits, 0.9)
    kept = torch.isfinite(out)[0]
    assert bool(kept[1]) and bool(kept[3]), "оставлять надо самые вероятные, но на их исходных местах"
    assert not bool(kept[0])


def test_repetition_penalty():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    generated = torch.tensor([[0, 1]])
    out = repetition_penalty(logits, generated, 2.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, -4.0, 1.0]]))


def test_sample_from_logits_is_greedy_at_zero_temperature():
    logits = torch.tensor([[0.1, 0.2, 5.0], [7.0, 0.0, 0.0]])
    out = sample_from_logits(logits, temperature=0.0)
    assert out.shape == (2,)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([2, 0]))


def test_sample_from_logits_respects_top_k():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[3.0, 2.9, 0.0, -5.0]])
    samples = torch.stack([sample_from_logits(logits, top_k=2, generator=g) for _ in range(200)])
    assert set(samples.flatten().tolist()) <= {0, 1}, "top-k пропустил отрезанные токены"


def test_sample_from_logits_distribution():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    samples = torch.stack([sample_from_logits(logits, generator=g) for _ in range(4000)])
    counts = torch.bincount(samples.flatten(), minlength=4).float() / 4000
    assert (counts - 0.25).abs().max().item() < 0.03


def test_kv_cache_append():
    cache = KVCache()
    assert len(cache) == 0
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k, v = cache.append(k1, v1)
    assert k.shape == (1, 2, 3, 4) and len(cache) == 3

    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = cache.append(k2, v2)
    assert k.shape == (1, 2, 4, 4) and len(cache) == 4
    torch.testing.assert_close(k[:, :, :3], k1)
    torch.testing.assert_close(v[:, :, 3:], v2)


def test_incremental_decoding_matches_full_forward():
    """Главный тест топика: пошаговая генерация с кешем обязана дать те же числа,
    что и один проход по всей последовательности."""
    torch.manual_seed(0)
    attn = CachedSelfAttention(32, 4)
    x = torch.randn(2, 6, 32)

    full = attn(x)

    cache = KVCache()
    steps = [attn(x[:, t : t + 1], cache=cache) for t in range(6)]
    incremental = torch.cat(steps, dim=1)

    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
    assert len(cache) == 6


def test_prefill_then_decode():
    """Реальный режим: сначала префилл всего промпта, потом по одному токену."""
    torch.manual_seed(0)
    attn = CachedSelfAttention(16, 2)
    x = torch.randn(1, 5, 16)
    full = attn(x)

    cache = KVCache()
    prefill = attn(x[:, :3], cache=cache)
    assert len(cache) == 3
    torch.testing.assert_close(prefill, full[:, :3], atol=1e-5, rtol=1e-5)

    rest = torch.cat([attn(x[:, t : t + 1], cache=cache) for t in (3, 4)], dim=1)
    torch.testing.assert_close(rest, full[:, 3:], atol=1e-5, rtol=1e-5)


def test_cache_memory_bytes():
    # LLaMA-2-7B: 32 слоя, 32 kv-головы, head_dim 128, bf16, батч 1, контекст 4096
    mha = cache_memory_bytes(1, 32, 32, 4096, 128, torch.bfloat16)
    assert mha == 2 * 1 * 32 * 32 * 4096 * 128 * 2
    # GQA с 8 kv-головами уменьшает кеш ровно в 4 раза
    gqa = cache_memory_bytes(1, 32, 8, 4096, 128, torch.bfloat16)
    assert mha // gqa == 4


def test_greedy_generate():
    """Модель-заглушка всегда предсказывает "следующий токен = предыдущий + 1"."""

    def step_fn(ids: Tensor) -> Tensor:
        B, T = ids.shape
        logits = torch.full((B, T, 10), -10.0)
        nxt = (ids + 1) % 10
        logits.scatter_(-1, nxt.unsqueeze(-1), 10.0)
        return logits

    prompt = torch.tensor([[0, 1]])
    out = greedy_generate(step_fn, prompt, max_new_tokens=3)
    assert out.shape == (1, 5)
    torch.testing.assert_close(out, torch.tensor([[0, 1, 2, 3, 4]]))


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
def test_top_p_never_empties_the_row(p):
    logits = torch.randn(4, 20)
    out = top_p_filter(logits, p)
    assert (torch.isfinite(out).sum(dim=-1) >= 1).all()
    probs = out.softmax(dim=-1)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)
