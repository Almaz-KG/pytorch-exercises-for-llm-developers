"""Топик 14 - внимание.

Формула целиком:
    Attention(Q, K, V) = softmax(Q K^T / sqrt(Dh) + mask) V

Что здесь стоит понимать не по слогам, а целиком:
  - деление на sqrt(Dh) держит дисперсию логитов ~1, иначе softmax насыщается и градиент умирает;
  - маска аддитивная (-inf), а не мультипликативная: её надо добавить ДО softmax;
  - причинность - это не "свойство модели", а именно эта маска;
  - матрица внимания весит B*H*T*T - квадрат по длине, отсюда вся возня с flash-attention.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 14.0 Вспомогательное (повторение топика 02, понадобится дальше)
# ==========================================================================


def split_heads(x: Tensor, num_heads: int) -> Tensor:
    """(B, T, D) -> (B, H, T, Dh)."""
    raise NotImplementedError


def merge_heads(x: Tensor) -> Tensor:
    """(B, H, T, Dh) -> (B, T, D)."""
    raise NotImplementedError


# ==========================================================================
# 14.1 Ядро внимания
# ==========================================================================


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    """q: (..., T, Dh), k, v: (..., S, Dh), mask: bool broadcast-ящаяся к (..., T, S),
    где True = разрешено смотреть.

    Вернуть (..., T, Dh). Softmax считать по оси ключей.
    """
    raise NotImplementedError


def attention_weights(q: Tensor, k: Tensor, mask: Tensor | None = None) -> Tensor:
    """Те же вычисления, но вернуть саму матрицу внимания (..., T, S) после softmax.

    Полезно для отладки и для картинок в отчётах.
    """
    raise NotImplementedError


def causal_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Внимание с причинной маской: позиция i видит только ключи j <= i.

    Маску построй внутри, по длинам q и k (считаем, что они равны).
    """
    raise NotImplementedError


def attention_with_padding(q: Tensor, k: Tensor, v: Tensor, pad_mask: Tensor) -> Tensor:
    """pad_mask: (B, S) bool, True = настоящий токен. Причинность плюс запрет смотреть на padding.

    q, k, v: (B, H, T, Dh), T == S.
    """
    raise NotImplementedError


# ==========================================================================
# 14.2 Multi-head attention как модуль
# ==========================================================================


class MultiHeadSelfAttention(nn.Module):
    """Многоголовое причинное самовнимание.

    Требования (имена важны - это контракт тестов CS336):
      - подмодули q_proj, k_proj, v_proj, output_proj: nn.Linear(d_model, d_model, bias=False);
      - forward(x: (B, T, D)) -> (B, T, D);
      - внимание причинное.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class GroupedQueryAttention(nn.Module):
    """GQA: голов запросов num_heads, а голов ключей/значений num_kv_heads (num_heads кратно ему).

    Каждая kv-голова обслуживает группу из num_heads / num_kv_heads голов запросов.
    Это стандарт в современных моделях: KV-кеш уменьшается в разы почти без потери качества.

    Проекции: q_proj: d_model -> num_heads * Dh, k_proj и v_proj: d_model -> num_kv_heads * Dh,
    output_proj: num_heads * Dh -> d_model, все без bias, Dh = d_model // num_heads.
    """

    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 14.3 Стоимость
# ==========================================================================


def attention_matrix_bytes(batch: int, heads: int, seq_len: int, dtype: torch.dtype) -> int:
    """Сколько байт занимает материализованная матрица внимания.

    Посчитай для себя, во что превращается 32k контекста - станет понятно, зачем flash-attention.
    """
    raise NotImplementedError


def attention_flops(batch: int, heads: int, seq_len: int, d_head: int) -> int:
    """FLOPs внимания без проекций: Q K^T плюс weights V.

    Оба умножения: (T, Dh) x (Dh, T) и (T, T) x (T, Dh), для каждой головы и каждого примера.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_heads_roundtrip():
    x = torch.randn(2, 5, 12)
    torch.testing.assert_close(merge_heads(split_heads(x, 3)), x)


def test_sdpa_matches_torch():
    q, k, v = torch.randn(2, 3, 5, 8), torch.randn(2, 3, 7, 8), torch.randn(2, 3, 7, 8)
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        atol=1e-5,
        rtol=1e-5,
    )


def test_sdpa_with_mask_matches_torch():
    q, k, v = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
    idx = torch.arange(6)
    mask = idx[:, None] >= idx[None, :]
    torch.testing.assert_close(
        scaled_dot_product_attention(q, k, v, mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask),
        atol=1e-5,
        rtol=1e-5,
    )


def test_attention_weights_are_a_distribution():
    q, k = torch.randn(2, 2, 4, 16), torch.randn(2, 2, 6, 16)
    w = attention_weights(q, k)
    assert w.shape == (2, 2, 4, 6)
    torch.testing.assert_close(w.sum(dim=-1), torch.ones(2, 2, 4), atol=1e-5, rtol=0)
    assert (w >= 0).all()


def test_attention_is_uniform_for_zero_queries():
    """Нулевые запросы -> все логиты равны -> равномерное внимание. Хорошая проверка формулы."""
    q = torch.zeros(1, 1, 3, 8)
    k = torch.randn(1, 1, 5, 8)
    w = attention_weights(q, k)
    torch.testing.assert_close(w, torch.full((1, 1, 3, 5), 0.2), atol=1e-6, rtol=0)


def test_scaling_by_sqrt_dh():
    """Без деления на sqrt(Dh) логиты растут как sqrt(Dh) и softmax насыщается.
    Проверяем, что деление действительно есть."""
    torch.manual_seed(0)
    q = torch.randn(1, 1, 1, 1024)
    k = torch.randn(1, 1, 2, 1024)
    w = attention_weights(q, k)
    logits_scale = (q[0, 0, 0] @ k[0, 0].T) / math.sqrt(1024)
    expected = logits_scale.softmax(dim=-1)
    torch.testing.assert_close(w[0, 0, 0], expected, atol=1e-5, rtol=1e-5)


def test_causal_attention_masks_the_future():
    q, k, v = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8)
    out = causal_attention(q, k, v)
    assert out.shape == (1, 1, 4, 8)
    # первый выход обязан быть в точности первым значением: смотреть больше не на что
    torch.testing.assert_close(out[0, 0, 0], v[0, 0, 0], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5, rtol=1e-5)


def test_causality_via_gradients():
    """Настоящая проверка причинности: выход в позиции t не должен зависеть от входов после t."""
    x = torch.randn(1, 6, 16, requires_grad=True)
    attn = MultiHeadSelfAttention(16, 4)
    out = attn(x)
    out[0, 2].sum().backward()
    assert x.grad is not None
    assert x.grad[0, 3:].abs().max().item() == 0.0, "информация утекла из будущего"
    assert x.grad[0, :3].abs().max().item() > 0.0


def test_attention_with_padding():
    q = k = v = torch.randn(2, 2, 4, 8)
    pad = torch.tensor([[True, True, False, False], [True, True, True, True]])
    out = attention_with_padding(q, k, v, pad)
    assert out.shape == (2, 2, 4, 8)
    # для первой последовательности реальных ключей два, значит выход позиции 1
    # обязан совпасть с вниманием, посчитанным только по этим двум ключам
    ref = scaled_dot_product_attention(q[0:1, :, 1:2], k[0:1, :, :2], v[0:1, :, :2])
    torch.testing.assert_close(out[0:1, :, 1:2], ref, atol=1e-5, rtol=1e-5)


def test_mha_module_shapes_and_names():
    attn = MultiHeadSelfAttention(32, 4)
    names = {n for n, _ in attn.named_parameters()}
    assert names == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight"}
    x = torch.randn(2, 7, 32)
    assert attn(x).shape == (2, 7, 32)
    assert sum(p.numel() for p in attn.parameters()) == 4 * 32 * 32


def test_mha_matches_manual_composition():
    """Собираем то же самое из уже реализованных кирпичей - результат обязан совпасть."""
    torch.manual_seed(0)
    d_model, heads = 16, 4
    attn = MultiHeadSelfAttention(d_model, heads)
    x = torch.randn(2, 5, d_model)

    q = split_heads(x @ attn.q_proj.weight.T, heads)
    k = split_heads(x @ attn.k_proj.weight.T, heads)
    v = split_heads(x @ attn.v_proj.weight.T, heads)
    expected = merge_heads(causal_attention(q, k, v)) @ attn.output_proj.weight.T

    torch.testing.assert_close(attn(x), expected, atol=1e-5, rtol=1e-5)


def test_gqa_shapes_and_savings():
    gqa = GroupedQueryAttention(64, num_heads=8, num_kv_heads=2)
    x = torch.randn(2, 5, 64)
    assert gqa(x).shape == (2, 5, 64)
    assert gqa.k_proj.weight.shape == (2 * 8, 64), "kv-проекция должна быть узкой"
    assert gqa.q_proj.weight.shape == (64, 64)

    # при num_kv_heads == num_heads GQA вырождается в обычное MHA
    full = GroupedQueryAttention(64, num_heads=8, num_kv_heads=8)
    assert full.k_proj.weight.shape == (64, 64)
    assert full(x).shape == (2, 5, 64)


def test_gqa_is_causal():
    x = torch.randn(1, 6, 32, requires_grad=True)
    gqa = GroupedQueryAttention(32, num_heads=4, num_kv_heads=2)
    gqa(x)[0, 1].sum().backward()
    assert x.grad[0, 2:].abs().max().item() == 0.0


def test_attention_costs():
    # bf16, batch 8, 32 головы, контекст 32768
    b = attention_matrix_bytes(8, 32, 32768, torch.bfloat16)
    assert b == 8 * 32 * 32768 * 32768 * 2
    assert b > 500e12 / 1e3  # это сотни терабайт - материализовать нельзя, отсюда flash-attention

    assert attention_flops(1, 1, 128, 64) == 2 * (2 * 128 * 128 * 64)


@pytest.mark.parametrize("num_heads", [1, 2, 8])
def test_mha_head_counts(num_heads):
    attn = MultiHeadSelfAttention(16, num_heads)
    x = torch.randn(3, 4, 16)
    assert attn(x).shape == (3, 4, 16)
