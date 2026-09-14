"""Топик 13 - эмбеддинги и позиционные кодировки, включая RoPE.

Позиционная информация в трансформер попадает одним из трёх способов:
  1. абсолютные синусоиды (2017): прибавляются к эмбеддингам один раз;
  2. обучаемые абсолютные позиции (GPT-2): просто ещё одна таблица эмбеддингов;
  3. RoPE (LLaMA, Qwen, CS336): поворот q и k на угол, пропорциональный позиции.

Почему победил RoPE: после поворота скалярное произведение q_i . k_j зависит ТОЛЬКО
от разности позиций i - j. То есть модель получает относительные позиции бесплатно,
не тратя ни одного параметра, и это переносится на длины, которых не было в обучении.

Конвенция RoPE в этом файле (она же в CS336): вектор головы длины Dh (Dh чётное)
разбивается на СОСЕДНИЕ пары (x0, x1), (x2, x3), ... Пара номер i поворачивается
на угол pos * theta^(-2i/Dh).
"""

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 13.1 Эмбеддинги
# ==========================================================================


def scaled_token_embedding(table: Tensor, ids: Tensor) -> Tensor:
    """Lookup, умноженный на sqrt(d_model) - так делает оригинальный трансформер,
    чтобы эмбеддинги и позиционные синусоиды были сравнимы по масштабу.
    """
    raise NotImplementedError


def learned_positional(embed: Tensor, pos_table: Tensor, offset: int = 0) -> Tensor:
    """embed: (B, T, D), pos_table: (max_len, D) -> (B, T, D): прибавить позиционные векторы
    для позиций offset, offset+1, ..., offset+T-1.

    offset нужен при генерации: на шаге декодирования у нас один токен, но его позиция не нулевая.
    """
    raise NotImplementedError


def sinusoidal_pe(seq_len: int, d_model: int) -> Tensor:
    """Синусоидальные позиционные кодировки (T, D) из статьи 2017 года:
    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
    """
    raise NotImplementedError


# ==========================================================================
# 13.2 RoPE: частоты и кеш
# ==========================================================================


def rope_inv_freq(d_head: int, theta: float = 10000.0) -> Tensor:
    """(d_head // 2,): обратные частоты theta^(-2i/d_head) для i = 0, 1, ..., d_head/2 - 1.

    Первая пара крутится быстро (частота 1), последняя - очень медленно.
    """
    raise NotImplementedError


def rope_cos_sin(seq_len: int, d_head: int, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    """Кеш поворотов: два тензора (seq_len, d_head // 2) с косинусами и синусами углов
    (offset + pos) * inv_freq[i].

    В реальной модели это считают один раз и держат как буфер.
    """
    raise NotImplementedError


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (..., T, Dh), cos/sin: (T, Dh // 2) -> тензор той же формы.

    Каждая соседняя пара (x[..., 2i], x[..., 2i+1]) поворачивается на свой угол:
      out_even = x_even * cos - x_odd * sin
      out_odd  = x_even * sin + x_odd * cos

    Функция должна работать и для (T, Dh), и для (B, H, T, Dh) - за счёт broadcasting.
    """
    raise NotImplementedError


def apply_rope_to_qk(q: Tensor, k: Tensor, theta: float = 10000.0, offset: int = 0) -> tuple[Tensor, Tensor]:
    """Удобная обёртка: q, k формы (B, H, T, Dh) -> повёрнутые q, k.

    Обрати внимание: RoPE применяют к q и k, но НЕ к v.
    """
    raise NotImplementedError


# ==========================================================================
# 13.3 Свойства RoPE, которые и делают его полезным
# ==========================================================================


def rope_dot_product(q: Tensor, k: Tensor, pos_q: int, pos_k: int, theta: float = 10000.0) -> Tensor:
    """q, k: (Dh,). Повернуть q на позицию pos_q, k на позицию pos_k, вернуть их скалярное произведение.

    Понадобится, чтобы проверить относительность.
    """
    raise NotImplementedError


def rope_norm_change(x: Tensor, pos: int, theta: float = 10000.0) -> Tensor:
    """Насколько поворот меняет норму вектора. Ожидаемо - никак: это чистый поворот."""
    raise NotImplementedError


# ==========================================================================
# 13.4 Модуль
# ==========================================================================


class RotaryEmbedding(nn.Module):
    """Модуль, который держит кеши cos и sin как БУФЕРЫ (не параметры!) до max_len
    и применяет их к тензору (B, H, T, Dh) с учётом смещения позиций.

    Буферы должны называться `cos_cached` и `sin_cached`, обучаемых параметров быть не должно.
    """

    def __init__(self, d_head: int, max_len: int, theta: float = 10000.0):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_scaled_token_embedding():
    table = torch.randn(10, 16)
    ids = torch.tensor([[1, 2]])
    out = scaled_token_embedding(table, ids)
    torch.testing.assert_close(out, table[ids] * math.sqrt(16))


def test_learned_positional():
    embed = torch.zeros(2, 3, 4)
    pos = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    out = learned_positional(embed, pos)
    torch.testing.assert_close(out[0], pos[:3])
    out2 = learned_positional(embed, pos, offset=2)
    torch.testing.assert_close(out2[0], pos[2:5])


def test_sinusoidal_pe():
    pe = sinusoidal_pe(6, 8)
    assert pe.shape == (6, 8)
    torch.testing.assert_close(pe[0], torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]))
    # первая пара каналов крутится с частотой 1
    torch.testing.assert_close(pe[1, 0], torch.sin(torch.tensor(1.0)))
    torch.testing.assert_close(pe[1, 1], torch.cos(torch.tensor(1.0)))
    # последняя пара - очень медленно
    assert pe[1, -1].item() > 0.999


def test_rope_inv_freq():
    f = rope_inv_freq(8, theta=10000.0)
    assert f.shape == (4,)
    torch.testing.assert_close(f[0], torch.tensor(1.0))
    torch.testing.assert_close(f[3], torch.tensor(10000.0 ** (-6 / 8)))
    assert (f[1:] < f[:-1]).all(), "частоты должны убывать"


def test_rope_cos_sin():
    cos, sin = rope_cos_sin(4, 8)
    assert cos.shape == sin.shape == (4, 4)
    torch.testing.assert_close(cos[0], torch.ones(4))
    torch.testing.assert_close(sin[0], torch.zeros(4))
    torch.testing.assert_close(cos.pow(2) + sin.pow(2), torch.ones(4, 4), atol=1e-6, rtol=0)

    cos_off, sin_off = rope_cos_sin(2, 8, offset=2)
    torch.testing.assert_close(cos_off[0], cos[2])
    torch.testing.assert_close(sin_off[1], sin[3])


def test_apply_rope_position_zero_is_identity():
    x = torch.randn(2, 3, 5, 8)
    cos, sin = rope_cos_sin(5, 8)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    torch.testing.assert_close(out[..., 0, :], x[..., 0, :], atol=1e-6, rtol=0)


def test_apply_rope_explicit_small_case():
    """Dh = 2, одна пара, inv_freq = [1]. Значит поворот на угол, равный самой позиции."""
    x = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # (T=3, Dh=2)
    cos, sin = rope_cos_sin(3, 2)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out[0], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(out[1], torch.tensor([math.cos(1.0), math.sin(1.0)]))
    torch.testing.assert_close(out[2], torch.tensor([-math.sin(2.0), math.cos(2.0)]))


def test_apply_rope_preserves_norm():
    x = torch.randn(2, 4, 6, 16)
    cos, sin = rope_cos_sin(6, 16)
    out = apply_rope(x, cos, sin)
    torch.testing.assert_close(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5, rtol=1e-5)


def test_rope_is_relative():
    """Ключевое свойство: скалярное произведение зависит только от разности позиций."""
    q, k = torch.randn(16), torch.randn(16)
    base = rope_dot_product(q, k, 5, 3)
    for pq, pk in [(2, 0), (10, 8), (101, 99)]:
        torch.testing.assert_close(rope_dot_product(q, k, pq, pk), base, atol=1e-4, rtol=1e-4)
    # а при другой разности значение обязано отличаться
    assert not torch.allclose(rope_dot_product(q, k, 5, 0), base, atol=1e-3)


def test_rope_norm_change():
    x = torch.randn(32)
    assert rope_norm_change(x, 17).abs().item() < 1e-5


def test_apply_rope_to_qk():
    B, H, T, Dh = 2, 3, 4, 8
    q, k = torch.randn(B, H, T, Dh), torch.randn(B, H, T, Dh)
    qr, kr = apply_rope_to_qk(q, k)
    assert qr.shape == q.shape and kr.shape == k.shape
    cos, sin = rope_cos_sin(T, Dh)
    torch.testing.assert_close(qr, apply_rope(q, cos, sin))

    # смещение позиций - режим генерации по одному токену
    q1 = q[:, :, 3:4]
    q1_shifted, _ = apply_rope_to_qk(q1, q1, offset=3)
    torch.testing.assert_close(q1_shifted, qr[:, :, 3:4], atol=1e-5, rtol=1e-5)


def test_rotary_embedding_module():
    rope = RotaryEmbedding(8, max_len=16)
    assert sum(p.numel() for p in rope.parameters()) == 0, "у RoPE нет обучаемых параметров"
    assert set(rope.state_dict().keys()) == {"cos_cached", "sin_cached"}

    x = torch.randn(2, 3, 5, 8)
    cos, sin = rope_cos_sin(5, 8)
    torch.testing.assert_close(rope(x), apply_rope(x, cos, sin))

    x1 = torch.randn(2, 3, 1, 8)
    cos1, sin1 = rope_cos_sin(1, 8, offset=4)
    torch.testing.assert_close(rope(x1, offset=4), apply_rope(x1, cos1, sin1))


@pytest.mark.parametrize("d_head", [2, 8, 64])
def test_rope_relative_property_across_dims(d_head):
    q, k = torch.randn(d_head), torch.randn(d_head)
    torch.testing.assert_close(
        rope_dot_product(q, k, 7, 4),
        rope_dot_product(q, k, 3, 0),
        atol=1e-4,
        rtol=1e-4,
    )
