"""Топик 15 - собираем блок трансформера и целиком языковую модель.

Да, часть кирпичей ты уже писал в топиках 11-14. Здесь они пишутся ещё раз, и это
намеренно: цель упражнений - чтобы блок трансформера набирался руками без раздумий.

Архитектура ровно та, что в задании CS336 (pre-norm, RMSNorm, RoPE опущен для краткости):

    x = x + attn(ln1(x))
    x = x + ffn(ln2(x))
    ...
    logits = lm_head(ln_final(x))

Имена подмодулей повторяют эталонный state_dict курса:
    token_embeddings, layers.{i}.attn, layers.{i}.ffn, layers.{i}.ln1, layers.{i}.ln2,
    ln_final, lm_head
"""

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 15.1 Кирпичи (повторение - мать мышечной памяти)
# ==========================================================================


class RMSNorm(nn.Module):
    """Параметр `weight` формы (d_model,), инициализирован единицами."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class SwiGLU(nn.Module):
    """Подмодули w1, w2, w3 - nn.Linear без bias. out = w2(silu(w1(x)) * w3(x))."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """Подмодули q_proj, k_proj, v_proj, output_proj - nn.Linear(d_model, d_model, bias=False).
    Причинное многоголовое внимание.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 15.2 Блок
# ==========================================================================


class TransformerBlock(nn.Module):
    """Pre-norm блок. Подмодули обязаны называться ln1, attn, ln2, ffn.

    x = x + attn(ln1(x))
    x = x + ffn(ln2(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 15.3 Модель целиком
# ==========================================================================


class TransformerLM(nn.Module):
    """Языковая модель.

    Подмодули: token_embeddings (nn.Embedding), layers (nn.ModuleList из TransformerBlock),
    ln_final (RMSNorm), lm_head (nn.Linear(d_model, vocab_size, bias=False)).

    forward(ids: (B, T) int64) -> logits (B, T, vocab_size).

    Инициализация: эмбеддинги и все линейные веса нормальным с std=0.02.
    Это важно для теста "лосс необученной модели равен log(V)".
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, ids: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 15.4 Арифметика модели
# ==========================================================================


def count_lm_params(vocab_size: int, d_model: int, d_ff: int, num_layers: int, tied_embeddings: bool) -> int:
    """Число параметров модели по формуле, без создания модели.

    Считаем: эмбеддинги + num_layers * (4 * d^2 внимания + 3 * d * d_ff у SwiGLU) + lm_head.
    Нормализации и bias-ы игнорируем. При tied_embeddings=True lm_head не считается отдельно.
    """
    raise NotImplementedError


def activation_memory_bytes(batch: int, seq_len: int, d_model: int, num_layers: int, dtype: torch.dtype) -> int:
    """Грубая оценка памяти под активации остаточного потока: по одному тензору (B, T, D)
    на каждый слой. Реальность хуже в разы, но порядок величины эта формула даёт.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_block_structure():
    block = TransformerBlock(32, 4, 64)
    children = dict(block.named_children())
    assert set(children.keys()) == {"ln1", "attn", "ln2", "ffn"}
    x = torch.randn(2, 5, 32)
    assert block(x).shape == x.shape


def test_block_is_pre_norm_with_residuals():
    torch.manual_seed(0)
    block = TransformerBlock(16, 2, 32)
    x = torch.randn(2, 4, 16)
    h = x + block.attn(block.ln1(x))
    expected = h + block.ffn(block.ln2(h))
    torch.testing.assert_close(block(x), expected, atol=1e-5, rtol=1e-5)


def test_block_identity_when_sublayers_are_zero():
    """Занулим выходные проекции - остаточный поток обязан пройти вход без изменений."""
    block = TransformerBlock(16, 2, 32)
    with torch.no_grad():
        block.attn.output_proj.weight.zero_()
        block.ffn.w2.weight.zero_()
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(block(x), x, atol=1e-6, rtol=0)


def test_lm_shapes_and_names():
    model = TransformerLM(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    ids = torch.randint(0, 100, (2, 7))
    logits = model(ids)
    assert logits.shape == (2, 7, 100)

    names = {n for n, _ in model.named_parameters()}
    assert "token_embeddings.weight" in names
    assert "ln_final.weight" in names
    assert "lm_head.weight" in names
    assert "layers.0.attn.q_proj.weight" in names
    assert "layers.1.ffn.w1.weight" in names
    assert "layers.0.ln1.weight" in names


def test_lm_is_causal():
    model = TransformerLM(vocab_size=50, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    ids = torch.randint(0, 50, (1, 6))
    emb = model.token_embeddings(ids).detach().requires_grad_(True)

    h = emb
    for layer in model.layers:
        h = layer(h)
    logits = model.lm_head(model.ln_final(h))
    logits[0, 2].sum().backward()

    assert emb.grad[0, 3:].abs().max().item() == 0.0, "модель подглядывает в будущее"
    assert emb.grad[0, :3].abs().max().item() > 0.0


def test_untrained_loss_is_log_vocab():
    """Первая проверка любого запуска обучения: лосс на старте должен быть ~log(V)."""
    torch.manual_seed(0)
    V = 512
    model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    ids = torch.randint(0, V, (4, 16))
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 0.3, f"лосс {loss.item():.3f}, ожидалось ~{math.log(V):.3f}"


def test_lm_can_overfit_one_batch():
    """Модель обязана уметь запомнить один батч - иначе где-то оборван граф или маска."""
    torch.manual_seed(0)
    V = 32
    model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    ids = torch.randint(0, V, (2, 8))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(60):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < 0.1 * first, f"лосс не упал: {first:.3f} -> {loss.item():.3f}"


def test_count_lm_params():
    # упрощённый GPT-2 small: 12 слоёв, d=768, но с SwiGLU FFN
    n = count_lm_params(vocab_size=50257, d_model=768, d_ff=3072, num_layers=12, tied_embeddings=True)
    per_layer = 4 * 768 * 768 + 3 * 768 * 3072
    assert n == 50257 * 768 + 12 * per_layer

    untied = count_lm_params(50257, 768, 3072, 12, tied_embeddings=False)
    assert untied - n == 50257 * 768


def test_count_matches_real_model():
    model = TransformerLM(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    formula = count_lm_params(100, 32, 64, 2, tied_embeddings=False)
    real = sum(p.numel() for p in model.parameters())
    # разница - только веса нормализаций: 2 на слой плюс финальная
    assert real - formula == (2 * 2 + 1) * 32


def test_activation_memory_bytes():
    b = activation_memory_bytes(8, 2048, 4096, 32, torch.bfloat16)
    assert b == 8 * 2048 * 4096 * 32 * 2


@pytest.mark.parametrize("num_layers", [1, 3])
def test_lm_depth(num_layers):
    model = TransformerLM(vocab_size=40, d_model=16, num_heads=2, d_ff=32, num_layers=num_layers)
    assert len(model.layers) == num_layers
    assert model(torch.randint(0, 40, (1, 5))).shape == (1, 5, 40)
