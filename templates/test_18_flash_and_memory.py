"""Топик 18 (сложный) - память и численность: online softmax, flash-attention, чекпоинтинг.

Все задачи этого файла об одном: как посчитать то же самое, но не аллоцируя
огромный промежуточный тензор.

  - матрица внимания T x T не влезает в память -> online softmax и блочное внимание;
  - логиты (N, 50257) не влезают -> кросс-энтропия по кускам;
  - активации всех слоёв не влезают -> gradient checkpointing;
  - fp16 переполняется -> loss scaling.

Это уже уровень ассайнмента 2 CS336. Если проходит - с трансформерами ты на "ты".
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# ==========================================================================
# 18.1 Online softmax
# ==========================================================================


def online_softmax_update(m_prev: Tensor, l_prev: Tensor, scores_block: Tensor) -> tuple[Tensor, Tensor]:
    """Инкрементальный пересчёт статистик softmax при добавлении нового блока логитов.

    Вход: m_prev (..., 1) - текущий максимум, l_prev (..., 1) - текущая сумма exp
    относительно ЭТОГО максимума, scores_block (..., Sb) - новые логиты.

    Вернуть (m_new, l_new), где
        m_new = max(m_prev, max(scores_block))
        l_new = l_prev * exp(m_prev - m_new) + sum(exp(scores_block - m_new))

    Это ядро flash-attention: сумма экспонент обновляется без хранения всех логитов.
    """
    raise NotImplementedError


def chunked_softmax(x: Tensor, chunk: int) -> Tensor:
    """Softmax по последней оси, посчитанный за два прохода блоками размера chunk:
    сначала online-статистики (m, l), потом сами вероятности.

    Результат обязан совпасть с обычным softmax до последнего бита точности float32.
    """
    raise NotImplementedError


# ==========================================================================
# 18.2 Flash-attention (упрощённый, только forward)
# ==========================================================================


def flash_attention_forward(q: Tensor, k: Tensor, v: Tensor, block_size: int = 32, causal: bool = False) -> Tensor:
    """Блочное внимание без материализации матрицы (T, S).

    q: (B, H, T, Dh), k, v: (B, H, S, Dh) -> (B, H, T, Dh).

    Схема (flash-attention v2):
      для каждого блока запросов держим бегущие (m, l, накопленный выход);
      идём по блокам ключей, для каждого считаем локальные логиты,
      обновляем m и l по online-правилу, корректируем накопленный выход
      множителем exp(m_old - m_new), в конце делим на l.

    Полностью замаскированные блоки (при causal=True) можно пропускать -
      это заодно спасает от nan вида exp(-inf - (-inf)).

    Размеры НЕ обязаны делиться на block_size.
    """
    raise NotImplementedError


# ==========================================================================
# 18.3 Кросс-энтропия без материализации логитов
# ==========================================================================


def chunked_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor, chunk: int) -> Tensor:
    """hidden: (N, D), weight: (V, D) - вес lm_head, targets: (N,).

    Посчитать средний CE, НЕ создавая тензор логитов (N, V) целиком:
    идти кусками по токенам, для каждого куска считать свои логиты и его вклад в лосс.

    Градиенты должны течь и в hidden, и в weight.
    Для 8k токенов и словаря 128k такой тензор логитов весит 4 ГБ - отсюда и задача.
    """
    raise NotImplementedError


def chunked_logsumexp(x: Tensor, chunk: int) -> Tensor:
    """logsumexp по последней оси блоками, через online-обновление. (N, V) -> (N,)."""
    raise NotImplementedError


# ==========================================================================
# 18.4 Gradient checkpointing
# ==========================================================================


def checkpointed_sequential(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool) -> Tensor:
    """Прогнать x через список блоков. При use_checkpoint=True каждый блок обёрнут
    в torch.utils.checkpoint.checkpoint (use_reentrant=False), то есть его активации
    не сохраняются, а пересчитываются на backward.

    Компромисс: примерно +30% времени, но память под активации падает с O(L) до O(sqrt(L))
    или даже O(1) при наивной схеме.
    """
    raise NotImplementedError


def count_saved_tensors(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool) -> int:
    """Сколько тензоров граф автограда сохранил для backward. Это прямое измерение
    того, что даёт чекпоинтинг.

    Подсказка: контекст torch.autograd.graph.saved_tensors_hooks(pack, unpack) вызывает
    pack на каждом сохраняемом тензоре. Считаем вызовы pack во время forward.
    """
    raise NotImplementedError


# ==========================================================================
# 18.5 Смешанная точность
# ==========================================================================


def scale_and_check(grads: list[Tensor], scale: float) -> tuple[list[Tensor], bool]:
    """Разделить градиенты на scale (unscale) и сказать, встретились ли inf/nan.

    Это сердце GradScaler: если found_inf, шаг оптимизатора пропускают, а scale уменьшают.
    Возвращать (список раз-масштабированных градиентов, found_inf).
    """
    raise NotImplementedError


def next_scale(scale: float, found_inf: bool, growth_interval_reached: bool) -> float:
    """Правило GradScaler: при found_inf уменьшить масштаб вдвое,
    иначе при достижении интервала роста - удвоить, иначе оставить как есть.
    """
    raise NotImplementedError


def safe_softmax_fp16(logits: Tensor) -> Tensor:
    """logits приходят в float16 и могут быть большими. Посчитать softmax так,
    чтобы не было переполнения, и вернуть результат в float16.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_online_softmax_update():
    x = torch.randn(4, 20)
    m = torch.full((4, 1), float("-inf"))
    l = torch.zeros(4, 1)
    for start in range(0, 20, 5):
        m, l = online_softmax_update(m, l, x[:, start : start + 5])
    torch.testing.assert_close(m.squeeze(-1), x.max(dim=-1).values)
    torch.testing.assert_close(l.squeeze(-1).log() + m.squeeze(-1), torch.logsumexp(x, dim=-1))


def test_chunked_softmax():
    x = torch.randn(8, 37) * 5
    torch.testing.assert_close(chunked_softmax(x, chunk=8), x.softmax(dim=-1))
    big = torch.randn(4, 16) + 5000.0
    assert torch.isfinite(chunked_softmax(big, chunk=4)).all()


def test_chunked_logsumexp():
    x = torch.randn(6, 41) * 3
    torch.testing.assert_close(chunked_logsumexp(x, chunk=7), torch.logsumexp(x, dim=-1))


def test_flash_attention_matches_reference():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 40, 16), torch.randn(2, 3, 40, 16), torch.randn(2, 3, 40, 16)
    torch.testing.assert_close(
        flash_attention_forward(q, k, v, block_size=8),
        F.scaled_dot_product_attention(q, k, v),
        atol=1e-5,
        rtol=1e-5,
    )


def test_flash_attention_causal():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 8)
    out = flash_attention_forward(q, k, v, block_size=8, causal=True)
    assert torch.isfinite(out).all(), "где-то получился exp(-inf - (-inf))"
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
def test_flash_attention_any_block_size(block_size):
    q, k, v = torch.randn(1, 1, 20, 8), torch.randn(1, 1, 20, 8), torch.randn(1, 1, 20, 8)
    torch.testing.assert_close(
        flash_attention_forward(q, k, v, block_size, causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        atol=1e-5,
        rtol=1e-5,
    )


def test_flash_attention_is_stable_on_large_scores():
    q = torch.full((1, 1, 4, 8), 30.0)
    k = torch.full((1, 1, 4, 8), 30.0)
    v = torch.randn(1, 1, 4, 8)
    out = flash_attention_forward(q, k, v, block_size=2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, v.mean(dim=-2, keepdim=True).expand_as(out), atol=1e-4, rtol=1e-4)


def test_chunked_cross_entropy_value_and_grads():
    torch.manual_seed(0)
    N, D, V = 20, 16, 50
    hidden = torch.randn(N, D, requires_grad=True)
    weight = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(0, V, (N,))

    reference = F.cross_entropy(hidden @ weight.T, targets)
    reference.backward()
    ref_h, ref_w = hidden.grad.clone(), weight.grad.clone()

    hidden.grad = None
    weight.grad = None
    loss = chunked_cross_entropy(hidden, weight, targets, chunk=7)
    torch.testing.assert_close(loss, reference.detach(), atol=1e-5, rtol=1e-5)

    loss.backward()
    torch.testing.assert_close(hidden.grad, ref_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight.grad, ref_w, atol=1e-5, rtol=1e-5)


def test_checkpointing_gives_identical_results():
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(4)])
    x = torch.randn(3, 8, requires_grad=True)

    plain = checkpointed_sequential(blocks, x, use_checkpoint=False)
    plain.sum().backward()
    plain_grads = [p.grad.clone() for p in blocks.parameters()]
    plain_x_grad = x.grad.clone()

    for p in blocks.parameters():
        p.grad = None
    x.grad = None

    ckpt = checkpointed_sequential(blocks, x, use_checkpoint=True)
    torch.testing.assert_close(ckpt, plain, atol=1e-6, rtol=1e-6)
    ckpt.sum().backward()
    torch.testing.assert_close(x.grad, plain_x_grad, atol=1e-6, rtol=1e-6)
    for p, g in zip(blocks.parameters(), plain_grads, strict=True):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-6)


def test_checkpointing_shrinks_the_graph():
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(6)])
    x = torch.randn(3, 8, requires_grad=True)
    plain = count_saved_tensors(blocks, x, use_checkpoint=False)
    ckpt = count_saved_tensors(blocks, x, use_checkpoint=True)
    assert ckpt < plain, f"чекпоинтинг не сократил сохранённое: {ckpt} против {plain}"
    assert ckpt <= len(blocks), "с чекпоинтингом хранится примерно по одному входу на блок"


def test_scale_and_check():
    grads = [torch.tensor([1024.0, -2048.0]), torch.tensor([512.0])]
    unscaled, found = scale_and_check(grads, 1024.0)
    assert found is False
    torch.testing.assert_close(unscaled[0], torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(unscaled[1], torch.tensor([0.5]))

    bad = [torch.tensor([float("inf"), 1.0])]
    _, found = scale_and_check(bad, 1024.0)
    assert found is True

    nan = [torch.tensor([float("nan")])]
    _, found = scale_and_check(nan, 2.0)
    assert found is True


def test_next_scale():
    assert next_scale(1024.0, found_inf=True, growth_interval_reached=False) == 512.0
    assert next_scale(1024.0, found_inf=True, growth_interval_reached=True) == 512.0
    assert next_scale(1024.0, found_inf=False, growth_interval_reached=True) == 2048.0
    assert next_scale(1024.0, found_inf=False, growth_interval_reached=False) == 1024.0


def test_safe_softmax_fp16():
    logits = torch.tensor([[60000.0, 60000.0, 0.0]], dtype=torch.float16)
    out = safe_softmax_fp16(logits)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), "fp16 переполнился на exp"
    torch.testing.assert_close(out.float().sum(), torch.tensor(1.0), atol=1e-3, rtol=1e-3)
    assert abs(out[0, 0].item() - 0.5) < 1e-2
