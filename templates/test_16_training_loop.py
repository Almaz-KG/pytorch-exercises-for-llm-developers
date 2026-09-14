"""Топик 16 - цикл обучения: AdamW, расписание lr, клиппинг, накопление градиента.

Здесь заканчиваются "тензорные упражнения" и начинается инженерия обучения.
Всё, что ниже, ты будешь писать руками в ассайнменте 1 CS336.

AdamW пошагово (это стоит уметь воспроизвести по памяти):
    m = b1*m + (1-b1)*g
    v = b2*v + (1-b2)*g^2
    m_hat = m / (1 - b1^t)
    v_hat = v / (1 - b2^t)
    p = p - lr * m_hat / (sqrt(v_hat) + eps)
    p = p - lr * wd * p      <- decoupled weight decay, не через градиент
"""

import itertools
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 16.1 Шаги оптимизации руками
# ==========================================================================


def sgd_step(params: list[Tensor], lr: float) -> None:
    """Обновить параметры на месте: p -= lr * p.grad.

    Обязательно под torch.no_grad(), иначе обновление попадёт в граф.
    """
    raise NotImplementedError


def adamw_step(
    p: Tensor,
    grad: Tensor,
    m: Tensor,
    v: Tensor,
    t: int,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> tuple[Tensor, Tensor, Tensor]:
    """Один шаг AdamW для одного тензора. Вернуть (новый p, новый m, новый v).

    t - номер шага, начиная с 1 (нужен для bias correction).
    Ничего не менять на месте: возвращай новые тензоры, так проще проверять.
    """
    raise NotImplementedError


class AdamW(torch.optim.Optimizer):
    """Полноценный оптимизатор в интерфейсе PyTorch.

    Требования:
      - __init__(params, lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01);
      - состояние (m, v, число шагов) хранится в self.state[p];
      - step() обновляет параметры и возвращает loss, если передан closure;
      - результат обязан численно совпадать с torch.optim.AdamW.
    """

    def __init__(self, params, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        raise NotImplementedError

    def step(self, closure=None):
        raise NotImplementedError


# ==========================================================================
# 16.2 Расписание learning rate
# ==========================================================================


def cosine_lr(step: int, lr_max: float, lr_min: float, warmup_steps: int, cosine_steps: int) -> float:
    """Расписание из задания CS336:

      step < warmup_steps:                    линейный разогрев от 0 до lr_max
      warmup_steps <= step <= cosine_steps:   косинусное затухание от lr_max до lr_min
      step > cosine_steps:                    константа lr_min

    Косинус: lr_min + 0.5 * (1 + cos(pi * (step - warmup) / (cosine_steps - warmup))) * (lr_max - lr_min)
    """
    raise NotImplementedError


# ==========================================================================
# 16.3 Клиппинг градиента
# ==========================================================================


def clip_grad_norm(params: list[Tensor], max_norm: float, eps: float = 1e-6) -> float:
    """Обрезать градиенты по глобальной L2-норме и вернуть норму ДО обрезки (python float).

    Если норма больше max_norm, все градиенты домножаются на max_norm / (norm + eps).
    Реализовать самому, без torch.nn.utils.clip_grad_norm_.
    """
    raise NotImplementedError


# ==========================================================================
# 16.4 Накопление градиента
# ==========================================================================


def accumulate_gradients(model: nn.Module, batches: list[tuple[Tensor, Tensor]]) -> float:
    """Прогнать несколько микробатчей, накапливая градиент так, чтобы итог совпал
    с одним большим батчем той же суммарной длины. Вернуть суммарный (усреднённый) лосс.

    Ключевая деталь: лосс каждого микробатча надо поделить на число микробатчей,
    иначе градиент окажется в N раз больше. На этом спотыкаются все.

    Лосс - MSE между model(x) и y. Градиенты обнулить в начале.
    """
    raise NotImplementedError


def train_steps(model: nn.Module, x: Tensor, y: Tensor, steps: int, lr: float) -> list[float]:
    """Простейший цикл обучения на своём AdamW: вернуть список лоссов по шагам.

    Порядок операций внутри шага - тоже часть мышечной памяти:
    forward -> loss -> zero_grad -> backward -> clip -> step.
    """
    raise NotImplementedError


# ==========================================================================
# 16.5 Арифметика обучения
# ==========================================================================


def training_flops(num_params: int, num_tokens: int) -> int:
    """Правило 6ND: forward примерно 2ND, backward примерно 4ND."""
    raise NotImplementedError


def chinchilla_tokens(num_params: int) -> int:
    """Compute-оптимальное число токенов по Chinchilla: примерно 20 токенов на параметр."""
    raise NotImplementedError


def tokens_per_step(batch_size: int, seq_len: int, grad_accum: int) -> int:
    """Сколько токенов реально видит модель за один шаг оптимизатора."""
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_sgd_step():
    p = torch.tensor([1.0, 2.0], requires_grad=True)
    p.grad = torch.tensor([0.5, -0.5])
    sgd_step([p], lr=0.1)
    torch.testing.assert_close(p.detach(), torch.tensor([0.95, 2.05]))
    assert p.requires_grad and p.is_leaf, "параметр должен остаться листом графа"


def test_adamw_step_matches_torch():
    torch.manual_seed(0)
    p_ref = torch.randn(5, requires_grad=True)
    p_mine = p_ref.detach().clone()
    grads = [torch.randn(5) for _ in range(4)]

    opt = torch.optim.AdamW([p_ref], lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    m = torch.zeros(5)
    v = torch.zeros(5)
    for t, g in enumerate(grads, start=1):
        p_ref.grad = g.clone()
        opt.step()
        p_mine, m, v = adamw_step(p_mine, g, m, v, t, lr=0.1)
        torch.testing.assert_close(p_mine, p_ref.detach(), atol=1e-6, rtol=1e-5)


def test_adamw_weight_decay_is_decoupled():
    """Даже при нулевом градиенте вес обязан уменьшиться - это и значит decoupled."""
    p = torch.ones(3)
    out, _, _ = adamw_step(p, torch.zeros(3), torch.zeros(3), torch.zeros(3), t=1, lr=0.1, weight_decay=0.1)
    torch.testing.assert_close(out, torch.full((3,), 1.0 - 0.1 * 0.1), atol=1e-6, rtol=0)


def test_adamw_optimizer_matches_torch():
    torch.manual_seed(0)
    mine = nn.Linear(4, 3)
    ref = nn.Linear(4, 3)
    ref.load_state_dict(mine.state_dict())

    opt_mine = AdamW(mine.parameters(), lr=0.05, weight_decay=0.05)
    opt_ref = torch.optim.AdamW(ref.parameters(), lr=0.05, weight_decay=0.05)

    x = torch.randn(16, 4)
    y = torch.randn(16, 3)
    for _ in range(8):
        for model, opt in ((mine, opt_mine), (ref, opt_ref)):
            loss = F.mse_loss(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        torch.testing.assert_close(mine.weight, ref.weight, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(mine.bias, ref.bias, atol=1e-6, rtol=1e-5)


def test_cosine_lr_schedule():
    lr_max, lr_min, warmup, cosine = 1.0, 0.1, 10, 110
    assert cosine_lr(0, lr_max, lr_min, warmup, cosine) == 0.0
    assert abs(cosine_lr(5, lr_max, lr_min, warmup, cosine) - 0.5) < 1e-9
    assert abs(cosine_lr(10, lr_max, lr_min, warmup, cosine) - lr_max) < 1e-9
    # середина косинуса
    assert abs(cosine_lr(60, lr_max, lr_min, warmup, cosine) - (lr_min + 0.5 * (lr_max - lr_min))) < 1e-9
    assert abs(cosine_lr(110, lr_max, lr_min, warmup, cosine) - lr_min) < 1e-9
    assert cosine_lr(500, lr_max, lr_min, warmup, cosine) == lr_min
    # монотонное убывание на косинусном участке
    values = [cosine_lr(s, lr_max, lr_min, warmup, cosine) for s in range(10, 111)]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(values))


def test_clip_grad_norm_matches_torch():
    torch.manual_seed(0)
    mine = [torch.randn(4, 3), torch.randn(3)]
    for p in mine:
        p.grad = torch.randn_like(p) * 10
    ref = [p.detach().clone() for p in mine]
    for p, q in zip(ref, mine, strict=True):
        p.grad = q.grad.clone()

    returned = clip_grad_norm(mine, 1.0)
    expected = torch.nn.utils.clip_grad_norm_(ref, 1.0).item()
    assert abs(returned - expected) < 1e-4
    for p, q in zip(mine, ref, strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)

    total = torch.sqrt(sum(p.grad.pow(2).sum() for p in mine)).item()
    assert total <= 1.0 + 1e-4


def test_clip_grad_norm_leaves_small_gradients_alone():
    p = torch.zeros(3)
    p.grad = torch.tensor([0.1, 0.0, 0.0])
    before = p.grad.clone()
    norm = clip_grad_norm([p], 1.0)
    assert abs(norm - 0.1) < 1e-6
    torch.testing.assert_close(p.grad, before)


def test_gradient_accumulation_equals_big_batch():
    torch.manual_seed(0)
    model = nn.Linear(4, 2)
    x = torch.randn(16, 4)
    y = torch.randn(16, 2)

    loss = F.mse_loss(model(x), y)
    model.zero_grad()
    loss.backward()
    full_grads = [p.grad.clone() for p in model.parameters()]

    micro = [(x[i : i + 4], y[i : i + 4]) for i in range(0, 16, 4)]
    accum_loss = accumulate_gradients(model, micro)
    assert abs(accum_loss - loss.item()) < 1e-5
    for p, g in zip(model.parameters(), full_grads, strict=True):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-5)


def test_train_steps_reduces_loss():
    torch.manual_seed(0)
    model = nn.Linear(4, 1)
    x = torch.randn(32, 4)
    y = x @ torch.randn(4, 1) + 0.1
    losses = train_steps(model, x, y, steps=50, lr=0.05)
    assert len(losses) == 50
    assert losses[-1] < 0.2 * losses[0], f"лосс почти не упал: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_training_arithmetic():
    # 7B параметров, 1.4T токенов - примерно LLaMA-7B
    assert training_flops(7e9, 1.4e12) == 6 * 7e9 * 1.4e12
    assert chinchilla_tokens(7_000_000_000) == 140_000_000_000
    assert tokens_per_step(8, 1024, 4) == 32768


@pytest.mark.parametrize("warmup,cosine", [(0, 100), (10, 100), (50, 50)])
def test_cosine_lr_bounds(warmup, cosine):
    for step in range(cosine + 20):
        lr = cosine_lr(step, 1.0, 0.1, warmup, cosine)
        assert 0.0 <= lr <= 1.0 + 1e-12
        assert not math.isnan(lr)
