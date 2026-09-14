"""Топик 08 - autograd.

Модель мышления: каждая операция над тензором с requires_grad=True дописывает узел
в граф. backward() идёт по графу назад и КОПИТ (не перезаписывает!) градиенты в .grad
у листовых тензоров.

Здесь же - ручной вывод градиентов, который дальше понадобится для flash-attention
и кастомных ядер: если ты умеешь написать backward сам, ты понимаешь слой по-настоящему.
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 8.1 Базовый цикл backward
# ==========================================================================


def grad_of_sum_of_squares(x: Tensor) -> Tensor:
    """x - тензор без requires_grad. Посчитать d(sum(x^2))/dx и вернуть градиент.

    Ожидаемо это 2*x, но получить его надо через autograd, а не формулой.
    """
    raise NotImplementedError


def grad_functional(x: Tensor) -> Tensor:
    """То же самое, но через torch.autograd.grad (без .backward() и без .grad).

    Функциональный API удобен, когда не хочется трогать .grad у параметров.
    """
    raise NotImplementedError


def grads_accumulate(x: Tensor) -> tuple[Tensor, Tensor]:
    """Взять y = sum(x^2), вызвать backward ДВАЖДЫ (сохрани граф) и вернуть
    (градиент после первого вызова, градиент после второго).

    Смысл: показать себе руками, что второй backward прибавляется к первому.
    Именно поэтому в цикле обучения есть optimizer.zero_grad().
    """
    raise NotImplementedError


def stop_gradient(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """a, b - листья с requires_grad=True. Посчитать y = sum(a * b), но так,
    чтобы градиент НЕ шёл в b. Вернуть (a.grad, b.grad) - второй должен быть None.
    """
    raise NotImplementedError


def forward_without_graph(x: Tensor, w: Tensor) -> Tensor:
    """Посчитать x @ w так, чтобы результат не требовал градиента и граф не строился.

    Это режим inference: экономит память в разы.
    """
    raise NotImplementedError


# ==========================================================================
# 8.2 Производные посложнее
# ==========================================================================


def second_derivative(x: Tensor) -> Tensor:
    """Вторая производная f(x) = sum(x^3) по x, то есть 6*x, через autograd.

    Подсказка: create_graph=True на первом дифференцировании.
    """
    raise NotImplementedError


def vjp(f: Callable[[Tensor], Tensor], x: Tensor, v: Tensor) -> Tensor:
    """Vector-Jacobian product: (v^T J) для f в точке x. Это ровно то, что делает backward.

    x без requires_grad, вернуть тензор формы x.
    """
    raise NotImplementedError


def numeric_grad(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-4) -> Tensor:
    """Численный градиент скалярной функции центральными разностями.

    (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps) для каждой координаты.
    Здесь цикл по элементам разрешён - это отладочный инструмент, им проверяют
    правильность аналитических градиентов.
    """
    raise NotImplementedError


def softmax_vjp(probs: Tensor, grad_out: Tensor) -> Tensor:
    """Градиент softmax, выведенный руками (без autograd).

    probs - УЖЕ посчитанный softmax по последней оси, grad_out - градиент сверху.
    Известный результат: grad_in = probs * (grad_out - sum(grad_out * probs, dim=-1, keepdim=True)).

    Выведи его сам из J_ij = p_i (delta_ij - p_j) - эта формула ещё встретится
    в backward attention.
    """
    raise NotImplementedError


# ==========================================================================
# 8.3 Кастомные autograd-функции
# ==========================================================================


class SquareFunction(torch.autograd.Function):
    """y = x^2 со своим backward. Реализуй forward и backward.

    Правила: forward принимает ctx первым аргументом, сохраняет нужное через
    ctx.save_for_backward, backward получает grad_output и возвращает градиент
    по каждому входу forward.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        raise NotImplementedError


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator: округляем в forward, а в backward делаем вид,
    что операция была тождественной.

    На этом трюке держится квантизационно-осознанное обучение (QAT):
    round недифференцируем, но обучать модель как-то надо.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 8.4 Градиенты в реальном коде модели
# ==========================================================================


def embedding_grad(table_size: tuple[int, int], ids: Tensor) -> Tensor:
    """Создать таблицу эмбеддингов нужного размера, требующую градиент,
    сделать lookup по ids, взять сумму всех значений, вызвать backward
    и вернуть градиент таблицы.

    Итог поучителен: градиент строки равен числу её употреблений.
    Именно поэтому редкие токены учатся медленно.
    """
    raise NotImplementedError


def freeze_all_but_last(model: nn.Sequential) -> int:
    """Заморозить все параметры, кроме параметров последнего модуля,
    и вернуть число обучаемых параметров (элементов, не тензоров).

    Это буквально linear probing / частичный fine-tuning.
    """
    raise NotImplementedError


def grad_norm_after_backward(model: nn.Module, x: Tensor, target: Tensor) -> float:
    """Прогнать x через model, взять MSE-лосс к target, backward, вернуть
    глобальную L2-норму градиентов всех параметров как python float.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_grad_of_sum_of_squares():
    x = torch.randn(4, 3)
    g = grad_of_sum_of_squares(x)
    torch.testing.assert_close(g, 2 * x)


def test_grad_functional():
    x = torch.randn(5)
    torch.testing.assert_close(grad_functional(x), 2 * x)


def test_grads_accumulate():
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    first, second = grads_accumulate(x)
    torch.testing.assert_close(first, torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(second, torch.tensor([4.0, 8.0]))


def test_stop_gradient():
    a = torch.randn(3, requires_grad=True)
    b = torch.randn(3, requires_grad=True)
    ga, gb = stop_gradient(a, b)
    torch.testing.assert_close(ga, b.detach())
    assert gb is None, "градиент в b просочился"


def test_forward_without_graph():
    x, w = torch.randn(2, 3), torch.randn(3, 4, requires_grad=True)
    out = forward_without_graph(x, w)
    assert out.requires_grad is False
    assert out.grad_fn is None
    torch.testing.assert_close(out, x @ w.detach())


def test_second_derivative():
    x = torch.tensor([1.0, 2.0, 3.0])
    torch.testing.assert_close(second_derivative(x), 6 * x)


def test_vjp():
    def f(t: Tensor) -> Tensor:
        return t.sin() * 2.0

    x = torch.randn(4)
    v = torch.randn(4)
    torch.testing.assert_close(vjp(f, x, v), v * 2.0 * x.cos(), atol=1e-5, rtol=1e-5)


def test_numeric_grad():
    def f(t: Tensor) -> Tensor:
        return (t**3).sum()

    x = torch.randn(5, dtype=torch.float64)
    torch.testing.assert_close(numeric_grad(f, x), 3 * x**2, atol=1e-5, rtol=1e-5)


def test_softmax_vjp():
    logits = torch.randn(2, 5, requires_grad=True)
    probs = logits.softmax(dim=-1)
    g = torch.randn(2, 5)
    probs.backward(g)
    torch.testing.assert_close(softmax_vjp(probs.detach(), g), logits.grad, atol=1e-5, rtol=1e-5)


def test_square_function():
    x = torch.randn(3, 4, requires_grad=True)
    y = SquareFunction.apply(x)
    torch.testing.assert_close(y, x.detach() ** 2)
    g = torch.randn_like(y)
    y.backward(g)
    torch.testing.assert_close(x.grad, 2 * x.detach() * g)

    x64 = torch.randn(4, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(SquareFunction.apply, (x64,)), "gradcheck не сошёлся"


def test_ste_quantize():
    x = torch.tensor([0.2, 1.7, -0.4], requires_grad=True)
    y = STEQuantize.apply(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 2.0, -0.0]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(3), msg="STE должен пропускать градиент как есть")


def test_embedding_grad():
    ids = torch.tensor([[0, 2, 2], [2, 1, 0]])
    g = embedding_grad((4, 3), ids)
    assert g.shape == (4, 3)
    torch.testing.assert_close(g[:, 0], torch.tensor([2.0, 1.0, 3.0, 0.0]))


def test_freeze_all_but_last():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    trainable = freeze_all_but_last(model)
    assert trainable == 8 * 2 + 2
    assert model[0].weight.requires_grad is False
    assert model[2].weight.requires_grad is True


def test_grad_norm_after_backward():
    torch.manual_seed(0)
    model = nn.Linear(3, 1, bias=False)
    x = torch.randn(10, 3)
    target = torch.randn(10, 1)
    norm = grad_norm_after_backward(model, x, target)
    assert isinstance(norm, float)
    assert norm > 0

    expected = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()
    assert abs(norm - expected) < 1e-6


@pytest.mark.parametrize("shape", [(3,), (2, 4)])
def test_softmax_vjp_property(shape):
    logits = torch.randn(*shape, requires_grad=True)
    probs = logits.softmax(dim=-1)
    g = torch.randn(*shape)
    probs.backward(g)
    torch.testing.assert_close(softmax_vjp(probs.detach(), g), logits.grad, atol=1e-5, rtol=1e-5)
