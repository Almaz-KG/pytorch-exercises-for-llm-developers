"""Topic 15 - autograd.

The mental model: every operation on a tensor with requires_grad=True appends a node
to the graph. backward() walks the graph backwards and ACCUMULATES (does not overwrite!)
the gradients in .grad on the leaf tensors.

Here too is the manual derivation of gradients, which will be needed later for flash-attention
and custom kernels: if you can write a backward yourself, you understand the layer for real.
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 15.1 The basic backward cycle
# ==========================================================================


def grad_of_sum_of_squares(x: Tensor) -> Tensor:
    """x is a tensor without requires_grad. Compute d(sum(x^2))/dx and return the gradient.

    It is 2*x as expected, but it has to be obtained through autograd, not by the formula.
    """
    raise NotImplementedError


def grad_functional(x: Tensor) -> Tensor:
    """The same thing, but through torch.autograd.grad (without .backward() and without .grad).

    The functional API is convenient when you do not want to touch .grad on the parameters.
    """
    raise NotImplementedError


def grads_accumulate(x: Tensor) -> tuple[Tensor, Tensor]:
    """Take y = sum(x^2), call backward TWICE (keep the graph) and return
    (the gradient after the first call, the gradient after the second).

    The point: to show yourself by hand that the second backward is added to the first.
    That is exactly why the training loop has optimizer.zero_grad().
    """
    raise NotImplementedError


def stop_gradient(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor | None]:
    """a, b are leaves with requires_grad=True. Compute y = sum(a * b), but in such a way
    that the gradient does NOT flow into b. Return (a.grad, b.grad) - the second must be None.
    """
    raise NotImplementedError


def forward_without_graph(x: Tensor, w: Tensor) -> Tensor:
    """Compute x @ w so that the result does not require a gradient and no graph is built.

    This is inference mode: it saves memory several times over.
    """
    raise NotImplementedError


# ==========================================================================
# 15.2 Harder derivatives
# ==========================================================================


def second_derivative(x: Tensor) -> Tensor:
    """The second derivative of f(x) = sum(x^3) with respect to x, that is 6*x, through autograd.

    Hint: create_graph=True on the first differentiation.
    """
    raise NotImplementedError


def vjp(f: Callable[[Tensor], Tensor], x: Tensor, v: Tensor) -> Tensor:
    """Vector-Jacobian product: (v^T J) for f at the point x. This is exactly what backward does.

    x comes without requires_grad, return a tensor with the shape of x.
    """
    raise NotImplementedError


def numeric_grad(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-4) -> Tensor:
    """The numeric gradient of a scalar function by central differences.

    (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps) for every coordinate.
    A loop over the elements is allowed here - this is a debugging instrument,
    the one used to check that analytic gradients are right.
    """
    raise NotImplementedError


def softmax_vjp(probs: Tensor, grad_out: Tensor) -> Tensor:
    """The gradient of softmax, derived by hand (without autograd).

    probs is the ALREADY computed softmax along the last axis, grad_out is the gradient from above.
    The known result: grad_in = probs * (grad_out - sum(grad_out * probs, dim=-1, keepdim=True)).

    Derive it yourself from J_ij = p_i (delta_ij - p_j) - this formula will come up again
    in the backward pass of attention.
    """
    raise NotImplementedError


# ==========================================================================
# 15.3 Custom autograd functions
# ==========================================================================


class SquareFunction(torch.autograd.Function):
    """y = x^2 with its own backward. Implement forward and backward.

    The rules: forward takes ctx as its first argument and saves what it needs through
    ctx.save_for_backward, backward receives grad_output and returns the gradient
    with respect to every input of forward.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator: we round in forward, and in backward we pretend
    the operation was the identity.

    Quantization-aware training (QAT) rests on this trick:
    round is not differentiable, but the model has to be trained somehow.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError


class RMSNormFunction(torch.autograd.Function):
    """y_i = g_i * x_i * r, where r = 1 / sqrt(mean(x^2) + eps), the statistics over the last axis.

    The derivation of backward (do it yourself, the answer below is only for cross-checking):
        r = (mean(x^2) + eps)^(-1/2)
        dr/dx_j = -x_j * r^3 / D
        let a_i = grad_y_i * g_i, then
        dL/dx_j = a_j * r - x_j * r^3 * sum_i(a_i * x_i) / D
        dL/dg_i = the sum over all the leading axes of grad_y_i * x_i * r

    Implement forward and backward. backward is obliged to return the gradients for (x, weight, eps),
    and for eps that is None.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out: Tensor):  # pyright: ignore[reportIncompatibleMethodOverride]
        raise NotImplementedError


# ==========================================================================
# 15.4 Gradients in real model code
# ==========================================================================


def embedding_grad(table_size: tuple[int, int], ids: Tensor) -> Tensor:
    """Create an embedding table of the given size that requires a gradient,
    do a lookup by ids, take the sum of all the values, call backward
    and return the gradient of the table.

    The result is instructive: the gradient of a row equals the number of its uses.
    That is exactly why rare tokens learn slowly.
    """
    raise NotImplementedError


def freeze_all_but_last(model: nn.Sequential) -> int:
    """Freeze every parameter except the parameters of the last module,
    and return the number of trainable parameters (elements, not tensors).

    This is literally linear probing / partial fine-tuning.
    """
    raise NotImplementedError


def grad_norm_after_backward(model: nn.Module, x: Tensor, target: Tensor) -> float:
    """Run x through model, take the MSE loss against target, backward, and return
    the global L2 norm of the gradients of all the parameters as a python float.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
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
    assert gb is None, "the gradient leaked into b"


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


@pytest.mark.parametrize("shape", [(3,), (2, 4)])
def test_softmax_vjp_property(shape):
    logits = torch.randn(*shape, requires_grad=True)
    probs = logits.softmax(dim=-1)
    g = torch.randn(*shape)
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
    assert torch.autograd.gradcheck(SquareFunction.apply, (x64,)), "gradcheck did not converge"


def test_ste_quantize():
    x = torch.tensor([0.2, 1.7, -0.4], requires_grad=True)
    y = STEQuantize.apply(x)
    torch.testing.assert_close(y, torch.tensor([0.0, 2.0, -0.0]))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones(3), msg="an STE must pass the gradient through as is")


def test_rms_norm_function_forward():
    x = torch.randn(2, 3, 8)
    w = torch.randn(8)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(RMSNormFunction.apply(x, w, 1e-5), expected, atol=1e-6, rtol=1e-5)


def test_rms_norm_function_backward_matches_autograd():
    x = torch.randn(4, 6, 16, requires_grad=True)
    w = torch.randn(16, requires_grad=True)
    g = torch.randn(4, 6, 16)

    # the reference is ordinary autograd through elementary operations
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    ref.backward(g)
    assert x.grad is not None and w.grad is not None
    ref_gx, ref_gw = x.grad.clone(), w.grad.clone()

    x.grad = None
    w.grad = None
    out = RMSNormFunction.apply(x, w, 1e-5)
    out.backward(g)
    torch.testing.assert_close(x.grad, ref_gx, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(w.grad, ref_gw, atol=1e-5, rtol=1e-4)


def test_rms_norm_function_gradcheck():
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: RMSNormFunction.apply(a, b, 1e-5), (x, w))


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

    expected = torch.cat([p.grad.flatten() for p in model.parameters()]).norm().item()  # pyright: ignore[reportOptionalMemberAccess]
    assert abs(norm - expected) < 1e-6
