# Topic 08 - Autograd

> **Exercise** `test_08_autograd.py`


Autograd is the thing you stop thinking about until it costs you money.
A graph you forgot to disable doubles the memory of an evaluation pass.
A missing `zero_grad` silently accumulates gradients across steps.
A `.detach()` in the wrong place stops half the model from learning.
And a hand-written backward is how every serious optimization in topics 18 and 19 gets implemented.

This file makes the graph visible: you build it, you cut it, you differentiate twice, and you write a backward yourself.

## The mental model

### What the graph is

Every tensor has `requires_grad`.
When an operation has at least one input with `requires_grad=True` and grad mode is enabled, the output gets `requires_grad=True` and a `grad_fn` pointing at a node that knows how to compute the backward of that operation.
The nodes chain into a directed acyclic graph whose leaves are the tensors you created yourself.

A **leaf** is a tensor not produced by a differentiable operation: parameters, and anything you built with `requires_grad_(True)` directly.
`.grad` is only populated on leaves.
Asking for `.grad` on an intermediate gives `None` and a warning, unless you called `.retain_grad()` on it.

`loss.backward()` walks the graph from `loss` backwards, and at each node multiplies the incoming gradient by that node's local Jacobian.
It never builds a Jacobian matrix; it computes a **vector-Jacobian product**, which is the entire reason reverse-mode differentiation is affordable.

Two consequences worth stating plainly:

- `backward()` **accumulates** into `.grad`, it does not overwrite.
  That makes gradient accumulation across micro-batches free, and it is exactly why every training loop calls `optimizer.zero_grad()`.
- `backward()` **frees the graph** by default.
  A second `backward()` on the same graph raises unless the first was called with `retain_graph=True`.

### 8.1 The basic cycle

```
x.requires_grad_(True)    # or torch.tensor(..., requires_grad=True)
y = f(x)                  # graph is built
y.backward()              # y must be a scalar, or pass a gradient argument
x.grad                    # populated
```

`backward()` on a non-scalar needs a `gradient=` argument of matching shape, which is the `v` in the vector-Jacobian product.
The usual way around that is to reduce to a scalar first, which is why losses are means and sums.

**The functional API** is `torch.autograd.grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None)`.
It returns a tuple of gradients and leaves `.grad` untouched.
Use it whenever you want a gradient without disturbing the state the optimizer is looking at.

**Stopping gradients** has several spellings that do different things:

| Call | Effect |
| --- | --- |
| `t.detach()` | a tensor sharing storage, cut out of the graph, `requires_grad=False` |
| `with torch.no_grad():` | no graph is built inside the block at all |
| `with torch.inference_mode():` | stronger than `no_grad`, also disables version counting, results cannot later require grad |
| `p.requires_grad_(False)` | this leaf stops receiving gradients, the graph still flows through it |

`detach()` on one operand of a product means the gradient reaches the other operand and nothing reaches the detached one, so its `.grad` stays `None`.
That `None` is informative: it means the leaf was never on any path to the loss.

`no_grad` is the inference setting, and the memory difference is large, because building no graph means saving no intermediate activations.

### 8.2 Harder derivatives

**Second derivatives** require the first backward to itself be differentiable, which is what `create_graph=True` does: it builds a graph for the gradient computation as well.
Then differentiating the gradient gives the second derivative.
For `f(x) = sum(x^3)` the first derivative is `3x^2` and the second is `6x`, which is a good sanity check because it is elementwise and unambiguous.
`create_graph=True` implies `retain_graph=True`.

**The vector-Jacobian product** is the primitive autograd actually computes.
For `f : R^n -> R^m` with Jacobian `J`, `vjp(f, x, v) = v^T J`, a vector of length `n`.
You get it by running the forward pass and then calling backward with `v` as the incoming gradient.
Recognizing that `backward()` on a scalar loss is just a VJP with `v = 1.0` unifies the whole picture.

**Numeric gradients by central differences** are the ground truth you check analytic gradients against:

```
grad_i ~= (f(x + eps*e_i) - f(x - eps*e_i)) / (2*eps)
```

The central form has error `O(eps^2)` where the forward difference has `O(eps)`, which is why it is worth the second evaluation.
`eps` around `1e-4` in float32 and `1e-6` in float64 balances truncation error against cancellation.

**The softmax Jacobian** is the one derivation in this file that pays off repeatedly.
With `p = softmax(x)` along the last axis:

```
dp_i / dx_j = p_i * (delta_ij - p_j)
```

Contracting that with an incoming gradient `g` gives

```
grad_x = p * (g - sum_j(g_j * p_j))
```

where the sum is over the softmax axis, kept with `keepdim=True` so it broadcasts.
Read the structure rather than memorizing it: the gradient of the softmax is the incoming gradient with its *probability-weighted mean* subtracted, then scaled by the probabilities.
Because a constant shift of the input does not change the output, the gradient has to be orthogonal to the all-ones direction in exactly this sense.
This same expression reappears inside the backward pass of attention in topic 19.

### 8.3 Custom autograd functions

`torch.autograd.Function` lets you insert a node with a backward you wrote yourself.

```python
class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, some_flag):
        ctx.save_for_backward(a, b)   # tensors only
        ctx.some_flag = some_flag     # non-tensors go on ctx directly
        return result

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_a, grad_b, None   # one return per forward input
```

Rules that are easy to get wrong:

- `backward` must return **exactly as many values as `forward` took inputs**, not counting `ctx`.
  Non-tensor inputs, and tensor inputs that need no gradient, get `None`.
- Save tensors with `ctx.save_for_backward` rather than as plain attributes, so autograd's correctness checks for in-place modification still work.
- Call it as `MyOp.apply(a, b, flag)`, never `MyOp()(a, b)`.
- Inside `forward`, grad mode is off, so operations there build no graph.

Two reasons to write one.
Memory or speed: a fused implementation can avoid saving intermediates that the default decomposition would save, which is exactly what flash attention does.
Or a **deliberately wrong gradient**, which is the straight-through estimator.

**The straight-through estimator** solves an impossible problem.
`round`, `sign` and quantization have a derivative that is zero almost everywhere and undefined at the steps, so the true gradient kills all learning.
The STE keeps the discrete operation in the forward pass and pretends the backward pass saw the identity, so the gradient passes through unchanged.
It is not the true gradient of anything.
The implementation is as short as the idea: forward rounds, backward returns `grad_output`.

### 8.4 Gradients in real model code

**Embedding gradients are sparse and additive.**
A lookup is an indexing operation, whose backward is a scatter-add.
If token 7 appears five times in a batch, row 7 of the table receives five contributions and every unused row gets exactly zero.
Summing the looked-up embeddings and calling backward gives a gradient whose row `i` is the number of occurrences of `i`, which is a direct demonstration of why rare tokens learn slowly.

**Freezing** is setting `requires_grad=False` on the parameters you do not want to train.
`model.parameters()` and `model.named_parameters()` iterate everything recursively; a submodule's own `.parameters()` gives just its share.
Counting trainable parameters is a sum of `p.numel()` over the ones with `requires_grad=True`.

Freezing a leaf does not stop the graph from flowing *through* it: gradients still propagate to whatever is behind it, they just are not stored on that leaf.
To also save the backward compute you would have to detach the input to the frozen section.

**The global gradient norm** after a backward is the quantity from topic 04 applied to `model.parameters()`: square every element of every `p.grad`, sum, take the square root.
Watching it during training is the cheapest diagnostic there is, and clipping it is topic 16.

## API you need

### Building and cutting the graph

| Call | Result |
| --- | --- |
| `x.requires_grad_(True)` | mark a leaf, in-place, returns `x` |
| `torch.tensor(..., requires_grad=True)` | the constructor form |
| `x.detach()` | same storage, no graph |
| `x.detach().clone()` | independent and no graph |
| `with torch.no_grad():` | build no graph inside |
| `with torch.inference_mode():` | stronger, faster, results are inference-only |
| `with torch.enable_grad():` | re-enable inside a `no_grad` block |
| `x.grad_fn` | the node that produced `x`, `None` for a leaf |
| `x.is_leaf` | whether `.grad` will be populated |
| `x.retain_grad()` | keep `.grad` on a non-leaf |

### Differentiating

| Call | Result |
| --- | --- |
| `loss.backward(retain_graph=False, create_graph=False)` | accumulate into `.grad` |
| `y.backward(gradient=v)` | for non-scalar `y`, the `v` of the VJP |
| `torch.autograd.grad(outputs, inputs, grad_outputs=None, create_graph=False)` | returns a tuple, touches no `.grad` |
| `torch.autograd.gradcheck(f, inputs)` | numeric check, wants float64 |
| `torch.func.jvp`, `torch.func.vjp`, `torch.func.jacrev` | the functional transforms |

### Custom functions

| Call | Result |
| --- | --- |
| `class F(torch.autograd.Function)` with static `forward`/`backward` | a custom node |
| `ctx.save_for_backward(*tensors)` / `ctx.saved_tensors` | stash and retrieve |
| `ctx.needs_input_grad` | tuple of bools, skip computing what nobody wants |
| `F.apply(*args)` | invoke it |

### Parameters

| Call | Result |
| --- | --- |
| `model.parameters()` / `model.named_parameters()` | recursive iterators |
| `p.requires_grad_(False)` | freeze |
| `p.grad` | the accumulated gradient, `None` until a backward |
| `optimizer.zero_grad(set_to_none=True)` | clear gradients, the default and the faster option |
| `model.zero_grad()` | the same without an optimizer |

### A worked example on neutral data

```python
>>> x = torch.tensor([1.0, 2.0], requires_grad=True)
>>> y = (x ** 2).sum()
>>> y.backward(retain_graph=True)
>>> x.grad
tensor([2., 4.])
>>> y.backward()                 # accumulates, does not replace
>>> x.grad
tensor([4., 8.])

>>> a = torch.tensor([3.0], requires_grad=True)
>>> b = torch.tensor([5.0], requires_grad=True)
>>> (a * b.detach()).sum().backward()
>>> a.grad, b.grad
(tensor([5.]), None)
```

## Traps

- Gradients accumulate.
  Without `zero_grad()` your second step uses the sum of two batches' gradients at the same learning rate.
- `set_to_none=True` makes `.grad` be `None` rather than a zero tensor, so code that reads `.grad` unconditionally has to handle it.
- A second `backward()` raises `RuntimeError: Trying to backward through the graph a second time` unless `retain_graph=True`.
- `retain_graph=True` left in a training loop leaks memory, because every step's graph stays alive.
- `.grad` on a non-leaf is `None` with a warning, not an error.
- An in-place operation on a tensor that autograd saved raises a version counter error.
  The fix is to stop doing the in-place write, not to clone the error away.
- `backward` on a custom `Function` must return one value per forward input, `None` included.
  A wrong count gives a confusing arity error.
- `torch.no_grad()` around a forward pass whose output you later want to differentiate silently gives you no gradients.
- `inference_mode` outputs cannot be used in an autograd graph later, even outside the block.
- `numeric_grad` with too small an `eps` in float32 is dominated by cancellation rather than truncation.
  If a gradient check fails suspiciously, try float64 first.

## Ready when

Without looking anything up you can:

- [ ] explain what a leaf is and why only leaves get `.grad`
- [ ] name three ways to stop a gradient and say how they differ
- [ ] say what `backward()` does to an existing `.grad` and why the training loop is written the way it is
- [ ] write the softmax VJP from the Jacobian `p_i(delta_ij - p_j)`
- [ ] write the skeleton of a `torch.autograd.Function` including the return arity of `backward`
- [ ] explain the straight-through estimator in two sentences, including why it is not a real gradient
- [ ] say what the gradient of an embedding table looks like after summing a lookup

## Context



**Why reverse mode and not forward mode.**
The other direction of differentiation computes `J u`, the Jacobian-vector product, by propagating derivatives forward alongside the values.
Forward mode costs one pass per *input* dimension; reverse mode costs one pass per *output* dimension.
A neural network has millions of inputs, meaning parameters, and exactly one output, the scalar loss, so reverse mode wins by a factor of millions.
`torch.func.jvp` implements forward mode, and it is the right tool when the situation is inverted, such as differentiating a small parameterization of a large simulation.

**Where the straight-through estimator is actually used.**
Quantization-aware training runs the forward pass through fake-quantized weights so the model learns to tolerate the rounding it will meet at inference time, and the STE is what lets the gradient reach the underlying float weights at all.
VQ-VAE uses the same trick to backpropagate through a discrete codebook lookup.
In both cases the estimator is biased and everyone knows it; it survives because the alternative is no gradient at all.

**`no_grad` versus `inference_mode`.**
`no_grad` stops the graph from being recorded.
`inference_mode` additionally skips the version counter and view tracking that autograd maintains on every tensor, which is a measurable speedup on small operations.
The price is that its outputs are permanently marked and cannot participate in a later autograd computation, even outside the block, so it fits a serving loop and not a training loop that evaluates mid-epoch.

## Going deeper

- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html), the normative document, including the in-place correctness rules.
- [Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html) for the full `Function` contract.
- The [`torch.func`](https://pytorch.org/docs/stable/func.html) transforms, if you want the JAX-style functional view of the same machinery.
