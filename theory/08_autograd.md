# Topic 08 - Autograd

> **Exercise** `test_08_autograd.py`

## Why this topic

A neural network is a big function with a lot of adjustable numbers inside it.
Those numbers are called weights, or parameters.
Training means changing the weights a little at a time so that the network's answers get less wrong.

To do that you need two things.
First, a single number that says how wrong the network is right now.
That number is called the loss.
Second, for every weight, an answer to the question "if I nudge this weight up a tiny bit, does the loss go up or down, and by how much?"
That answer is the derivative of the loss with respect to that weight, and the collection of all of them is called the gradient.

Once you have the gradient, the update is simple: move every weight a small step in the direction that makes the loss smaller.
Repeat that a few million times and you have a trained model.

Working out derivatives by hand for a model with millions of weights is not realistic.
Autograd is the part of PyTorch that does it for you.
You write the forward computation as ordinary tensor code, and PyTorch works out the derivatives automatically.

You can use autograd for a long time without knowing how it works.
But sooner or later it surprises you: gradients that keep growing, a part of the model that never learns, an evaluation run that eats all the GPU memory.
Each of those comes from not knowing what autograd does behind your back.
This topic walks through what it does, and then has you write a backward pass by hand, which is the skill topics 18 and 19 build on.

## The mental model

### A two-minute refresher on derivatives

If you have not touched calculus in a while, this is all you need for this topic.

The derivative of a function at a point tells you how fast the output changes when you move the input a little.
For `f(x) = x^2` the derivative is `2x`.
At `x = 3` it is `6`, which means that moving `x` from `3` to `3.001` moves `f` by about `0.006`.

A few derivatives you will meet in this file:

| Function | Derivative |
| --- | --- |
| `c * x` (a constant times x) | `c` |
| `x^2` | `2x` |
| `x^3` | `3x^2` |
| `sin(x)` | `cos(x)` |
| `exp(x)` | `exp(x)` |
| `round(x)` | `0` almost everywhere, and undefined at the jumps |

When a function has many inputs, you take one derivative per input, holding the others still.
These are called partial derivatives.
For `f(a, b) = a * b`, the derivative with respect to `a` is `b`, and the derivative with respect to `b` is `a`.

When a computation is a chain of steps, the derivatives multiply.
This is the chain rule, and it is the only idea autograd really relies on.
Take `x = 2`, then `u = 3 * x` (so `u = 6`), then `y = u^2` (so `y = 36`).
How much does `y` change per unit change in `x`?
Step one says `u` moves 3 times as fast as `x`.
Step two says `y` moves `2u = 12` times as fast as `u`.
So `y` moves `3 * 12 = 36` times as fast as `x`.
You walked the chain backwards, from `y` to `x`, multiplying the local derivative of each step.
That walk is exactly what PyTorch's `backward()` does.

### What the graph is

PyTorch only tracks tensors you ask it to track.
Every tensor has a flag called `requires_grad`, which is `False` by default.
When you set it to `True` on a tensor, you are saying "I will want derivatives with respect to this."

From then on, every operation that uses that tensor gets recorded.
The result of the operation also gets `requires_grad=True`, plus an attribute called `grad_fn` that remembers which operation produced it and what its inputs were.
Those records link together into a graph: a map of how the final number was computed from the starting tensors.

```python
>>> w = torch.tensor(2.0, requires_grad=True)
>>> u = w * 3
>>> y = u + 1
>>> y.grad_fn
<AddBackward0 ...>
>>> w.grad_fn is None
True
```

The tensors you created yourself, like `w` above, sit at the start of the graph.
They are called **leaves**.
In a real model the leaves you care about are the weights.
Everything computed from them, like `u` and `y`, is an intermediate result, not a leaf.

When you call `y.backward()`, PyTorch walks the graph from `y` back to the leaves, applying the chain rule at every step.
It stores the final derivative in the `.grad` attribute of each leaf.
Only leaves get a `.grad`.
If you read `.grad` on an intermediate tensor you get `None` and a warning, because PyTorch throws those values away to save memory.
If you really need one, call `.retain_grad()` on that tensor before the backward.

Two behaviours of `backward()` matter for everything that follows.

It **adds** to `.grad` instead of replacing it.
If `.grad` already holds a value, the new gradient is added on top.
This is on purpose, because it lets you split a big batch into smaller pieces and sum their gradients for free.
It is also why every training loop clears the gradients before each step, usually with `optimizer.zero_grad()`.
Forget it, and each step uses the sum of all previous gradients.

It **throws the graph away** when it finishes.
The graph holds the intermediate values each step needs to compute its derivative, and those can take a lot of memory, so PyTorch frees them as soon as the backward pass is done.
Calling `backward()` a second time on the same result then fails.
If you need to go backwards twice through the same graph, pass `retain_graph=True` to the first call.

### 8.1 The basic cycle

The whole cycle is four lines:

```python
x = torch.tensor([1.0, 2.0], requires_grad=True)  # 1. mark the leaf
y = (5 * x).sum()  # 2. compute; the graph is recorded
y.backward()  # 3. walk it backwards
x.grad  # 4. tensor([5., 5.])
```

Each element of `x` was multiplied by 5 and then summed, so nudging any one element by a tiny amount moves `y` by 5 times that amount.
That is why the gradient is 5 everywhere.

**`backward()` wants a single number.**
A plain `y.backward()` only works when `y` holds exactly one value.
That is why losses are always reduced to one number with `.sum()` or `.mean()`.
Section 8.2 explains what happens when the output has more than one value.

**When the input does not require a gradient yet.**
Several stubs receive an `x` with `requires_grad=False`.
You can switch the flag on in place with `x.requires_grad_(True)`, but that changes the caller's tensor too, which is rude.
The polite way is to make your own copy and mark that: `x.detach().clone().requires_grad_(True)`.
`detach()` drops any graph history, `clone()` copies the data, and the trailing underscore in `requires_grad_` means "change this tensor in place".

**The functional API.**
`torch.autograd.grad(outputs, inputs)` computes the same derivatives as `backward()`, but returns them to you instead of writing them into `.grad`.

```python
>>> x = torch.tensor([1.0, 2.0], requires_grad=True)
>>> (g,) = torch.autograd.grad((5 * x).sum(), x)
>>> g
tensor([5., 5.])
>>> x.grad is None
True
```

It returns a tuple, one gradient per input you asked about, so note the `(g,) =` unpacking.
It is the right tool when you want a gradient without disturbing the `.grad` values an optimizer is about to read.

**Accumulation, seen directly.**

```python
>>> x = torch.tensor([1.0, 2.0], requires_grad=True)
>>> y = (5 * x).sum()
>>> y.backward(retain_graph=True)   # keep the graph for a second pass
>>> x.grad
tensor([5., 5.])
>>> y.backward()
>>> x.grad                          # added, not replaced
tensor([10., 10.])
```

One detail if you want to return the value after the first call: `.grad` is the same tensor object across both calls, and the second backward updates it in place.
If you keep a reference to it and return it later, it will already show the second value.
Take a copy with `.clone()` when you want a snapshot.

**Stopping a gradient.**
Sometimes you want a value to take part in the forward computation but not receive any gradient.
There are a few ways to do that, and they do different things:

| Call | What it does |
| --- | --- |
| `t.detach()` | a new tensor with the same values and the same memory, but with no history; autograd treats it as a constant |
| `with torch.no_grad():` | inside this block nothing is recorded at all, so no graph is built |
| `with torch.inference_mode():` | a stricter `no_grad`, slightly faster; its results can never be used for gradients later |
| `p.requires_grad_(False)` | this leaf stops collecting a gradient; covered in 8.4 |

Here is `detach()` on a neutral example:

```python
>>> a = torch.tensor(3.0, requires_grad=True)
>>> y = a + 2 * a.detach()
>>> y.backward()
>>> a.grad
tensor(1.)
```

`y` equals `3a` numerically, so you might expect a gradient of 3.
But the `2 * a.detach()` part is treated as a plain constant, so only the first `a` counts, and the gradient is 1.

If a leaf never contributes to the loss through any recorded path, its `.grad` stays `None` after the backward, not zero.
`None` means "nothing ever reached me".

**Running without a graph.**
When you only want the answer and not the gradient, for example when you evaluate a trained model, wrap the computation in `torch.no_grad()`.
Nothing gets recorded, so the result has `requires_grad=False` and `grad_fn=None`, even if its inputs require gradients.
This also saves a lot of memory, because PyTorch no longer keeps intermediate values around for a backward pass that will never happen.

```python
>>> w = torch.tensor(2.0, requires_grad=True)
>>> with torch.no_grad():
...     z = w * 4
>>> z.requires_grad, z.grad_fn
(False, None)
```

### 8.2 Harder derivatives

**Second derivatives.**
A second derivative is the derivative of the derivative.
For `f(x) = sin(x)` the first derivative is `cos(x)` and the second is `-sin(x)`.

By default, PyTorch computes the gradient without recording that computation, so the gradient you get back has no graph and cannot be differentiated again.
Passing `create_graph=True` to `torch.autograd.grad` (or to `backward`) tells PyTorch to record the gradient computation too.
The gradient you get back then has its own `grad_fn`, and you can call `torch.autograd.grad` on it a second time.

The second call has the same rule as any other: the thing you differentiate must be a single number.
When the function works element by element, like `sin` or `x^3`, each element's derivative depends only on its own input.
So summing the first gradient and differentiating that sum gives you every element's second derivative at once.

**The Jacobian and the vector-Jacobian product.**
So far every output has been a single number.
Now take a function with several inputs and several outputs, say `f(x1, x2) = (x1 * x2, x1 + x2)`.
There are now four derivatives: each output with respect to each input.
Arrange them in a table where row `i` is output `i` and column `j` is input `j`:

```
J = | d(x1*x2)/dx1   d(x1*x2)/dx2 |   =   | x2   x1 |
    | d(x1+x2)/dx1   d(x1+x2)/dx2 |       | 1    1  |
```

That table is called the **Jacobian**.
For a function with `n` inputs and `m` outputs it has `m * n` entries, which for a real network is far too many to ever build.

PyTorch never builds it.
Instead, `backward()` needs one extra piece of information: a vector `v` with one number per output, saying how much you care about each output.
It then computes `v^T J`, the weighted sum of the Jacobian's rows, which has one number per input.
That is called the **vector-Jacobian product**, or VJP.

At `x1 = 2, x2 = 5` with `v = (1, 10)`, the VJP is `1 * (5, 2) + 10 * (1, 1) = (15, 12)`.

You pass `v` as the `gradient` argument: `y.backward(gradient=v)`, where `v` has the same shape as `y`.
With `torch.autograd.grad`, the same thing is called `grad_outputs`.

This also explains the rule from 8.1.
When `y` is a single number, the Jacobian has one row, and a plain `y.backward()` is just a VJP with `v = 1`.
Every backward pass in PyTorch is a VJP.

**Checking a gradient by brute force.**
How do you know an analytic gradient is right?
You compare it with a numeric one, computed straight from the definition of the derivative.
Move one input a little up, move it a little down, and see how much the output changed:

```
derivative ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
```

For `f(x) = x^2` at `x = 3` with `eps = 0.01`, that is `(9.0601 - 8.9401) / 0.02 = 6.0`, which matches `2x = 6`.

For a function of many inputs, you repeat this once per input element: nudge only element `i`, leave the others alone, and the result is element `i` of the gradient.
A plain Python loop over the elements is fine here, because this is a debugging tool, not training code.
A few practical points:

- Work on copies of `x` so the caller's tensor is never modified, and so the "plus" and "minus" evaluations do not disturb each other.
- If `x` has more than one dimension, it is easiest to loop over a flattened view (`reshape(-1)`) and give the result the shape of `x` at the end.
- `f` returns a one-element tensor; `.item()` turns it into a Python float.

Why `eps` both up and down, instead of only `(f(x + eps) - f(x)) / eps`?
The symmetric version is much more accurate for the same `eps`.
And why not make `eps` tiny?
Because floating-point numbers have limited precision, and subtracting two almost-equal numbers loses most of it.
`1e-4` is a reasonable choice in float32, and `1e-6` in float64.

**The softmax derivative.**
Softmax turns a row of arbitrary scores into probabilities: all positive, summing to 1.
For scores `x_1, ..., x_n`:

```
p_i = exp(x_i) / S        where S = exp(x_1) + ... + exp(x_n)
```

Bigger scores get bigger probabilities, and every probability depends on every score through `S`.
You will see softmax constantly from topic 10 on.

Its derivative is worth working out once, slowly.
Take the derivative of `p_i` with respect to one score `x_j`.
Two cases:

When `j = i`, both the top `exp(x_i)` and the bottom `S` depend on `x_i`.
Using the quotient rule, `d p_i / d x_i = (exp(x_i) * S - exp(x_i) * exp(x_i)) / S^2`, which simplifies to `p_i - p_i * p_i`.

When `j ≠ i`, only `S` depends on `x_j`.
So `d p_i / d x_j = -exp(x_i) * exp(x_j) / S^2`, which is `-p_i * p_j`.

Both cases fit one formula, using `delta_ij` to mean 1 when `i = j` and 0 otherwise:

```
d p_i / d x_j = p_i * (delta_ij - p_j)
```

That is the softmax Jacobian.
Now do what `backward()` does: take the incoming gradient `g` (one number per output `p_i`) and compute the VJP, that is, for each input `j` sum `g_i * J_ij` over all `i`.
Plug in the formula and split the sum into the `delta` part and the rest.
The `delta` part keeps only the term `i = j`.
The rest contains a sum over `i` of `g_i * p_i`, which is the same number for every `j`.
Do that algebra on paper before you write any code, and check the result against the formula in the stub's docstring.

In code, softmax runs along the last axis, so that shared sum is a reduction over the last axis, kept with `keepdim=True` so it broadcasts back against the rows (topics 03 and 04).

The result has an intuitive reading.
Softmax ignores a constant added to all its scores, because that constant cancels out between the top and the bottom.
So the gradient can never push all the scores up together.
Subtracting the probability-weighted average of `g` is exactly what removes that "all together" direction.
The same expression shows up again in the backward pass of attention in topic 19.

### 8.3 Custom autograd functions

Sometimes you want to decide the backward pass yourself.
`torch.autograd.Function` lets you write an operation as a pair: a `forward` that computes the result, and a `backward` that receives the gradient arriving from later steps and returns the gradient for each input.

Here is the shape of one, for `y = exp(x)`.
Its derivative is `exp(x)` again, which is the output itself, so it saves the output instead of the input:

```python
class MyExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = x.exp()
        ctx.save_for_backward(out)  # stash what backward will need
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (out,) = ctx.saved_tensors
        return grad_output * out  # chain rule: incoming gradient times local derivative


y = MyExp.apply(x)  # call it with .apply, never MyExp()(x)
```

`ctx` is a scratch object that carries information from `forward` to `backward`.
`grad_output` has the shape of the output, and it is the "how much does the loss care about each output element" that came back from later in the graph.
The local derivative is multiplied by it element by element, which is the chain rule from the start of this file.

The rules that are easy to get wrong:

- `backward` returns **one value per `forward` input**, not counting `ctx`, in the same order.
  For inputs that are not tensors, or do not need a gradient, return `None` in that position.
- Save tensors with `ctx.save_for_backward(...)` and read them back with `ctx.saved_tensors`.
  Other things, like a Python flag, can be stored directly as `ctx.something = ...`.
- Nothing inside `forward` is recorded in the graph.
  PyTorch only knows about the operation through your `backward`, so if `backward` is wrong, the gradients are wrong.
- `torch.autograd.gradcheck(MyExp.apply, (x,))` compares your `backward` with numeric gradients, like the ones from 8.2.
  It needs float64 inputs to be precise enough.

There are two common reasons to write one.
The first is efficiency: a hand-written backward can avoid saving large intermediate values, which is how flash attention saves memory in topic 18.
The second is to give an operation a gradient it does not really have.

**The straight-through estimator.**
Some operations, such as rounding, are flat almost everywhere and jump at a few points.
The derivative of `round(x)` is zero almost everywhere.
If you put rounding in the middle of a network, the zero gradient stops everything before it from learning.

The straight-through estimator, or STE, is a deliberate cheat.
In `forward` you really do round.
In `backward` you pretend the operation was `y = x`, whose derivative is 1, so the incoming gradient passes through unchanged.
It is not the true gradient of anything.
It works well enough in practice, and the alternative is no learning at all.
It is used in quantization-aware training, where the model learns to cope with the rounding it will meet when it runs in low precision, and in VQ-VAE models that pick an entry from a discrete codebook.

### 8.4 Gradients in real model code

**Embedding tables.**
A language model cannot do arithmetic on words, so it gives every token in its vocabulary a number, an id, and keeps a table with one row of learned weights per id.
That table is just a matrix of shape `(vocab size, vector size)`.
Turning a batch of ids into vectors is plain indexing: `table[ids]`, exactly as in topic 05.
If `ids` has shape `(B, T)` and the table has shape `(V, D)`, the result has shape `(B, T, D)`.

What is its gradient?
Indexing copies rows out of the table.
In the backward pass, each copied row sends its gradient back to the row it came from, and if the same row was copied several times, the gradients add up.
Rows that were never looked up get exactly zero.

A neutral example with a table of 5 rows and ids `[4, 1, 4]`: if you sum everything you looked up and call backward, the incoming gradient is 1 for every looked-up value.
Row 4 was used twice, so every entry in row 4 gets 2.
Row 1 gets 1.
Rows 0, 2 and 3 get 0.
This is why rare tokens learn slowly: a row only gets a learning signal when its token actually appears.

**Freezing.**
To stop part of a model from training, set `requires_grad=False` on its parameters.
Those parameters stop collecting gradients, so an optimizer leaves them alone.
This is how you fine-tune only the last layer of a large model.

You need two bits of module API for this, both covered properly in topic 09:

- `model.parameters()` goes through every weight tensor in a model, including all the layers inside it.
  Calling `.parameters()` on one layer gives you only that layer's weights.
- `nn.Sequential` is a list of layers run one after another, and it can be indexed like a list: `model[0]` is the first layer and `model[-1]` the last.
  Some layers, like `nn.ReLU`, have no parameters at all.

To count trainable parameters, count numbers, not tensors: add up `p.numel()` (the number of elements in a tensor) over the parameters that still have `requires_grad=True`.

A frozen weight does not stop gradients from flowing *through* it to earlier parts of the graph.
It just does not store a gradient of its own.

**The global gradient norm.**
After a backward pass, one quick health check is the overall size of the gradient.
Square every gradient value of every parameter, add them all up, and take the square root.
That is the L2 norm from topic 04, applied to all the parameters at once as if they were one long vector.
If it explodes or drops to zero during training, something is wrong.
Topic 16 uses the same number to clip gradients that get too large.

The loss for that stub is the mean squared error: the average of `(prediction - target)^2` over all elements.
PyTorch has it as `torch.nn.functional.mse_loss(prediction, target)`.
A one-element tensor turns into a Python float with `.item()`.

## API you need

### Building and cutting the graph

| Call | Result |
| --- | --- |
| `torch.tensor(..., requires_grad=True)` | a new leaf that will collect a gradient |
| `x.requires_grad_(True)` | switch the flag on, in place, returns `x` |
| `x.detach()` | same values and memory, no history |
| `x.detach().clone()` | an independent copy with no history |
| `with torch.no_grad():` | record nothing inside the block |
| `with torch.inference_mode():` | a stricter, faster `no_grad` |
| `x.grad_fn` | the operation that produced `x`, `None` for a leaf |
| `x.is_leaf` | whether `x` is a leaf |
| `x.retain_grad()` | keep `.grad` on an intermediate tensor |

### Computing gradients

| Call | Result |
| --- | --- |
| `y.backward()` | `y` must hold one value; adds into `.grad` of the leaves |
| `y.backward(gradient=v)` | for `y` with many values; `v` has the shape of `y` |
| `y.backward(retain_graph=True)` | keep the graph for another backward |
| `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False)` | returns a tuple of gradients, does not touch `.grad` |
| `torch.autograd.gradcheck(f, (x,))` | compare a backward with numeric gradients; use float64 |

### Custom functions

| Call | Result |
| --- | --- |
| `class F(torch.autograd.Function)` with static `forward(ctx, ...)` and `backward(ctx, grad_output)` | your own operation |
| `ctx.save_for_backward(*tensors)` / `ctx.saved_tensors` | stash tensors in forward, read them in backward |
| `F.apply(*args)` | run it |
| `torch.round(x)` | round to the nearest integer, halves to even |

### Parameters and losses

| Call | Result |
| --- | --- |
| `model.parameters()` | every parameter tensor, recursively |
| `model[-1]` | the last layer of an `nn.Sequential` |
| `p.requires_grad_(False)` | freeze `p` |
| `p.numel()` | number of elements in `p` |
| `p.grad` | the stored gradient, `None` before any backward |
| `torch.nn.functional.mse_loss(pred, target)` | mean squared error, one value |
| `t.item()` | a one-element tensor as a Python number |
| `optimizer.zero_grad()` / `model.zero_grad()` | clear gradients before the next step |

## Traps

- Gradients add up.
  Without clearing them, the second training step uses the sum of two steps' gradients.
- `.grad` is updated in place, so a reference you kept earlier changes under you.
  `.clone()` it if you want a snapshot.
- A second `backward()` on the same graph raises `RuntimeError: Trying to backward through the graph a second time` unless the first one had `retain_graph=True`.
- Leaving `retain_graph=True` in a training loop keeps every step's graph alive and slowly fills the memory.
- `.grad` on an intermediate tensor is `None` with a warning, not an error.
- `x.requires_grad_(True)` on a tensor someone passed you changes their tensor.
  Work on a copy.
- Computing something inside `torch.no_grad()` and later trying to differentiate it gives you no gradient, silently.
- Results created inside `inference_mode` can never be used in a graph, even after the block ends.
- Changing a tensor in place after autograd saved it for the backward pass raises an error about a version counter.
  The fix is to stop modifying it in place.
- In a custom `Function`, returning the wrong number of values from `backward` gives a confusing error about the number of gradients.
- A numeric gradient with a very small `eps` in float32 is mostly rounding noise.
  If a check fails in a suspicious way, try float64 first.

## Ready when

Without looking anything up you can:

- [ ] explain in plain words what a gradient is and why training needs one
- [ ] say what a leaf is and why only leaves get `.grad`
- [ ] say what `backward()` does to an existing `.grad`, and why training loops clear it
- [ ] name three ways to stop a gradient and how they differ
- [ ] explain what the `gradient=` argument of `backward()` means, using the word Jacobian
- [ ] derive the softmax Jacobian `p_i (delta_ij - p_j)` on paper and turn it into the VJP
- [ ] write the skeleton of a `torch.autograd.Function`, including how many values `backward` returns
- [ ] explain the straight-through estimator in two sentences, including why it is not a real gradient
- [ ] say what the gradient of an embedding table looks like after summing a lookup

## Going deeper

- [A gentle introduction to `torch.autograd`](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html), the official beginner tutorial, if this file moved too fast.
- [Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html), the reference document, including the rules for in-place operations.
- [Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html), for the full contract of `torch.autograd.Function`.
- Why PyTorch goes backwards rather than forwards: computing derivatives forwards costs one pass per input, and backwards costs one pass per output.
  A network has millions of inputs (its weights) and one output (the loss), so backwards wins by a huge margin.
  `torch.func.jvp` does the forward direction for the rare cases where the numbers are the other way around.
