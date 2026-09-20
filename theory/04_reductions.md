# Topic 04 - Reductions

> **Exercise** `test_04_reductions.py`

Every loss is a reduction.
Every normalization layer is two reductions.
Gradient clipping is a reduction over every parameter in the model, and greedy decoding is an `argmax`.

The mechanics are simple and the mistakes are not.
Averaging over padding tokens inflates or deflates your loss by a factor nobody notices.
Using the unbiased variance in LayerNorm puts you slightly off every reference implementation.
A naive `log(sum(exp(x)))` overflows to `inf` on perfectly ordinary logits.

## The mental model

### dim and keepdim

Every reduction takes `dim` and `keepdim`.

- No `dim`: reduce everything, get a 0-d scalar tensor.
- `dim=k`: reduce that one axis, which then disappears, lowering the rank by one.
- `dim=(a, b)`: reduce several axes at once.
- `keepdim=True`: leave the reduced axes in place with size 1, keeping the rank.

`keepdim=True` exists for exactly one reason: so the result still broadcasts against the input.

```
x                        (B, T, D)
x.mean(-1)               (B, T)        does NOT broadcast back against x
x.mean(-1, keepdim=True) (B, T, 1)     does
```

Negative `dim` counts from the end, and `-1` meaning "the feature axis" is the convention throughout this repository.
Writing `-1` rather than `2` makes the same function work for `(T, D)` and `(B, T, D)` alike.

### 4.1 Means, variances, RMS

`x.var(dim, correction=...)` defaults to the **unbiased** estimator, dividing by `N - 1`.
LayerNorm uses the **biased** one, dividing by `N`, which is `correction=0` (or the older `unbiased=False`).
Getting it wrong gives an answer close enough to pass an eyeball check and fails `assert_close`.

The biased variance is `((x - x.mean(dim, keepdim=True)) ** 2).mean(dim, keepdim=True)`, which is worth being able to write directly.

RMS is `sqrt(mean(x^2))`, with no mean subtraction at all.
That difference is the entire distinction between LayerNorm and RMSNorm, and it is why RMSNorm is cheaper: one pass instead of two, and no centering.

### 4.2 Reductions with a mask

Padded batches are the normal case, and a padded position must not contribute.

A **masked mean** is a sum of the masked-in values divided by the *count* of masked-in values, never by the axis length.
Zero out the excluded entries, sum, divide by the mask's sum.
For `x` of `(B, T, D)` and `mask` of `(B, T)`, the mask needs an axis to become `(B, T, 1)`, the product sums over `T` to `(B, D)`, and the divisor `mask.sum(1, keepdim=True)` is `(B, 1)`.
Guard the divisor against zero if an all-padding row is possible.

A **masked max** cannot be done by zeroing, because zero might be larger than every real value.
Fill the excluded positions with `-inf`, or `torch.finfo(dtype).min`, and then take the max.
A masked min fills with `+inf`.

Counting is summing a bool: `mask.sum(dim)` gives per-row lengths as int64, `mask.sum()` gives the total, and `.item()` turns that into a Python `int`.

### 4.3 max, argmax, topk

`x.max(dim)` returns a **named tuple** `(values, indices)`, not a tensor.
`x.max()` with no `dim` returns a plain scalar tensor.
That asymmetry catches everyone once.
`x.amax(dim)` returns only the values and is the right call when you do not need the indices.

`x.argmax(dim)` gives the indices alone, and `argmax` with no `dim` flattens first, so the index it returns is into the flattened tensor.
Recovering a 2-D position from a flat index is `divmod(flat, n_cols)`.

`torch.topk(x, k, dim=-1, largest=True, sorted=True)` returns `(values, indices)` sorted descending by default.
This is the primitive behind top-k sampling and behind beam search.

Greedy decoding is `logits.argmax(-1)`.
Because softmax is monotone, the argmax of the logits and the argmax of the probabilities are the same index, so taking a softmax first is wasted work.

**Accuracy with an ignore index** is a masked mean of a boolean: compare predictions to targets, exclude positions where the target equals `ignore_index`, and divide correct non-ignored positions by non-ignored positions.
`-100` is PyTorch's conventional ignore index, chosen because it is never a valid class id.

### 4.4 logsumexp and entropy

`log(sum(exp(x)))` appears in cross-entropy, in the softmax denominator and in flash attention.
Computed naively it overflows: `exp(1000.0)` is `inf` in float32, and logits of that magnitude are not unusual in an untrained model.

The stable form subtracts the maximum first:

```
m = x.max(dim, keepdim=True).values
logsumexp(x) = m + log(sum(exp(x - m)))
```

This is exact, not an approximation.
`sum_i exp(x_i) = exp(m) * sum_i exp(x_i - m)`, and taking the log turns the factor into the additive `m`.
Every exponent is now at most `exp(0) == 1`, so nothing overflows, and the largest term is exactly 1, so the total cannot underflow to zero either.
Remember to drop the kept axis at the end if the signature asks for the reduced shape.

**Entropy** in nats is `-sum(p * log(p))` over the last axis.
The problem is `p == 0`, where `log(0)` is `-inf` and `0 * -inf` is NaN, even though the mathematical limit is 0.
Three clean fixes: `torch.where(p > 0, p * p.log(), zeros)` so the bad term is never multiplied, `torch.xlogy(p, p)` which is defined to return 0 when the first argument is 0, or `p.clamp(min=1e-12).log()`.

### 4.5 Norms

The L2 norm of a vector is `sqrt(sum(x^2))`, and `torch.linalg.vector_norm(x, ord=2, dim=..., keepdim=...)` is the modern call.
`x.norm()` still exists and is deprecated.

**L2 normalization** divides each vector by its own norm, with an `eps` in the denominator so a zero vector does not produce NaN.

**Global gradient norm** treats every parameter's gradient as part of one giant vector and takes its L2 norm.
Sum the squares of every element of every tensor, add those sums, take the square root.
Equivalently, take the per-tensor norms and then the norm of *those*, which avoids building any large intermediate.

**Counting tokens** is `torch.bincount(ids.flatten(), minlength=vocab_size)`.
`bincount` only accepts a 1-D non-negative integer tensor, and without `minlength` the result is only as long as the largest id present.

**Cumulative statistics** come from `torch.cumsum(x, dim)`.
A running mean is the running sum divided elementwise by a running count, which is just `arange(1, T+1)`.

## API you need

### Basic reductions

| Call | Result |
| --- | --- |
| `x.sum(dim=None, keepdim=False)` | sum |
| `x.mean(dim=None, keepdim=False)` | mean, floats only |
| `x.prod(dim)` | product |
| `x.var(dim, correction=1, keepdim=False)` | variance, `correction=0` for the biased one |
| `x.std(dim, correction=1)` | standard deviation, same convention |
| `torch.var_mean(x, dim, correction=0, keepdim=True)` | both in one pass |
| `x.cumsum(dim)`, `x.cumprod(dim)` | running totals, same shape as the input |

### Extrema

| Call | Result |
| --- | --- |
| `x.max(dim, keepdim=False)` | named tuple `(values, indices)` |
| `x.max()` | a scalar tensor |
| `x.amax(dim)` / `x.amin(dim)` | values only |
| `x.argmax(dim)` / `x.argmin(dim)` | indices only |
| `torch.topk(x, k, dim=-1, largest=True, sorted=True)` | `(values, indices)` |
| `torch.sort(x, dim=-1, descending=False)` | `(values, indices)` |
| `torch.maximum(a, b)` / `torch.minimum(a, b)` | elementwise between two tensors |

`torch.max(a, b)` with two tensors is the elementwise version, while `torch.max(x, 0)` with an int is the reduction.
The overload depends on the second argument's type, so prefer `amax` and `maximum` when you want to be explicit.

### Stability

| Call | Result |
| --- | --- |
| `torch.logsumexp(x, dim, keepdim=False)` | the stable form, built in |
| `torch.xlogy(a, b)` | `a * log(b)`, returning 0 where `a == 0` |
| `x.clamp(min=eps)` | floor before a log |
| `torch.finfo(dtype).min` / `.max` / `.eps` / `.tiny` | dtype limits |

### Norms and counting

| Call | Result |
| --- | --- |
| `torch.linalg.vector_norm(x, ord=2, dim=None, keepdim=False)` | Lp norm |
| `F.normalize(x, p=2, dim=-1, eps=1e-12)` | divide by the norm |
| `torch.bincount(x, minlength=0)` | counts of a 1-D non-negative int tensor |
| `mask.sum(dim)` | count of True values |
| `mask.any(dim)` / `mask.all(dim)` | boolean reductions |

### A worked example on neutral data

```python
>>> x = torch.tensor([[1000.0, 1001.0], [0.0, 0.0]])
>>> x.exp().sum(-1).log()                    # naive
tensor([inf, 0.6931])
>>> m = x.amax(-1, keepdim=True)
>>> (m + (x - m).exp().sum(-1, keepdim=True).log()).squeeze(-1)
tensor([1.0013e+03, 6.9315e-01])          # 1001.3133 and 0.6931
>>> torch.logsumexp(x, -1)
tensor([1.0013e+03, 6.9315e-01])
```

## Traps

- `x.var()` divides by `N - 1` by default; normalization layers want `N`.
- `x.max(dim)` returns a tuple, so `x.max(-1) + 1` raises a confusing error.
  Use `.values`, or `amax`.
- `x.mean()` on an int64 tensor raises; cast first.
- Forgetting `keepdim=True` breaks the broadcast in the very next line.
- Averaging over a padded axis by dividing by `T` instead of by `mask.sum()` silently scales the loss.
- A masked max done by zeroing instead of filling with `-inf` returns 0 for all-negative rows.
- `0 * log(0)` is NaN, not 0.
- `bincount` without `minlength` returns a shorter tensor whenever the highest ids are absent from the batch, and refuses anything but a 1-D non-negative integer input.
- `argmax` with no `dim` flattens; the index it returns is into the flattened tensor.
- `.item()` on a reduction result forces a device synchronization, which is why training loops accumulate loss tensors and call `.item()` once rather than per step.

## Ready when

Without looking anything up you can:

- [ ] write the stable `logsumexp` from memory and explain why subtracting the max is exact
- [ ] compute a masked mean over a padded batch, including the shape of every intermediate
- [ ] say what `x.max(-1)` returns and how it differs from `x.amax(-1)`
- [ ] give the biased variance of `(B, T, D)` along the last axis with the right `correction`
- [ ] describe how gradient clipping computes one number out of a whole model's gradients
- [ ] compute entropy without producing a NaN on a zero probability

## Context



**Why logsumexp keeps coming back.**
It is the same trick three times over.
In topic 10 it is the softmax denominator, and cross-entropy is written as `logsumexp(logits) - logits[target]` precisely so the exponentials never have to be materialized.
In topic 18 it becomes *online* softmax: a running maximum and a running sum that can be updated as new blocks of scores arrive, which is what lets flash attention compute an exact softmax without ever holding the full score row in memory.
Learning to write it by hand here is what makes topic 18 readable.

**What `clip_grad_norm_` actually does.**
It computes the per-tensor norms, stacks them, takes the norm of that stack, and if the result exceeds the threshold it multiplies every gradient by `threshold / total_norm`.
The direction of the update is preserved and only its length is capped, which is why clipping stabilizes training without biasing it.
Topic 16 implements this.

## Going deeper

- [`torch.logsumexp`](https://pytorch.org/docs/stable/generated/torch.logsumexp.html) and the log-sum-exp trick, the single most reused idea in this repository.
- [`torch.nn.utils.clip_grad_norm_`](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html), whose source is short and worth reading before topic 16.
