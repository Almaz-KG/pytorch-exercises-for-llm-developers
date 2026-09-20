# Topic 11 - Normalization

> **Exercise** `test_11_normalization.py`

Normalization is what makes deep transformers trainable at all.
It is also three lines of code, which is why it is so easy to get subtly wrong: the unbiased variance instead of the biased one, the statistics computed in bfloat16, the norm on the wrong side of the residual connection. Each of those produces a model that trains, and trains worse.

Two normalization layers in every block, one before attention and one before the feed-forward branch, plus one more at the very end of the model.

Knowing why RMSNorm replaced LayerNorm and why pre-norm replaced post-norm is also the shortest possible summary of what changed between the 2017 transformer and a modern LLM.

## The mental model

### 11.1 The two formulas

**LayerNorm**, over the last axis:

```
mu    = mean(x, dim=-1, keepdim=True)
var   = mean((x - mu)^2, dim=-1, keepdim=True)        # biased, denominator D
y     = (x - mu) / sqrt(var + eps) * weight + bias
```

**RMSNorm**, over the last axis:

```
rms   = sqrt(mean(x^2, dim=-1, keepdim=True) + eps)
y     = x / rms * weight
```

The differences are exactly two: RMSNorm does not subtract the mean, and it has no bias.
Everything else is identical.

`weight` and `bias` are `(D,)`, one per channel, broadcasting against `(B, T, D)` by the rules of topic 03.
`weight` is initialized to ones and `bias` to zeros, so a freshly initialized norm is the pure normalization.

Three details the tests will catch:

- The variance is **biased**, denominator `D`, not `D - 1`.
- `eps` goes **inside** the square root, added to the variance: `sqrt(var + eps)`, never `sqrt(var) + eps`.
  The two differ by an amount that matters when `var` is tiny, which is exactly the case `eps` exists for.
- The statistics are over the **last axis only**, per token and per example.
  That is what makes it *layer* norm rather than batch norm, and it is why it behaves identically at batch size 1 and needs no running statistics at inference.

RMSNorm is cheaper: one reduction instead of two, no subtraction, and `D` fewer parameters per layer.
It matches LayerNorm's quality empirically, which is why LLaMA, Mistral, Qwen and essentially every model since 2022 use it.

### Precision, and why every real implementation upcasts

This is the part that is not in the formula and is in every production codebase.

When the model runs in bfloat16, `x` has about 7 mantissa bits, so two to three decimal digits.
Squaring such values and summing `D` of them accumulates error fast, and that sum of squares is exactly the quantity the whole operation depends on.
The result is a variance estimate off by a percent or more, which is noise injected into every layer.

So the pattern is universal:

```
in_dtype = x.dtype
x = x.to(torch.float32)
... compute the normalization in float32 ...
return result.to(in_dtype)
```

The `weight` multiplication may happen in either precision; what matters is that the reduction and the reciprocal square root happen in float32.
Returning in the original dtype is required, because the rest of the model is running in bfloat16 and a float32 tensor would silently promote everything downstream.

### 11.2 The modules

Both are minimal `nn.Module` subclasses from topic 09: a `weight` parameter of `(d_model,)` initialized to ones, plus a `bias` of zeros for LayerNorm, an `eps` stored as a plain float attribute, and a `forward` that calls the functional version.
No buffers.

The parameter names `weight` and `bias` are load-bearing, because tests in later topics assert on `state_dict` keys.

`eps` as a plain Python float is correct: it is a configuration constant, not a tensor, so it is neither a parameter nor a buffer.

### 11.3 Pre-norm versus post-norm

**Post-norm**, the original 2017 transformer:

```
x = norm(x + sublayer(x))
```

**Pre-norm**, everything since roughly 2019:

```
x = x + sublayer(norm(x))
```

The difference is not cosmetic.

In pre-norm there is a path from the input of the network to its output that passes through **only additions**.
The residual stream is never normalized, never scaled, never squashed.
During the backward pass the gradient flows along that path with a Jacobian of exactly the identity, so it arrives at layer 1 with the same magnitude it had at layer 48.
Deep pre-norm models train stably from scratch, often with little or no learning rate warmup.

In post-norm every residual addition is immediately followed by a normalization, so the gradient is rescaled at every layer on the way back.
Those factors compound multiplicatively across depth, and past a couple of dozen layers training becomes fragile.

The one thing pre-norm needs that post-norm does not is a **final normalization** after the last block, `ln_final`, because the residual stream reaching the output head was never normalized on the way.
That is why a pre-norm model has `2L + 1` norms rather than `2L`.

### 11.4 Properties worth feeling

**RMSNorm is scale invariant.**
`rms_norm(c * x) == rms_norm(x)` for any `c > 0`, because both the numerator and the RMS scale linearly in `c` and they cancel.
The `eps` breaks this exactly, so the measured gap is small but not zero.

That invariance is the mechanism by which normalization stabilizes training: whatever magnitude the previous layer happened to produce, the next layer sees something of controlled scale.
It also means the *direction* of the activation vector carries all the information and its length carries none, which is a useful way to think about what the residual stream is doing.

LayerNorm is invariant to both scale and shift, since it removes the mean too.

**After LayerNorm with weight 1 and bias 0**, the last axis has mean exactly 0 and biased standard deviation exactly 1, up to the effect of `eps`.
RMSNorm gives a root mean square of 1 but generally not a mean of 0, which is the observable difference between the two.

## API you need

| Call | Result |
| --- | --- |
| `F.layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5)` | the reference LayerNorm; `normalized_shape` is a tuple like `(D,)` |
| `F.rms_norm(x, normalized_shape, weight=None, eps=None)` | available in PyTorch 2.4+ |
| `nn.LayerNorm(d, eps=1e-5, elementwise_affine=True)` | the module |
| `nn.RMSNorm(d, eps=None)` | the module, 2.4+ |
| `x.mean(-1, keepdim=True)` | the centering statistic |
| `x.var(-1, correction=0, keepdim=True)` | the biased variance |
| `torch.var_mean(x, -1, correction=0, keepdim=True)` | both in one pass |
| `x.pow(2).mean(-1, keepdim=True)` | the RMS statistic |
| `torch.rsqrt(t)` | `1 / sqrt(t)`, one op instead of two |
| `x.to(torch.float32)` / `.to(in_dtype)` | the upcast sandwich |

### A worked example on neutral data

```python
>>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
>>> y = F.layer_norm(x, (4,))
>>> y.mean(-1), y.var(-1, correction=0)
(tensor([0.]), tensor([1.0000]))

>>> r = x / x.pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
>>> r.pow(2).mean(-1)                    # RMS is 1 ...
tensor([1.0000])
>>> r.mean(-1)                           # ... but the mean is not 0
tensor([0.9129])
```

## Traps

- `x.var()` defaults to `correction=1`.
  Normalization wants `correction=0`.
- `sqrt(var) + eps` is not `sqrt(var + eps)`.
- Computing the statistics in bfloat16 gives a visibly noisy variance.
  Upcast, then cast the result back.
- Returning float32 from a norm inside a bfloat16 model promotes everything downstream and silently doubles memory.
- Normalizing over the wrong axis.
  It is always the last one here, per token.
- Forgetting `keepdim=True` breaks the broadcast in the subtraction or the division.
- Pre-norm without a final `ln_final` leaves the output head reading an unnormalized residual stream.
- RMSNorm has no bias.
  Adding one is not RMSNorm and will fail a `state_dict` key assertion later.

## Ready when

Without looking anything up you can:

- [ ] write LayerNorm and RMSNorm from memory, with `eps` in the right place
- [ ] say which variance convention normalization uses and why the default is wrong
- [ ] explain the upcast-to-float32 pattern and what goes wrong without it
- [ ] argue why pre-norm trains more stably than post-norm, in terms of the gradient path
- [ ] say why a pre-norm model needs one extra norm at the end
- [ ] state what is invariant under RMSNorm and what is additionally invariant under LayerNorm

## Context



**What the re-centering was doing, apparently nothing.**
LayerNorm was introduced with two justifications, re-centering and re-scaling, and RMSNorm was the experiment of dropping the first.
Quality held, which is the empirical finding; the interpretation people settled on is that the mean of a high-dimensional activation vector carries almost no information, so removing it removes almost nothing.
It took three years from the RMSNorm paper in 2019 to it being the default, which is roughly the field's normal latency for a change that is cheap and boring.

**How post-norm was diagnosed.**
"On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020) showed analytically that in post-norm the expected gradient at initialization scales with depth, and that warmup exists mainly to survive the early steps where that is worst.
The same paper showed pre-norm has depth-independent gradients and can drop warmup entirely.
In practice large models keep a short warmup anyway, for reasons more to do with the optimizer's moment estimates than with the norms.

**Why autocast will not do the upcast for you, except that it does.**
`torch.autocast` keeps normalization layers in float32 deliberately, which is a strong hint about how much the precision matters.
That only covers the library's own `F.layer_norm` and `nn.LayerNorm`; a hand-written norm inside an autocast region gets no such protection, and must do its own upcasting.
Topic 18 covers what autocast does and does not touch.

## Going deeper

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), Zhang and Sennrich 2019.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), Xiong et al. 2020, which is the pre-norm argument in full.
