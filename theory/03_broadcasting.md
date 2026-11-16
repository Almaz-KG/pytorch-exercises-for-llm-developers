# Topic 03 - Broadcasting

> **Exercise** `test_03_broadcasting.py`

Broadcasting is the reason transformer code has no loops in it.
Adding a bias, applying a causal mask, scaling each head differently, building ALiBi, blocking attention to padding: every one of those is a shape alignment followed by an elementwise operation.
The rule is short enough to memorize in a minute, and the bugs it causes are the expensive kind, because a wrong broadcast usually produces a tensor of the wrong shape that still flows through the rest of the model.

## The mental model

### The one rule

Align the two shapes **from the right**.
Missing leading axes on the shorter shape count as 1.
Two axes are compatible when they are equal, or when at least one of them is 1.
The output size on an axis is the larger of the two.
If any aligned pair is neither equal nor contains a 1, the operation raises.

```
    (B, H, T, T)          scores
          (T, T)          mask, right-aligned
--------------------
    (B, H, T, T)          result, mask reused for every batch and head
```

```
        (3, 1)
           (3,)  ->  right-aligned as (1, 3)
--------------
        (3, 3)
```

That second example is the trap the exercise names explicitly.
A column vector plus a row vector is a matrix, not a vector.
It is the mechanism behind outer products and pairwise distances, and it is also the mechanism behind a loss that quietly becomes `(N, N)` instead of `(N,)` and averages to something plausible but wrong.

### 3.1 Doing the rule by hand

Pad the shorter shape on the *left* with ones until the two have equal length, then walk the pairs.
`torch.broadcast_shapes(*shapes)` is the built-in that does this, and `torch.broadcast_tensors(a, b)` returns actual expanded views, both useful for checking yourself afterwards.

Broadcasting never allocates the expanded tensor in an elementwise operation.
It is implemented as a zero stride, exactly like `expand` in topic 02.

### 3.2 The everyday uses

**Adding a bias.**
`(B, T, D) + (D,)` right-aligns to `(B, T, D)` with no help.
This is the only case that needs no `unsqueeze` at all, and it works because the feature axis is last.

**Per-head scaling.**
`(B, H, T, Dh) * (H,)` right-aligns `(H,)` against the *last* axis, `Dh`, which is wrong and usually raises.
The fix is to shape the scales as `(1, H, 1, 1)`, or equivalently `scales[None, :, None, None]`.
Whenever a vector indexes an axis that is not the last one, you have to say so with explicit size-1 axes.

**keepdim.**
Reductions drop the reduced axis by default, so `x.mean(-1)` on `(B, T, D)` gives `(B, T)`, which right-aligns against `D` and does the wrong thing.
`x.mean(-1, keepdim=True)` gives `(B, T, 1)`, which broadcasts back against the original correctly.
This is why `keepdim=True` appears in every normalization implementation you will ever read.

**Weighted sums.**
Weights `(B, T)` against values `(B, T, D)` need an axis: `(B, T, 1) * (B, T, D)`, then a sum over `T`.
Recognizing that shape is most of understanding attention.

### 3.3 Making matrices out of vectors

An outer product is `a[:, None] * b[None, :]`: `(N, 1) * (1, M) -> (N, M)`.

Squared pairwise distances follow the same skeleton with a reduction on the end.
`a` is `(N, D)` and `b` is `(M, D)`, so `a[:, None, :] - b[None, :, :]` is `(N, M, D)`, and summing the squares over the last axis gives `(N, M)`.

A one-hot encoding is a comparison against `arange`.
`ids[..., None] == torch.arange(V)` right-aligns `(..., 1)` against `(V,)` and gives `(..., V)` of bools, which a cast turns into floats.

### 3.4 Masks and relative positions

A **causal mask** compares two aranges shaped as a column and a row.
With $i$ the query position and $j$ the key position, "query $i$ may attend to key $j$" is $j \le i$, which is `torch.arange(T)[:, None] >= torch.arange(T)[None, :]`.
`torch.tril(torch.ones(T, T, dtype=torch.bool))` is the same thing.
The comparison form is preferred here because it generalizes directly to windows and to `q_len != k_len`.

**Relative positions** are $i - j$, so `arange(q_len)[:, None] - arange(k_len)[None, :]`.
Positive below the diagonal, zero on it, negative above.

**ALiBi** turns that into an additive bias with a per-head slope: `bias[h, i, j] = -slope[h] * (i - j)`.
Shapes: `(H, 1, 1) * (1, T, T) -> (H, T, T)`.
The effect is a linear penalty that grows with distance, different per head.

**Applying a mask** to scores is `scores.masked_fill(~mask, float("-inf"))`, where `mask` is True at *allowed* positions.
Two things matter.
Forbidden positions must become a large negative value rather than zero, because the mask is applied to the scores *before* the softmax, and $e^{-\infty} = 0$ is what actually removes them.
Setting them to zero instead gives them weight $e^{0} = 1$, the opposite of masking.
And `masked_fill` takes a mask of positions to *fill*, so a boolean that marks what is allowed has to be inverted.

**Padding masks** refer to keys, and keys live on the last axis of `(B, H, T_q, T_k)`.
A `(B, T)` padding mask therefore has to become `(B, 1, 1, T)`, so that it broadcasts across heads and across every query.
Written as `(B, 1, T, 1)` it masks padded *queries* instead, which produces rows of all `-inf` and a softmax full of NaN rather than the silent wrongness of the other mistake.

## API you need

### Shape alignment

| Call | Result |
| --- | --- |
| `torch.broadcast_shapes(*shapes)` | the resulting shape, raises on incompatibility |
| `torch.broadcast_tensors(a, b, ...)` | expanded views of all inputs |
| `torch.broadcast_to(x, shape)` | a single expanded view |
| `x[:, None]`, `x[None, :]`, `x[..., None]` | insert size-1 axes inline |
| `x.unsqueeze(dim)`, `x.view(...)` | the explicit spellings |

`x[..., None]` is the safe way to append an axis to a tensor of unknown rank.

### Masking

| Call | Result |
| --- | --- |
| `x.masked_fill(mask, value)` | copy with `value` where `mask` is True |
| `x.masked_fill_(mask, value)` | the in-place version |
| `torch.where(cond, a, b)` | elementwise select, broadcasts all three |
| `torch.tril(x, diagonal=0)` / `torch.triu(x)` | lower / upper triangle, rest zeroed |
| `torch.ones(T, T, dtype=torch.bool).tril()` | the canonical causal mask, the non-comparison way |
| `float("-inf")`, `torch.finfo(dtype).min` | the two usual mask values |

`torch.finfo(x.dtype).min` is what you use instead of `-inf` when the tensor is float16 and a later operation might produce `inf - inf`.

### Comparisons

| Call | Result |
| --- | --- |
| `a == b`, `a < b`, `a >= b` | elementwise bool tensors, broadcasting |
| `~mask`, `mask & other`, `mask \| other` | boolean logic, elementwise |
| `mask.any(dim)`, `mask.all(dim)` | boolean reductions |

Use `&` and `|`, not `and` and `or`; the Python keywords try to convert the whole tensor to a single bool and raise.

### A worked example on neutral data

```python
>>> a = torch.tensor([1.0, 2.0, 3.0])
>>> a[:, None].shape, a[None, :].shape
(torch.Size([3, 1]), torch.Size([1, 3]))
>>> (a[:, None] * a[None, :]).shape
torch.Size([3, 3])
>>> scores = torch.zeros(2, 4, 5, 5)          # (B, H, T, T)
>>> keep = torch.ones(5, 5, dtype=torch.bool).tril()
>>> scores.masked_fill(~keep, float("-inf")).shape
torch.Size([2, 4, 5, 5])
```

## Traps

- `(3, 1)` plus `(3,)` is `(3, 3)`.
  Whenever a result is unexpectedly square, this is why.
- A reduction without `keepdim=True` breaks the alignment for the subtraction that follows it.
- A `(H,)` vector multiplied into `(B, H, T, Dh)` aligns against `Dh`, not `H`.
- A padding mask shaped `(B, 1, T, 1)` masks queries instead of keys and produces NaN after the softmax.
- Masking with `0` instead of `-inf` does nothing, because the mask goes on before the softmax.
- `masked_fill` fills where the mask is True, so an "allowed" mask needs inverting.
- `and` / `or` / `not` on tensors raise; use `&` / `|` / `~`.
- In float16, `-inf` in every position of a row gives NaN after softmax.
  Use `torch.finfo(dtype).min` when that is possible.
- Building `(N, M, D)` for pairwise distances is quadratic in memory.
  Fine at test sizes, not fine at real ones.

## Ready when

Without looking anything up you can:

- [ ] state the broadcasting rule in one sentence and apply it to `(4, 1, 6)` against `(3, 6)`
- [ ] say why `x - x.mean(-1)` is wrong and `x - x.mean(-1, keepdim=True)` is right
- [ ] shape a per-head vector so it multiplies into `(B, H, T, Dh)` correctly
- [ ] write a causal mask two different ways, and say which axis is queries and which is keys
- [ ] explain why a key padding mask becomes `(B, 1, 1, T)`

## Context



**Pairwise distances at scale.**
The `(N, M, D)` intermediate this topic builds is the clear way to write the formula and the wrong way to run it.
Production code expands the square into $\lVert a \rVert^2 - 2 a \cdot b + \lVert b \rVert^2$, which turns the whole thing into one matmul plus two cheap row norms and never materializes the third axis.
Topic 06 does that expansion.

**Why anyone wanted ALiBi.**
A model trained with absolute position embeddings has never seen position 5000 and behaves unpredictably there.
ALiBi encodes only *distance*, as a penalty that grows linearly and is fixed rather than learned, so a model trained at 2k can be run at 8k and degrade gracefully instead of falling apart.
It was one of the first credible answers to the long-context problem and lost to RoPE, which topic 12 builds, mostly because RoPE extrapolates about as well and composes better with the rest of the architecture.

## Going deeper

- [Broadcasting semantics](https://pytorch.org/docs/stable/notes/broadcasting.html) in the PyTorch docs, which is the normative statement of the rule.
