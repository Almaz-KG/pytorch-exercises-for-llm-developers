# Topic 02 - Shapes and views

> **Exercise** `test_02_shapes_and_views.py`

A transformer is mostly reshaping.
The residual stream is `(B, T, D)`, attention wants `(B, H, T, Dh)`, the loss wants `(B*T, V)`, a KV cache grows along `T`, and grouped-query attention has to fan `H_kv` key heads out to `H` query heads.
None of that is arithmetic.
It is bookkeeping, and getting it wrong produces a model that trains to a mediocre loss instead of crashing.

The most common real bug in hand-written attention lives here: splitting heads in the wrong order mixes different heads' channels together while every shape assertion stays happy.
Topic 13 does the head split again from memory, topic 17 does the `cat` that grows a cache, topic 20 does the windowing that cuts training examples out of a token stream.

What you need from topic 01: storage, strides, and the difference between a copy and a view.

## The mental model

### Strides, and why `view` sometimes refuses

Row-major strides for a shape are the running products of the sizes to the right.
For `(2, 3, 4)` the stride along the last axis is 1, along the middle axis 4, along the first axis 12, so `x.stride() == (12, 4, 1)`.
Strides are counted in *elements*, not bytes.

A tensor is **contiguous** when its strides are exactly those row-major ones.
`transpose` and `permute` only swap entries in the shape and stride tuples, so they are free, and the result is almost never contiguous.

`view` requires that the requested shape be expressible with some stride pattern over the *existing* storage without moving data.
When it is not, `view` raises.
`reshape` is the forgiving version: a view when it can, a silent copy when it cannot.

```
x = torch.arange(6).reshape(2, 3)     # strides (3, 1), contiguous
y = x.t()                             # shape (3, 2), strides (1, 3), NOT contiguous
y.view(6)                             # RuntimeError
y.reshape(6)                          # works, copies
y.contiguous().view(6)                # works, the explicit form
```

Rule of thumb for this repository: `reshape` when you just want the shape, `contiguous().view(...)` when you want to be told that a copy is happening.

### 2.1 reshape, view, flatten

`-1` in a shape means "infer this axis from the total element count", and exactly one axis may be `-1`.
`x.reshape(-1, x.shape[-1])` flattens every leading axis and keeps the feature axis, which is what you do before a loss or a linear layer that does not care about batch structure.

`flatten(start_dim, end_dim)` collapses a contiguous run of axes and is the readable spelling of the same thing.
`unflatten(dim, sizes)` is its inverse.

`unsqueeze(dim)` inserts an axis of size 1 at position `dim`, and a negative `dim` counts from the end, where `-1` means "after the last axis".
`squeeze()` with no argument removes *every* axis of size 1, which is a trap when a batch of size 1 disappears along with the axis you meant to drop.
`squeeze(dim)` removes only that one, and quietly does nothing if that axis is not of size 1.

### 2.2 Splitting heads, and why the order matters

The whole of multi-head attention rests on this one rearrangement:

```
(B, T, D)  --reshape-->  (B, T, H, Dh)  --transpose(1,2)-->  (B, H, T, Dh)
```

with `D == H * Dh`.

The order is not interchangeable.
The last axis of size `D` is laid out as head 0's `Dh` channels, then head 1's, and so on, so splitting `D` into `(H, Dh)` keeps each head's channels together.
Reshape to `(B, T, Dh, H)` instead, or permute before reshaping, and channels of different heads land inside one head.
Every shape still matches and the model just learns worse.
Check yourself with a roundtrip: split, merge, compare against the original.

The inverse is the mirror image:

```
(B, H, T, Dh)  --transpose(1,2)-->  (B, T, H, Dh)  --reshape-->  (B, T, D)
```

After that transpose the tensor is not contiguous, which is exactly where `view` fails and `reshape` or `contiguous()` is required.

`transpose(-2, -1)` on `(B, H, T, Dh)` gives `(B, H, Dh, T)`, the $K^\top$ of the attention score computation.
Negative indices mean the same code works for `(T, Dh)` and `(B, H, T, Dh)` alike.

### 2.3 Joining and cutting

`torch.cat(tensors, dim)` joins along an existing axis and keeps the rank.
`torch.stack(tensors, dim)` creates a new axis and raises the rank by one.
Appending a new token's keys to a KV cache is `cat` along the time axis.
Collecting a list of per-example tensors into a batch is `stack`.

Cutting is `torch.split(x, size_or_sizes, dim)` or `torch.chunk(x, n, dim)`.
`chunk(3, dim=-1)` on a last axis of size `3*D` gives the three equal pieces of a fused QKV projection.
`split` takes either a single size, repeated, or an explicit list of sizes that must sum to the axis length.
Both return tuples of views, not copies.

### 2.4 expand versus repeat

`expand` sets the stride of a size-1 axis to **zero** and reports a larger size.
No data is copied, and every "row" of the result is the same memory read again.
It only works on axes that are currently of size 1.

`repeat` actually allocates and copies, tiling the tensor a given number of times per axis.

```
x = torch.arange(3).reshape(1, 3)
x.expand(4, 3).stride()    # (0, 1)   -> one row of storage, read four times
x.repeat(4, 1).stride()    # (3, 1)   -> twelve elements in memory
```

A zero stride means the result is not contiguous and cannot be written into safely, and a later `reshape` on it will have to copy.
That is fine and expected: the point is that the *intermediate* stays free.

Where this matters is grouped-query attention.
With `H` query heads and `H_kv` key/value heads, each KV head serves `G = H / H_kv` query heads, and the KV tensor has to be widened from `(B, H_kv, T, Dh)` to `(B, H_kv * G, T, Dh)` with each KV head repeated `G` times **consecutively**.
Insert an axis, expand along it, then collapse:

```
(B, Hkv, T, Dh) -> unsqueeze -> (B, Hkv, 1, T, Dh) -> expand -> (B, Hkv, G, T, Dh) -> reshape -> (B, Hkv*G, T, Dh)
```

Doing it with `repeat` on the head axis instead gives the order `[h0, h1, h0, h1]` rather than `[h0, h0, h1, h1]`, which pairs every query head with the wrong keys.

### 2.5 Windows and padding

`Tensor.unfold(dim, size, step)` produces all sliding windows along an axis as a **view**, appending a new last axis of length `size`.
A 1-D tensor of length `T` unfolded with `size=w, step=1` becomes `(T - w + 1, w)`.
Because it is a view it costs nothing until you materialize it.

`F.pad(x, pad, mode="constant", value=0)` takes its padding spec **from the last axis backwards**, in pairs of `(before, after)`.
For a 2-D `(B, T)` tensor, `F.pad(x, (0, n))` adds `n` columns on the right of the time axis, while `F.pad(x, (0, 0, 0, n))` would add `n` rows to the batch axis instead.
This reversed order is the single most misremembered signature in PyTorch.

Shifting right by one with a fill on the left is padding plus a slice: pad one on the left, then drop the last column.

## API you need

### Reshaping

| Call | Result |
| --- | --- |
| `x.reshape(*shape)` | view if possible, copy otherwise |
| `x.view(*shape)` | view only, raises if impossible |
| `x.flatten(start_dim=0, end_dim=-1)` | collapse a run of axes |
| `x.unflatten(dim, sizes)` | split one axis into several |
| `x.unsqueeze(dim)` / `x[None]` / `x[:, None]` | insert an axis of size 1 |
| `x.squeeze()` / `x.squeeze(dim)` | drop all size-1 axes / one specific axis |
| `x.contiguous()` | row-major copy if needed |

### Axis order

| Call | Result |
| --- | --- |
| `x.transpose(d0, d1)` | swap two axes, a view |
| `x.permute(*dims)` | arbitrary reordering, a view |
| `x.movedim(src, dst)` | move one axis, leaving the rest in order |
| `x.mT` | transpose of the last two axes |
| `x.stride()`, `x.is_contiguous()` | the diagnostics |

### Joining and cutting

| Call | Result |
| --- | --- |
| `torch.cat(seq, dim=0)` | join along an existing axis |
| `torch.stack(seq, dim=0)` | join along a new axis |
| `torch.split(x, size_or_sizes, dim=0)` | tuple of views |
| `torch.chunk(x, chunks, dim=0)` | tuple of roughly equal views |
| `torch.unbind(x, dim=0)` | tuple of slices with that axis removed |

### Expanding

| Call | Result |
| --- | --- |
| `x.expand(*shape)` | zero-stride view, size-1 axes only, `-1` means "keep" |
| `x.expand_as(other)` | the same, shape taken from `other` |
| `x.repeat(*reps)` | tile and copy, `reps` is per-axis repeat counts |
| `torch.repeat_interleave(x, n, dim)` | repeat each slice `n` times consecutively, copies |
| `torch.broadcast_to(x, shape)` | the functional spelling of `expand` |

### Windows and padding

| Call | Result |
| --- | --- |
| `x.unfold(dim, size, step)` | sliding windows as a view, new last axis |
| `F.pad(x, pad, value=...)` | pad, spec runs last axis first |
| `torch.roll(x, shifts, dims)` | circular shift, wraps around rather than filling |

## Traps

- `view` after `transpose` or `permute` raises `RuntimeError: view size is not compatible with input tensor's size and stride`.
  That message always means "insert a `.contiguous()` or use `reshape`".
- Splitting heads with a single `permute`, or in the wrong order, silently mixes channels across heads.
  Test it with a roundtrip, not with an assertion on the shape.
- `squeeze()` without an argument removes every size-1 axis, including a batch of size 1.
- `F.pad` pairs run from the last axis backwards.
- `expand` only works on axes of size 1, and gives a non-writable, non-contiguous result.
- `repeat` on the head axis of a GQA KV tensor produces the wrong head order.
  Interleaving is what you want, not tiling.
- `torch.split(x, 3)` splits into pieces of size 3, while `torch.chunk(x, 3)` splits into 3 pieces.
  The argument means opposite things.
- `unfold` returns a view with overlapping windows, so writing into it aliases.

## Ready when

Without looking anything up you can:

- [ ] write the split-heads and merge-heads rearrangements and explain why the transpose comes after the reshape
- [ ] say which of `view`, `reshape`, `expand`, `repeat`, `transpose` copy and which do not
- [ ] give `x.stride()` for a contiguous `(2, 3, 4)` tensor and for its transpose
- [ ] pad the time axis of a `(B, T)` tensor on the right with one `F.pad` call, and say why the pair is `(0, n)`
- [ ] explain why GQA uses unsqueeze/expand/reshape rather than `repeat`

## Context



**Why fused QKV exists.**
One matmul of width `3*D` is meaningfully faster than three of width `D`, because each kernel launch has a fixed cost and the larger matrix uses the hardware better.
So real implementations keep a single `(D, 3*D)` weight, project once, and cut the result into three.
`split_qkv` is that cut.
This repository's later topics keep `q_proj`, `k_proj` and `v_proj` separate instead, because the CS336 naming contract wants them separate and because three small matmuls are easier to read.

**Why GQA expands rather than interleaves.**
`torch.repeat_interleave` produces the correct head order in one call, and it copies.
The unsqueeze/expand/reshape form keeps the intermediate free and lets the final `reshape` be the only allocation, which at generation time runs once per token per layer.
Both are correct; the second is what production code uses.

## Going deeper

- [`torch.Tensor.unfold`](https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html), which generalizes to the `im2col` trick in convolutions.
- The GQA paper, [Ainslie et al. 2023](https://arxiv.org/abs/2305.13245), for why the KV heads are shared at all.
