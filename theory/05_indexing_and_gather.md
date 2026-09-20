# Topic 05 - Indexing and gather

> **Exercise** `test_05_indexing_and_gather.py`

An embedding layer is an indexing operation, so the first thing the model does is this topic.
Cross-entropy is a `gather`, so the last thing it does is too.
In between, top-k sampling is a `scatter`, beam search reorders a KV cache by index every step, and packing a padded batch before the loss is a boolean mask.

Indexing is the one part of tensor programming with no analogue in ordinary array code, because PyTorch has three separate systems for it that look alike and behave differently.
Knowing which of the three you are in is most of the skill.

## The mental model

### The three systems

**Basic indexing** uses integers and slices, and always returns a **view**.
`x[0]`, `x[1:3]`, `x[:, 2]`, `x[..., -1]`.
Shares storage, costs nothing.

**Advanced (fancy) indexing** uses tensors or lists of integers, and always returns a **copy**.
`x[torch.tensor([0, 2, 0])]`, `x[ids]`, `x[rows, cols]`.
The result's shape is built from the shape of the index tensors, not from the shape of `x`.

**Boolean masking** uses a bool tensor and returns a copy whose size is not known until runtime.
`x[mask]`.

Mixing basic and advanced in one subscript is legal, and is where the rules stop being intuitive.

### 5.1 Indexing as an embedding lookup

The single most important fact in this file: if `table` is `(V, D)` and `ids` is any integer tensor of shape `S`, then `table[ids]` has shape `S + (D,)`.

```
table (V, D) indexed by ids (B, T)   ->   (B, T, D)
```

The index tensor's shape replaces the axis being indexed, and the remaining axes of `table` are appended.
That is literally all `nn.Embedding` does in its forward pass, and it works for `ids` of any rank.

**Multiple index tensors** index different axes *in parallel*, not as a cross product.
Given `x` of `(B, T, D)` and two 1-D index tensors `bi` and `pi` of the same length `N`, `x[bi, pi]` has shape `(N, D)` and element `n` is `x[bi[n], pi[n]]`.
The index tensors are broadcast against each other first, so `x[bi[:, None], pi[None, :]]` would instead give the full `(len(bi), len(pi), D)` cross product.
Choosing between those two by shaping the indices is the technique worth internalizing.

Negative indices work and wrap from the end, which is convenient and is also how an off-by-one turns into wrong data instead of an exception.

### 5.2 gather

`torch.gather(input, dim, index)` answers a question that fancy indexing answers clumsily: pick one element along `dim` for every position of every other axis.

The contract is strict:

- `index` must have the **same number of dimensions** as `input`.
- `index` must have the same size as `input` on every axis **except** `dim`, where it may be anything.
- The output has exactly the shape of `index`.
- `out[i][j][k] = input[i][index[i][j][k]][k]` when `dim == 1`, and analogously for other `dim`.

The idiom for cross-entropy: `logits` is `(B, T, V)`, `targets` is `(B, T)`, and you want `(B, T)` holding the logit of the correct token.
`targets` is one axis short, so it becomes `(B, T, 1)`, `gather(-1, ...)` returns `(B, T, 1)`, and a final squeeze gives `(B, T)`.
Unsqueeze, gather, squeeze: memorize that as one unit.

**Selecting the last real token** of each padded sequence is the same pattern with `lengths - 1` as the index, expanded to `(B, 1, D)` to gather along the time axis of `(B, T, D)`.
The alternative spelling with advanced indexing is `h[torch.arange(B), lengths - 1]`, which is shorter and equally correct.
Both are worth having.

**Reordering rows** by an index vector is plain advanced indexing: `x[idx]` where `idx` is `(B,)`.
`torch.index_select(x, dim, idx)` is the explicit function form and requires `idx` to be 1-D.

### 5.3 scatter

`scatter_` is the inverse of `gather`: instead of reading at the indices, it writes at them.

`dst.scatter_(dim, index, src)` sets `dst[i][index[i][j][k]][k] = src[i][j][k]` for `dim == 1`.
`src` may be a tensor shaped like `index`, or a constant through the `value=` form.
Like `gather`, `index` must have the same rank as `dst`.

A one-hot encoding is a scatter of `1.0` into a zero tensor at the positions given by the ids: start from `zeros(B, T, V)`, index `ids.unsqueeze(-1)` of `(B, T, 1)`, scatter along `-1`.

`scatter_add_` accumulates instead of overwriting, which is what you need when several indices collide.
A bag-of-words count is `scatter_add_` of ones into a `(B, V)` zero tensor at the ids, and duplicates are exactly the point: a token appearing three times should add three, where a plain `scatter_` would leave 1.

`index_add_(dim, index, source)` is the same accumulation with a 1-D index, and is the call that shows up in Mixture-of-Experts when each expert's outputs are added back into a shared buffer.

**Top-k filtering** combines the two directions.
Take `topk` to get values and indices, create a tensor filled with `-inf`, and scatter the kept values back into it at their indices.
Building a boolean keep-mask and `masked_fill`ing its complement gives the same answer and is often clearer.

All of `scatter_`, `scatter_add_`, `index_add_` are in-place and return the modified tensor, so they chain as `torch.zeros(...).scatter_(...)`.

### 5.4 Boolean masks

`x[mask]` where `mask` is bool selects the positions where `mask` is True and flattens them.
When `mask` has fewer axes than `x`, it applies to the leading axes: for `x` of `(B, T, D)` and `mask` of `(B, T)`, `x[mask]` is `(N, D)` where `N == mask.sum()`.
That is packing, which is how you drop padding before a loss so that no compute or gradient is wasted on it.

`torch.nonzero(cond, as_tuple=True)` gives the *coordinates* where a condition holds rather than the values.
For a 1-D tensor, `cond.nonzero(as_tuple=True)[0]` is the list of positions.

`torch.where(cond, a, b)` is the non-destructive elementwise select, as opposed to `masked_fill_` which writes in place.

**Masking everything after the first EOS** is the cumsum trick, and it generalizes widely.
`is_eos` is a bool tensor, and `is_eos.cumsum(dim)` counts how many EOS tokens have been seen up to and including each position.
That count is 0 before the first EOS, 1 at the EOS itself, and grows after.
Keeping positions where the count is `< 1` excludes the EOS.
An exclusive cumsum is the inclusive one minus the current element, so `cumsum - is_eos` is "how many EOS strictly before here", and comparing that to zero keeps the EOS and drops everything after.
Working this out on a length-5 toy row on paper is faster than guessing.

### 5.5 Training pairs

Next-token prediction turns a sequence of `T + 1` ids into inputs `[0 : T]` and targets `[1 : T+1]`.
Two slices of the same tensor, offset by one, and both are views so it costs nothing.
Position `i` of the input predicts position `i` of the target, which is the original token `i + 1`.

**Expanding for beams** widens `(B, T)` to `(B * num_beams, T)` with each example's beams *adjacent*.
`repeat_interleave(num_beams, dim=0)` gives `[a, a, b, b]`; `repeat(num_beams, 1)` gives `[a, b, a, b]`.
This is the same ordering trap as GQA head expansion in topic 02, and it is worth noticing that it is the same trap.

## API you need

### Indexing

| Call | Result |
| --- | --- |
| `x[i]`, `x[a:b]`, `x[..., -1]` | basic, a view |
| `table[ids]` | advanced, a copy, shape `ids.shape + table.shape[1:]` |
| `x[idx_a, idx_b]` | parallel advanced indexing, indices broadcast together |
| `x[mask]` | boolean selection, flattens to `(N, ...)` |
| `torch.index_select(x, dim, idx)` | explicit form, `idx` must be 1-D |
| `torch.take_along_dim(x, idx, dim)` | like `gather` but broadcasts the index |

### gather and scatter

| Call | Result |
| --- | --- |
| `torch.gather(x, dim, index)` | read along `dim`, output shaped like `index` |
| `dst.scatter_(dim, index, src)` | write along `dim` |
| `dst.scatter_(dim, index, value=v)` | write a constant |
| `dst.scatter_add_(dim, index, src)` | accumulate along `dim` |
| `dst.index_add_(dim, index, source)` | accumulate with a 1-D index |
| `dst.index_copy_(dim, index, source)` | overwrite with a 1-D index |
| `dst.index_fill_(dim, index, value)` | fill with a 1-D index |

`gather` and `scatter_` both require `index.ndim == input.ndim`.
That single requirement is behind most of the errors this file produces.

### Conditions

| Call | Result |
| --- | --- |
| `torch.where(cond, a, b)` | elementwise select, returns a copy |
| `x.masked_fill(mask, value)` | copy with fills |
| `x.masked_fill_(mask, value)` | in-place |
| `torch.nonzero(cond, as_tuple=True)` | tuple of coordinate tensors |
| `cond.cumsum(dim)` | running count of True values |
| `F.one_hot(ids, num_classes)` | the library one-hot, int64; the exercise wants it by hand |
| `torch.repeat_interleave(x, n, dim)` | each slice repeated `n` times consecutively |

### A worked example

```python
>>> table = torch.arange(12.0).reshape(4, 3)     # (V=4, D=3)
>>> ids = torch.tensor([[0, 2], [3, 0]])         # (B=2, T=2)
>>> table[ids].shape
torch.Size([2, 2, 3])

>>> x = torch.tensor([[10., 11., 12.], [20., 21., 22.]])
>>> idx = torch.tensor([[2], [0]])
>>> torch.gather(x, 1, idx)                      # output shaped like idx
tensor([[12.],
        [20.]])

>>> torch.zeros(2, 4).scatter_add_(1, torch.tensor([[0, 0, 1], [3, 3, 3]]), torch.ones(2, 3))
tensor([[2., 1., 0., 0.],
        [0., 0., 0., 3.]])
```

## Traps

- `gather` and `scatter_` raise unless `index` has the same rank as the input.
  `unsqueeze(-1)` before, `squeeze(-1)` after.
- `scatter_` overwrites on duplicate indices and the winner is unspecified; use `scatter_add_` when duplicates carry meaning.
- Fancy indexing returns a copy, so `x[idx] += 1` updates a temporary and does nothing to `x`.
  Use `index_add_` for that.
- `x[a, b]` with two index tensors pairs them elementwise.
  If you wanted a cross product, shape them as `(N, 1)` and `(1, M)`.
- `x[mask]` flattens, so all shape information beyond the masked axes is gone.
- `x[mask]` has a data-dependent size and forces a synchronization.
- `repeat` and `repeat_interleave` differ in ordering, and beam search only works with the latter.
- Negative indices silently wrap, so an off-by-one reads the wrong end of the sequence rather than raising.
- `index_select` refuses a multi-dimensional index; `gather` refuses a 1-D one.
  They are not interchangeable.

## Ready when

Without looking anything up you can:

- [ ] give the output shape of `table[ids]` for any `table` and `ids`
- [ ] write the unsqueeze/gather/squeeze idiom that pulls target logits out of `(B, T, V)`
- [ ] say what `scatter_add_` does that `scatter_` does not, and when the difference matters
- [ ] explain what `x[mask]` returns for `x` of `(B, T, D)` and `mask` of `(B, T)`
- [ ] use a cumsum to mask everything after the first occurrence of a token
- [ ] say why beam expansion uses `repeat_interleave`

## Context



**Packing versus keeping the padding.**
`select_valid` produces a tensor whose length depends on the data, which forces a device synchronization and rules the operation out of any graph compiled with static shapes.
Production code therefore often keeps the padded shape and zeroes the loss at padded positions instead, paying for compute it throws away in exchange for shapes that never change.
Which one wins depends on how much padding there is, and the modern answer for training is a third option: pack multiple documents into one full-length sequence so there is no padding to skip.
Topic 20 builds that.

**Why the beam ordering is not a matter of taste.**
Beam search keeps `B * num_beams` rows and, after scoring, needs to know which example and which beam a flat index belongs to.
With beams adjacent, that decoding is `divmod(flat_index, num_beams)` and the cache reorder is a single `gather`.
With the other ordering every step needs an index remap, and the bug it produces is a model that generates fluent text for the wrong prompt.
Topic 19 builds the search itself.

## Going deeper

- [Tensor indexing API](https://pytorch.org/docs/stable/generated/torch.Tensor.index_select.html) and the NumPy [advanced indexing](https://numpy.org/doc/stable/user/basics.indexing.html) documentation, which PyTorch follows closely.
- [`torch.gather`](https://pytorch.org/docs/stable/generated/torch.gather.html), whose doc page states the index equation explicitly and is worth rereading whenever the axes stop making sense.
