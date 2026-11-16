# Topic 05 - Indexing and gather

> **Exercise** `test_05_indexing_and_gather.py`

## Why this topic

Topics 01 to 04 treated a tensor as a whole: its shape, its axes, a sum along one of them.
This topic is about reaching inside and picking out particular pieces of it.

That sounds like `x[3]`, and sometimes it is.
But a language model needs much more than "element number 3".
It needs "for each of these 512 token ids, fetch the matching row of a table", "for every position in the batch, fetch the score of the token that actually came next", "throw away everything except the 50 best scores in each row".
Each of those is one line of PyTorch, with no Python loop, as long as you know which indexing tool to reach for.

PyTorch has several, and they look alike on the page while behaving quite differently.
Square brackets alone mean three different things depending on what you put inside them.
Then there is `gather`, which reads one value per row, and `scatter`, which writes one value per row.
Knowing which one you are using, what shape it returns, and whether it shares memory with the original is most of the skill.

Almost everything later in the repository leans on this file.
The embedding layer in topic 12 is one line of indexing.
The loss in topic 09 is a `gather`.
Top-k sampling in topic 17 is a `topk` plus a `scatter`.
Mixture-of-experts in topic 19 is an `index_add_`.

## The mental model

### A few words you will meet

The exercise is phrased in language-model terms, so here is the vocabulary, before any of it is explained properly in later topics.

| Word | Meaning here |
| --- | --- |
| token id | an integer that stands for one piece of text; a sentence becomes a row of them |
| vocabulary, `V` | how many different token ids exist, so every id is in `[0, V)` |
| embedding table | a `(V, D)` tensor with one row of `D` numbers per token id |
| hidden state | the `D` numbers the model holds for one position, so a batch of them is `(B, T, D)` |
| logits | the model's output: one score per vocabulary entry, per position, `(B, T, V)` |
| target | the id of the token that actually came next at a position |
| padding | filler positions added at the end of short sequences so a batch is a rectangle |
| EOS | a special "end of sequence" token id |

Shapes follow the alphabet from the theory index: `B` batch, `T` sequence length, `D` width, `V` vocabulary, `N` a count of selected items.

### A refresher: views and copies

From topic 01: a **view** is a new tensor object that looks at the same memory as the original, and a **copy** has memory of its own.
Writing into a view changes the original; writing into a copy does not.
A view costs nothing to make, whatever its size; a copy costs memory and time proportional to its size.

Every indexing tool below falls clearly on one side of that line, and it matters for both speed and correctness.

### The three things square brackets can mean

Here is one small tensor that the examples in this section use:

```python
>>> x = torch.arange(12).reshape(3, 4)
>>> x
tensor([[ 0,  1,  2,  3],
        [ 4,  5,  6,  7],
        [ 8,  9, 10, 11]])
```

**1. Basic indexing: integers and slices. Returns a view.**

```python
>>> x[1]          # row 1
tensor([4, 5, 6, 7])
>>> x[:, 2]       # column 2
tensor([ 2,  6, 10])
>>> x[0:2, 1:3]   # rows 0-1, columns 1-2
tensor([[1, 2],
        [5, 6]])
>>> x[..., -1]    # last entry along the last axis, whatever comes before
tensor([ 3,  7, 11])
```

Every one of these is a view.
What you can select is always a regular grid: a fixed stride along each axis, which is exactly what a view can describe (topic 02).
An integer removes its axis, and a slice keeps it.
Negative numbers count from the end, so `-1` is the last element.

**2. Advanced indexing: a tensor or list of integers. Returns a copy.**

```python
>>> x[torch.tensor([2, 0, 2])]     # rows 2, 0 and 2 again, in that order
tensor([[ 8,  9, 10, 11],
        [ 0,  1,  2,  3],
        [ 8,  9, 10, 11]])
```

Now you can pick any rows, in any order, with repeats.
That cannot be described as a regular grid over the original memory, so PyTorch has to copy the selected rows into a new tensor.
A plain Python list of integers works the same as an integer tensor: `x[[2, 0, 2]]` gives the same result.

**3. Boolean masking: a tensor of `True`/`False`. Returns a copy.**

```python
>>> x[x > 6]
tensor([ 7,  8,  9, 10, 11])
```

It keeps the elements where the mask is `True` and lines them up in one flat row.
How many that is depends on the data, so the size of the result is not known until the mask is looked at.
Section 5.4 covers it.

You can mix the kinds in one subscript, for example `x[ids, 1:3]`.
That is legal, and it is also where the shape rules get hard to predict, so this exercise keeps them separate.

### 5.1 Indexing as an embedding lookup

**The one rule to remember.**
When you index the first axis of a tensor with an integer tensor, the index's shape replaces that axis, and the remaining axes are kept at the end:

```
table          (V, D)
ids            any shape S, values in [0, V)
table[ids]     S + (D,)
```

So with a table of shape `(V, D)` and ids of shape `(B, T)`, `table[ids]` is `(B, T, D)`: every id in the grid is replaced by its row of `D` numbers.

```python
>>> table = torch.arange(12.0).reshape(4, 3)   # 4 ids, 3 numbers each
>>> ids = torch.tensor([[0, 2], [3, 0]])       # a 2 x 2 grid of ids
>>> table[ids].shape
torch.Size([2, 2, 3])
>>> table[ids][1, 0]                           # the id at [1, 0] is 3
tensor([ 9., 10., 11.])
```

That is the entire forward pass of an embedding layer: no arithmetic, just a lookup.
It works for an index of any shape, `(N,)`, `(B, T)` or anything else.

**Picking rows in a given order** is the same rule with a 1-D index.
The order of the index is the order of the output, and an id that appears twice gives two copies of its row.
`torch.index_select(x, 0, idx)` does the same thing as a function call, and insists that `idx` is 1-D.

**Two index tensors at once: paired, not crossed.**
You can pass one index tensor per axis.
They are read **side by side**, like the columns of a table, not combined in every possible way:

```python
>>> rows = torch.tensor([0, 2, 1])
>>> cols = torch.tensor([3, 0, 0])
>>> x[rows, cols]          # x[0, 3], x[2, 0], x[1, 0]
tensor([3, 8, 4])
```

Three pairs in, three values out.
If the tensor has more axes than you index, the leftover axes come along whole: with `h` of shape `(B, T, D)` and two index tensors of length `N`, `h[b_idx, t_idx]` is `(N, D)`, one full `D`-vector per pair.

Before pairing, the index tensors are broadcast against each other (topic 03).
That is how you get the cross product when you do want it: shape one index as a column and the other as a row.

```python
>>> x[rows[:, None], cols[None, :]]    # (3, 1) and (1, 3) broadcast to (3, 3)
tensor([[ 3,  0,  0],
        [11,  8,  8],
        [ 7,  4,  4]])
```

Choosing between "paired" and "crossed" just by shaping the indices is the technique to remember from this section.

A very common pair is `torch.arange(B)` as the first index, meaning "every row, in order", with a second index holding one chosen position per row.
`torch.arange(n)` is the tensor `[0, 1, ..., n - 1]`.

### 5.2 gather

**The problem it solves.**
You have a score for every class in every row, and for each row you want the score of one particular class, a different one per row.
With rows `[10, 11, 12]` and `[20, 21, 22]`, and wanted columns `2` and `0`, the answer is `[12, 20]`.

**The contract.**
`torch.gather(input, dim, index)` walks over `index`, and for every entry reads `input` at the same position, except that along `dim` it uses the entry's value instead:

```
dim == 1:   out[i][j] = input[i][ index[i][j] ]
dim == 0:   out[i][j] = input[ index[i][j] ][j]
```

The rules that follow from that, and that cause nearly every error with it:

- `index` must have the **same number of axes** as `input`.
- On every axis other than `dim`, `index` must not be larger than `input`.
- The output has **exactly the shape of `index`**.
- `index` must be an `int64` tensor, and every value must be a valid position along `dim`.

```python
>>> s = torch.tensor([[10., 11., 12.],
...                   [20., 21., 22.]])
>>> torch.gather(s, 1, torch.tensor([[2], [0]]))       # one pick per row
tensor([[12.],
        [20.]])
>>> torch.gather(s, 1, torch.tensor([[2, 2, 0], [1, 0, 0]]))   # three picks per row
tensor([[12., 12., 10.],
        [21., 20., 20.]])
>>> torch.gather(s, 1, torch.tensor([2, 0]))           # one axis short
RuntimeError: Index tensor must have the same number of dimensions as input tensor
```

The first call is the problem from above.
Notice the index had to be `(2, 1)`, not `(2,)`, and the output came back as `(2, 1)`.
That gives the standard three-step pattern for "one value per row along the last axis":

1. `unsqueeze(-1)` the index, to add the missing axis of size 1;
2. `gather` along `-1`;
3. `squeeze(-1)` the result, to remove that axis again.

The same three steps work with any number of leading axes: `(B, T, V)` scores with `(B, T)` chosen ids is exactly the same situation as `(2, 3)` with `(2,)`.
Topic 09 uses this to read out the score of the correct token at every position, and it is the most common `gather` in all of language-model training.

**gather along a different axis.**
Nothing forces `dim` to be the last axis.
To pick one **position** per sequence out of `h` of shape `(B, T, D)`, you gather along the time axis, `dim=1`.
Then the index needs shape `(B, 1, D)`: one position per row, repeated across all `D` features so that every feature of that position is read.
`idx.view(B, 1, 1).expand(B, 1, D)` builds that from a `(B,)` tensor without copying (topic 02), and a final `squeeze(1)` leaves `(B, D)`.

For this particular job, paired advanced indexing from 5.1 is the shorter spelling: an `arange` for the rows and the chosen positions for the second axis.
Both are worth knowing, because the `gather` form generalizes to cases advanced indexing does not handle neatly.

**Lengths are counts, positions start at 0.**
A sequence with 3 real tokens has them at positions 0, 1 and 2.
Whenever a quantity is a length or a count, check whether you need it as is or as a position.
An off-by-one here does not raise: it silently reads padding instead of the last real token, or, with a negative index, wraps around and reads the other end of the sequence.

**Reordering rows.**
"Put row `idx[0]` first, then row `idx[1]`, and so on" is just `x[idx]` from 5.1.
Beam search, which keeps several candidate continuations per prompt and drops the bad ones after each step, reorders its stored state this way after every step.

### 5.3 scatter

`gather` reads at the positions an index names.
`scatter` is the mirror image: it **writes** at them.

```
dim == 1:   dst[i][ index[i][j] ] = src[i][j]
```

`dst.scatter_(dim, index, src)` writes the values of `src`, which has the shape of `index`.
`dst.scatter_(dim, index, value=v)` writes the same number `v` everywhere instead.
The same shape rules as `gather` apply: `index` has as many axes as `dst`.

```python
>>> torch.zeros(2, 5).scatter_(1, torch.tensor([[4], [1]]), value=7.0)
tensor([[0., 0., 0., 0., 7.],
        [0., 7., 0., 0., 0.]])
```

The trailing underscore means it changes `dst` in place, and it also returns `dst`, so you can write it straight after the tensor you create: `torch.zeros(...).scatter_(...)`.
The result's dtype is the dtype of `dst`, so if you want floats, start from a float tensor; `torch.zeros` gives `float32` by default.

**One-hot vectors.**
A one-hot vector of length `V` is all zeros except a single 1 at a chosen position.
It is how "the correct answer is class 3" is written as a tensor, and topic 09 needs it.
`F.one_hot` exists, but it is nothing more than a scatter of 1s into zeros, which is what the exercise asks you to write.
The output should be the shape of the ids plus one new last axis of size `V`.

**Collisions: `scatter_` versus `scatter_add_`.**
If two entries of `index` point at the same place, `scatter_` writes twice, and only one write survives.
Which one is not specified, and can differ between CPU and GPU.

`scatter_add_` adds instead of overwriting, so collisions accumulate:

```python
>>> torch.zeros(6).scatter_add_(0, torch.tensor([1, 4, 1, 1]), torch.ones(4))
tensor([0., 3., 0., 0., 1., 0.])
```

Position 1 was named three times and received three 1s.
That is a histogram: counting how often each value appears, in one call.
For `scatter_add_`, `src` must be a tensor (there is no `value=` form) with the same shape as `index`, and the same dtype as `dst`.

`dst.index_add_(dim, index, source)` does the same accumulation with a 1-D index that selects whole slices.
The example above gives exactly the same result with `index_add_`, and it is the call mixture-of-experts uses to add each expert's outputs back into a shared buffer (topic 19).

**Keeping only the k largest.**
Top-k sampling (topic 17) keeps the `k` best scores in each row and makes every other one impossible.
"Impossible" means `-inf`: topic 09's softmax turns `-inf` into a probability of exactly 0.

`torch.topk(x, k, dim=-1)` returns two tensors, the `k` largest values per row and their positions, both of shape `(..., k)`:

```python
>>> torch.topk(torch.tensor([[0.3, 2.0, -1.0, 0.7, 1.5]]), 2, dim=-1)
torch.return_types.topk(
values=tensor([[2.0000, 1.5000]]),
indices=tensor([[1, 4]]))
```

Those positions are exactly the shape of index that `scatter_` wants.
`torch.full(shape, value)` and `torch.full_like(x, value)` make a tensor filled with one number, `float("-inf")` included.

A tempting alternative is to find the `k`-th largest value and keep everything at least that large.
That keeps more than `k` entries whenever there is a tie at the boundary.
Writing back exactly the positions `topk` returned keeps exactly `k`.

### 5.4 Boolean masks

**Selecting with a mask.**
A boolean mask with the same shape as `x` picks individual elements, and the result is always flat:

```python
>>> m = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
>>> m[m > 2.5]
tensor([3., 4., 5., 6.])
```

A mask with **fewer** axes than `x` applies to the leading axes, and keeps the rest whole.
For `x` of shape `(B, T, D)` and a mask of shape `(B, T)`, each `True` selects one entire `D`-vector:

```python
>>> x3 = torch.arange(12).reshape(2, 3, 2)
>>> keep = torch.tensor([[True, False, True], [False, False, True]])
>>> x3[keep]               # (N, 2) with N = keep.sum() = 3
tensor([[ 0,  1],
        [ 4,  5],
        [10, 11]])
```

The selected vectors come out in reading order, row by row.
This is called **packing**: the padding is dropped, the real tokens from the whole batch end up in one list, and nothing is computed on filler afterwards.

**Packing or keeping the padding.**
The size of a packed tensor depends on the data.
On a GPU that is expensive: the CPU has to wait for the GPU to finish counting the `True` values before it knows how much memory to allocate.
It also rules the operation out of compiled code that expects fixed shapes.
So production code often does the opposite: it keeps the padded shape and multiplies the loss at padded positions by zero, paying for wasted computation in exchange for shapes that never change.
For training, the modern answer is a third option: fill every sequence to full length with several documents back to back, so there is no padding at all.
Topic 20 builds that.

**Replacing values without selecting them.**
Sometimes you want to keep the shape and only change some entries.
Two tools do that:

```python
>>> torch.where(m > 2.5, m, 0.0)          # where true take m, else 0.0
tensor([[0., 0., 3.],
        [4., 5., 6.]])
>>> m.masked_fill(m > 2.5, -1.0)          # where true put -1.0, else keep m
tensor([[ 1.,  2., -1.],
        [-1., -1., -1.]])
```

Both return a new tensor and leave `m` alone.
`masked_fill_`, with the underscore, writes into `m` itself, and so changes the caller's tensor too.
When a function is handed a tensor and asked for a modified copy, the in-place form is a bug, even though it returns the right values.

**Where is it true?**
`torch.nonzero(cond)` returns the **positions** where a condition holds, rather than the values:

```python
>>> w = torch.tensor([3, 7, 3, 3, 1])
>>> torch.nonzero(w == 3)
tensor([[0],
        [2],
        [3]])
>>> torch.nonzero(w == 3, as_tuple=True)
(tensor([0, 2, 3]),)
```

The default form returns one row per hit and one column per axis, so even for a 1-D input it is `(N, 1)`.
`as_tuple=True` returns one 1-D tensor per axis instead, which for a 1-D input is a tuple with a single tensor in it.
`torch.where(cond)` with only one argument is the same as `nonzero(cond, as_tuple=True)`.
Positions always come back as `int64`.

**Counting as you go: `cumsum` on booleans.**
Some masks depend on what came **earlier** in the row, like "everything after the first EOS".
A loop would check token by token; the vectorized way is a running count.

`cumsum` on a boolean tensor treats `True` as 1 and `False` as 0, and at each position gives the number of `True` values seen so far, **including** the current one:

```python
>>> b = torch.tensor([False, True, False, True, False])
>>> b.cumsum(0)
tensor([0, 1, 1, 2, 2])
```

Read the result as a question you can ask at every position at once.
"Has a `True` appeared at or before here?" is `count > 0`.
"Has exactly one appeared so far?" is `count == 1`.
Sometimes you want the count **before** the current position, not including it.
That is the inclusive count minus the current element itself:

```python
>>> b.cumsum(0) - b.long()
tensor([0, 0, 1, 1, 2])
```

The difference between those two counts is exactly the difference between "is this position strictly after a marker" and "is this position at or after one".
For a mask that must keep the marker itself but drop what follows, write out both counts for a short toy row on paper, and pick the comparison that gives the row you want.
Rows with no marker at all should come out right without any special case; check that on paper too.

### 5.5 Training pairs

**Next-token prediction.**
A language model learns by guessing each next token from the ones before it.
Take a row of `T + 1` token ids.
The model reads the first `T` of them as input, and at every input position $i$ its target is the token at position $i + 1$:

| position $i$ | 0 | 1 | 2 |
| --- | --- | --- | --- |
| input | `a` | `b` | `c` |
| target | `b` | `c` | `d` |

for the original row `a b c d`.
So the inputs are the row without its last token, and the targets are the row without its first token.
Both are basic slices along the time axis, which means both are views: preparing training pairs copies nothing.

**Expanding for beams.**
Beam search keeps several candidate continuations, the beams, for each prompt, so before it starts, every prompt has to be copied once per beam.
There are two ways to order the copies:

```python
>>> a = torch.tensor([[1, 2], [3, 4]])
>>> a.repeat_interleave(2, dim=0)      # each row twice in a row: a a b b
tensor([[1, 2],
        [1, 2],
        [3, 4],
        [3, 4]])
>>> a.repeat(2, 1)                     # the whole block twice: a b a b
tensor([[1, 2],
        [3, 4],
        [1, 2],
        [3, 4]])
```

Beam search wants the copies of one prompt **next to each other**, the first form.
Then the beams of prompt `p` are rows `p * num_beams` up to `p * num_beams + num_beams - 1`, and a flat row number splits back into "which prompt, which beam" with a single `divmod(row, num_beams)`.
With the other ordering every step needs an extra index remapping, and getting it wrong produces fluent text for the wrong prompt, with no error.
It is the same ordering trap as expanding key/value heads in grouped-query attention (topic 02 and topic 13).

## API you need

### Indexing

| Call | Result |
| --- | --- |
| `x[i]`, `x[a:b]`, `x[:, j]`, `x[..., -1]` | basic indexing, a view |
| `table[ids]` | advanced indexing, a copy of shape `ids.shape + table.shape[1:]` |
| `x[[2, 0, 2]]` | a Python list of ints works like an int tensor |
| `x[idx_a, idx_b]` | paired advanced indexing; the indices broadcast against each other |
| `x[mask]` | boolean selection, a flat copy of shape `(N, ...)` |
| `torch.index_select(x, dim, idx)` | rows by index as a function, `idx` must be 1-D |
| `torch.arange(n)` | `[0, 1, ..., n - 1]`, int64 |
| `torch.tensor(list_of_ints)` | int64 tensor from a Python list |

### gather and scatter

| Call | Result |
| --- | --- |
| `torch.gather(x, dim, index)` / `x.gather(dim, index)` | read along `dim`; output shaped like `index` |
| `torch.take_along_dim(x, idx, dim)` | like `gather`, but broadcasts the index |
| `dst.scatter_(dim, index, src)` | write `src` along `dim`, in place |
| `dst.scatter_(dim, index, value=v)` | write the constant `v` |
| `dst.scatter_add_(dim, index, src)` | accumulate `src` along `dim`, collisions add up |
| `dst.index_add_(dim, index, source)` | accumulate whole slices with a 1-D index |
| `dst.index_copy_(dim, index, source)` / `dst.index_fill_(dim, index, value)` | overwrite or fill whole slices with a 1-D index |
| `torch.topk(x, k, dim=-1)` | `(values, indices)` of the `k` largest, both `(..., k)` |
| `torch.zeros(shape)`, `torch.ones(shape)` | float32 by default; pass `dtype=` otherwise |
| `torch.full(shape, v)` / `torch.full_like(x, v)` | a tensor of one value, `float("-inf")` included |
| `torch.zeros_like(x)` / `torch.ones_like(x)` | same shape, dtype and device as `x` |
| `t.unsqueeze(d)` / `t.squeeze(d)` | add or remove an axis of size 1 |
| `t.expand(*sizes)` | broadcast a size-1 axis without copying |

`gather` and `scatter_` both require `index.ndim == input.ndim`.
That one rule is behind most of the errors this file produces.

### Conditions

| Call | Result |
| --- | --- |
| `torch.where(cond, a, b)` | elementwise select, a new tensor; `a` or `b` may be a plain number |
| `torch.where(cond)` | same as `nonzero(cond, as_tuple=True)` |
| `x.masked_fill(mask, value)` | a copy with fills |
| `x.masked_fill_(mask, value)` | the same, written into `x` |
| `torch.nonzero(cond)` | `(N, ndim)` positions; `as_tuple=True` gives one 1-D tensor per axis |
| `cond.cumsum(dim)` | running count of `True` values, inclusive, int64 |
| `b.long()` / `b.int()` | booleans as 1s and 0s |
| `torch.equal(a, b)` | `True` if same shape and same values |

### Repeating

| Call | Result |
| --- | --- |
| `x.repeat_interleave(n, dim)` | each slice `n` times in a row: `a a b b` |
| `x.repeat(*times)` | the whole tensor tiled: `a b a b` |

### A worked example on neutral data

```python
>>> grid = torch.tensor([[5., 1., 9.], [7., 3., 2.]])
>>> grid[torch.tensor([1, 1, 0])]                 # rows in a chosen order, with a repeat
tensor([[7., 3., 2.],
        [7., 3., 2.],
        [5., 1., 9.]])
>>> grid[torch.tensor([0, 1]), torch.tensor([2, 0])]   # paired: grid[0, 2], grid[1, 0]
tensor([9., 7.])
>>> grid.gather(1, torch.tensor([[1, 1], [2, 0]]))     # two picks per row
tensor([[1., 1.],
        [2., 7.]])
>>> torch.zeros(2, 4).scatter_add_(1, torch.tensor([[0, 0, 1], [3, 3, 3]]), torch.ones(2, 3))
tensor([[2., 1., 0., 0.],
        [0., 0., 0., 3.]])
>>> torch.where(grid > 4, grid, -1.0)
tensor([[ 5., -1.,  9.],
        [ 7., -1., -1.]])
```

## Traps

- `gather` and `scatter_` raise unless `index` has the same number of axes as the input.
  Add the missing axis with `unsqueeze`, remove it after with `squeeze`.
- A `gather` index must be `int64` and in range; a negative or too-large index raises.
- `scatter_` with repeated indices keeps one write, and which one is unspecified; use `scatter_add_` when repeats should add up.
- `scatter_add_` takes a `src` tensor of the same shape as `index` and the same dtype as the destination, not a plain number.
- `x[idx] += 1` with repeated indices adds 1 only **once** per distinct position, not once per occurrence.
  Use `index_add_` when every occurrence should count.
- `x[a, b]` with two index tensors pairs them elementwise.
  For the cross product, shape them as `(N, 1)` and `(1, M)`.
- `x[mask]` flattens: the axes covered by the mask are gone, and their information with them.
- `x[mask]` has a size that depends on the data, which on a GPU forces a wait.
- `masked_fill_` and `scatter_` change their tensor in place; on an argument that was handed to you, that changes the caller's data.
- `torch.nonzero` without `as_tuple=True` returns `(N, 1)` for a 1-D input, not `(N,)`.
- A length is a count; the last real position is one less.
  An off-by-one reads padding, and a negative index wraps around to the other end without an error.
- Keeping "everything at least as large as the k-th value" keeps more than `k` on ties.
- `repeat` and `repeat_interleave` order the copies differently, and beam search needs the copies of one row next to each other.
- `index_select` refuses a multi-dimensional index and `gather` refuses one with the wrong number of axes.
  They are not interchangeable.

## Ready when

Without looking anything up you can:

- [ ] say which of `x[1:3]`, `x[[1, 2]]` and `x[x > 0]` returns a view, and why the others cannot
- [ ] give the output shape of `table[ids]` for any `table` and `ids`
- [ ] predict the result of `x[rows, cols]`, and change the shapes of the indices to get the cross product instead
- [ ] state the four rules of `gather`, and use unsqueeze, gather, squeeze to take one value per row out of `(B, T, V)`
- [ ] gather one position per sequence out of `(B, T, D)` both with `gather` and with paired indexing
- [ ] build a one-hot tensor and a count per id with `scatter_` and `scatter_add_`, and say when the difference matters
- [ ] replace everything but the top `k` scores per row with `-inf` and explain why ties do not matter
- [ ] say what `x[mask]` returns for `x` of `(B, T, D)` and `mask` of `(B, T)`, and why production code often avoids it
- [ ] return a modified copy without touching the input
- [ ] use an inclusive and an exclusive running count to build a mask that depends on the first marker in a row
- [ ] split a row of `T + 1` ids into inputs and targets without copying
- [ ] say why beam expansion uses `repeat_interleave` and not `repeat`

## Going deeper

- [Tensor indexing in PyTorch](https://pytorch.org/docs/stable/tensor_view.html) for which operations return views, and the NumPy [indexing guide](https://numpy.org/doc/stable/user/basics.indexing.html), which PyTorch follows closely and which explains advanced indexing in full.
- [`torch.gather`](https://pytorch.org/docs/stable/generated/torch.gather.html) and [`Tensor.scatter_`](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html), whose doc pages state the index equations for every `dim`; reread them whenever the axes stop making sense.
- [`torch.topk`](https://pytorch.org/docs/stable/generated/torch.topk.html), including the `sorted` and `largest` arguments.
