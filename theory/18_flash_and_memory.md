# Topic 18 - Flash attention and memory

> **Exercise** `test_18_flash_and_memory.py`

## Why this topic

There are no new parts of the model in this topic.
It takes the parts that already exist and makes them fit in memory.

Three earlier topics each ended on a number too big to allocate.
Topic 14 computed the attention score matrix of a 32k context: about 550 GB per layer.
Topic 15 computed the activation memory of a whole model, and called its own formula an underestimate.
Topic 10 built a loss over the vocabulary, and at 128k entries its logits are the single largest tensor in a training step.
This file answers all three, and every task in it asks the same question: **how do you compute exactly the same result without ever allocating the huge intermediate?**

- The `(T, S)` attention matrix does not fit, so the softmax is computed online, one block of keys at a time.
- The `(N, V)` logits do not fit, so cross-entropy is computed in chunks of tokens.
- The activations of all `L` layers do not fit, so most of them are recomputed instead of stored.
- fp16 flushes small gradients to zero, so the loss is scaled up before the backward pass and the gradients scaled back down after it.

The idea behind the first two is the most reusable one in the repository: a reduction whose running state is small enough to carry, updated block by block, that ends on the same answer as the one-shot version.
The idea behind the third is the oldest trade in computing, memory for time.
The fourth is about the number format itself.

This is [CS336](https://cs336.stanford.edu/) assignment 2 territory, and it is the hardest file in the repository.
Nothing in it is long; every function is a dozen lines at most.
What makes it hard is that each one has an exact invariant, and a version that is off by one rescaling still runs and still returns numbers of the right shape.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| materialize | actually allocate a tensor in memory, as opposed to computing its effect piece by piece |
| block, chunk, tile | a slice of an axis processed at once; a tile is a 2D block of a matrix, rows from one axis and columns from another |
| running statistic | a small tensor carried through a loop over blocks and updated after each one, such as a running maximum |
| online | computed in one left-to-right pass over the blocks, seeing each block once |
| `m`, `l` | the running maximum and the running sum of exponentials relative to it, one of each per row |
| saved tensor | a tensor that autograd keeps from the forward pass because the backward pass needs it |
| checkpointing | not saving a region's intermediates, and recomputing them during the backward pass instead |
| fp16, bf16 | the two 16-bit float formats: same size, different split between range and precision |
| overflow, underflow | a result too large for the format becomes `inf`; one too small becomes `0` |
| loss scale | the constant the loss is multiplied by so that small gradients survive in fp16 |

### A refresher: the stable softmax and logsumexp

Topic 10 built both, and everything here is a rearrangement of them.

`exp` overflows quickly: in float32 $\exp(89)$ is already `inf`.
The fix is that subtracting the same constant from every entry of a row changes nothing, so you subtract the row's maximum `m` first:

```math
\begin{aligned}
\mathrm{softmax}(x)_i &= \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)} \\
\mathrm{logsumexp}(x) &= m + \log\Big(\sum_j \exp(x_j - m)\Big)
\end{aligned}
```

After the subtraction the largest exponent is $\exp(0) = 1$ and every other one is at most 1, so nothing can overflow, and the sum is at least 1, so the `log` is safe.

The catch is that this needs the maximum **before** any exponential is taken.
That means two passes over the row, and it means holding the whole row.
For a row of a few thousand logits that is nothing.
For the `(T, S)` scores of attention, or the `(N, V)` logits of a large vocabulary, holding all of it at once is exactly what does not fit.

### A refresher: what `-inf` does in this arithmetic

Masks write `-inf` into scores (topic 14), and the online algorithms below subtract running maxima that start at `-inf`.
Four facts decide whether such code returns numbers or NaN:

```python
>>> ninf = torch.tensor(float("-inf"))
>>> torch.exp(ninf - torch.tensor(3.0))       # -inf minus anything finite: exp gives exactly 0
tensor(0.)
>>> torch.exp(ninf - ninf)                     # -inf minus -inf is NaN
tensor(nan)
>>> torch.maximum(ninf, torch.tensor(-2.0))    # a finite value always wins against -inf
tensor(-2.)
>>> torch.maximum(ninf, ninf)
tensor(-inf)
```

So `-inf` is harmless as long as there is a finite number on the other side of the subtraction.
NaN appears only when a row has seen nothing but `-inf` so far and is asked to compare against itself.

### 18.1 Online softmax

The online version of the softmax carries two running statistics per row instead of the whole row:

- `m`, the maximum seen so far;
- `l`, the sum of $\exp(x - m)$ over everything seen so far, **relative to the current `m`**.

Start with `m = -inf` and `l = 0`, which stand for "nothing seen yet".
When a new block of scores arrives, the maximum may go up, and every exponential already summed into `l` was taken relative to the old maximum.
Fixing that takes a single multiplication:

```math
\begin{aligned}
m_{\text{new}} &= \max\big(m_{\text{prev}}, \max(\text{block})\big) \\
l_{\text{new}} &= l_{\text{prev}} \cdot \exp(m_{\text{prev}} - m_{\text{new}}) + \sum \exp(\text{block} - m_{\text{new}})
\end{aligned}
```

The factor $\exp(m_{\text{prev}} - m_{\text{new}})$ is at most 1, and it moves the old sum from the old reference point to the new one.
It is exact, because $\exp(a - m_{\text{old}}) \cdot \exp(m_{\text{old}} - m_{\text{new}}) = \exp(a - m_{\text{new}})$ for every $a$ already summed.
When the block does not raise the maximum, $m_{\text{new}} = m_{\text{prev}}$, the factor is $\exp(0) = 1$, and the update is a plain addition.

**One row by hand.**
Take the row `[1, 3, 0.5, 4, 2, -1, 0, 2.5]` in blocks of 3:

| after block | block | `m` | `l` |
| --- | --- | --- | --- |
| 1 | `[1, 3, 0.5]` | 3 | `e^-2 + e^0 + e^-2.5` = 1.2174 |
| 2 | `[4, 2, -1]` | 4 | `1.2174 * e^-1 + e^0 + e^-2 + e^-5` = 1.5899 |
| 3 | `[0, 2.5]` | 4 | `1.5899 * e^0 + e^-4 + e^-1.5` = 1.8314 |

At the end, $m + \log(l) = 4 + \log(1.8314) = 4.6051$, which is `torch.logsumexp` of the whole row to the last printed digit.
The first block also shows why `-inf` and `0` are the right starting values: the correction $\exp(-\infty - 3)$ is 0, and multiplying the starting `l = 0` by it leaves 0.

**Why this is exact and not an approximation.**
Nothing is dropped or truncated, the rescaling is an identity, and every exponent is at most zero, so nothing overflows on the way.
The last bit may differ from a one-shot computation because the additions happen in a different order, which is ordinary floating point, not an error in the algorithm.

**Shapes.**
`online_softmax_update` works on `(..., Sb)` blocks of any number of leading axes, and its `m` and `l` are `(..., 1)`: one value per row, with the reduced axis kept.
That is `amax(dim=-1, keepdim=True)` and `sum(dim=-1, keepdim=True)`.
Keeping the axis is what lets `block - m_new` broadcast row by row; the test squeezes it away itself before comparing.
Blocks may have different widths; nothing in the update depends on `Sb`.

**What the update gives you at the end.**
Once every block has gone through, `m` is the true row maximum and `l` the true sum relative to it, so:

```math
\begin{aligned}
\mathrm{logsumexp}(x) &= m + \log(l) \qquad (\dots, 1) \\
\mathrm{softmax}(x)_i &= \exp(x_i - m) / l
\end{aligned}
```

The second line still needs every $x_i$, and the statistics are only final after the last block.
So `chunked_softmax` takes **two passes** over the blocks: the first carries `(m, l)` through all of them, the second goes back over each block, turns it into probabilities with the final statistics, and the blocks are joined back along the last axis.
Its working memory beyond the input and output is one block plus two numbers per row, and the test asks for the same answer as `x.softmax(-1)` within float32 tolerance, including on rows near 5000 that a naive `exp` would turn into `inf`.

A logsumexp needs only the first pass, which is why section 18.3 can compute it online in one sweep.

**Iterating over blocks when the length does not divide evenly.**
The tests use a width of 37 in blocks of 8 and a width of 41 in blocks of 7.
Slicing past the end of a tensor is not an error in PyTorch; the slice is simply shorter, so `range(0, n, chunk)` together with `x[..., i : i + chunk]` produces the short last block for free:

```python
>>> x = torch.arange(10.0)
>>> [(i, min(i + 4, 10)) for i in range(0, 10, 4)]
[(0, 4), (4, 8), (8, 10)]
>>> x[8:12]                                    # asks for 4, gets the 2 that exist
tensor([8., 9.])
>>> [t.tolist() for t in x.split(4)]           # split does the same
[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0]]
```

Writing a loop over blocks as Python is allowed in this file: the loop is over blocks, not over elements, and each step is still a tensor operation on a whole block.

### 18.2 Flash attention

Attention for one query is a softmax-weighted average of the value vectors:

```math
\text{out} = \sum_j \frac{\exp(s_j - m)}{l} \, v_j \qquad \text{where } s_j = \frac{q \cdot k_j}{\sqrt{d_{\text{head}}}}
```

Flash attention applies the online softmax to this and adds one thing: an **output accumulator**, rescaled together with `l`.
Carry, per query row, the unnormalized sum $o = \sum_j \exp(s_j - m) v_j$ alongside `m` and `l`.
Every term in `o` was weighted relative to the current `m`, exactly like every term in `l`, so when `m` goes up, `o` takes the same correction factor $\exp(m_{\text{old}} - m_{\text{new}})$.
Dividing `o` by `l` once, after the last key block, gives the normalized output.

The `(T, S)` score matrix never exists in full.
Only one tile of it does at a time, `(block of queries, block of keys)`, and per query row the state is `m` and `l`, of shape `(..., rows, 1)`, and `o`, of shape `(..., rows, Dh)`.

The loop structure of FlashAttention-2, the version in the exercise, as pseudocode:

```
for each block of query rows:
    m = -inf, l = 0, o = 0                        state for these rows only
    for each block of key columns:
        s = (query block) @ (key block)ᵀ / sqrt(Dh)       one tile of scores
        if causal: -inf on the forbidden entries of this tile
        m_new      = max(m, rowmax(s))
        correction = exp(m - m_new)
        p          = exp(s - m_new)
        l = l * correction + rowsum(p)
        o = o * correction + p @ (value block)
        m = m_new
    this block's output = o / l
```

`o` and `l` take the *same* correction, which is the whole trick in one line.
The outer loop is over query blocks because each query row's result depends only on its own `(m, l, o)`: the rows never interact, so each block of rows is finished and written out before the next one starts.
The finished blocks are written into a preallocated output (`torch.empty_like(q)` and slice assignment) or collected and joined with `torch.cat` along the query axis.
The tests check the forward pass only, so either way is fine.

The scale is $1 / \sqrt{d_{\text{head}}}$, with `Dh` the last axis of `q`, as in topic 14; the reference `F.scaled_dot_product_attention` uses exactly that.
Leading axes `(B, H)` need no special handling: `@` treats them as batch axes, and slicing a block means slicing the second-to-last axis, `q[..., i : i + block, :]`.

**Large scores.**
One test fills `q` and `k` with 30, so every score is $30 \cdot 30 \cdot 8 / \sqrt{8}$, about 2546.
$\exp(2546)$ is `inf` in any float format, and a version that exponentiates before subtracting the running maximum returns NaN.
With the online rule every exponent is $s - m_{\text{new}} \le 0$, so the scores can be any size.
All scores equal means a uniform softmax, so the expected output is the plain mean of the value vectors.

#### The causal mask, tile by tile

With `causal=True`, query $i$ may see key $j$ only when $j \le i$, as in topic 14.
Inside a tile, $i$ and $j$ are **absolute** positions, not positions within the tile.
A tile covering query rows $[i_0, i_1)$ and key columns $[j_0, j_1)$ builds its mask from `arange(i0, i1)` and `arange(j0, j1)`, compared by broadcasting exactly as in topic 14.
A neutral tile, rows 2 to 4 and columns 3 to 5, with `True` marking the **forbidden** entries $j > i$:

```python
>>> rows = torch.arange(2, 5)[:, None]
>>> cols = torch.arange(3, 6)[None, :]
>>> cols > rows
tensor([[ True,  True,  True],
        [False,  True,  True],
        [False, False,  True]])
```

Comparing tile-local indices, `arange(0, 3)` against `arange(0, 3)`, would produce the mask of a diagonal tile on every tile, which is wrong for all the others.

Each tile is one of three kinds:

- entirely below the diagonal, every key visible to every row: no mask needed;
- crossing the diagonal: some entries masked;
- entirely above it, every key in the future of every row: fully masked.

A tile is fully masked when its first key comes after its last query, $j_0 > i_1 - 1$.
Those tiles contribute nothing, so they can be **skipped** without computing them at all.
For `T == S` that is almost half of all tiles, and it is where a causal flash kernel gets close to a 2x speedup over the non-causal one.

**Where the NaN of the docstring comes from.**
Folding a fully masked tile into a row that has already seen at least one visible key is harmless: `rowmax(s)` is `-inf`, `m_new` stays the old finite `m`, the correction is $\exp(0) = 1$, and $p = \exp(-\infty - m)$ is 0.
The NaN appears when a row meets a fully masked tile while its `m` is still `-inf`, because then `m_new` is `-inf` too and both $\exp(m - m_{\text{new}})$ and $\exp(s - m_{\text{new}})$ are $\exp(-\infty - (-\infty))$.
Whether that can happen depends on the order of the loop, which is fragile; skipping fully masked tiles removes the possibility instead of relying on the order.
The same hazard sits inside `online_softmax_update` itself: called on a row that is all `-inf` with `m_prev = -inf`, it returns NaN, and it is the caller's job never to do that.

**Ragged sizes.**
`T` and `S` need not divide by `block_size`, and the tests use 37 and 20 with blocks of 1, 4, 8, 16 and 64.
Slicing gives the short last block, as in 18.1, so the tile mask has to be built from the actual size of the slice, not from `block_size`.
`block_size = 64` with `T = 20` is a single tile, which must still work, and `block_size = 1` is a tile per score.

**Which diagonal `is_causal` means.**
`F.scaled_dot_product_attention(..., is_causal=True)` builds its mask as `torch.ones(T, S).tril()`, that is, query `i` sees keys `0..i` counted from the top-left corner.
With `T == S`, as in every test here, that is the ordinary causal mask.
With `T != S` it is not the "last `T` positions" alignment that topic 14 needed for a KV cache, so the built-in is only a valid reference for causal attention when the lengths match.

#### Why the real kernel is fast

The exercise is about the algorithm, and written in Python it is slower than `F.scaled_dot_product_attention`.
The real kernel wins for a reason the Python version cannot show.

On a GPU, attention is limited by memory traffic, not by arithmetic.
The naive version writes the whole score matrix out to the GPU's main memory (HBM), reads it back for the softmax, writes the weights, reads them back for the matmul with `v`.
That traffic is $O(T^2)$ and it dominates the runtime.
A flash kernel keeps each tile in on-chip SRAM, roughly an order of magnitude faster to access but only a few hundred kilobytes in size, and never writes the scores anywhere.
The extra arithmetic of the rescaling is cheap next to the traffic it saves.

Memory drops from $O(T^2)$ to $O(T)$: per query row, `(m, l)` and the output.
The backward pass recomputes each tile from `q`, `k`, `v` and the saved `m` and `l` instead of storing the scores, which is the same compute-for-memory trade as 18.4.

### 18.3 Cross-entropy without materializing the logits

**The size of the problem.**
`lm_head` turns `N` hidden vectors into `(N, V)` logits.
At `N = 8192` tokens and `V = 128256` (Llama 3) in float32 that is $8192 \cdot 128256 \cdot 4$ bytes, about 4.2 GB, and the backward pass needs a tensor of the same size again for the gradient of the logits.
On a large-vocabulary model it is the biggest tensor in the whole step, bigger than any layer's activations.

**The decomposition.**
From topic 10, the loss of one token is

```math
\text{loss}_n = \mathrm{logsumexp}(\text{logits}_n) - \text{logits}_n[\text{target}_n] \qquad \text{logits}_n = \text{weight} \cdot \text{hidden}_n, \ \text{shape } (V,)
```

and the mean loss is $\sum_n \text{loss}_n / N$.
Every term depends on its own row of `hidden` only, so the rows can be processed in groups: take rows `[i, i + chunk)` of `hidden`, `(chunk, D)`, multiply by `weight.T`, `(D, V)`, to get that group's `(chunk, V)` logits, reduce them to that group's sum of losses, and go on to the next group.
`weight` is `(V, D)`, the `nn.Linear` layout of `lm_head` (topic 09), so the logits of a chunk are `hidden_chunk @ weight.T`, the same expression the test uses for its reference.
`targets` is sliced with the same bounds as `hidden`.

The per-group reduction is any of the loss formulations from topic 10.
`F.cross_entropy(logits, targets, reduction="sum")` returns the sum over the rows instead of the mean, which is the form a chunked total needs; `logsumexp` and a `gather` of the target logits per row is the same thing by hand.

**The averaging.**
Divide the total by `N`, once, at the end.
The test uses `N = 20` in chunks of 7, so the chunks are 7, 7 and 6 rows.
Averaging the per-chunk means gives $(m_1 + m_2 + m_3) / 3$, which weighs each of the 6 tokens in the last chunk more than each of the 14 others: wrong whenever the chunks differ in size.
This is the same trap as the gradient-accumulation division in topic 16.

**The gradients.**
They must reach both `hidden` and `weight`, and they do if the chunked computation is an ordinary differentiable graph built from slicing, `@` and the loss.
Slicing `hidden` is differentiable, so each chunk's gradient lands in its own rows of `hidden.grad`.
`weight` is used once per chunk, and autograd adds up the contributions from every use, so `weight.grad` comes out as the sum over all chunks, which is the gradient of the full loss (topic 08).
Anything that breaks the graph breaks this: `torch.no_grad()`, `.detach()`, `.item()` on an intermediate, or building the total with a Python float.
Summing tensors with `+` or `sum(list_of_tensors)` keeps it intact.

**The honest memory picture.**
Chunking as described keeps the forward pass from ever holding the full logits at once, but in training it is not the whole story.
Autograd saves, for every chunk, what that chunk's backward pass needs, and for a softmax-based loss that includes a `(chunk, V)` tensor.
Summed over the chunks, the saved tensors are again at least `N * V` numbers: measured on a toy size, the saved total was slightly larger than for the unchunked loss.
What the chunking alone buys is a smaller peak during the forward pass; it does not shrink what waits for the backward pass.

The real memory saving combines this with 18.4: wrap each chunk's computation in `checkpoint`, so that autograd keeps only the chunk's inputs and recomputes its logits during the backward pass, one chunk at a time.
In the same measurement, that cut the saved total from about `3 * N * V` numbers to roughly the size of the inputs.
Production kernels go one step further and compute each chunk's gradient right inside the forward pass, so that no logits are ever saved.
The test checks the value and the gradients, not the memory, so the plain chunked graph passes, and checkpointing each chunk passes too.

**Chunked logsumexp.**
`chunked_logsumexp` reduces `(N, V)` to `(N,)` along the last axis, and it chunks the **vocabulary**, the columns, not the rows.
It is the first pass of 18.1 with the final $m + \log(l)$ from the formula above, squeezed from `(N, 1)` to `(N,)`.
It is the piece a loss kernel needs when even one row of logits is too wide to hold, the opposite case from chunking over tokens.

### 18.4 Gradient checkpointing

**What autograd saves.**
To compute gradients, the backward pass needs values from the forward pass, and autograd keeps them as **saved tensors** attached to the graph.
Each operation saves only what its own derivative needs.
`y = x @ w` saves `x` and `w`, because $\partial y / \partial w$ needs `x` and $\partial y / \partial x$ needs `w`.
`tanh` saves its output, because its derivative is $1 - \tanh(x)^2$.
For a stack of `nn.Linear` followed by `nn.Tanh`, that is three saved tensors per block: the input, the weight and the tanh output.

For a transformer the saved tensors of all `L` layers are alive at the same moment, between the end of the forward pass and the point where the backward pass reaches each layer.
That is the activation memory of topic 15, and for long sequences it is the largest memory consumer in training.

**The trade.**
Checkpointing a region means: during the forward pass, run it **without** saving its intermediates, keeping only its inputs; during the backward pass, when the gradient reaches that region, run its forward pass again from the kept inputs, this time saving everything, and then backpropagate through it as usual.
The answer is identical, because the same operations run on the same inputs.
The cost is one extra forward pass for every checkpointed region.
A backward pass costs about twice a forward pass, so a training step is about three forward passes' worth of work, and recomputing all of them adds about one more: roughly 33% more compute.

```python
from torch.utils.checkpoint import checkpoint

y = checkpoint(fn, *inputs, use_reentrant=False)  # y equals fn(*inputs); fn's intermediates are not saved
```

`fn` can be any callable, a module included, and its result is what `checkpoint` returns.
`use_reentrant=False` selects the current implementation.
The older reentrant one, the default for years, does not propagate gradients correctly when none of the inputs requires grad, does not support `torch.autograd.grad`, and is on its way out; recent PyTorch versions warn when the argument is left out and announce that leaving it out will become an error.

**How much it saves.**
Checkpointing each of `L` blocks separately keeps one input tensor per block, the block's boundary, and recomputes one block at a time during the backward pass.
So the saved tensors drop from "all intermediates of every block" to "one tensor per block", plus the intermediates of a single block while it is being recomputed.
That is still $O(L)$ boundary tensors, but each block's inner intermediates (attention weights, FFN hidden states, norm statistics in a real block) are no longer multiplied by `L`.
Checkpointing segments of $\sqrt{L}$ blocks instead stores $\sqrt{L}$ boundaries and recomputes one segment of $\sqrt{L}$ blocks at a time, which brings the total to $O(\sqrt{L})$.
That is the result of Chen et al. 2016 and the source of the $\sqrt{L}$ in the folklore.

**Measuring it.**
`torch.autograd.graph.saved_tensors_hooks(pack, unpack)` is a context manager.
While it is active, autograd calls `pack(tensor)` for every tensor it saves, and stores whatever `pack` returns in place of the tensor; during the backward pass it calls `unpack` on the stored thing to get a tensor back.
Real uses offload saved tensors to CPU or compress them.
For counting, `pack` records the call and returns the tensor unchanged, and `unpack` is the identity.

On a neutral function, `sin` saves its input and `*` saves both of its operands; under `checkpoint`, only the two inputs are saved:

```python
>>> a, b = torch.randn(4, requires_grad=True), torch.randn(4, requires_grad=True)
>>> seen = []
>>> def pack(t):
...     seen.append(tuple(t.shape))
...     return t
>>> with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
...     y = (a.sin() * b).sum()
>>> len(seen)
3
>>> seen.clear()
>>> with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
...     y = checkpoint(lambda u, w: (u.sin() * w).sum(), a, b, use_reentrant=False)
>>> len(seen)
2
```

Three details decide whether a count like that means anything.

- The count is taken during the **forward** pass: the hooks must be active while the graph is built, and no backward pass is needed.
- The forward pass must run with gradients enabled and with something that requires grad: under `torch.no_grad()` nothing is saved and the count is 0 for both versions.
- The counter has to be something `pack` can change, such as a list it appends to or a one-element list it increments, because a plain local integer cannot be reassigned from inside a nested function without `nonlocal`.

The test runs six `Linear + Tanh` blocks and expects the checkpointed count to be smaller than the plain one and at most one per block.

**RNG and determinism.**
The recomputation must produce the same activations as the original forward pass, or the gradients are for a different function.
For dropout that means the same random mask, and `checkpoint` saves and restores the RNG state for exactly that reason (`preserve_rng_state=True`, the default).
A region whose forward pass depends on anything else that changes between the two runs, a counter or a global flag, gets a wrong gradient silently.

### 18.5 Mixed precision

**Two 16-bit formats.**
Both spend 16 bits per number, half of float32, and halve memory and memory traffic.
They split those bits differently:

| | float32 | float16 | bfloat16 |
| --- | --- | --- | --- |
| largest value | 3.4e38 | 65504 | 3.4e38 |
| smallest normal value | 1.2e-38 | 6.1e-5 | 1.2e-38 |
| `eps`, the gap after 1.0 | 1.2e-7 | 9.8e-4 | 7.8e-3 |

bf16 keeps float32's exponent, so its **range** is float32's, and pays with precision, about two and a half significant digits.
fp16 keeps more precision but has a narrow range: anything above 65504 is `inf`, and values below about 6e-5 lose precision and, below about 6e-8, become exactly 0.
`torch.finfo(dtype)` reports all of these (`.max`, `.smallest_normal`, `.eps`).

**The problem with fp16 in training is underflow.**
Gradients in a large model are routinely around `1e-6` to `1e-8`.
In fp16 they flush to zero, and a zero gradient means that weight simply does not learn:

```python
>>> g = torch.tensor(1e-4, dtype=torch.float16)
>>> g * g                                          # 1e-8 is below fp16's range
tensor(0., dtype=torch.float16)
>>> ((g * 1024) * g).float() / 1024                # scaled up first, it survives
tensor(1.0012e-08)
```

**Loss scaling** is that second line applied to a whole backward pass.
Multiply the loss by a large constant `scale` before `backward()`; by the chain rule every gradient comes out multiplied by `scale`, which lifts the small ones back into range.
Before the optimizer step, divide every gradient by `scale` again, in float32.
The optimizer sees the true gradients, and the model never notices the trick.

**The dynamic part.**
Too small a scale does not cure the underflow; too large a scale pushes the biggest gradients past 65504 into `inf`, and an `inf` anywhere in the backward pass tends to become NaN further down.
No fixed value is right for a whole run, so `torch.amp.GradScaler` adjusts it:

- Unscale every gradient and check whether any of them contains `inf` or NaN.
- If one does, **skip the optimizer step entirely** and halve the scale.
- If none does, take the step; and once `growth_interval` steps in a row have gone by without an overflow, double the scale.

The defaults are a starting scale of $2^{16} = 65536$, a factor of 2 up and 0.5 down, and a growth interval of 2000 steps.
So the scale sits just below the point where overflows start, and probes upward now and then.
An overflow takes priority over growth: if both happen on the same step, the scale goes down.

Skipping the step is not optional.
A gradient containing `inf` or NaN that reaches the parameters makes them `inf` or NaN, and from then on every output of the model is NaN: one bad step destroys the run.
Losing one step in a few thousand costs nothing.

The unscale and the check are one pass over the gradients.
Dividing by a finite scale keeps `inf` as `inf` and NaN as NaN, so the check finds an overflow whether it looks before or after the division; what matters is that it looks at **every** gradient.
`torch.isfinite(g)` is `True` where an entry is neither `inf`, `-inf` nor NaN, `.all()` reduces it to one element, and a gradient is bad if that element is `False`.

**A tensor is not a bool.**
`torch.isfinite(g).all()` returns a **0-dim tensor**, not a Python `bool`.
It works in an `if`, but it is not `True` or `False` itself, so a check like `found is False` fails on it:

```python
>>> t = torch.isfinite(torch.tensor([1.0, float("inf")])).all()
>>> t
tensor(False)
>>> t is False, t.item() is False, bool(t) is False
(False, True, True)
```

`.item()` or `bool(...)` turns it into a Python bool; `not` on a tensor also gives a Python bool.
The same applies to a flag combined from several tensors: combine the Python values, or convert at the end.

**Softmax in fp16.**
`exp` overflows in fp16 as soon as its argument passes $\log(65504)$, about 11.09: $\exp(11.0)$ is 59872, $\exp(11.1)$ is `inf`.
Logits of 60000, which fp16 can hold, are far beyond that.
The max subtraction from the refresher removes the overflow of the `exp`, but the sum of many exponentials, and any intermediate on the way, still lives in a format with three significant digits and a ceiling of 65504.

The robust pattern is the **upcast sandwich** from topic 11: convert the input to float32, do the whole reduction there, max subtraction included, and convert the result back to the input's dtype.
The result of a softmax is in $[0, 1]$, which fp16 represents well; only the intermediates needed the range.
The test checks all of it: the result is float16, every entry is finite, and the row sums to 1.
On neutral data, a norm of a vector whose squares overflow fp16:

```python
>>> v = torch.tensor([300.0, 400.0], dtype=torch.float16)
>>> v.pow(2).sum()                                  # 90000 + 160000 overflows
tensor(inf, dtype=torch.float16)
>>> v.float().pow(2).sum().sqrt().to(v.dtype)      # the sandwich
tensor(500., dtype=torch.float16)
```

PyTorch's own `softmax` already subtracts the maximum and accumulates in higher precision internally, so the built-in on fp16 logits is safe.
The danger is the hand-written `exp(x) / exp(x).sum()` in a half-precision region, and any custom kernel that does the same.

**Why everyone moved to bf16.**
The whole `GradScaler` state machine exists for one property of fp16, its narrow exponent range.
bf16 has float32's range, so gradients do not underflow, no scaling is needed, and the machinery disappears.
The price is three fewer mantissa bits than fp16, which turned out not to matter for training.
fp16 with loss scaling survives on hardware without bf16 support and in older code, which is why the mechanism is still worth knowing.

**What autocast does and does not touch.**
`torch.autocast(device_type, dtype)` runs matmuls and convolutions in half precision and keeps softmax, normalization and loss reductions in float32: the same list of "reductions that need range" this topic keeps arriving at.
A hand-written function inside an autocast region is not on that list and has to do its own upcasting, as topic 11 warns.
The parameters themselves usually stay in float32 as "master weights", and the optimizer updates those.

## API you need

### Blockwise computation

| Call | Result |
| --- | --- |
| `x.amax(dim=-1, keepdim=True)` | the row maximum of a block, `(..., 1)` |
| `torch.maximum(a, b)` | elementwise maximum of two tensors, for `m_new`; `torch.max(a, b)` is the same |
| `x.sum(dim=-1, keepdim=True)` | the row sum, `(..., 1)` |
| `torch.exp(t)`, `torch.log(t)` / `t.exp()`, `t.log()` | the online update, and $m + \log(l)$ |
| `torch.full(shape, float("-inf"))`, `torch.full_like(t, float("-inf"))` | the starting `m` |
| `torch.zeros(shape)`, `torch.zeros_like(t)` | the starting `l` and `o` |
| `range(0, n, chunk)` with `x[..., i : i + chunk]` | blocks of an axis, the last one short |
| `x.split(chunk, dim=-1)` | the same blocks as a tuple of views |
| `torch.cat(blocks, dim=-1)` | join blocks back along an axis |
| `torch.empty_like(q)`, `out[..., i : i + n, :] = block` | a preallocated result, filled block by block |
| `t.squeeze(-1)` | `(N, 1)` to `(N,)` |
| `q @ k.mT` | scores of a tile, `(..., rows, Dh) @ (..., Dh, cols)` |
| `math.sqrt(q.shape[-1])` | the attention scale's denominator |
| `torch.arange(i0, i1)[:, None]`, `torch.arange(j0, j1)[None, :]` | absolute row and column positions of a tile, for broadcasting |
| `t.masked_fill(cond, float("-inf"))` | `-inf` wherever `cond` is `True` |
| `F.scaled_dot_product_attention(q, k, v, is_causal=False)` | the tests' reference; `is_causal` is the top-left `tril` mask |

### Loss

| Call | Result |
| --- | --- |
| `hidden[i : i + chunk] @ weight.T` | one chunk's logits, `(chunk, V)` |
| `F.cross_entropy(logits, targets, reduction="sum")` | the sum of per-row losses instead of the mean |
| `torch.logsumexp(t, dim=-1)`, `t.gather(-1, idx[:, None])` | the two terms of the loss by hand (topic 10) |
| `sum(list_of_tensors)` | add tensors up, keeping the graph |

### Checkpointing

| Call | Result |
| --- | --- |
| `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)` | `fn(*args)`, recomputed instead of saved |
| `torch.utils.checkpoint.checkpoint_sequential(functions, segments, input, use_reentrant=False)` | the segmented version for a sequence of modules |
| `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` | context manager: `pack` sees every saved tensor |
| `torch.is_grad_enabled()` | whether the forward pass is building a graph at all |

### Precision

| Call | Result |
| --- | --- |
| `torch.isfinite(t)`, `torch.isinf(t)`, `torch.isnan(t)` | elementwise checks |
| `torch.isfinite(t).all()` | one 0-dim bool tensor for the whole tensor |
| `t.item()`, `bool(t)` | a 0-dim tensor to a Python value |
| `torch.finfo(torch.float16).max` | 65504.0 |
| `x.float()`, `x.to(torch.float32)`, `y.to(x.dtype)` | the upcast sandwich |
| `torch.amp.GradScaler(device, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000)` | the reference implementation of 18.5 |
| `torch.autocast(device_type, dtype)` | automatic mixed precision for a region |

## Traps

- Initializing `m` to 0 instead of `-inf`: a row whose values are all negative then reports 0 as its maximum, and the test compares `m` with the true row maximum.
- Forgetting the correction factor on the old `l`, or applying it to the new block's sum instead of the old one.
- Dropping `keepdim=True`, so `block - m` broadcasts `(N, Sb)` against `(N,)` along the wrong axis or fails.
- Normalizing each block of `chunked_softmax` by its own partial `l` instead of the final one.
- Exponentiating scores before subtracting the running maximum: large scores become `inf` and the output NaN.
- Rescaling `l` but not the output accumulator `o`, or rescaling them by different factors.
- Dividing by `l` inside the key loop instead of once after it.
- Building the causal mask of a tile from tile-local indices instead of absolute positions.
- Building the tile mask from `block_size` instead of the actual, possibly shorter, slice.
- Letting a row whose `m` is still `-inf` meet a fully masked tile: $\exp(-\infty - (-\infty))$ is NaN.
- Forgetting the $1 / \sqrt{d_{\text{head}}}$ scale, or using `D` instead of `Dh`.
- Averaging per-chunk mean losses, which is wrong whenever the last chunk is shorter, instead of summing and dividing by `N` once.
- Chunking the loss under `torch.no_grad()`, or detaching anything, and then wondering why nothing trains.
- Believing that chunking alone shrinks the memory autograd saves for the backward pass.
- Chunking `chunked_logsumexp` over rows instead of the vocabulary axis, or returning `(N, 1)` instead of `(N,)`.
- Leaving out `use_reentrant=False`.
- Counting saved tensors under `torch.no_grad()`, or after the forward pass has already run outside the hook.
- Returning a 0-dim tensor where a Python `bool` is expected: `tensor(False) is False` is `False`.
- Checking only some of the gradients for `inf`.
- Stepping the optimizer on a step where `inf` was found, or halving the scale but forgetting to skip the step.
- Doubling the scale on a step with an overflow because the growth interval was also reached.
- A hand-written softmax in fp16 with logits above about 11.
- Returning float32 from a function that a half-precision caller expects to return its own dtype.

## Ready when

Without looking anything up you can:

- [ ] write the online softmax update, run it on a short row by hand, and explain why it is exact rather than approximate
- [ ] say what `m` and `l` start at, and why those values make the first update correct
- [ ] say which three running statistics flash attention carries per query row, their shapes, and which two take the same correction
- [ ] build the causal mask of one tile from absolute positions, and say when a tile can be skipped
- [ ] say exactly when $\exp(-\infty - (-\infty))$ can occur in flash attention, and how skipping prevents it
- [ ] explain why flash attention is faster on a GPU despite doing more arithmetic
- [ ] compute the size of the logits for a given `N` and `V`, and chunk the loss over tokens with the right averaging
- [ ] say why gradients still reach `weight` when it is used once per chunk
- [ ] explain why chunking alone does not shrink saved memory, and what checkpointing each chunk changes
- [ ] list what `x @ w` and `tanh` save for the backward pass, and what `checkpoint` keeps instead
- [ ] state the time cost of checkpointing, and where $O(\sqrt{L})$ comes from
- [ ] count saved tensors with `saved_tensors_hooks`, and say why the count is 0 under `no_grad`
- [ ] explain loss scaling, the `GradScaler` rules and their priority, and why bf16 does not need any of it
- [ ] compute a reduction safely in fp16 with the upcast sandwich

## Going deeper

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Milakov and Gimelshein 2018: the online softmax on its own, three pages.
- [FlashAttention](https://arxiv.org/abs/2205.14135), Dao et al. 2022, and [FlashAttention-2](https://arxiv.org/abs/2307.08691), Dao 2023, the second of which has the loop order used here.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), Chen et al. 2016, where the $\sqrt{L}$ checkpointing result comes from.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740), Micikevicius et al. 2017, for fp16 training with master weights and loss scaling.
- [PyTorch's `torch.utils.checkpoint` documentation](https://pytorch.org/docs/stable/checkpoint.html), for the differences between the reentrant and non-reentrant versions.
- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009), Wijmans et al. 2024, on computing cross-entropy without ever materializing the logits, gradients included.
