# Topic 18 - Flash attention and memory

> **Exercise** `test_18_flash_and_memory.py`


No new parts of the model here. This topic makes the ones that already exist fit in memory.

Topic 14 computed the size of the attention matrix and found 550 GB.
Topic 15 computed the activation memory of a whole model.
Topic 10 built a loss over a vocabulary that at 128k entries produces the single largest tensor in a training step.
This file is the answer to all three, and every task in it asks the same question: **how do you compute exactly the same result without ever allocating the huge intermediate?**

- The `(T, T)` attention matrix does not fit, so the softmax is computed online, blockwise.
- The `(N, V)` logits do not fit, so cross-entropy is computed in chunks.
- The activations of all `L` layers do not fit, so they are recomputed instead of stored.
- fp16 underflows the gradients, so the loss is scaled.

The technique shared by the first two is the most reusable idea in the repository: a reduction whose running state is small enough to carry, updated block by block, giving an answer identical to the one-shot version.

This is [CS336](https://cs336.stanford.edu/) assignment 2 territory, and it is the hardest file here.

## The mental model

### 18.1 Online softmax

The stable softmax of topic 10 needs two passes: one to find the maximum, one to exponentiate and sum.
That means holding all the logits.

The online version carries two running statistics instead:

- `m`, the maximum seen so far;
- `l`, the sum of `exp(x - m)` over everything seen so far, **relative to the current `m`**.

When a new block arrives the maximum may increase, and every previously accumulated exponential was computed against the old maximum.
The correction is a single multiplication:

```
m_new = max(m_prev, max(block))
l_new = l_prev * exp(m_prev - m_new) + sum(exp(block - m_new))
```

Read the correction factor: `exp(m_prev - m_new)` is at most 1, and it rescales the old sum from the old reference point to the new one.
Because `exp(a - m_old) * exp(m_old - m_new) == exp(a - m_new)` exactly in real arithmetic, the result is algebraically identical to the one-pass computation.

At the end, `logsumexp = m + log(l)`, and the softmax of any element is `exp(x - m) / l`.

**Why this is exact and not an approximation.**
Nothing is dropped or truncated, the rescaling is an identity, and every exponent stays at most zero.
The final bits may differ from a different summation order, which is ordinary floating point, not an approximation in the algorithm.

`chunked_softmax` does this in **two passes** over the blocks: the first accumulates `(m, l)`, the second recomputes each block's exponentials and divides by the final `l`.
Two passes, `O(chunk)` memory, exact answer.

Initialize `m` to `-inf` and `l` to `0`.
`exp(-inf - (-inf))` is NaN, so a block that is entirely masked has to be skipped rather than folded in.

### 18.2 Flash attention

Flash attention applies the online softmax to attention and adds one thing: the **output accumulator is rescaled too**.

The output for a query is `sum_j (exp(s_j - m) / l) * v_j`.
Carry the unnormalized accumulator `o = sum_j exp(s_j - m) * v_j` alongside `m` and `l`, and updating `m` requires rescaling `o` by the same `exp(m_old - m_new)`.
Dividing by `l` at the very end produces the correct output.
The `(T, S)` score matrix never exists in full; only a `(block_T, block_S)` tile does.

The v2 loop structure:

```
for each block of queries (rows):
    m = -inf, l = 0, o = 0
    for each block of keys (columns):
        s = q_block @ k_block^T / sqrt(Dh)
        apply the mask to this tile
        m_new = max(m, rowmax(s))
        correction = exp(m - m_new)
        p = exp(s - m_new)
        l = l * correction + rowsum(p)
        o = o * correction + p @ v_block
        m = m_new
    write o / l
```

`o` and `l` take the *same* correction factor, which is the whole trick in one line.

Three implementation details the exercise calls out.

**Causal blocks.**
With `causal=True`, a key block entirely above the diagonal for this query block is fully masked, and folding it in produces `exp(-inf - (-inf))`, which is NaN.
Skip such blocks entirely.
This is also where the real speedup comes from: in a causal model roughly half the tiles need never be computed.

**Ragged sizes.**
`T` and `S` are not required to divide evenly by `block_size`, so the loop has to handle a short final block.
Python slicing already does the right thing, which is why writing the bounds as slices rather than as arithmetic is easier to get right.

**Performance is not the goal here.**
Written in Python this will be slower than `F.scaled_dot_product_attention`.
The exercise is about understanding the algorithm; why the real kernel is fast is in [Context](#context).

### 18.3 Cross-entropy without materializing the logits

For a vocabulary of 128k and 8k tokens in a batch, the logit tensor is about 4 GB in float32, and both the forward activation and its gradient have to exist.
On a large-vocabulary model this is the single biggest tensor in the whole training step.

The fix is to chunk over **tokens**: for each chunk of rows of `hidden`, compute that chunk's logits, compute its contribution to the loss, and let those logits be freed before the next chunk is allocated.

Per topic 10, the loss for one token is `logsumexp(logits) - logits[target]`, so each chunk contributes a sum over its own rows, and the total divided by `N` is the mean.

The subtlety is **gradients**.
They must still flow into `hidden` and into `weight`.
This works naturally as long as the chunked computation is a normal differentiable graph: autograd keeps each chunk's small intermediates and sums the contributions into `weight.grad` automatically, because `weight` is used `N/chunk` times.
Peak memory is one chunk's logits rather than all of them.

Be careful with the averaging.
Summing per-chunk means and dividing by the number of chunks is only correct when the chunks are equal in size, which the last one is not.
Summing the per-token losses and dividing once by `N` is always correct.
This is the same trap as the gradient-accumulation division in topic 16.

**Chunked logsumexp** on its own is the online update from 18.1 applied along the vocabulary axis.

### 18.4 Gradient checkpointing

Autograd saves the tensors it needs for the backward pass.
For a stack of `L` transformer blocks that is `O(L)` sets of activations, and they are typically the largest memory consumer in training.

**Checkpointing** trades compute for memory: wrap a block so its intermediate activations are *not* saved, and recompute them by re-running the block's forward pass during the backward pass.
Memory drops from `O(L)` to `O(1)` per checkpointed segment, plus the block boundaries, at the cost of roughly one extra forward pass, which is about 30% more time end to end.

```python
from torch.utils.checkpoint import checkpoint

x = checkpoint(block, x, use_reentrant=False)
```

`use_reentrant=False` is the newer implementation and should always be used: the reentrant version does not support many ordinary things and is on its way out.

**Measuring what it buys.**
`torch.autograd.graph.saved_tensors_hooks(pack, unpack)` is a context manager whose `pack` callable is invoked for every tensor autograd saves.
Counting `pack` calls during a forward pass is a direct measurement of the graph's memory footprint, and it is the right way to *demonstrate* that checkpointing works rather than assert it.
`pack` must return something `unpack` can turn back into the tensor; returning the tensor itself and the identity is the trivial pair.

### 18.5 Mixed precision

**The problem with fp16.**
Its exponent range is narrow: the smallest normal value is about `6e-5`, and gradients in a large model routinely go below that, where they flush to zero and vanish.
bfloat16 does not have this problem, since it has float32's exponent range, which is why it is preferred whenever the hardware supports it.

**Loss scaling** is the fix for fp16.
Multiply the loss by a large constant `S` before `backward()`; by the chain rule every gradient is multiplied by `S`, lifting the small ones back into representable range.
Before the optimizer step, divide the gradients by `S` again.
The model never notices.

**The dynamic part.**
Too small an `S` does not solve the underflow, too large an `S` overflows to `inf`.
`GradScaler` adapts:

- Unscale the gradients and check for `inf` or `nan`.
- If any are found, **skip the optimizer step entirely** and halve the scale.
- If none are found for `growth_interval` consecutive steps, double the scale.

Skipping the step is important: gradients containing `inf` must never reach the parameters, because one such step destroys the model.
Losing an occasional step costs nothing.

The check is `torch.isfinite(g).all()` across all gradients, and it must happen after unscaling, on the values the optimizer would actually use.

**Softmax in fp16.**
Logits in fp16 with magnitudes above about 11 already overflow when exponentiated, since `exp(11) > 65504`.
The fix is the upcast sandwich from topic 11: compute in float32, which includes the max subtraction that makes the largest exponent exactly `exp(0)`, and cast the result back to float16.
The output probabilities are all in `[0, 1]` and represent perfectly well in fp16; it is only the intermediate exponentials that overflow.

## API you need

### Blockwise computation

| Call | Result |
| --- | --- |
| `torch.maximum(a, b)` | elementwise max, for `m_new` |
| `x.amax(dim, keepdim=True)` | the block maximum |
| `torch.full_like(x, float("-inf"))` | initialize `m` |
| `x.split(chunk, dim)` / `range(0, n, chunk)` | iterate blocks |
| `x[i : i + chunk]` | ragged-safe slicing |
| `torch.exp`, `torch.log` | the online update |

### Checkpointing

| Call | Result |
| --- | --- |
| `torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)` | recompute instead of save |
| `torch.utils.checkpoint.checkpoint_sequential(modules, segments, x)` | the segmented helper |
| `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` | observe every saved tensor |

### Precision

| Call | Result |
| --- | --- |
| `torch.isfinite(x)`, `torch.isinf(x)`, `torch.isnan(x)` | the checks |
| `x.isfinite().all()` | a single bool for a whole tensor |
| `torch.finfo(torch.float16).max` | 65504 |
| `torch.amp.GradScaler` | the reference implementation |
| `torch.autocast(device_type, dtype)` | automatic mixed precision |
| `x.to(torch.float32)` / `.to(torch.float16)` | the manual sandwich |

### A worked example on neutral data

```python
>>> x = torch.randn(1, 8)
>>> m, l = torch.full((1, 1), float("-inf")), torch.zeros(1, 1)
>>> for blk in x.split(3, dim=-1):
...     m_new = torch.maximum(m, blk.amax(-1, keepdim=True))
...     l = l * (m - m_new).exp() + (blk - m_new).exp().sum(-1, keepdim=True)
...     m = m_new
>>> torch.allclose(m + l.log(), torch.logsumexp(x, -1, keepdim=True))
True
```

`m` starting at `-inf` makes the first `(m - m_new).exp()` equal `exp(-inf)`, which is 0, so the term vanishes correctly.
It is only `-inf` minus `-inf` that produces NaN, which is the fully-masked-block case.

## Traps

- `exp(-inf - (-inf))` is NaN.
  Skip fully masked blocks; do not fold them in.
- Forgetting to rescale the output accumulator `o` by the same factor as `l`.
- Assuming the block sizes divide evenly.
- Averaging chunked losses by dividing per chunk instead of summing and dividing once by `N`.
- Chunking cross-entropy inside `torch.no_grad()` and then wondering why nothing trains.
- `use_reentrant=True`, the old default, breaks in ways that are hard to diagnose; pass `False`.
- Checkpointing a block whose forward pass is not deterministic gives different activations on the recompute unless the RNG state is preserved, which `checkpoint` handles but only if you let it.
- Checking for `inf` before unscaling tests the wrong numbers.
- Stepping the optimizer on a step where `inf` was found destroys the model.
- Halving the scale but forgetting to skip the step, or skipping the step but forgetting to halve.
- Computing a softmax in fp16 with logits above about 11.
- Returning float32 from a function that a bf16 model expects to return bf16.

## Ready when

Without looking anything up you can:

- [ ] write the online softmax update and explain why it is exact rather than approximate
- [ ] say what three running statistics flash attention carries per query block, and which two get rescaled together
- [ ] explain why flash attention is faster despite doing more arithmetic
- [ ] say why a fully masked causal block must be skipped
- [ ] describe how chunked cross-entropy keeps gradients flowing into both `hidden` and `weight`
- [ ] state the memory and compute trade of gradient checkpointing and how you would measure it
- [ ] explain loss scaling, the two `GradScaler` rules, and why fp16 needs it while bf16 does not

## Context



**Why flash attention is faster while doing more arithmetic.**
Attention is memory-bandwidth bound, not compute bound.
The naive version writes an `O(T^2)` matrix out to HBM and reads it back twice, and that traffic dominates the runtime.
Flash attention keeps each tile in SRAM, roughly an order of magnitude faster to access, and never writes the score matrix anywhere.
Memory drops from `O(T^2)` to `O(T)` and wall clock improves several times over, which is an unusually good trade.
The backward pass recomputes the tiles rather than storing them, saving only `(m, l)` per row, which is the same compute-for-memory bargain as checkpointing.

**Where the `O(sqrt(L))` in the folklore comes from.**
Checkpointing every layer gives `O(1)` activations but pays a full extra forward pass.
Checkpointing every `sqrt(L)`-th layer instead stores `sqrt(L)` boundaries and recomputes segments of `sqrt(L)` layers, which minimizes total memory for a given recompute budget.
That result is from Chen et al. 2016, and it is why the figure is quoted as `sqrt(L)` even though a naive per-layer scheme is `O(1)`.

**Why everyone moved from fp16 to bf16.**
All of section 18.5 is a workaround for one property: fp16's narrow exponent range.
bf16 keeps float32's exponent bits, so gradients do not underflow, no scaling is needed, and the whole `GradScaler` state machine disappears.
The price is three fewer mantissa bits, which turned out not to matter for training.
fp16 plus loss scaling remains on hardware without bf16 support, and the machinery is worth understanding because it still shows up in older code and in inference stacks.

**What autocast does and does not touch.**
It runs matmuls and convolutions in half precision and keeps softmax, normalization, and loss reductions in float32, which is the same list of "reductions that need range" this topic keeps arriving at.
A hand-written kernel inside an autocast region is not on that list and has to do its own upcasting, as topic 11 warns.

## Going deeper

- [FlashAttention](https://arxiv.org/abs/2205.14135) and [FlashAttention-2](https://arxiv.org/abs/2307.08691), Dao et al.
- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867), Milakov and Gimelshein, which is the online softmax on its own, three pages.
- [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174), Chen et al., where the `sqrt(L)` checkpointing result comes from.
- [Mixed Precision Training](https://arxiv.org/abs/1710.03740), Micikevicius et al., for loss scaling.
