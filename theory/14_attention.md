# Topic 14 - Attention

> **Exercise** `test_14_attention.py` · **Reference** `solutions/sol_14_attention.py` · **Assumes** topics 01-13

## Where this fits

This is attention itself, and it is the only place in the whole model where one token reads another.
Topic 13 gave every token a vector.
Everything those vectors have been through so far happened to each token in isolation.
Here they finally interact, and in topic 15 the result gets wrapped in a norm, a residual add and a feed-forward branch to become one transformer block.

What you need from earlier topics:

| From | What exactly |
| --- | --- |
| 02 | Splitting the last axis into heads and putting the head axis in front |
| 03 | Broadcasting a `(B, S)` mask against `(B, H, T, S)` scores |
| 06 | Batched matmul, and why a dot product over `Dh` terms has variance `Dh` |
| 10 | Softmax along one chosen axis |

The exercise re-declares `split_heads` and `merge_heads` at the top.
That is deliberate: you write them again from memory rather than importing them.

## What you build here

Ten components, twelve stubs: the two modules each have an `__init__` and a `forward`.
Read the table first, then read the mental model for whatever you cannot already picture.

### 14.0 Helpers

| Component | In -> out | What it must do |
| --- | --- | --- |
| `split_heads(x, num_heads)` | `(B, T, D) -> (B, H, T, Dh)` | Cut the last axis into `H` parts, then move the head axis in front of `T`. `Dh = D // H` |
| `merge_heads(x)` | `(B, H, T, Dh) -> (B, T, D)` | The exact inverse, so that `merge_heads(split_heads(x, H))` returns `x` |

### 14.1 The core

| Component | In -> out | What it must do |
| --- | --- | --- |
| `scaled_dot_product_attention(q, k, v, mask=None)` | `q (..., T, Dh)`, `k, v (..., S, Dh)` -> `(..., T, Dh)` | Score, mask, softmax, weighted sum. `mask` is a bool broadcastable to `(..., T, S)` where `True` means allowed |
| `attention_weights(q, k, mask=None)` | `q (..., T, Dh)`, `k (..., S, Dh)` -> `(..., T, S)` | The same computation, stopped at the softmax and returned as is. Every row sums to 1 |
| `causal_attention(q, k, v)` | `(B, H, T, Dh) -> (B, H, T, Dh)` | Build the mask inside from the lengths of `q` and `k`, which here are equal. Position `i` may use keys `j <= i` |
| `attention_with_padding(q, k, v, pad_mask)` | `q, k, v (B, H, T, Dh)`, `pad_mask (B, S)` bool -> `(B, H, T, Dh)` | Causal and padding at once. `True` in `pad_mask` means a real token. The mask has to broadcast over heads and over all queries |

### 14.2 The modules

| Component | In -> out | What it must do |
| --- | --- | --- |
| `MultiHeadSelfAttention(d_model, num_heads)` | `(B, T, D) -> (B, T, D)` | Four submodules named `q_proj`, `k_proj`, `v_proj`, `output_proj`, each `nn.Linear(D, D, bias=False)`. Causal. The names are asserted by the tests here and in topic 15 |
| `GroupedQueryAttention(d_model, num_heads, num_kv_heads)` | `(B, T, D) -> (B, T, D)` | Same, except `k_proj` and `v_proj` produce only `num_kv_heads` heads. Each KV head serves `G = H / H_kv` consecutive query heads. Causal, no biases |

### 14.3 Cost

| Component | In -> out | What it must do |
| --- | --- | --- |
| `attention_matrix_bytes(batch, heads, seq_len, dtype)` | ints -> `int` | The bytes one materialized `(B, H, T, T)` score tensor occupies in that dtype |
| `attention_flops(batch, heads, seq_len, d_head)` | ints -> `int` | The FLOPs of the two matmuls, projections excluded, at `2mnk` per matmul |

## The mental model

### 14.0 Heads

A head is a slice of the `D` channels that attends on its own.
Splitting is pure bookkeeping: `D` becomes `(H, Dh)` and the head axis moves in front of the sequence axis, so that every matmul afterwards batches over `(B, H)` and never mixes heads.
No parameters are involved and no numbers change.

### 14.1 Scaled dot-product attention

```
scores  = q @ k.transpose(-2, -1) / sqrt(Dh)        (..., T, Dh) x (..., Dh, S) -> (..., T, S)
scores  = scores.masked_fill(~mask, -inf)           optional
weights = softmax(scores, dim=-1)                   over the KEY axis
out     = weights @ v                               (..., T, S) x (..., S, Dh) -> (..., T, Dh)
```

Read it as a soft dictionary lookup.
Each query is compared against every key by a dot product, the comparisons are turned into a distribution, and the output is the average of the values under that distribution.
The softmax makes the keys compete: attention hands out a fixed budget of 1 per query.

Four decisions, each of which is a bug if you get it wrong.

**The scale is `1/sqrt(Dh)`**, the head dimension, never `d_model`.
A dot product of `Dh` unit-variance terms has variance `Dh`, so without the correction the softmax saturates and the gradient dies.

**The softmax is over `dim=-1`**, the keys.
Each query's weights sum to 1 across the keys.
Softmaxing over the query axis gives a function that still runs, still trains a little, and is not attention.

**The mask goes on the scores, before the softmax**, as `-inf` at forbidden positions.
`exp(-inf) == 0` removes them, and the survivors renormalize on their own because the denominator only sums what is left.
Masking after the softmax needs a manual renormalization, and getting that wrong is how future information leaks in.

**`q` and `k` share a length only in self-attention.**
Writing the function for `(..., T, Dh)` and `(..., S, Dh)` separately costs nothing and makes it correct for cross-attention and for cached decoding, where `T == 1` and `S` is the whole history.

The `(..., T, S)` weight matrix is worth returning from a function of its own.
It is the standard debugging picture, and one look at it tells you whether a mask does what you think.

#### Causal masks

Query `i` may see key `j` when `j <= i`.
As a comparison of two aranges that is `arange(T)[:, None] >= arange(S)[None, :]`, a lower-triangular bool of shape `(T, S)` that broadcasts against `(B, H, T, S)` unaided.

When `T != S`, as in cached decoding, the alignment matters: the query at relative position `i` sits at absolute position `S - T + i`, so the mask becomes `arange(S - T, S)[:, None] >= arange(S)[None, :]`.
This exercise assumes `T == S`, but the general form is what topic 17 needs.

#### Padding masks

A `(B, S)` mask says which key positions are real tokens rather than filler.
Reshaped to `(B, 1, 1, S)` it broadcasts over heads and over every query, which is the pattern from topic 03.

Combining causal and padding is a logical `&`.
The shapes `(T, S)` and `(B, 1, 1, S)` broadcast together to `(B, 1, T, S)` without help.

### 14.2 Multi-head attention

The module, in order:

```
x  (B, T, D)
 -> q_proj, k_proj, v_proj                each nn.Linear(D, D, bias=False)
 -> split_heads                           (B, H, T, Dh)
 -> scaled dot-product attention, causal  (B, H, T, Dh)
 -> merge_heads                           (B, T, D)
 -> output_proj                           nn.Linear(D, D, bias=False)
```

One softmax can only express one thing: a single distribution over keys, so a single-head layer sees one relationship per position.
Splitting `D` into `H` independent subspaces lets the layer track several at once, and `output_proj` mixes their results back together.
This costs no extra parameters, because `H * Dh == D`.

#### Grouped-query attention

GQA changes exactly one thing.
There are `H` query heads but only `H_kv` key/value heads, with `H` a multiple of `H_kv`, and each KV head is shared by a group of `G = H / H_kv` query heads.

```
q_proj:      D -> H * Dh
k_proj:      D -> H_kv * Dh
v_proj:      D -> H_kv * Dh
output_proj: H * Dh -> D
```

`k` and `v` are then expanded from `(B, H_kv, S, Dh)` to `(B, H, S, Dh)`, each KV head repeated `G` times **consecutively**, which is the unsqueeze/expand/reshape from topic 02.
From there it is ordinary multi-head attention.
`H_kv == H` is plain MHA and `H_kv == 1` is multi-query attention; everything in between is GQA.
Why anyone does this is in [Context](#context) below.

### 14.3 Cost

Two numbers, both of which you compute rather than look up.

The score matrix is `(B, H, T, S)` and is materialized in memory:

```
bytes = B * H * T * S * element_size
```

The matmuls around it, excluding the projections and counting `2mnk` per matmul:

```
Q K^T:       2 * B * H * T * S * Dh
weights V:   2 * B * H * T * S * Dh
total:       4 * B * H * T * S * Dh
```

Both are quadratic in the sequence length when `T == S`.
Put a 32k context into the first formula before you read the Context section, because the number is the point.

## API you need

| Call | Result |
| --- | --- |
| `q @ k.transpose(-2, -1)` or `k.mT` | the score matmul |
| `math.sqrt(d_head)` | the scale |
| `scores.masked_fill(~mask, float("-inf"))` | apply a boolean mask |
| `torch.finfo(dtype).min` | the fp16-safe alternative to `-inf` |
| `F.softmax(scores, dim=-1)` | over the key axis |
| `torch.ones(T, S, dtype=torch.bool).tril()` | causal mask |
| `torch.arange(T)[:, None] >= torch.arange(S)[None, :]` | the same, generalizable to `T != S` |
| `mask_a & mask_b` | combine masks |
| `nn.Linear(d_in, d_out, bias=False)` | the projections |
| `x.reshape(B, T, H, Dh).transpose(1, 2)` | split heads |
| `rearrange(x, "b t (h d) -> b h t d", h=H)` | split heads, einops |
| `x.unsqueeze(2).expand(...).reshape(...)` | the GQA KV expansion |
| `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)` | the fused built-in, used by the tests as the reference |
| `x.element_size()` | bytes per element, for the memory arithmetic |

### A worked example on neutral data

```python
>>> B, H, T, Dh = 1, 2, 4, 8
>>> q, k, v = (torch.randn(B, H, T, Dh) for _ in range(3))
>>> causal = torch.ones(T, T, dtype=torch.bool).tril()
>>> s = (q @ k.mT / math.sqrt(Dh)).masked_fill(~causal, float("-inf"))
>>> w = s.softmax(-1)
>>> w[0, 0, 0]                    # the first query can only see itself
tensor([1., 0., 0., 0.])
>>> w.sum(-1)                     # every row sums to 1
tensor([[[1., 1., 1., 1.], [1., 1., 1., 1.]]])
>>> torch.allclose(w @ v, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-6)
True
```

That `w[0, 0, 0] == [1, 0, 0, 0]` is the fastest test of a causal mask there is: the first query has nowhere else to put its weight.

## Traps

- Scaling by `sqrt(d_model)` rather than `sqrt(d_head)`.
- Softmax over the wrong axis.
- Applying the mask after the softmax.
- Masking with `0` instead of `-inf`.
- `masked_fill` fills where `True`, so an "allowed" mask needs `~`.
- A fully masked row gives NaN, not zero.
  It happens when a padded query is also masked out as a key, so its own row has no survivors.
  Production code either never masks the diagonal or drops padded query rows from the loss.
- The head split done in the wrong order mixes channels across heads and never raises.
- Forgetting `contiguous()`, or using `view` instead of `reshape`, after a transpose when merging heads.
- In GQA, expanding KV heads with `repeat` instead of an interleaving expand pairs every query head with the wrong keys.
- Forgetting `output_proj`.
  Without it the heads never interact and the layer is much weaker, while every shape still checks out.
- Using biases in the projections, which breaks the `state_dict` contract for later topics.
- `F.scaled_dot_product_attention` reads a bool mask as "allowed" and a float mask as "added".
  Passing a bool where a float was meant silently inverts your intent.

## Ready when

Without looking anything up you can:

- [ ] write scaled dot-product attention in four lines, with the right scale and the right softmax axis
- [ ] say what each of the four axes of the score tensor means
- [ ] build a causal mask for `T != S` and say which positions the queries actually occupy
- [ ] shape a key padding mask so it broadcasts correctly, and say what the wrong shape does
- [ ] write the full MHA module with the canonical parameter names, from memory
- [ ] say what GQA changes in the projections and in the KV expansion
- [ ] compute the size of the score matrix for a 32k context

## Context


It is why the mechanics above look the way they do.

**The score matrix is the whole problem.**
Take `B=8`, `H=32`, `T=S=32768`, bfloat16.
That is `8 * 32 * 32768^2 * 2` bytes, around 550 GB, for one layer, for one of several intermediates the backward pass would need.
No GPU has that.
Doubling the context quadruples it.
This single number is why flash attention exists, and topic 18 is the response to it.

**Attention is not always the expensive part.**
The projections cost `4 * 2 * B * T * D^2`, and attention itself `4 * B * T^2 * D`, so their ratio is `T / (2 * D)`.
Attention only dominates once the sequence is longer than roughly twice `d_model`, which for `D = 4096` is about 8k tokens.
Below that, a transformer is mostly matmuls against weights, and the quadratic term is a rounding error.

**GQA is about the KV cache, not about FLOPs.**
During generation the cache holds `2 * B * H_kv * S * Dh` values, and every generated token reads all of it out of memory.
At long context and large batch that traffic, not arithmetic, is what caps decoding throughput.
Going from 32 KV heads to 8 shrinks the cache fourfold for a quality loss close to noise, which is why Llama 2 70B, Mistral and essentially everything since use GQA.
Topic 17 builds the cache this is about.

**The built-in.**
`F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)` dispatches to a flash attention kernel when the shapes and dtypes allow it.
Real code calls it, and prefers `is_causal=True` to an explicit mask so the kernel can skip masked blocks entirely.
Knowing it exists is essential; writing it by hand once is what this topic is for.

## Going deeper

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), still the clearest statement of the mechanism.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245), Ainslie et al. 2023.
- [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), whose doc page includes the reference implementation in Python.
