# Topic 17 - Generation: sampling and the KV cache

> **Exercise** `test_17_sampling_and_kv_cache.py`

Topic 15 built a model that maps a sequence to logits, and topic 16 trained it.
Generation is what happens after: pick a token from the last position's logits, append it, and run again.

Two independent things live here.
Sampling is four lines of code that decide every user-visible property of the model, whether it rambles, repeats itself, or sounds robotic, with no involvement from the weights.
The KV cache is the single optimization that makes autoregressive generation practical, turning a quadratic cost per token into a linear one.

The invariant the tests check, and the one that matters, is that caching must produce **exactly** the same numbers as a full forward pass.

## The mental model

### 17.1 Filtering the logits

The pipeline is always applied in this order:

```
logits -> temperature -> top-k -> top-p -> softmax -> multinomial
```

Every filter operates on **logits**, and everything filtered out is set to `-inf` so the softmax assigns it exactly zero and, crucially, **renormalizes what remains**.
That automatic renormalization is why filtering in logit space is the right design: you never divide by a surviving-probability sum yourself.

**Temperature** divides the logits, per topic 10.
Below 1 the distribution sharpens toward the argmax, above 1 it flattens.
`T = 0` is a special case, since dividing by zero is undefined; it means greedy decoding, and the conventional implementation produces a distribution with all its mass on the argmax, for example by filling with `-inf` and placing `0` at the argmax.

**Top-k** keeps the `k` highest logits and sets the rest to `-inf`.
The idiom is `topk`, then either scatter the kept values into an `-inf` tensor or compare against the `k`-th largest value.
It is crude: `k` is fixed regardless of how confident the model is at this position.

**Top-p, or nucleus sampling**, fixes exactly that by keeping the smallest set of tokens whose cumulative probability reaches `p`.
When the model is confident that might be two tokens; when it is uncertain, several hundred.

The algorithm:

1. Sort the logits descending and softmax them.
   Sorting logits and sorting probabilities give the same order, since softmax is monotone.
2. Take the cumulative sum of the sorted probabilities.
3. Keep every token whose cumulative sum **before it** is less than `p`.
   That formulation is what keeps the threshold-crossing token itself, which is required: otherwise with `p = 0.9` and a single token of probability 0.95 you would keep nothing at all.
   In practice it is a shift of the cumsum by one position, or a comparison of `cumsum - probs < p`.
4. Scatter the keep-mask back into the original token order, and `masked_fill` the rest with `-inf`.

That last step is the one to be careful with: after `sort` you are in sorted order, and the mask has to be mapped back through the sort indices before it can be applied.

**Repetition penalty** divides the logits of already-generated tokens by `penalty` when they are positive and multiplies when they are negative.
The asymmetry exists because a plain division would *increase* a negative logit toward zero, making an unlikely token more likely, which is the opposite of the intent.
Both branches push the logit down.
`penalty = 1.0` is a no-op, and typical values are 1.05 to 1.2.

**Sampling itself** is `torch.multinomial(probs, num_samples=1)`, which takes probabilities, not logits and not log-probabilities.
Passing a `generator=` is what makes a test reproducible.

### 17.2 The KV cache

**The problem.**
Generating token `t` naively means running the whole prefix of length `t` through the model, which costs `O(t^2)` in attention per step and `O(T^3)` over a whole generation.
Almost all of that is recomputation: the keys and values of positions `0..t-1` do not change when token `t` is appended, because attention is causal and nothing looks forward.

**The fix.**
Keep `k` and `v` for every layer and every past position.
At each step, run only the new token through the projections, append its `k` and `v` to the store, and attend from a single query against the whole accumulated history.

```
step t:   x is (B, 1, D)
          q is (B, H, 1, Dh)
          k, v after append are (B, H, t+1, Dh)
          scores are (B, H, 1, t+1)
          out is (B, H, 1, Dh)
```

Cost per step drops from `O(t^2)` to `O(t)`, and the whole generation from cubic to quadratic.

**The mask.**
When `T == 1`, no causal mask is needed at all: the single new query is at the newest position and is allowed to see every cached key.
The mask is only needed on the prefill call, when the whole prompt is processed at once with `T > 1`.

**Positions.**
If the model uses RoPE or learned positional embeddings, the new token's position is `len(cache)`, not zero.
This is the offset from topic 13, and forgetting it is the classic reason a cached generation diverges from the uncached one.

**What the cache is not.**
Queries are not cached: each new token produces exactly one query and never needs it again.
And nothing is cached across *sequences*; the cache belongs to one generation.

**The equivalence test** is the whole point.
A full forward pass over a length-`T` sequence, compared at the last position against a step-by-step cached generation, must agree to floating point tolerance.
When it does not, the suspects in order are: the position offset, the mask on the prefill call, appending along the wrong axis, and applying RoPE to `v`.

**Cache memory:**

```
bytes = 2 * B * L * H_kv * T * Dh * element_size
```

The `2` is `k` and `v`.
Work an example: a 70B-class model with `L = 80`, `H = 64`, `Dh = 128`, `B = 1`, `T = 8192`, bfloat16, without GQA, is about 21 GB.
For a single user.
With `H_kv = 8` instead of 64, it is 2.7 GB.

That is the entire argument for GQA from topic 14.
Note that it is linear in batch size, so serving many users concurrently multiplies it.

### 17.3 The generation loop

The naive loop, without a cache, is the right thing to write first, because it is obviously correct and is what the cached version gets checked against:

```
for _ in range(max_new_tokens):
    logits = step_fn(ids)          # (B, T, V) for the whole current sequence
    next_logits = logits[:, -1]    # only the last position matters
    next_id = pick(next_logits)    # argmax, or the sampling pipeline
    ids = torch.cat([ids, next_id[:, None]], dim=1)
```

Two things to notice.
Only the **last** position's logits are used, and everything else the model computed is thrown away, which is exactly the waste the cache eliminates.
And the new token is appended along the time axis with `cat`, which is topic 02's KV-cache observation showing up literally.

In a real implementation this loop also tracks an EOS token per sequence and stops when all sequences have finished, using the cumsum trick from topic 05 to mask anything generated after an EOS.

## API you need

### Sampling

| Call | Result |
| --- | --- |
| `logits / temperature` | temperature |
| `torch.topk(logits, k, dim=-1)` | `(values, indices)`, descending |
| `torch.sort(logits, dim=-1, descending=True)` | `(values, indices)` |
| `probs.cumsum(-1)` | the nucleus threshold |
| `mask.scatter(-1, sort_idx, sorted_mask)` | map a mask back to the original order |
| `logits.masked_fill(drop, float("-inf"))` | apply the filter |
| `torch.full_like(logits, float("-inf"))` | the base for a scatter-based filter |
| `torch.multinomial(probs, 1, generator=g)` | sample, takes probabilities |
| `logits.argmax(-1)` | greedy |
| `torch.gather(logits, -1, generated)` | read the logits of generated tokens |
| `logits.scatter_(-1, generated, penalized)` | write them back |
| `torch.Generator().manual_seed(n)` | reproducible sampling |

### The cache

| Call | Result |
| --- | --- |
| `torch.cat([cached, new], dim=-2)` | append along time for `(B, H, T, Dh)` |
| `cache.k.shape[-2]` | the current length |
| `torch.no_grad()` / `torch.inference_mode()` | generation never needs gradients |
| `x.element_size()` | the memory arithmetic |

### A worked example on neutral data

```python
>>> logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]])
>>> probs = logits.softmax(-1)
>>> probs
tensor([[0.6439, 0.2369, 0.0871, 0.0321]])
>>> probs.cumsum(-1)
tensor([[0.6439, 0.8808, 0.9679, 1.0000]])
```

With `p = 0.9`: the tokens whose *preceding* cumulative sum is below 0.9 are indices 0, 1 and 2, so three tokens survive and the fourth is dropped.
Using the inclusive cumsum with `<= p` instead would keep only two, dropping the token that crosses the threshold, which is the off-by-one to watch for.

## Traps

- `torch.multinomial` takes probabilities, not logits and not log-probabilities.
- Filters must be applied to logits before the softmax, so that the renormalization is automatic.
- Top-p off by one: the token that crosses the threshold is kept.
- Forgetting to map the top-p mask back from sorted order to original order.
- `temperature = 0` divides by zero; handle it as greedy.
- A repetition penalty that only divides makes negative logits *more* likely.
- Forgetting the position offset when generating with a cache, which silently changes the output.
- Building a causal mask when `T == 1`, which either does nothing or masks the only query out entirely.
- Appending to the cache along the wrong axis; for `(B, H, T, Dh)` it is `dim=-2`.
- Caching `q`, which is never needed again.
- Generating without `torch.no_grad()`, which builds a graph that grows with every token.
- Using `logits[:, -1]` versus `logits[-1]`: the first is the last time step, the second is the last batch element.

## Ready when

Without looking anything up you can:

- [ ] list the sampling pipeline in order and say why each filter writes `-inf` rather than renormalizing
- [ ] implement top-p including the sort, the cumsum, the off-by-one and the scatter back
- [ ] say why the repetition penalty has two branches
- [ ] explain what the KV cache stores, what it does not store, and the cost it saves
- [ ] say why no causal mask is needed for a single decoded token
- [ ] write the KV cache memory formula and compute it for a realistic model
- [ ] name the three most likely causes when cached and uncached generation disagree

## Context



**The KV cache is what limits a serving system.**
Weights are loaded once and shared by every concurrent request; the cache is per request and grows with every token.
So the number of users a GPU can serve at once is set by how many caches fit beside the weights, and the whole field of inference serving is about that number.
vLLM's paged attention was the breakthrough: instead of reserving a contiguous block for each sequence's maximum length, it allocates the cache in fixed-size pages like virtual memory, which removes the internal fragmentation and raises concurrency by several times on the same hardware.

**Why the repetition penalty is unfashionable.**
It cannot tell a repeated phrase from the word "the", so pushing it hard produces text that avoids ordinary function words and reads strangely.
It was necessary when base models genuinely looped, and instruction-tuned models mostly do not, so modern stacks either leave it at 1.0 or apply gentler variants such as a frequency penalty that scales with the number of occurrences rather than penalizing the first repeat as hard as the tenth.

**Why greedy decoding is not obviously best.**
Maximizing likelihood token by token produces text that is bland and prone to loops, which is the observation the nucleus sampling paper opens with: human text is not the most probable text, and sits in a region of moderate rather than maximal likelihood.
That is why sampling with a truncated tail beats both pure sampling, which occasionally picks nonsense from the tail, and pure greedy, which never surprises.

## Going deeper

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), Holtzman et al., which introduced nucleus sampling.
- [Fast Transformer Decoding](https://arxiv.org/abs/1911.02150), Shazeer 2019, the multi-query attention paper, where the KV cache memory argument is made most clearly.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180), the vLLM paper.
