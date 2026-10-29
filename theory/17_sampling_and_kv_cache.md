# Topic 17 - Generation: sampling and the KV cache

> **Exercise** `test_17_sampling_and_kv_cache.py`

## Why this topic

Topic 15 built a model that maps a row of token ids to logits, and topic 16 trained it.
A trained model still does not write anything.
**Generation** is the loop around it: run the model, take the scores at the last position, pick one token, append it to the input, and run again, until enough tokens have been produced.

Two independent things live inside that loop, and this topic builds both.

**Sampling** is how one token gets picked out of `V` scores.
It is a handful of tensor operations with no weights involved, and yet it decides most of what a user notices about a model: whether it rambles, repeats itself, or sounds robotic.
Always picking the most likely token sounds like the safe choice and is not.
Text built from the most probable token at every step is bland and falls into loops, and human text sits in a region of moderate rather than maximal likelihood.
Sampling at random from the full distribution fails the other way: once in a while it picks one of the thousands of tokens in the long tail, and the sentence derails.
The filters in 17.1 are the compromise: sample, but only from the part of the distribution that makes sense.

The **KV cache** is the one optimization that makes generation practical.
Without it, every new token reruns the model over the whole prefix, and almost all of that work recomputes numbers that were already computed on the previous step.
With it, each step runs the model over one token only.
The invariant this topic tests, and the one that matters, is that the cache changes the cost and **nothing else**: step-by-step generation with a cache must give exactly the numbers a single full forward pass gives.

The cache also explains a good part of LLM serving.
Its size, not the weights, decides how many users fit on a GPU at once, and grouped-query attention from topic 14 exists to make it smaller.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| logits | the model's output, `(B, T, V)`: one unnormalized score per vocabulary entry, for the token after each position |
| next-token logits | the scores at the last position only, `(B, V)`; the only part generation uses |
| autoregressive | each new token is predicted from all the tokens before it, including the ones the model just generated |
| greedy decoding | always pick the highest-scoring token; no randomness |
| sampling | draw the token at random, with the probabilities the softmax of the logits gives |
| temperature | a number the logits are divided by before the softmax, to sharpen or flatten the distribution |
| top-k | keep only the `k` highest-scoring tokens as candidates |
| top-p, nucleus | keep only the smallest set of top tokens whose probabilities add up to at least `p` |
| repetition penalty | push down the scores of tokens that were already generated |
| KV cache | the stored keys and values of every past position, for every layer |
| prefill | the first call of a cached generation: the whole prompt at once, which fills the cache |
| decode | every later call: one new token, attending to everything in the cache |
| EOS | the "end of sequence" token id; a sequence that produced it is finished |

### A refresher: `-inf` is how a token is removed

Topic 10 built the softmax, and topic 14 used the one property of it that this whole topic relies on.
`exp(-inf) = 0`, so a logit of `-inf` gets probability exactly 0, and the surviving tokens share the whole 1 among themselves in the same proportions as before:

```python
>>> torch.tensor([2.0, float("-inf"), 1.0, float("-inf")]).softmax(-1)
tensor([0.7311, 0.0000, 0.2689, 0.0000])
>>> torch.tensor([2.0, 1.0]).softmax(-1)              # the same two shares
tensor([0.7311, 0.2689])
```

That is why every filter here works on **logits** and writes `-inf` into what it removes.
The softmax at the end renormalizes the survivors automatically, and no code ever divides by "the probability that is left".

Two more facts from earlier topics come back:

- adding the same constant to every logit changes nothing after the softmax, but **multiplying** all of them by a constant does, which is what temperature is;
- the softmax preserves order: the largest logit has the largest probability, so sorting logits and sorting probabilities give the same order.

### 17.1 Filters for the logits

Every function in this section takes next-token logits of shape `(B, V)`, one row per sequence in the batch, and treats each row on its own.
They are applied in a fixed order, and `sample_from_logits` chains them:

```
logits -> temperature -> top-k -> top-p -> softmax -> multinomial
```

The order matters because top-p looks at probabilities, and the probabilities depend on the temperature and on what top-k already removed.

#### Temperature

The logits are divided by a positive number `temperature` before the softmax.

```python
>>> l = torch.tensor([2.0, 1.0, 0.0])
>>> (l / 0.5).softmax(-1)      # cold: sharper, the top token takes more
tensor([0.8668, 0.1173, 0.0159])
>>> (l / 1.0).softmax(-1)      # the model's own distribution
tensor([0.6652, 0.2447, 0.0900])
>>> (l / 2.0).softmax(-1)      # hot: flatter, the tail gets more
tensor([0.5065, 0.3072, 0.1863])
```

Dividing by a number below 1 stretches the gaps between logits, which the softmax turns into a sharper distribution; dividing by a number above 1 shrinks the gaps and flattens it.
The order of the tokens never changes, only how decisively the top one wins.
As the temperature goes to 0, all the probability moves onto the largest logit; as it goes to infinity, the distribution becomes uniform.

**Temperature 0 cannot be computed by division.**
Dividing by zero does not raise in PyTorch, it produces `inf`, `-inf` or `nan` depending on the sign, and the softmax of that is garbage:

```python
>>> torch.tensor([[-1.0, 0.0, 3.0]]) / 0.0
tensor([[-inf, nan, inf]])
>>> (torch.tensor([[1.0, 2.0, 3.0]]) / 0.0).softmax(-1)
tensor([[nan, nan, nan]])
```

So temperature 0 is a special case, by convention meaning **greedy**: the limit of the distribution as the temperature falls, all mass on the argmax.
The exercise asks `apply_temperature` to return, at temperature 0, logits whose softmax is that limit, and suggests one form: `-inf` everywhere except a `0` at the argmax of each row.
The test checks that the argmax is unchanged and that the softmax of the result puts more than 0.999 on it.
A row of `-inf` with one `0` softmaxes to an exact one-hot:

```python
>>> torch.tensor([[float("-inf"), float("-inf"), 0.0]]).softmax(-1)
tensor([[0., 0., 1.]])
```

Building it is topic 05 material: `argmax` gives one index per row, and a `scatter` writes a value at one index per row into a tensor filled with `-inf`.

#### Top-k

Keep the `k` highest logits in each row and set every other one to `-inf`.
The test gives four logits and `k = 2`, and checks that exactly the two largest stay finite.
By convention `k <= 0` means "filter off", and the function returns the logits unchanged; `sample_from_logits` uses that, since its default is `top_k=0`.

`torch.topk(x, k, dim=-1)` returns the `k` largest values and their indices along the last axis, sorted descending.
Two ways to use it:

- write the kept values back at their indices into a tensor full of `-inf` (a `scatter`, topic 05);
- take the `k`-th largest value of each row, the last column of the `topk` values, kept as a `(B, 1)` column so it broadcasts against `(B, V)`, and replace every logit strictly below it with `-inf`.

The two differ only on ties: with the threshold, every logit equal to the `k`-th largest survives, so more than `k` tokens can be kept.
Real implementations accept that; the tests use distinct values.

`topk` raises if `k` is larger than the row, so production code clamps it with `min(k, V)`.

Top-k is crude, and that is its known weakness: `k` is fixed whatever the model's confidence.
When the model is sure, two tokens deserve consideration and `k = 50` lets in 48 bad ones; when it is unsure, hundreds are reasonable and `k = 50` cuts most of them off.

#### Top-p (nucleus sampling)

Top-p fixes exactly that by letting the number of candidates adapt.
It keeps the **smallest set of most probable tokens whose probabilities add up to at least `p`**.
When the model is confident that set might be one or two tokens; when it is uncertain, several hundred.

Done in tensors, per row:

1. Sort the logits in descending order, and keep the sort indices.
2. Softmax the sorted logits and take the running sum along the row, `cumsum`.
3. For each sorted token, look at the sum of the probabilities **before** it, not including itself.
   Drop it if that sum is already `>= p`: the tokens before it already reached the target.
   Keep it otherwise.
4. Map the keep/drop decision from sorted order back to the original vocabulary order, and write `-inf` at the dropped positions.

The "before it" in step 3 is the important detail.
It is what keeps the token that **crosses** the threshold.
On the probabilities of the toy row `[3.0, 2.0, 1.0, 0.0]`:

```python
>>> p = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).softmax(-1)
>>> p
tensor([[0.6439, 0.2369, 0.0871, 0.0321]])
>>> p.cumsum(-1)                 # inclusive: up to and including each token
tensor([[0.6439, 0.8808, 0.9679, 1.0000]])
>>> p.cumsum(-1) - p             # exclusive: strictly before each token
tensor([[0.0000, 0.6439, 0.8808, 0.9679]])
```

With `p = 0.9`, the exclusive sums are below 0.9 for the first three tokens, so three survive: the first two reach only 0.88, and the third is the one that crosses 0.9.
Comparing the **inclusive** sum against `p` instead would keep only two tokens and stop short of the target, which is the off-by-one to avoid.

Two consequences of the exclusive sum, both tested:

- the first token's exclusive sum is 0, so for any `p > 0` it is always kept, and a row can never end up all `-inf` (which would softmax to NaN, as topic 14 showed);
- with `p = 1.0` nothing should be cut: every exclusive sum is below 1 in exact arithmetic.

The exclusive sum is either `cumsum - probs`, or the inclusive `cumsum` shifted one place to the right with a `0` put in front.

**Floating point at `p = 1.0`.**
In float32 the running sum of a large row can drift slightly above 1, and then tokens far in the tail get dropped at `p = 1.0`.
On 8 rows of 50257 random logits with standard deviation 3, the inclusive sum ended as high as `1.0000076`, and a plain comparison at `p = 1.0` dropped 34834 tail tokens in total, each with a vanishing probability.
On other rows it drops none; it depends on how the rounding falls.
The four-token test does not see this, but it is why real implementations treat `p >= 1.0` as "filter off" and return the logits unchanged, the same way `k <= 0` switches top-k off.

**Back to the original order.**
After `sort`, position `j` of a row holds the token whose original index is `indices[j]`.
A mask computed in sorted order is therefore in the wrong order for the logits, and applying it directly marks the wrong tokens: the test with logits `[0.1, 5.0, 0.2, 4.0]` checks that the kept tokens are 1 and 3, in their original places.
`scatter` along the last axis with the sort indices writes each sorted entry back to where it came from (topic 05).
On neutral data, sorting values and scattering them back restores the row exactly:

```python
>>> v = torch.tensor([[10.0, 30.0, 20.0]])
>>> s, i = v.sort(-1, descending=True)
>>> s, i
(tensor([[30., 20., 10.]]), tensor([[1, 2, 0]]))
>>> torch.empty_like(s).scatter(-1, i, s)
tensor([[10., 30., 20.]])
```

A boolean tensor scatters the same way.
The other route is `gather` with the inverse permutation, `i.argsort(-1)`, which gives the same result.

#### Repetition penalty

Every token that already appears in the generated sequence gets its logit pushed down:

```
logit / penalty     if logit > 0
logit * penalty     if logit < 0
```

with `penalty > 1`; a logit of exactly 0 stays 0 either way.
The two branches exist because a plain division only lowers positive logits.
Dividing `-2` by `2` gives `-1`, which **raises** it, making an unlikely token more likely: the opposite of a penalty.
Multiplying a negative logit moves it further below zero, so both branches push down.
`penalty = 1.0` is a no-op, and values in use are around 1.05 to 1.2.

`torch.where(cond, a, b)` picks elementwise from two tensors of the same shape (or broadcastable ones), `a` where `cond` is `True` and `b` elsewhere:

```python
>>> x = torch.tensor([3.0, -1.0, 0.0])
>>> torch.where(x > 0, x * 10, x - 10)
tensor([ 30., -11., -10.])
```

The penalty touches only the tokens in `generated`, a `(B, T)` tensor of ids.
Those ids are column indices into the `(B, V)` logits, one row each, which is exactly the shape `gather` reads and `scatter` writes (topic 05):

```python
>>> logits = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
>>> ids = torch.tensor([[3, 0, 3]])
>>> logits.gather(-1, ids)                      # read one value per id
tensor([[8., 5., 8.]])
>>> logits.scatter(-1, ids, torch.zeros(1, 3))  # write one value per id
tensor([[0., 6., 7., 0.]])
```

**Repeated ids.**
A token generated twice appears twice in `generated`.
`gather` reads it twice, the same value both times, and `scatter` writes the same result twice, so the token is penalized **once**, which is the standard behaviour.
Anything that accumulates over the occurrences, such as a loop that penalizes in place once per id or a `scatter_add`, penalizes it once per repetition instead.

**Out of place.**
`scatter_` and `masked_fill_` change their tensor in place, so calling them on the `logits` argument changes the caller's tensor too (topic 05).
`scatter` and `masked_fill` without the underscore return a new tensor; use them, or `clone` first.

The repetition penalty has fallen out of fashion.
It cannot tell a repeated phrase from a repeated "the", so a strong penalty produces text that avoids ordinary function words and reads strangely.
It was needed when base models genuinely looped, and instruction-tuned models mostly do not, so modern stacks leave it at 1.0 or use frequency penalties that grow with the number of occurrences.

#### Sampling

The last two steps turn filtered logits into token ids.
`softmax(-1)` gives probabilities, and `torch.multinomial(probs, num_samples, generator=None)` draws from them:

- it takes **probabilities**, non-negative, not necessarily summing to 1; logits with negative entries make it raise, and so does a row whose entries are all 0;
- each row of a `(B, V)` input is one distribution, and the result is `(B, num_samples)` int64 indices, so one sample per row is `(B, 1)`, and the exercise wants `(B,)`;
- a probability of exactly 0, from a `-inf` logit, is never drawn.

```python
>>> g = torch.Generator().manual_seed(0)
>>> torch.multinomial(torch.tensor([[0.2, 0.8], [0.5, 0.5]]), 1, generator=g).shape
torch.Size([2, 1])
```

**The generator.**
A `torch.Generator` is a random number source of its own, independent of the global seed.
Passing the same freshly seeded generator gives the same draws every time, and repeated calls that share one generator continue its stream rather than repeat it.
The tests pass one in and call `sample_from_logits` hundreds of times with it: once with uniform logits, expecting each of four tokens about 25% of the time, and once with `top_k=2`, expecting nothing outside the top two ever to appear.

**Temperature 0 means no randomness.**
At temperature 0 the result is the argmax of each row, and the test checks the values, the `(B,)` shape and the int64 dtype.
`argmax(-1)` returns exactly that: a `(B,)` int64 tensor.
The defaults `top_k=0` and `top_p=1.0` mean both filters are off.

### 17.2 The KV cache

**The problem.**
Generating with the naive loop from 17.3, step `t` runs the model over the whole prefix of `t` tokens, and only the last position's logits are used.
Attention over `t` tokens costs on the order of `t²` per layer, so a whole generation of `T` tokens costs on the order of `T³`.
Almost all of it is recomputation.
Because attention is causal (topic 14), position `i` never looks at anything after it, so its keys and values, and indeed its whole column of activations, are the same on every step after it first appeared.

**The fix.**
For each layer, keep the `k` and `v` of every position seen so far.
On each new step, run only the new token through the layer: project it to its own `q`, `k`, `v`, **append** its `k` and `v` to the stored ones, and attend from its single query to the whole accumulated history.

```
decode step, cache already holds S - 1 positions:

x               (B, 1, D)          the new token's residual stream
q               (B, H, 1, Dh)      projected and split into heads
k, v (new)      (B, H, 1, Dh)
k, v (cached)   (B, H, S, Dh)      after the append
scores          (B, H, 1, S)       one query against every key
out             (B, H, 1, Dh)  ->  merged and projected back to (B, 1, D)
```

Per step, the attention cost drops from `t²` to `t`, and the whole generation from cubic to quadratic.
Everything else in the model (projections, FFN, norms) also runs on one token instead of `t`.

**What is not cached.**
Queries: each token's query is used once, on the step it arrives, and never again, because later tokens attend **to** it through its key and value, not from it.
And nothing is shared across sequences: a cache belongs to one generation, and a new prompt starts a new one.

#### `KVCache`

The exercise's cache holds the keys and values of **one** layer; a model would keep one per layer.
Its contract:

- it starts empty, and `len(cache)` is then 0;
- `append(k, v)` takes `(B, H, T_new, Dh)` tensors, adds them after what is stored along the time axis, and returns the full stored `(k, v)`;
- `len(cache)` is the number of positions stored, the size of the time axis.

The test appends 3 positions, then 1, and checks the shapes, the lengths, and that the first 3 positions of the result are the first `k` and the last position of `v` is the second `v`.

`torch.cat(tensors, dim)` joins tensors that agree on every axis except `dim`; for `(B, H, T, Dh)` the time axis is `dim=2`, or equivalently `dim=-2`:

```python
>>> torch.cat([torch.zeros(1, 2, 3, 4), torch.ones(1, 2, 1, 4)], dim=-2).shape
torch.Size([1, 2, 4, 4])
```

`cat` along the wrong axis either raises or, when the sizes happen to agree, silently glues heads or channels together.

**The empty state.**
Before the first append there is nothing to concatenate with.
The clean representation is `None`: the first append simply stores its tensors, and every later one concatenates.
`__init__` takes no arguments, so it can only set up that empty state, and `__len__` has to handle the empty case too and return a Python `int` either way (`x.shape[-2]` is one).
An empty tensor as the starting value is possible but fragile, because its shape has to match the others on every axis except time, which the cache does not know before the first call.

**The cost of `cat`.**
`cat` allocates a new tensor and copies everything into it, so appending one token copies the whole cache, and a long generation does that thousands of times.
Production caches preallocate a `(B, H, T_max, Dh)` buffer once and write each new token into the next free slot.
For this exercise `cat` is the right tool: the copying costs time, not correctness.

#### `CachedSelfAttention`

This is the multi-head causal self-attention of topics 14 and 15, with one new argument.
The four projections are `q_proj`, `k_proj`, `v_proj`, `output_proj`, each `nn.Linear(D, D, bias=False)`; the test checks `named_parameters()` against exactly those four names.
The head split cuts `D` as `(H, Dh)` and then moves the head axis in front of time, the order from topic 14; the test builds its reference with `unflatten(-1, (H, Dh)).transpose(1, 2)`, and with the scale `1 / sqrt(Dh)` that `F.scaled_dot_product_attention` uses by default.

`forward(x, cache=None)`:

- without a cache, it is plain causal self-attention over `x`, and the test compares it against `F.scaled_dot_product_attention(q, k, v, is_causal=True)` built from your own projections;
- with a cache, the new `k` and `v` (already split into heads) are appended, and attention runs from the new queries to **everything** the cache returns.

**The mask.**
`T` is the number of new tokens in `x`, and `S` the number of keys after the append.
Which mask is needed depends on the call:

| Call | `T` | `S` | Mask |
| --- | --- | --- | --- |
| no cache, or prefill into an empty cache | `T` | `S = T` | the ordinary lower-triangular causal mask |
| decode, one token | 1 | `S` | none: the new token is the latest position and may see every key, including itself |
| a chunk of `T > 1` tokens into a non-empty cache | `T` | `S > T` | causal with an offset: query `i` sits at absolute position `S - T + i` |

The tests exercise the first two rows: a full pass, six one-token steps from an empty cache, and a prefill of 3 positions followed by two one-token steps.
The third row is the general case, and topic 14 built its table by comparing `arange(S - T, S)` against `arange(S)`.

Two ways to get the decode row wrong.
A causal mask built as `tril` of a `(1, S)` table is not "no mask", it allows only the **first** key:

```python
>>> torch.ones(1, 5, dtype=torch.bool).tril()
tensor([[ True, False, False, False, False]])
```

And `F.scaled_dot_product_attention(..., is_causal=True)` builds exactly that kind of mask when `T != S`: its causal mask is aligned to the **top-left** corner, `tril` of the `(T, S)` table, not the bottom-right one a cache needs.
Comparing the two on random tensors with `T = 2`, `S = 5` confirms it: `is_causal=True` matches `attn_mask=torch.ones(2, 5, dtype=torch.bool).tril()` and not the offset table.
So `is_causal=True` is correct only when `T == S`; with a cache and `T > 1` the offset mask has to be passed explicitly as `attn_mask`.

**Positions.**
This module has no positional encoding, so its keys and values depend on nothing but the token.
A model with RoPE or a learned position table must give the new token its true position, `len(cache)` before the append, not 0.
That is the offset from topic 13, and forgetting it is the classic reason cached generation diverges from uncached generation.

**The equivalence test** is the main test of the topic.
Running the module on all six positions at once and running it one position at a time with a cache must agree to `1e-5`.
When they disagree, the suspects in order are: a mask on the one-token steps (or a missing one on the prefill), appending along the wrong axis, appending before splitting heads or after merging them, and, in a model with positions, the offset.

#### Cache memory

The cache stores, for every layer, a key and a value per KV head, per position, per sequence in the batch, each `Dh` numbers long:

```
bytes = 2 * B * L * H_kv * T * Dh * bytes_per_element
```

The `2` is `k` and `v`.
`bytes_per_element` comes from the dtype, `dtype.itemsize` (topic 14): 2 for bfloat16 and float16, 4 for float32.
`H_kv` is the number of **key/value** heads: with grouped-query attention it is smaller than the number of query heads, and the cache shrinks by exactly their ratio.

A worked example: a 70B-class model with `L = 80`, `Dh = 128`, a single sequence of `T = 8192` tokens in bfloat16.
With full multi-head attention, `H_kv = 64`, the cache is `2 * 1 * 80 * 64 * 8192 * 128 * 2 = 21,474,836,480` bytes, about 21.5 GB, for one user.
With `H_kv = 8`, it is 2.7 GB.
That one line is the entire argument for GQA.

The formula is linear in batch and in sequence length, and that is what limits a serving system.
The weights are loaded once and shared by every request, but the cache is per request and grows with every token, so the number of users a GPU can serve at once is the number of caches that fit beside the weights.
vLLM's paged attention raised that number several times over on the same hardware by allocating the cache in fixed-size pages, like virtual memory, instead of reserving one contiguous block for each sequence's maximum length.

### 17.3 Generation

`greedy_generate(step_fn, prompt, max_new_tokens)` is the naive loop, without a cache.
`step_fn` stands for the model: given ids of shape `(B, T)`, it returns logits `(B, T, V)` for every position.
The prompt is `(B, T_prompt)` int64, and the result is the prompt with `max_new_tokens` generated tokens after it, `(B, T_prompt + max_new_tokens)`.

Each step does the same four things, with these shapes:

| What | Shape |
| --- | --- |
| the current ids | `(B, T)` |
| the model's logits for them | `(B, T, V)` |
| the logits at the last position only | `(B, V)` |
| the greedy pick for each row | `(B,)` int64 |
| that pick as a column, ready to append | `(B, 1)` |
| the ids with it appended along time | `(B, T + 1)` |

Notes on each line:

- `step_fn` gets the **whole** current sequence every time, prompt plus everything generated so far; that is what makes this version slow and obviously correct.
- the last position is `logits[:, -1]`, which indexes the time axis; `logits[-1]` would take the last **sequence** in the batch.
- the pick is an `argmax` over the vocabulary axis, the last one.
- `argmax` drops the axis it reduces, so the `(B,)` result needs a size-1 axis put back, `[:, None]` or `unsqueeze(-1)`, before it lines up with `(B, T)` for `torch.cat(..., dim=1)`.

A Python `for` loop over the generation steps is fine and unavoidable: each step needs the token the previous step produced.
The ban on Python loops is about looping over the **elements** of a tensor, not over time steps.

The test's `step_fn` is a stand-in that always predicts "previous token + 1, modulo 10", so from the prompt `[0, 1]` three steps must give `[0, 1, 2, 3, 4]`.
It only predicts that correctly when it reads the logits at the **last** position.

**Real generation loops add three things** the exercise leaves out:

- they run under `torch.no_grad()` or `torch.inference_mode()`, because generation never calls `backward`, and otherwise autograd keeps the graph of every step alive;
- they replace `argmax` with the sampling pipeline of 17.1 whenever the temperature is above 0;
- they stop early once every sequence in the batch has produced EOS, and mask whatever a finished sequence generates afterwards, which is the "everything after the first EOS" `cumsum` trick from topic 05.

With a cache, the loop changes shape: one prefill call on the whole prompt, then each step feeds only the newest token, `(B, 1)`, and the cache supplies the rest.

## API you need

### Sampling

| Call | Result |
| --- | --- |
| `logits / temperature` | a new tensor; division by `0.0` gives `inf`/`nan`, not an error |
| `x.argmax(dim=-1)` | index of the largest entry per row, int64, the axis dropped |
| `torch.full_like(x, float("-inf"))` | a tensor of `x`'s shape, dtype and device, all `-inf` |
| `torch.topk(x, k, dim=-1)` | `(values, indices)` of the `k` largest, sorted descending; raises if `k` exceeds the axis |
| `values[..., -1:]` | the last column kept as `(B, 1)`, so it broadcasts against `(B, V)` |
| `torch.sort(x, dim=-1, descending=True)` | `(values, indices)`; `indices[j]` is where sorted entry `j` came from |
| `p.cumsum(dim=-1)` | running sum along the row, inclusive |
| `x.scatter(dim, index, src)` | a copy of `x` with `src[i][j]` written at column `index[i][j]`; `scatter_` is in place |
| `x.gather(dim, index)` | `out[i][j] = x[i][index[i][j]]` |
| `idx.argsort(dim=-1)` | the inverse of a permutation |
| `x.masked_fill(cond, float("-inf"))` | `-inf` where `cond` is `True`, a new tensor |
| `torch.where(cond, a, b)` | `a` where `cond`, `b` elsewhere |
| `torch.isfinite(x)` | `True` where `x` is not `inf`, `-inf` or `nan` |
| `x.softmax(dim=-1)` | probabilities along the last axis |
| `torch.multinomial(probs, num_samples, generator=g)` | `(B, num_samples)` int64 draws, one distribution per row; takes probabilities |
| `x.squeeze(-1)`, `x[:, 0]` | `(B, 1)` to `(B,)` |
| `torch.Generator().manual_seed(n)` | an independent, reproducible random source |

### The cache and the loop

| Call | Result |
| --- | --- |
| `torch.cat([old, new], dim=-2)` | append along time for `(B, H, T, Dh)` |
| `x.shape[-2]` | the time length of a `(B, H, T, Dh)` tensor, a Python `int` |
| `def __len__(self) -> int` | what `len(obj)` calls; must return an `int` |
| `x.unflatten(-1, (H, Dh))`, `x.transpose(1, 2)` | the head split, as the tests' reference writes it |
| `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)` | built-in attention; bool mask `True` = allowed; `is_causal` is top-left aligned |
| `torch.arange(S - T, S)[:, None] >= torch.arange(S)[None, :]` | the causal mask for `T` new queries over `S` keys |
| `logits[:, -1]` | the last time step of `(B, T, V)`, giving `(B, V)` |
| `x[:, None]` / `x.unsqueeze(-1)` | `(B,)` to `(B, 1)` |
| `torch.cat([ids, new_col], dim=1)` | append a column of ids |
| `dtype.itemsize` | bytes per element |
| `torch.no_grad()`, `torch.inference_mode()` | context managers that switch autograd off |

### A worked example on neutral data

A filter that writes `-inf`, and what the sampler then does with it:

```python
>>> x = torch.tensor([[1.0, 4.0, 2.0, 3.0]])
>>> x.masked_fill(x < 2.5, float("-inf"))           # keep entries of at least 2.5
tensor([[-inf, 4., -inf, 3.]])
>>> _.softmax(-1)
tensor([[0.0000, 0.7311, 0.0000, 0.2689]])
>>> g = torch.Generator().manual_seed(0)
>>> torch.multinomial(_, 1000, replacement=True, generator=g).unique()   # only 1 and 3 are ever drawn
tensor([1, 3])
```

Bytes per element, and a cache length that grows:

```python
>>> torch.bfloat16.itemsize, torch.float32.itemsize
(2, 4)
>>> stored = torch.zeros(1, 2, 3, 4)
>>> torch.cat([stored, torch.zeros(1, 2, 1, 4)], dim=-2).shape[-2]
4
```

## Traps

- Dividing by a temperature of 0 does not raise; it produces `inf` and `nan`, and the softmax of them is NaN. Temperature 0 means greedy.
- `torch.multinomial` takes probabilities: logits make it raise on any negative entry, and log-probabilities are all negative.
- `multinomial` returns `(B, 1)`; the exercise wants `(B,)`.
- Filtering after the softmax instead of on logits leaves rows that no longer sum to 1.
- Top-p compared on the inclusive cumsum drops the token that crosses the threshold.
- Applying a top-p mask computed in sorted order straight to the unsorted logits removes the wrong tokens.
- A top-p filter that does not short-circuit `p >= 1.0` drops tail tokens in float32 on a large vocabulary.
- `topk` with `k` larger than the vocabulary raises; `k <= 0` must mean "off".
- A repetition penalty that only divides makes negative logits **more** likely.
- A penalty applied in a loop over the generated ids, or with `scatter_add`, hits a repeated token several times.
- `scatter_` or `masked_fill_` on the `logits` argument changes the caller's tensor.
- `logits[-1]` is the last sequence in the batch; the last time step is `logits[:, -1]`.
- Appending to the cache along the wrong axis; for `(B, H, T, Dh)` it is `dim=-2`.
- A causal mask on a one-token decode step, which lets the new token see only the first key.
- `F.scaled_dot_product_attention(..., is_causal=True)` with `T != S` uses a top-left mask, which is wrong for a cache.
- Caching `q`, which is never used again.
- Forgetting the position offset in a model with RoPE or learned positions.
- Generating without `torch.no_grad()`, which keeps a growing graph alive across every step.

## Ready when

Without looking anything up you can:

- [ ] list the sampling pipeline in order and say why every filter writes `-inf` instead of renormalizing
- [ ] say what temperature does to the distribution, and why temperature 0 needs special handling
- [ ] implement top-k, and say what `k <= 0` and ties do
- [ ] implement top-p including the sort, the exclusive cumsum, and the mapping back to vocabulary order
- [ ] say why the repetition penalty has two branches, and what happens to a repeated id
- [ ] draw samples with `multinomial` and a generator, with the right shape and dtype
- [ ] explain what the KV cache stores, what it does not, and the cost it removes
- [ ] write a cache that starts empty, appends along time and reports its length
- [ ] say which mask each attention call needs: no cache, prefill, one-token decode, a chunk into a non-empty cache
- [ ] say why `is_causal=True` is wrong once `T != S`
- [ ] write the KV cache memory formula and compute it for a real model, with and without GQA
- [ ] write the naive greedy generation loop with the right shapes at every step

## Going deeper

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), Holtzman et al. 2019, which introduced nucleus sampling and shows why greedy text degenerates.
- [CTRL](https://arxiv.org/abs/1909.05858), Keskar et al. 2019, section 4.1, where the repetition penalty was introduced; the two-branch form for negative logits is the later fix that libraries adopted.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), Shazeer 2019, the multi-query attention paper, where the KV cache memory argument is made most clearly.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180), Kwon et al. 2023, the vLLM paper.
- [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), whose reference implementation shows the top-left causal mask.
- [Hugging Face `generate` logits processors](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py), the production versions of every filter in 17.1, to read once you have written your own.
