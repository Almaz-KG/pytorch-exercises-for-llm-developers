# Topic 17 - Generation: sampling and the KV cache

> **Exercise** `test_17_sampling_and_kv_cache.py`

## Why this topic

Topic 15 built a model that maps a row of token ids to logits, and topic 16 trained it.
A trained model still does not write anything.
**Generation** is the loop around it: run the model, take the scores at the last position, pick one token, append it to the input, and run again, until enough tokens have been produced or the model says it is done.
17.2 writes that loop three ways: greedy, sampled, and stopping at EOS, and 17.4 writes it once more for a model with a cache.

Two independent things live inside the loop, and this topic builds both.

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
$\exp(-\infty) = 0$, so a logit of `-inf` gets probability exactly 0, and the surviving tokens share the whole 1 among themselves in the same proportions as before:

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

Top-p adjusts the number of candidate tokens to the model's confidence.
Instead of a fixed number of tokens, it keeps the **smallest set of top tokens whose combined probability reaches at least `p`**.
That set is called the **nucleus**.

- When the model is confident, the nucleus might be just 1 or 2 tokens.
- When the model is uncertain, it might include hundreds.

At `p = 0.9`, on two rows of four probabilities already sorted:

| Row | Running total | Kept |
| --- | --- | --- |
| `0.9692, 0.0178, 0.0065, 0.0065` (confident) | `0.97` after the first token | 1 token |
| `0.25, 0.25, 0.25, 0.25` (uncertain) | `0.25, 0.50, 0.75, 1.00` | 4 tokens |

**How it works, per row of the tensor:**

1. Sort the logits in descending order and save the original indices.
2. Turn the sorted logits into probabilities (`softmax`) and take their running sum (`cumsum`).
3. Filter (the "before" rule): keep a token only if the probability accumulated **strictly** before it, not counting itself, is `< p`.
   Every token after the nucleus is marked for dropping.
   This decision is made in sorted order, so it is a `(B, V)` boolean mask over sorted positions.
4. Map that mask back to the original vocabulary order, and only then write `-inf` at the dropped tokens.

**Why check the sum *before* the token?**
It is what keeps the token that pushes the total over the threshold `p`.
Take the sorted probabilities `[0.64, 0.24, 0.09, 0.03]` with `p = 0.90`:

- Inclusive sum (`cumsum`): `[0.64, 0.88, 0.97, 1.00]`.
  Keeping tokens while this is `< 0.90` keeps only two, `0.88` in total, short of the target.
- Exclusive sum (`cumsum - probs`): `[0.00, 0.64, 0.88, 0.97]`.
  Keeping tokens while this is `< 0.90` keeps the first **three**, and the nucleus reaches `0.97`.

Both sums are one `cumsum` away:

```python
>>> p = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).softmax(-1)
>>> p
tensor([[0.6439, 0.2369, 0.0871, 0.0321]])
>>> p.cumsum(-1)                 # inclusive: up to and including each token
tensor([[0.6439, 0.8808, 0.9679, 1.0000]])
>>> p.cumsum(-1) - p             # exclusive: strictly before each token
tensor([[0.0000, 0.6439, 0.8808, 0.9679]])
```

The exclusive sum is either `cumsum - probs`, as above, or the inclusive `cumsum` shifted one place to the right with a `0` put in front.

**Two edge cases the exclusive sum handles, both tested:**

- **No empty rows.**
  The top token has nothing before it, so its exclusive sum is `0`, and for any `p > 0` at least one token survives.
  A row of all `-inf` would softmax to NaN (topic 14).
- **`p = 1.0` keeps everything.**
  Every exclusive sum is below `1.0`, so no token is dropped, at least in exact arithmetic.

In float32 that second guarantee can break.
The running sum of a large row can drift slightly above 1, and then tokens far in the tail get dropped at `p = 1.0`.
On 8 rows of 50257 random logits with standard deviation 3, the inclusive sum ended as high as `1.0000076`, and a plain comparison at `p = 1.0` dropped 34834 tail tokens in total, each with a vanishing probability.
On other rows it drops none; it depends on how the rounding falls.
The four-token test does not see this, but it is why real implementations treat `p >= 1.0` as "filter off" and return the logits unchanged, the same way `k <= 0` switches top-k off.

**Restoring the original order (step 4).**
This is the step that is easy to skip by accident, and the test with logits `[0.1, 5.0, 0.2, 4.0]` exists to catch it: the kept tokens must be 1 and 3, in their original places.
After `sort`, position `j` holds the token whose original index is `indices[j]`.
`scatter` along the last axis with those indices writes each sorted entry back to where it came from (topic 05).
On neutral data, sorting values and scattering them back restores the row exactly:

```python
>>> v = torch.tensor([[10.0, 30.0, 20.0]])
>>> s, i = v.sort(-1, descending=True)
>>> s, i
(tensor([[30., 20., 10.]]), tensor([[1, 2, 0]]))
>>> torch.empty_like(s).scatter(-1, i, s)
tensor([[10., 30., 20.]])
```

A boolean mask scatters the same way.
Here a mask that flags the two **last sorted positions** comes back as a mask on the two smallest **original** entries, wherever they sit:

```python
>>> x = torch.tensor([[0.0, 3.0, 1.0, 2.0]])
>>> _, i = x.sort(-1, descending=True)
>>> i
tensor([[1, 3, 2, 0]])
>>> flagged = torch.tensor([[False, False, True, True]])    # in sorted order
>>> torch.zeros_like(flagged).scatter(-1, i, flagged)        # in original order
tensor([[ True, False,  True, False]])
```

The other route is `gather` with the inverse permutation, `i.argsort(-1)`, which gives the same result.

#### Repetition penalty

Every token that already appears in the generated sequence gets its logit pushed down:

```math
\begin{cases}
\text{logit} / \text{penalty} & \text{if logit} > 0 \\
\text{logit} \cdot \text{penalty} & \text{if logit} < 0
\end{cases}
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

### 17.2 Generation

Every function in this section is the same outer loop with a different rule inside it.
They all take the model as a `step_fn`, a prompt of shape `(B, T_prompt)` int64, and a step budget `max_new_tokens`, and they all return the prompt with the generated tokens appended after it.

#### The model as a `StepFn`

The exercise declares one type alias for the model:

```python
type StepFn = Callable[[Tensor], Tensor]
```

`Callable[[A], R]`, from `collections.abc`, is the type of "anything that can be called with one `A` and returns an `R`".
The `type X = ...` statement (Python 3.12) gives that type a name, so every signature can say `step_fn: StepFn` instead of repeating it.
The contract the type cannot express is the shapes: ids `(B, T)` int64 in, logits `(B, T, V)` out, one row of scores for the token **after** each position.

A whole transformer from topic 15 is a `StepFn`, and so is any plain function with that contract.
That is the point of passing a function rather than a model: the loop never needs to know what is inside, and the tests can hand it tiny stand-ins whose answer is known in advance.
The stand-in most tests use always predicts "this token + 1, modulo 10", so from the prompt `[0, 1]` three greedy steps must give `[0, 1, 2, 3, 4]`.

```python
>>> def bump(ids):                          # a toy StepFn: always predicts "this token + 1"
...     return torch.nn.functional.one_hot((ids + 1) % 4, 4).float()
>>> bump(torch.tensor([[0, 2]])).shape
torch.Size([1, 2, 4])
```

#### Greedy: `greedy_generate`

The naive loop, without a cache.
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

- `step_fn` gets the **whole** current sequence every time, prompt plus everything generated so far; that is what makes this version slow and obviously correct, and the test records the lengths it was called with.
- the last position is `logits[:, -1]`, which indexes the time axis; `logits[-1]` would take the last **sequence** in the batch, and the test uses two rows to catch it.
- the pick is an `argmax` over the vocabulary axis, the last one.
- `argmax` drops the axis it reduces, so the `(B,)` result needs a size-1 axis put back, `[:, None]` or `unsqueeze(-1)`, before it lines up with `(B, T)` for `torch.cat(..., dim=1)`.

A Python `for` loop over the generation steps is fine and unavoidable: each step needs the token the previous step produced.
The ban on Python loops is about looping over the **elements** of a tensor, not over time steps.

**Autograd off.**
Generation never calls `backward`, but autograd does not know that.
With gradients on, every step records its graph, and the graph of a real model holds on to every activation of every step, so memory grows with each token for nothing.
Every generation function here must run its `step_fn` with autograd off, and the test's stand-ins check `torch.is_grad_enabled()` to make sure.
There are two ways to switch it off:

```python
>>> with torch.no_grad():                  # for a block
...     torch.is_grad_enabled()
False
>>> @torch.no_grad()                       # for a whole function
... def f():
...     return torch.is_grad_enabled()
>>> f(), torch.is_grad_enabled()
(False, True)
```

`torch.inference_mode()` is a stricter variant used the same way: it also skips some bookkeeping, and tensors created under it can never be used in autograd later.

#### Sampling: `sample_generate`

The same loop, with `argmax` replaced by `sample_from_logits` from 17.1 and its four settings passed through.
It is specified in terms of that function, so its tests stay skipped until `sample_from_logits` is written.
At `temperature = 0` the sampler is the argmax, and the result must equal greedy generation exactly.

**One generator for the whole generation.**
A `torch.Generator` is a stream of random numbers: every draw takes the next numbers from it and moves it forward.
Passing the caller's generator to every step continues one stream across the whole text, so a fixed seed reproduces the text exactly, and another seed gives another one.
Creating a fresh generator inside the loop, or reseeding it, restarts the stream at every step, and every step then makes the same draw:

```python
>>> probs = torch.full((1, 50), 0.02)
>>> g = torch.Generator().manual_seed(0)
>>> [torch.multinomial(probs, 1, generator=g).item() for _ in range(3)]       # one stream
[47, 8, 28]
>>> [torch.multinomial(probs, 1, generator=torch.Generator().manual_seed(0)).item() for _ in range(3)]
[47, 47, 47]
```

The tests check all three properties: the same seed twice gives the same ids, another seed gives different ones, and the tokens within one generation are not all the same.
A sampler that builds its own fixed-seed generator when given `None` has the restart bug built in: in a loop, every step draws identically.

#### Stopping at EOS: `generate_until_eos`

A real model says when it is done by producing the **EOS** ("end of sequence") token.
`generate_until_eos` is greedy generation with that rule added, and its result may be shorter than the budget: `(B, T_prompt + n)` with `n <= max_new_tokens`.

The difficulty is the batch.
Rows finish at different steps, but a tensor has one length, so a finished row cannot simply stop growing while the others go on.
The convention is to keep a **finished flag** per row, a `(B,)` bool tensor, and:

- on every step, a finished row gets `eos_id` appended **whatever the model predicted**, which pads it;
- a row becomes finished on the step its pick is `eos_id`;
- once every row is finished, the loop stops early, without spending the rest of the budget.

On the stand-in "+1" model with `eos_id = 5`, the prompt `[3, 4]` produces 5 on its first step, while the prompt `[0, 1]` needs four steps.
After its first step the model would continue the shorter row with 6, 7, ..., and the padding is what replaces those with 5.
The test checks both rows, and that `step_fn` was called four times and not ten.

Only **generated** tokens count.
An `eos_id` inside the prompt, which happens with chat templates, does not finish a row, so the flag must start as all `False` and be updated from the picks only, never computed from the whole sequence.

The pieces are all elementwise on `(B,)` tensors:

```python
>>> done = torch.tensor([False, True, False])
>>> pick = torch.tensor([7, 2, 5])
>>> torch.where(done, 9, pick)              # a Python scalar broadcasts as a branch
tensor([7, 9, 5])
>>> done | (pick == 5)                      # elementwise "or"
tensor([False,  True,  True])
>>> bool(done.all()), bool(done.any())      # reductions to one Python bool
(False, True)
```

The `cumsum` trick from topic 05, "everything after the first EOS", is the alternative for a sequence that was generated in full: it builds the same mask afterwards instead of step by step.

### 17.3 The KV cache

**The problem.**
Generating with the naive loop from 17.2, step `t` runs the model over the whole prefix of `t` tokens, and only the last position's logits are used.
Attention over `t` tokens costs on the order of $t^2$ per layer, so a whole generation of `T` tokens costs on the order of $T^3$.
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

Per step, the attention cost drops from $t^2$ to $t$, and the whole generation from cubic to quadratic.
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
The head split cuts `D` as `(H, Dh)` and then moves the head axis in front of time, the order from topic 14; the test builds its reference with `unflatten(-1, (H, Dh)).transpose(1, 2)`, and with the scale $1 / \sqrt{d_{\text{head}}}$ that `F.scaled_dot_product_attention` uses by default.

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
| a chunk of `T > 1` tokens into a non-empty cache | `T` | `S > T` | causal with an offset: query $i$ sits at absolute position $S - T + i$ |

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

```math
\text{bytes} = 2 \cdot B \cdot L \cdot H_{\text{kv}} \cdot T \cdot d_{\text{head}} \cdot \text{bytes\_per\_element}
```

The `2` is `k` and `v`.
`bytes_per_element` comes from the dtype, `dtype.itemsize` (topic 14): 2 for bfloat16 and float16, 4 for float32.
`H_kv` is the number of **key/value** heads: with grouped-query attention it is smaller than the number of query heads, and the cache shrinks by exactly their ratio.

A worked example: a 70B-class model with `L = 80`, `Dh = 128`, a single sequence of `T = 8192` tokens in bfloat16.
With full multi-head attention, `H_kv = 64`, the cache is $2 \cdot 1 \cdot 80 \cdot 64 \cdot 8192 \cdot 128 \cdot 2 = 21{,}474{,}836{,}480$ bytes, about 21.5 GB, for one user.
With `H_kv = 8`, it is 2.7 GB.
That one line is the entire argument for GQA.

The formula is linear in batch and in sequence length, and that is what limits a serving system.
The weights are loaded once and shared by every request, but the cache is per request and grows with every token, so the number of users a GPU can serve at once is the number of caches that fit beside the weights.
vLLM's paged attention raised that number several times over on the same hardware by allocating the cache in fixed-size pages, like virtual memory, instead of reserving one contiguous block for each sequence's maximum length.

### 17.4 Generation with the cache

17.2 reruns the whole prefix on every step, and 17.3 built the cache that makes that unnecessary.
`generate_with_cache` is the loop that drives a cached model, and its result must be the same as `greedy_generate`'s.

The difference is entirely in **what is passed to `step_fn`**.
A cached `step_fn` keeps its own KV cache between calls, so it must be given each position exactly once:

| Call | Input to `step_fn` | What the cache holds after it |
| --- | --- | --- |
| 1, the prefill | the whole prompt, `(B, T_prompt)` | `T_prompt` positions |
| 2 | the token picked on call 1, `(B, 1)` | `T_prompt + 1` |
| ... | the token picked on the previous call, `(B, 1)` | one more each time |
| `max_new_tokens` | the token picked on the previous call, `(B, 1)` | `T_prompt + max_new_tokens - 1` |

So there are `max_new_tokens` calls, as in the naive loop, but only the first one sees more than one position, and the logits it returns cover only the positions it was given; the last of them, `[:, -1]`, is still the one that predicts the next token.
The very last token picked is appended to the result and never fed back, because nothing is generated after it.
That is why the cache ends one position short of the output, and the test checks the exact length.

Two ways to get this loop wrong, both caught:

- passing the whole sequence on every call, as the naive loop does: the cache appends the same positions again, and from the second step on the model attends to duplicated keys and the numbers drift away from the uncached run;
- feeding the final token back after the loop, which costs a call and leaves the cache one position too long.

The first test of the section records every input the stand-in model receives and checks them one by one.
The second is the composition test of the whole topic: it builds a one-layer model from an embedding, your `CachedSelfAttention` and an output projection, wraps it twice, once as an uncached `StepFn` and once around a `KVCache`, and checks that `greedy_generate` and `generate_with_cache` produce identical ids.
It stays skipped until all four pieces are written.

A model with positions has one more thing to get right here: the token passed on call `t` sits at absolute position `T_prompt + t - 2`, not at 0, which is the offset from topic 13 and the "Positions" note in 17.3.

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

### The loop and the cache

| Call | Result |
| --- | --- |
| `type StepFn = Callable[[Tensor], Tensor]` | a named type for "called with a `Tensor`, returns a `Tensor`"; `Callable` from `collections.abc` |
| `@torch.no_grad()`, `with torch.no_grad():` | autograd off for a function or a block; `torch.inference_mode()` the same, stricter |
| `torch.is_grad_enabled()` | whether autograd is recording right now |
| `torch.where(done, eos_id, pick)` | a scalar in one branch broadcasts; pads finished rows |
| `a \| b`, `a == 5` | elementwise "or" of bool tensors, elementwise comparison |
| `bool(t.all())`, `bool(t.any())` | reduce a bool tensor to one Python `bool` |
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
- A generator created or reseeded inside the loop, or a fixed-seed default for `None`: every step then makes the same draw.
- A finished row that is not padded with EOS keeps whatever the model says after it.
- Counting an EOS from the prompt as finished; only generated tokens count.
- Running all `max_new_tokens` steps when every row has already produced EOS.
- Passing the whole sequence to a cached `step_fn` on every step, which appends the same positions to the cache again.
- Feeding the last picked token back after the loop; the cache ends at `T_prompt + max_new_tokens - 1`.

## Ready when

Without looking anything up you can:

- [ ] list the sampling pipeline in order and say why every filter writes `-inf` instead of renormalizing
- [ ] say what temperature does to the distribution, and why temperature 0 needs special handling
- [ ] implement top-k, and say what `k <= 0` and ties do
- [ ] implement top-p including the sort, the exclusive cumsum, and the mapping back to vocabulary order
- [ ] say why the repetition penalty has two branches, and what happens to a repeated id
- [ ] draw samples with `multinomial` and a generator, with the right shape and dtype
- [ ] read `Callable[[Tensor], Tensor]` and say what contract a `StepFn` adds on top of the type
- [ ] write the naive greedy generation loop with the right shapes at every step, with autograd off
- [ ] swap greedy for sampling and say why one generator must serve the whole generation
- [ ] stop a batch at EOS: pad finished rows, ignore the prompt, end once every row is done
- [ ] explain what the KV cache stores, what it does not, and the cost it removes
- [ ] write a cache that starts empty, appends along time and reports its length
- [ ] say which mask each attention call needs: no cache, prefill, one-token decode, a chunk into a non-empty cache
- [ ] say why `is_causal=True` is wrong once `T != S`
- [ ] write the KV cache memory formula and compute it for a real model, with and without GQA
- [ ] drive a cached model: prefill once, then one token per call, and say how long the cache ends up

## Going deeper

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), Holtzman et al. 2019, which introduced nucleus sampling and shows why greedy text degenerates.
- [CTRL](https://arxiv.org/abs/1909.05858), Keskar et al. 2019, section 4.1, where the repetition penalty was introduced; the two-branch form for negative logits is the later fix that libraries adopted.
- [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), Shazeer 2019, the multi-query attention paper, where the KV cache memory argument is made most clearly.
- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180), Kwon et al. 2023, the vLLM paper.
- [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), whose reference implementation shows the top-left causal mask.
- [Hugging Face `generate` logits processors](https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py), the production versions of every filter in 17.1, to read once you have written your own.
