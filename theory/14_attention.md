# Topic 14 - Attention

> **Exercise** `test_14_attention.py`

## Why this topic

Everything the model has done to its vectors so far happened to each token on its own.
An embedding looks up one token, a norm rescales one token's vector, the feed-forward network transforms one token's vector.
None of that lets the word "it" find out which noun it refers to, three words back.

**Attention** is the only place in the whole transformer where one token reads another.
Every token gets to look at the other tokens, decide which of them matter to it, and pull in a mix of their information.
In topic 15 the result gets wrapped in a norm, a residual add and a feed-forward branch to become one transformer block, and stacking those blocks is the whole model.

The mechanism is four lines of tensor algebra, and you will write it from scratch.
What makes it worth a whole topic is that each of those lines hides a decision that, made wrong, still runs, still produces numbers of the right shape, and quietly gives a worse or broken model:

- dividing by the wrong number;
- taking the softmax along the wrong axis;
- applying the mask at the wrong moment;
- splitting the heads in the wrong order.

After the core, you will wrap it into the multi-head module every LLM uses, then into its modern variant, grouped-query attention, and finally compute what attention costs, which explains a good part of the last five years of LLM engineering.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| query, `q` | what a token is looking for, one vector per token |
| key, `k` | what a token offers to be found by, one vector per token |
| value, `v` | what a token hands over once it is found, one vector per token |
| score | how well one query matches one key: their dot product, scaled |
| attention weights | the scores after a softmax: for each query, a set of non-negative numbers summing to 1 |
| mask | a boolean table saying which query may look at which key, `True` meaning allowed |
| causal | a token may look only at itself and at earlier tokens, never at later ones |
| padding | filler tokens added so that sequences of different lengths fit one rectangular batch |
| head, `H`, `Dh` | attention runs several times side by side, each run on its own slice of `Dh` channels; `H * Dh == D` |
| `T`, `S` | the number of queries and the number of keys; equal in ordinary self-attention |

### A refresher: the dot product as similarity

The dot product of two vectors multiplies them entry by entry and adds the results up: $(1, 2) \cdot (3, -1) = 1 \cdot 3 + 2 \cdot (-1) = 1$.
Geometrically it is $\lvert a \rvert \cdot \lvert b \rvert \cdot \cos(\text{angle between them})$.
So it is large and positive when the two vectors point the same way, near zero when they are unrelated (perpendicular), and negative when they point in opposite directions.
That makes it a cheap, differentiable measure of "how well do these two match", and attention uses nothing else to compare tokens.

The dot products of every row of a `(T, Dh)` matrix with every row of an `(S, Dh)` matrix are one matrix multiplication with the second one transposed: `(T, Dh) @ (Dh, S) -> (T, S)`.
Entry `[t, s]` of the result is "row `t` of the first, dot row `s` of the second".
`@` treats every axis before the last two as a batch axis (topic 06), so `(B, H, T, Dh) @ (B, H, Dh, S)` does that for every example and every head at once and gives `(B, H, T, S)`.

### A refresher: softmax turns scores into shares

Topic 10 built softmax: exponentiate, then divide by the sum.
For a row of scores it produces non-negative numbers that add up to 1, larger scores getting larger shares, and adding the same constant to every score changes nothing.

Two properties matter here.

**It is sensitive to scale.**
The scores `[1, 2, 3]` give the shares `[0.09, 0.24, 0.67]`: a preference, but every entry gets something.
The same scores times 8, `[8, 16, 24]`, give `[0.0000001, 0.0003, 0.9997]`: practically all of the weight on one entry.
A softmax that does this is called **saturated**, and a saturated softmax has almost no gradient: nudging the scores barely changes an output that is already pinned at 0 and 1, so learning stalls.

**`-inf` removes an entry.**
$e^{-\infty} = 0$, so a score of `-inf` gets a share of exactly 0, and the remaining entries share the whole 1 among themselves as if the removed one never existed:

```python
>>> torch.tensor([1.0, float("-inf"), 2.0]).softmax(-1)
tensor([0.2689, 0.0000, 0.7311])
>>> torch.tensor([1.0, 2.0]).softmax(-1)              # the same two shares
tensor([0.2689, 0.7311])
```

### 14.0 Heads

Multi-head attention runs several independent attentions side by side, each on its own slice of the `D` channels.
To do that with one batched matmul, the `(B, T, D)` activations are rearranged to `(B, H, T, Dh)`: the head axis moves in front of the sequence axis, so that `@` batches over `(B, H)` and never mixes heads.
That is **splitting heads**, and **merging heads** is the exact reverse, `(B, H, T, Dh)` back to `(B, T, D)`.
No parameters, no arithmetic, only a different view of the same numbers.

You did both in topic 02; here is the one rule that decides whether you did them right.
The last axis of size `D` holds head 0's `Dh` channels first, then head 1's, and so on.
So `D` must be cut as `(H, Dh)`, heads on the outside, and only after that is the head axis moved with a transpose.
The test checks precisely that: head 1 of the split must be channels `Dh` to `2*Dh - 1` of the input.
Cutting `D` as `(Dh, H)` instead, or permuting before cutting, gives the same shape with every head made of other heads' channels, and nothing raises.

**`view` versus `reshape` on the way back.**
A transpose does not move any data; it only changes the strides, the bookkeeping of how to step through memory (topic 02).
The result is **non-contiguous**, and `view` refuses to merge axes of such a tensor:

```python
>>> x = torch.randn(2, 3, 4).transpose(0, 1)          # (3, 2, 4), not contiguous
>>> x.view(3, 8)
RuntimeError: view size is not compatible with input tensor's size and stride (...). Use .reshape(...) instead.
>>> x.reshape(3, 8).shape                              # copies when it has to
torch.Size([3, 8])
```

So merging heads, which transposes first and then merges two axes, needs `reshape`, or `.contiguous()` followed by `view`.
Splitting heads cuts an axis first and transposes afterwards, so `view` works there.
`einops.rearrange` handles both directions and makes the axis order explicit in its pattern string.

### 14.1 Scaled dot-product attention

**The idea: a soft dictionary lookup.**
A Python dict takes a key and returns exactly one value: the one whose key matches exactly.
Attention is the same lookup made soft.
A query is compared against **every** key, every comparison gives a score, and the answer is a **weighted average of all the values**, each value weighted by how well its key matched.
A key that matches well contributes a lot, one that does not contributes almost nothing.

Every token produces all three vectors (in 14.2, by three separate learned projections of its activations):
its query says what it is looking for, its key says what it can be found by, and its value is what it passes on when found.
The key and the value are deliberately separate: what makes a token relevant and what it contributes need not be the same thing.

**The formula.**

```
scores  = q @ kᵀ / sqrt(Dh)                       (..., T, Dh) @ (..., Dh, S)  ->  (..., T, S)
scores  = where the mask says "not allowed", put -inf        (optional)
weights = softmax(scores) along the last axis, the keys        (..., T, S)
out     = weights @ v                              (..., T, S) @ (..., S, Dh)  ->  (..., T, Dh)
```

Or in one line, as the papers write it: $\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(Q K^\top / \sqrt{d_{\text{head}}} + \text{mask}\right) V$, where the mask is 0 at allowed positions and $-\infty$ at forbidden ones.

Read the shapes of the score tensor `(..., T, S)` as "for every query, one number per key".
Row `t` belongs to query `t`; column `s` to key `s`.

**One query, by hand.**
Take `Dh = 2`, one query and three keys, each key with its value:

```
q = (2, 0)
k0 = (1, 0)    v0 = (10, 0)       points the same way as q
k1 = (0, 1)    v1 = (0, 10)       perpendicular to q
k2 = (-1, 0)   v2 = (-10, -10)    points the opposite way

dot products:       2,      0,      -2
divided by sqrt(2): 1.41,   0,      -1.41
softmax:            0.768,  0.187,  0.045        sums to 1
output:             0.768 * v0 + 0.187 * v1 + 0.045 * v2  =  (7.23, 1.41)
```

The output is mostly `v0`, because `k0` matched the query best, with a little of the others mixed in.
That is attention: nothing more than this, done for every query in parallel.

Now the four decisions, each of which is a bug if you get it wrong.

**1. The scale is $1 / \sqrt{d_{\text{head}}}$.**
A dot product of two vectors with `Dh` random entries of variance 1 is a sum of `Dh` random products, and its standard deviation grows like $\sqrt{d_{\text{head}}}$.
Measured on random vectors:

| `Dh` | std of $q \cdot k$ | std after dividing by $\sqrt{d_{\text{head}}}$ |
| --- | --- | --- |
| 4 | 2.0 | 1.0 |
| 64 | 8.0 | 1.0 |
| 1024 | 32.0 | 1.0 |

Without the division, a model with `Dh = 64` feeds the softmax scores spread eight times wider than with it: exactly the `[8, 16, 24]` situation from the refresher, a saturated softmax and a dead gradient.
Dividing by $\sqrt{d_{\text{head}}}$ keeps the spread of the scores near 1 whatever the head size.
The `Dh` is the **head** dimension, the last axis of `q`, never `d_model`; the dot products are taken inside one head.
`q.shape[-1]` gives it, and `math.sqrt` of a Python int is a Python float.

**2. The softmax is along the last axis, the keys.**
For each query, the weights over all keys must sum to 1: each query hands out a budget of 1 among the keys.
That is `dim=-1` of the `(..., T, S)` scores.
A softmax along the query axis, `dim=-2`, also runs and also trains a little, but it makes every **key** distribute a budget among the queries, which is not attention.
The quick check is that `weights.sum(-1)` is all ones.

**3. The mask goes on the scores, before the softmax, as `-inf`.**
The refresher showed why: a `-inf` score gets weight exactly 0, and the survivors renormalize by themselves.
Masking with 0 instead does not remove anything; 0 is an ordinary score and still gets its share:

```python
>>> torch.tensor([1.0, 0.0, 2.0]).softmax(-1)        # "masked" middle entry still gets 9%
tensor([0.2447, 0.0900, 0.6652])
```

And zeroing weights **after** the softmax leaves rows that no longer sum to 1, unless you renormalize them by hand, which is a second place to get it wrong.

The mask in this exercise is boolean with `True` meaning "allowed to look".
`masked_fill(cond, value)` writes `value` where `cond` is `True`, so to fill the **forbidden** places you invert the mask first, `~mask`.
The mask only has to **broadcast** against `(..., T, S)` (topic 03): one `(T, S)` mask serves every example and every head.
PyTorch's own `F.scaled_dot_product_attention` reads a boolean `attn_mask` with the same meaning, `True` = allowed, which is how the tests compare against it.

**4. `q` and `k` need not have the same length.**
In ordinary self-attention every token is a query and a key, so `T == S`.
But nothing in the formula needs that: `q` is `(..., T, Dh)` and `k`, `v` are `(..., S, Dh)`.
Writing the function for separate `T` and `S` costs nothing and makes it right for cross-attention (queries from one sequence, keys from another) and for generation with a cache (topic 17), where one new query attends to the whole history.
`k` and `v` always have the same length: one value per key.

**The weights on their own.**
`attention_weights` stops before the last matmul and returns the `(..., T, S)` weight matrix.
It is the standard debugging picture: print it or plot it, and you see at a glance whether a mask does what you think.

Two checks the tests use, worth knowing because they test the formula rather than a particular number.
If all the queries are zero, every score is 0, all scores in a row are equal, and the weights are uniform: $1/S$ for every key.
And if you compute the scores of one query by hand, dividing by $\sqrt{d_{\text{head}}}$, the softmax of them must equal your row of weights; with no division, the two disagree.

#### Causal masks

A language model is trained to predict the next token from the previous ones, on every position of a sequence at once.
If position 5 could look at position 6, it could simply read the answer, and the model would learn nothing useful.
So in a decoder, query $i$ may see key $j$ only when $j \le i$.
"Causality" is not a property of the architecture: it is exactly this mask, and nothing else.

As a table, it is lower-triangular: row $i$ (the query) has `True` up to and including column $i$ (the key).
Comparing two `arange`s by broadcasting builds it (topic 03):

```python
>>> i = torch.arange(4)
>>> i[:, None] >= i[None, :]          # rows: queries, columns: keys
tensor([[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]])
```

`torch.ones(T, S, dtype=torch.bool).tril()` gives the same table when `T == S`.
Build the mask on the same device as the data (`device=q.device`), or a GPU run fails.

The first row is the fastest check of any causal implementation: query 0 may see only key 0, so all its weight is on that key, and its output is **exactly** `v` at position 0.

When `T != S`, as when one new token attends to a cache of `S` earlier ones, the queries are the **last** `T` positions, so query $i$ sits at absolute position $S - T + i$.
Comparing `arange(S - T, S)` with `arange(S)` gives the right table:

```python
>>> torch.arange(2, 5)[:, None] >= torch.arange(5)[None, :]      # T = 3 queries, S = 5 keys
tensor([[ True,  True,  True, False, False],
        [ True,  True,  True,  True, False],
        [ True,  True,  True,  True,  True]])
```

This exercise assumes `T == S`; topic 17 needs the general form.

**Testing causality with gradients.**
The strongest test of causality does not look at the mask at all.
It asks: does the output at position 2 depend on the inputs at positions 3, 4, 5?
Backpropagate from `out[:, 2]` only, and look at the gradient of the input: it must be exactly zero at every later position, and non-zero at the earlier ones.
With a `-inf` mask the forbidden weights are exactly 0, so the gradient through them is exactly 0, not merely small.
Every other part of the module (the projections, the head split) works on each position separately, so the mask is the only thing standing between a position and the future.

#### Padding masks

A batch is a rectangle, but sentences have different lengths.
The short ones are filled out with padding tokens, and a `(B, S)` boolean mask records which positions are real (`True`) and which are filler (`False`).
No query should take any information from a padding key.

The padding mask is about **keys**, so it has to line up with the last axis of the `(B, H, T, S)` scores, and be the same for every head and every query.
That is a `(B, 1, 1, S)` shape, which broadcasts over `H` and `T` (topic 03).
Getting there from `(B, S)` means inserting two axes of size 1 in the middle, with `None` indexing or `unsqueeze`.
Shaped `(B, S, 1, 1)` or `(1, 1, B, S)` instead, it either fails to broadcast or, worse, broadcasts against the wrong axes and silently masks the wrong thing.

Causal and padding together mean "allowed by both", a logical **and**, `&`.
A `(T, S)` causal mask and a `(B, 1, 1, S)` padding mask broadcast together to `(B, 1, T, S)` with no further help.

The test checks it with a sequence whose last two positions are padding: the queries at positions 2 and 3 must give exactly what plain attention over the first two keys gives.
A causal mask alone would let them see the padding keys at 2 and 3, and the numbers would differ.

**A fully masked row is NaN.**
If every key of some query is forbidden, its whole row of scores is `-inf`, and the softmax divides 0 by 0:

```python
>>> torch.full((3,), float("-inf")).softmax(-1)
tensor([nan, nan, nan])
```

With causal plus padding at the end, as here, this never happens: key 0 is real and visible to every query.
It does happen with left padding, where a padded query at position 0 can see only itself, a padding key.
Real code either never masks the diagonal or drops padded query positions from the loss.

### 14.2 Multi-head attention

**Where `q`, `k` and `v` come from.**
In self-attention all three are made from the same input `x` of shape `(B, T, D)`, each by its own learned linear map: `q_proj`, `k_proj`, `v_proj`.
The model learns, through those three matrices, what each token should look for, offer and pass on.

**Why several heads.**
One softmax gives one distribution over the keys, so a single attention can follow only one kind of relationship per token at a time.
A word may need to attend to its subject, to the previous word, and to a matching bracket all at once.
So the `D` channels are cut into `H` heads of `Dh = D / H` channels, attention runs inside each head independently, each head learning its own kind of relationship, and the results are put back side by side.
This costs no extra parameters: the projections are `D -> D` either way, only interpreted as `H` blocks.

**The module, step by step:**

```
x  (B, T, D)
 -> q_proj, k_proj, v_proj                 each nn.Linear(D, D, bias=False)        (B, T, D)
 -> split heads                                                                    (B, H, T, Dh)
 -> causal scaled dot-product attention                                            (B, H, T, Dh)
 -> merge heads                                                                    (B, T, D)
 -> output_proj                            nn.Linear(D, D, bias=False)             (B, T, D)
```

**`output_proj` is not optional.**
After the merge, the heads' results sit side by side in separate channel blocks and have never interacted.
`output_proj` is the matrix that mixes them, turning "what each head found" into one update for the token.
Leave it out and every shape still checks out, while the layer is noticeably weaker.

**`nn.Linear` refresher.**
`nn.Linear(d_in, d_out, bias=False)` stores its weight as `(d_out, d_in)` and computes `x @ weight.T` (topic 09).
The tests rebuild your module's output by hand that way, from `attn.q_proj.weight` and the others, so your forward must be exactly projections, split, causal attention, merge, output projection, with no other operations in between.

**The names are part of the contract.**
The four submodules must be called `q_proj`, `k_proj`, `v_proj` and `output_proj`, all without a bias.
The test checks that `named_parameters()` is exactly those four weights and that the parameter count is `4 * D * D`, and later topics (and CS336) load weights by these names.

`num_heads` has to be remembered in `__init__` (as an attribute), because `forward` needs it to split the heads.

#### Grouped-query attention

**The problem it solves.**
When a model generates text, it produces one token at a time, and to avoid recomputing the past it keeps every earlier token's keys and values in memory, the **KV cache** (topic 17).
For every generated token, the whole cache is read from GPU memory.
At long contexts and large batches, that memory traffic, not the arithmetic, limits generation speed, and the cache size limits how many requests fit on a GPU at once.

**The idea.**
Keep `H` query heads, but compute only `H_kv` key/value heads, with `H` a multiple of `H_kv`.
Each key/value head is then shared by a group of `G = H / H_kv` query heads.
The cache shrinks by the factor `G`, and in practice quality barely moves.

`H_kv == H` is ordinary multi-head attention.
`H_kv == 1`, one key/value head shared by all, is multi-query attention.
Everything in between is grouped-query attention, GQA, and it is the standard in Llama 2 70B, Llama 3, Mistral, Qwen and essentially everything since.

To see the effect: Llama 2 70B has 80 layers, 64 query heads and 8 key/value heads of `Dh = 128`.
Its cache holds a key and a value per layer, per KV head, per token: `2 * 80 * 8 * 128` numbers, 0.33 MB per token in bf16, 1.3 GB for a 4096-token sequence.
With 64 key/value heads it would be 2.6 MB per token, 10.7 GB per sequence: eight times more.

**What changes in the shapes.**
Only the key and value projections become narrower; `Dh = d_model // num_heads` stays the same for every head:

```
q_proj:       D       ->  H * Dh            (= D)
k_proj:       D       ->  H_kv * Dh
v_proj:       D       ->  H_kv * Dh
output_proj:  H * Dh  ->  D
```

Remember that the weight is stored as `(d_out, d_in)`: with `D = 64`, `H = 8`, `H_kv = 2`, `k_proj.weight.shape` is `(16, 64)`.

`k` and `v` are split into `H_kv` heads, `(B, H_kv, S, Dh)`.
Then each of them is widened to `H` heads by repeating every key/value head `G` times **consecutively**, giving `(B, H, S, Dh)`, and from there it is ordinary multi-head attention with the causal mask.
`split_heads` from 14.0 takes the number of heads as an argument, so it can do the split for `H_kv` heads as well.

**Consecutively, not cyclically.**
Query heads `0, 1, ..., G - 1` form group 0 and share key/value head 0; the next `G` share head 1; query head `h` uses key/value head `h // G`.
So the head axis must read `[kv0, kv0, kv1, kv1]`, not `[kv0, kv1, kv0, kv1]`:

```python
>>> heads = torch.tensor([0, 1])
>>> heads.repeat_interleave(2)            # each element G times in a row: the GQA order
tensor([0, 0, 1, 1])
>>> heads.repeat(2)                       # the whole thing G times: the wrong order
tensor([0, 1, 0, 1])
```

Both methods take a `dim` argument (topic 05); the head axis of `(B, H_kv, S, Dh)` is `dim=1`.
The `unsqueeze`/`expand`/`reshape` route from topic 02 gives the same consecutive order.
The wrong order pairs every query head with another group's keys: the shapes are all right, nothing raises, and a model loaded from a real checkpoint produces garbage.
The test compares your output against a reference built with the consecutive order, so it catches this.
With one key/value head, or with as many as query heads, the two orders coincide; the difference shows only for the groups in between.

### 14.3 Cost

Two numbers, both of which you compute from the shapes rather than look up.

**Memory of the score matrix.**
The weights tensor is `(B, H, T, S)`, and the straightforward implementation materializes it in memory.
Its size in bytes is the number of entries times the size of one entry:

```math
\text{bytes} = B \cdot H \cdot T \cdot S \cdot \text{bytes\_per\_element}
```

`bytes_per_element` comes from the dtype: 4 for float32, 2 for bfloat16 and float16.
`torch.bfloat16.itemsize` gives it directly (PyTorch 2.1 and later), and `torch.empty(0, dtype=dtype).element_size()` is the older spelling.

**FLOPs.**
Multiplying an $(m, k)$ matrix by a $(k, n)$ one costs $2 \cdot m \cdot k \cdot n$ floating-point operations: one multiply and one add per term (topic 06).
For attention without the projections, per example and per head:

```
q @ kᵀ:        (T, Dh) @ (Dh, S)   ->   2 * T * Dh * S
weights @ v:   (T, S)  @ (S, Dh)   ->   2 * T * S * Dh
total:                                  4 * T * S * Dh,   times B * H
```

The softmax and the scaling are a handful of operations per score and are left out by convention.
The exercise takes `T == S`.

**Why these two formulas matter.**
Both are quadratic in the sequence length: double the context and the score matrix and the attention FLOPs quadruple.
Take `B = 8`, `H = 32`, `T = S = 32768`, bf16: the score matrix is $8 \cdot 32 \cdot 32768^2 \cdot 2$ bytes, about 550 GB, for one layer, and the backward pass needs it again.
No GPU has that.
Even a single sequence is 69 GB per layer.

This one number is why **FlashAttention** exists.
It computes exactly the same result, but block by block, keeping a running softmax so the full `(T, S)` matrix never exists in memory at once.
Topic 18 comes back to it.
`F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)` is PyTorch's built-in, which dispatches to such a fused kernel when shapes and dtypes allow.
Real code calls it, and prefers `is_causal=True` over an explicit mask, so the kernel can skip whole masked blocks.
Writing attention by hand once is what this topic is for; afterwards, use the built-in.

**Attention is not always the expensive part.**
The four `D -> D` projections cost $4 \cdot 2 \cdot B \cdot T \cdot D^2$ FLOPs, and attention itself $4 \cdot B \cdot T^2 \cdot D$ (since `H * Dh = D`).
Their ratio is $T / (2D)$, so attention dominates only once the sequence is longer than about twice `d_model`: around 8k tokens for `D = 4096`.
Below that, a transformer is mostly matmuls against weights, and the quadratic term is a minor share.

## API you need

| Call | Result |
| --- | --- |
| `q @ k.transpose(-2, -1)` / `q @ k.mT` | every query dotted with every key, `(..., T, S)` |
| `q.shape[-1]`, `q.shape[-2]` | `Dh`, and `T` of a `(..., T, Dh)` tensor |
| `math.sqrt(n)` | square root of a Python number, a float |
| `scores.masked_fill(cond, float("-inf"))` | `-inf` wherever `cond` is `True`, a new tensor |
| `~mask`, `mask_a & mask_b` | invert, combine boolean masks |
| `t.softmax(dim=-1)` / `F.softmax(t, dim=-1)` | softmax along the last axis |
| `torch.arange(n, device=...)[:, None] >= torch.arange(m, device=...)[None, :]` | a causal `(n, m)` table |
| `torch.ones(T, S, dtype=torch.bool).tril()` | the same, when `T == S` |
| `mask[:, None, None, :]` / `mask.unsqueeze(1).unsqueeze(2)` | `(B, S)` to `(B, 1, 1, S)` |
| `x.view(...)`, `x.reshape(...)` | cut or merge axes; `view` only on compatible memory layouts |
| `x.transpose(1, 2)`, `x.contiguous()` | swap two axes; copy into standard layout |
| `rearrange(x, pattern, **sizes)` | einops: named-axis reshape and transpose in one call |
| `x.repeat_interleave(G, dim=1)` | every slice along `dim` `G` times in a row |
| `nn.Linear(d_in, d_out, bias=False)` | a projection, weight stored as `(d_out, d_in)` |
| `module.named_parameters()` | `(name, tensor)` pairs, e.g. `"q_proj.weight"` |
| `F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)` | the built-in, the tests' reference; bool mask `True` = allowed |
| `dtype.itemsize` / `torch.empty(0, dtype=dtype).element_size()` | bytes per element of a dtype |

### A worked example on neutral data

Masking, broadcasting and combining on toy tensors, none of them attention:

```python
>>> s = torch.tensor([[1.0, 2.0, 3.0],
...                   [4.0, 5.0, 6.0]])
>>> allowed = torch.tensor([True, False, True])
>>> s.masked_fill(~allowed, float("-inf"))           # the (3,) mask broadcasts over the rows
tensor([[1., -inf, 3.],
        [4., -inf, 6.]])
>>> s.masked_fill(~allowed, float("-inf")).softmax(-1)
tensor([[0.1192, 0.0000, 0.8808],
        [0.1192, 0.0000, 0.8808]])
>>> a = torch.tensor([[True, False], [True, True]])  # (2, 2)
>>> b = torch.tensor([[True, True], [False, True]])  # (2, 2), think (B, S)
>>> (a[None, :, :] & b[:, None, :]).shape            # (1, 2, 2) & (2, 1, 2)
torch.Size([2, 2, 2])
>>> torch.bfloat16.itemsize, torch.float32.itemsize
(2, 4)
```

## Traps

- Dividing by $\sqrt{d_{\text{model}}}$ instead of $\sqrt{d_{\text{head}}}$, or not dividing at all.
- Softmax along the query axis instead of the key axis; check `weights.sum(-1)` is all ones.
- Masking after the softmax, or with 0 instead of `-inf`.
- `masked_fill` fills where the condition is `True`, so an "allowed" mask needs `~` first.
- A row with no allowed keys at all gives NaN, not zero.
- A causal mask built on the CPU while the data is on the GPU.
- A padding mask of the wrong shape broadcasts against the wrong axes and masks the wrong things without raising; it must be `(B, 1, 1, S)`.
- Splitting heads as `(Dh, H)` instead of `(H, Dh)`, or permuting before reshaping, mixes channels of different heads with every shape correct.
- `view` after a transpose raises; use `reshape` when merging heads.
- Forgetting `output_proj`, or adding biases to the projections; the parameter names and count are tested.
- In GQA, `repeat` instead of `repeat_interleave` on the head axis pairs query heads with the wrong key/value heads.
- In GQA, making `k_proj` as wide as `q_proj`: only the key and value projections shrink, to `H_kv * Dh`.
- Memory is quadratic in the sequence length; a 32k context does not fit as a materialized score matrix.

## Ready when

Without looking anything up you can:

- [ ] explain attention as a soft dictionary lookup, and say what queries, keys and values are for
- [ ] work one query against three keys by hand and get the output as a weighted average of the values
- [ ] write scaled dot-product attention in four lines, with the right scale and the right softmax axis
- [ ] say why the scale is $\sqrt{d_{\text{head}}}$ and what happens to the softmax without it
- [ ] say what each axis of the `(B, H, T, S)` score tensor means
- [ ] say why the mask is `-inf` before the softmax, and what masking with 0 does instead
- [ ] build a causal mask, including for `T != S`, and say which positions the queries occupy
- [ ] shape a `(B, S)` padding mask so it broadcasts correctly, and combine it with a causal one
- [ ] test causality with gradients, and say why the gradient is exactly zero
- [ ] split and merge heads in the right order, and say where `view` breaks
- [ ] write the full multi-head module with the canonical parameter names, from memory
- [ ] say what GQA changes in the projections and in the key/value heads, and why it pays off
- [ ] compute the size of the score matrix for a 32k context, and the attention FLOPs

## Going deeper

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), Vaswani et al. 2017, sections 3.2 and 3.2.1, still the clearest statement of the mechanism.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar, the friendliest walk-through with pictures of queries, keys and values.
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245), Ainslie et al. 2023.
- [FlashAttention](https://arxiv.org/abs/2205.14135), Dao et al. 2022, for how the quadratic memory goes away.
- [`F.scaled_dot_product_attention`](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html), whose doc page includes a reference implementation in Python.
