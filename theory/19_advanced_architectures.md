# Topic 19 - Advanced mechanisms

> **Exercise** `test_19_advanced_architectures.py`

## Why this topic

Topics 10 to 17 built a plain transformer, trained it and generated text with it.
Most of what separates a 2019 transformer from a 2025 one is a handful of pieces swapped in on top of that plain design, and each of them started as one paper that became a standard.
This topic drills five of them.

- **A hand-derived backward pass**, for RMSNorm.
  Every fused kernel in a fast training stack carries a backward that somebody derived on paper, and the derivation has a shape that repeats for every normalization.
- **Structured masks**: the sliding window that lets Mistral attend over long contexts cheaply, and the document mask that makes packed pretraining batches correct.
- **Mixture of Experts**: how Mixtral holds 47B parameters while spending the compute of a 13B model per token, and the three pieces that make it work, routing, capacity and a balancing loss.
- **Speculative decoding**: free generation speed with a mathematical guarantee that the output distribution does not change.
- **Beam search**: the classical search for the most probable sequence, still the right tool for translation and anything with a scoring objective.

This is the hardest file in the repository, and its five sections are independent.
Take them one at a time, in any order; nothing in 19.3 needs anything from 19.1.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| backward, VJP | the function that takes the gradient of the loss with respect to an output and returns it with respect to each input (topic 08) |
| coupling term | the part of a gradient that comes from a statistic shared by many elements, like the `r` of RMSNorm |
| sliding window | a causal mask that additionally forbids keys more than `window - 1` positions back |
| packing | concatenating several short documents into one training example to avoid padding |
| expert | one of `E` independent feed-forward networks in a MoE layer |
| router | a `(E, D)` matrix that scores every expert for every token |
| slot | which of a token's `K` chosen experts we mean: slot 0 is its best expert, slot 1 its second best |
| capacity | the most tokens one expert accepts in a batch; the surplus is dropped |
| draft, target | the small fast model that proposes tokens, and the large model whose distribution we want |
| acceptance rate, $\alpha$ | the probability that the target accepts one draft token |
| beam | one partial sequence with its accumulated log probability |

### 19.1 A backward of your own for RMSNorm

**The forward.**
RMSNorm from topic 11, with statistics over the last axis of width `D`.
`x` is `(..., D)`, `weight` is `(D,)`:

```math
\begin{aligned}
r &= \big(\mathrm{mean}(x^2) + \varepsilon\big)^{-1/2} && \text{one number per row, } (\dots, 1) \\
y_i &= g_i \cdot x_i \cdot r && g \text{ is the weight}
\end{aligned}
```

`eps` sits **inside** the square root, and the test compares your forward with exactly `x * rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w`.

**The derivation.**
The exercise's docstring prints the result for cross-checking; derive it yourself first, it takes ten lines.
Work on one row, so `x`, `y`, `g` are vectors of length `D` and `r` is a scalar.

First, how `r` reacts to one input.
$\mathrm{mean}(x^2) = \frac{1}{D} \sum_i x_i^2$, so its derivative with respect to $x_j$ is $2 x_j / D$, and by the chain rule:

```math
\frac{\partial r}{\partial x_j} = -\frac{1}{2} \big(\mathrm{mean}(x^2) + \varepsilon\big)^{-3/2} \cdot \frac{2 x_j}{D} = -\frac{x_j \, r^3}{D}
```

Next, the chain rule through `y`.
Write $a_i = \frac{\partial L}{\partial y_i} \cdot g_i$, folding the weight into the incoming gradient.
The input $x_j$ reaches the loss along two paths: **directly**, through its own output $y_j = g_j \cdot x_j \cdot r$, and **indirectly**, through $r$, which every output $y_i$ depends on.
Sum both:

```math
\begin{aligned}
\frac{\partial L}{\partial x_j} &= a_j \cdot r && \text{the direct path, only } i = j \\
&\quad + \sum_i (a_i \cdot x_i) \cdot \frac{\partial r}{\partial x_j} && \text{the path through } r \text{, every } i \\[4pt]
&= a_j \cdot r - x_j \cdot r^3 \cdot \sum_i (a_i \cdot x_i) / D
\end{aligned}
```

The same formula in terms of the normalized input $\hat{x} = x \cdot r$ reads $\frac{\partial L}{\partial x} = r \cdot \big(a - \hat{x} \cdot \mathrm{mean}(a \odot \hat{x})\big)$, which is easier to remember: scale the incoming gradient, then remove its component along $\hat{x}$.
It has the same shape as the softmax backward in topic 08, for the same reason: a shared statistic takes one direction of freedom away.

The weight gradient is simpler, because $g_i$ touches only $y_i$.
Per row, $\frac{\partial L}{\partial g_i} = \frac{\partial L}{\partial y_i} \cdot x_i \cdot r$.
But one $g$ is shared by every row of the batch and every position, so the contributions of all those rows add up:

```math
\frac{\partial L}{\partial g_i} = \sum_{\text{over every leading axis}} \frac{\partial L}{\partial y_i} \cdot x_i \cdot r \qquad \text{shape } (D,)
```

**In tensor form.**
$\sum_i a_i \cdot x_i$ is a sum over the last axis, one number per row, and so is $r$.
Keep both as `(..., 1)` with `keepdim=True` so they broadcast against the `(..., D)` tensors; without it a `(B, T)` tensor meets a `(B, T, D)` one and either fails to broadcast or, worse, lines up against the wrong axis.
The weight gradient is a sum over every axis but the last: `t.sum(dim=tuple(range(t.ndim - 1)))`, or `t.reshape(-1, D).sum(0)`.

Two structural facts are worth extracting, because they generalize to every normalization you will ever differentiate.

- **A broadcast in the forward pass becomes a sum in the backward pass.**
  $g$ is broadcast over all leading axes, so its gradient sums over all of them; the bias of `linear_backward` in topic 06 did the same.
- **A shared statistic produces a coupling term.**
  Because $r$ depends on every $x_i$, the gradient at $j$ contains a sum over all $i$.
  LayerNorm has two shared statistics, the mean and the variance, and its backward has two such terms.

**As a `torch.autograd.Function`.**
The contract is topic 08's, with one new element, a non-tensor argument:

- it is called as `RMSNormFunction.apply(x, weight, eps)`, never by instantiating the class;
- `forward(ctx, x, weight, eps)` runs with gradient tracking off, so nothing inside it is recorded in the graph;
- tensors that `backward` needs go through `ctx.save_for_backward(...)` and come back from `ctx.saved_tensors`;
- `eps` is a Python float, not a tensor, so it is kept as a plain attribute, `ctx.eps = eps`;
- `backward(ctx, grad_out)` receives the gradient of the output, shaped like `y`, and returns **one value per `forward` input**, in order: the gradient for `x`, the gradient for `weight`, and `None` for `eps`.

What to save is a choice.
Saving `x` and `weight` and recomputing `r` in `backward` is the cheapest in memory; saving `r` as well, one number per row, saves the recomputation.
Production kernels typically save `r`, usually under the name `rstd`, rather than the normalized output.

One PyTorch detail, so that it does not surprise you: if `backward` returns a gradient that merely **broadcasts** to the input's shape, such as a `(B, T, D)` tensor for a `(D,)` weight, autograd quietly sums it down to the right shape for you.
Write the sum anyway.
It is the actual mathematics, and a raw kernel that writes into a `(D,)` buffer does no such favor.
A gradient that does not broadcast, or the wrong number of returned values, raises at `backward` time.

**Checking it.**
The exercise checks the function three ways: the forward against the elementary expression, the backward against plain autograd through that same expression, and `torch.autograd.gradcheck`.
`gradcheck` nudges every input element by a tiny step, measures the change of the output by finite differences, and compares that numeric Jacobian with what your `backward` produces.
It needs float64: the step is around `1e-6`, which in float32, with about seven significant digits, drowns the difference in rounding error, and the check fails even on a correct backward.
`gradcheck` only checks the inputs that require a gradient, so the test wraps the call in a lambda that fixes `eps`.

### 19.2 Harder masks

All the masks here follow topic 14's convention: a `(T, T)` boolean table, rows are queries `i`, columns are keys `j`, `True` means "may look".
They are built the way topic 14 built the causal mask, by comparing two `arange`s through broadcasting, so no Python loop is needed.

**Sliding window** (Mistral, Longformer).
Position $i$ may see $j$ when $i - \text{window} < j \le i$: causal, plus a floor on how far back.
So row $i$ has exactly $\min(i + 1, \text{window})$ allowed keys, which one of the tests checks for several windows.
`window = 1` leaves only the diagonal: every token sees only itself.
When `window >= T` the floor never bites, and the mask must coincide exactly with the plain causal mask.
The table of $i - j$ over all pairs is one broadcast subtraction, and both conditions are comparisons on it, joined with `&`.

The point is cost.
Each query attends to at most `window` keys, so attention costs $O(T \cdot \text{window})$ instead of $O(T^2)$, and the KV cache (topic 17) only ever needs the last `window` positions.
Information still travels further than `window`: layer 2 sees layer 1's outputs, each of which already summarized a window, so after `L` layers a token can be influenced by tokens roughly $L \cdot \text{window}$ positions back.
Mistral 7B uses a window of 4096 with 32 layers.

**Document masking**, the mask behind sequence packing.
Pretraining corpora are full of documents far shorter than the context.
Padding each one out to `T` would waste most of the compute on filler, so several documents are concatenated into one length-`T` example instead.
Attention must not cross a document boundary, or the model learns to condition on unrelated text that happened to precede it.

The packed example comes with `doc_ids` of `(T,)`: the index of the document each position belongs to.
Query $i$ may see key $j$ when **both** hold: $j \le i$, and $j$ belongs to the same document as $i$.
The second condition alone is not enough, since inside one document a position would see its own future; the first alone is the plain causal mask, which leaks across documents.
The "same document" table is an outer equality comparison of `doc_ids` with itself, and the result is causal triangles along the diagonal, one per document, and nothing between them.

The helper `block_diagonal_lengths_to_doc_ids` goes from a list of document lengths to `doc_ids`: document 0 repeated as many times as the first length, then document 1, and so on.
`torch.repeat_interleave(values, repeats)` repeats each element of `values` the number of times given by the matching element of `repeats`.
`repeats` must be a tensor or a single int; a Python list raises `TypeError`, so wrap it with `torch.tensor(...)` first.
The test compares against an int64 tensor, which is what `torch.arange` gives by default.

Real code hands the same information to a fused kernel as cumulative sequence lengths (FlashAttention's `cu_seqlens`) rather than building a `(T, T)` table, but the table is what that kernel computes implicitly.

### 19.3 Mixture of Experts

Replace one FFN with `E` FFNs, the **experts**, and send each token to only `K` of them, usually `K = 2`.
Parameters grow with `E`, while the compute per token grows only with `K`.
Mixtral 8x7B holds about 47B parameters and uses about 13B of them per token, so it costs roughly what a 13B model costs to run and behaves closer to a much larger one.
The catch is memory: all 47B must be resident, even though each token uses a fraction of them.
MoE trades memory, which is cheap to add in a datacenter, for compute, which is not; that is why it dominates served frontier models and is rare among models run on a laptop.

Four pieces make a MoE layer, and each is one stub.

#### Top-2 routing

A **router** is a `(E, D)` matrix, a bias-free linear map with no module around it; `x @ router.T` turns `(N, D)` tokens into `(N, E)` logits, one score per expert per token.
Mixtral's rule: take the 2 largest logits per token, and softmax **over those two only**, so the two weights of every token sum to 1.

Softmaxing over all `E` first and then selecting the top 2 is a different rule: the two weights then sum to less than 1, by an amount that differs from token to token, and the layer's output scale wobbles with the router's confidence.
The Switch Transformer, which routes to one expert, does take the full softmax; Mixtral renormalizes over the chosen ones, and the test checks Mixtral's.

`torch.topk(t, k, dim=-1)` returns the values and the int64 indices, already sorted from largest to smallest, which is the order the contract asks for.
Since the larger logit comes first, the larger weight comes first too.

#### Expert capacity

On real hardware every expert gets a fixed-size buffer, sized ahead of time as roughly `capacity_factor * N * K / E` tokens, so that the computation has a static shape.
An expert accepts at most `capacity` tokens; the surplus is **dropped**: its contribution through that expert is lost, and the token keeps only what its other expert gave it (and the residual connection around the layer).
This exercise passes `capacity` in, and asks only for the boolean "did this assignment fit" for each of the `(N, K)` assignments.

**The queue order.**
Each expert fills its slots in a fixed order:

1. first every token's **slot-0** choice, in increasing token order;
2. then every token's **slot-1** choice, in increasing token order.

So a token's best expert is honored before anybody's second choice, and a token that picked expert 3 as its second choice waits behind every token that picked expert 3 as its first.
The second test pins this: with capacity 1, a token arriving at an expert through slot 0 beats an earlier token arriving through slot 1.

**The rank trick.**
Ranking items within their category without a loop is a cumsum over a one-hot:

- `F.one_hot(labels, num_classes)` turns `(M,)` labels into an `(M, C)` int64 table with a single 1 per row;
- `cumsum(0)` over it gives, at row `m` and column `c`, how many of the first `m + 1` items had label `c`;
- read each row at its own label's column, with `gather` or by multiplying by the one-hot and summing the row, and you have each item's position in its category's queue.

The count is **inclusive**: the first item of a category gets position 1, not 0.
Either compare against `capacity` with that in mind, or subtract the one-hot to get an exclusive count that starts at 0.
This is the same counting trick as `mask_after_eos` in topic 05 and the nucleus filter in topic 17, and it is worth recognizing as a pattern: a cumsum over a one-hot or a boolean gives a rank without a loop.

**Carrying the load from slot 0 to slot 1.**
A slot-1 assignment's position in the queue is its rank among the slot-1 assignments to the same expert **plus** the number of slot-0 assignments that expert already took.
Two ways to get there:

- lay the `(N, K)` assignments out in queue order, slot-major, as one long `(K * N,)` sequence, rank that, and lay the result back out as `(N, K)`; note that plain `reshape(-1)` on `(N, K)` is token-major, the wrong order, and the transpose has to come first;
- rank each slot's `(N,)` column on its own, and add to slot 1's positions the per-expert totals of slot 0, which are the column sums of slot 0's one-hot.

A Python loop over the `K` slots is fine; a loop over the `N` tokens is what the cumsum replaces.
Keep the function general in `K`, read from `expert_idx.shape[1]`.

In this exercise `moe_forward` has no capacity: every token reaches both its experts.
`capacity_mask` stands alone.

#### The load balancing loss

Without help, routing collapses.
A few experts get chosen a little more often early on, get more gradient, improve faster, get chosen even more, and the rest sit idle.
The Switch Transformer's auxiliary loss pushes back:

```math
\text{loss} = E \cdot \sum_e f_e \cdot P_e
```

- $f_e$ is the **fraction of tokens whose top-1 expert is $e$**: count the tokens per expert from `top1_idx` and divide by `N`.
  `torch.bincount(idx, minlength=E)` counts directly; `F.one_hot(idx, E)` averaged over the tokens gives the same fractions.
- $P_e$ is the **mean router probability of expert $e$** over all tokens: the column means of `router_probs`, the full softmax over all `E` experts, not the renormalized top-2 weights.

Both vectors sum to 1 over experts, and the loss reads directly when logged:

- perfectly balanced routing, $f_e = P_e = 1/E$, gives $E \cdot E \cdot (1/E)^2 = 1$;
- total collapse onto one expert, $f = P = (1, 0, \dots, 0)$, gives $E$.

$f$ comes from an argmax, so it is discrete and carries no gradient; $P$ is a mean of softmax outputs, so it does.
The gradient of the loss with respect to $P_e$ is $E \cdot f_e$, which pushes down hardest on the probability of the experts that are actually receiving the most tokens, which is exactly the feedback that undoes collapse.
In training it is added to the language-model loss with a small coefficient, typically 0.01.

The function returns a 0-dim float tensor, and the test compares it with `torch.tensor(1.0)`.
A Python float, a `(1,)` tensor or an int64 tensor all fail that comparison on type, shape or dtype; `bincount` and `one_hot` return int64 counts, so divide them into a float.

#### The sparse forward

In this exercise an expert is a two-layer ReLU network without biases, with the weights of all experts stacked along a leading `E` axis:

```
x:       (N, D)
router:  (E, D)
w1:      (E, D_ff, D)        w1[e] is (D_ff, D), the nn.Linear layout: (out, in)
w2:      (E, D, D_ff)        w2[e] is (D, D_ff)
```

```math
\begin{aligned}
&\text{expert } e \text{ on one token:} && w_2[e] \, \mathrm{relu}(w_1[e] \, x_n) \qquad \text{a } (D,) \text{ vector} \\
&\text{layer output for token } n \text{:} && \sum_{\text{its two slots}} \text{weight}[n, \text{slot}] \cdot \text{expert}_{\text{idx}[n, \text{slot}]}(x_n)
\end{aligned}
```

The formula is written for one column vector `x_n`.
For a batch of `n` tokens stored as rows, `(n, D)`, the same map is `rows @ W.T` with `W` in `(out, in)` layout, exactly as `nn.Linear` computes it (topic 09).

**Gather, compute, scatter.**
The naive implementation runs every token through every expert and multiplies the unchosen ones by zero, which spends exactly the compute MoE exists to save.
The real one, and the one the docstring asks for, works expert by expert:

- **gather**: find the tokens routed to expert `e`;
- **compute**: run only those rows through expert `e`;
- **scatter**: scale each result by its routing weight and add it into the output rows of its tokens.

Finding the tokens is a comparison of the `(N, 2)` index table with `e`.
`torch.where(cond)` on a 2-D boolean table returns two index tensors, the rows and the columns of its `True` entries; here those are the token numbers **and** the slot through which each token reached `e`, which together pick out the matching routing weight.
A per-token weight is `(n,)` and has to become `(n, 1)` to scale `(n, D)` rows.
`out.index_add_(0, rows, values)` adds `values[k]` into `out[rows[k]]`, and adds rather than overwrites, which is what each token needs: it receives one contribution from each of its two experts.
An expert that no token chose gets an empty index tensor; every step above works on zero rows, so there is nothing to special-case.

A Python loop over the `E` experts is fine, and is what real implementations do; a loop over tokens is not.

The tests route with a reference top-2 of their own, so `moe_forward` can be written before `top2_routing`.
Inside `moe_forward` you may call your own `top2_routing`, or route inline by the same rule; the dependent test `test_moe_routing_agrees_with_top2_routing` turns on once both exist and checks that the two agree.
Another test zeroes the weights of every expert no token chose and expects the output not to move, which holds automatically for a sparse forward.

### 19.4 Speculative decoding

**Why it works at all.**
Generating one token with a large model is bound by memory bandwidth, not by arithmetic: every step reads all the weights out of GPU memory to do a small amount of work with each.
So scoring `K + 1` positions in one forward pass costs almost exactly what scoring one costs, as long as `K` is small.

The scheme:

1. a small **draft** model proposes `K` tokens, cheaply, one at a time, recording its distribution $p_{\text{draft}}$ at each position;
2. the large **target** model runs once over the context plus those `K` tokens and produces its distribution $p_{\text{target}}$ at all `K + 1` positions, the `K` proposed ones plus the one after them;
3. an acceptance rule decides how many of the `K` to keep, and which distribution the next token comes from.

`speculative_accept` is step 3.
It does not sample anything: it returns how many draft tokens survive and the distribution to sample the next token from.
`u` is passed in so the function is deterministic and testable.

**The acceptance rule.**
Walk the positions in order.
At position `i`, with `t` the draft's token there and `u[i]` uniform in $[0, 1)$:

- **accept** if $u[i] \le p_{\text{target}}(t) / p_{\text{draft}}(t)$.
  When the target likes $t$ at least as much as the draft did, the ratio is at least 1 and the token is accepted whatever `u` is; with `u = 0` every token is accepted, which is what the first test uses.
- On the **first rejection**, at position `j`, stop: return `j` and the **residual distribution** at position `j`,

  ```math
  \text{residual} = \frac{\mathrm{relu}(p_{\text{target}} - p_{\text{draft}})}{\sum \mathrm{relu}(p_{\text{target}} - p_{\text{draft}})}
  ```

  Everything after `j` is discarded, because the draft generated it conditioned on a token that was not kept.
- If all `K` are accepted, return `K` and $p_{\text{target}}$ at position `K`, the extra row of `target_probs`.
  That **bonus** token is free: the target's single pass already computed its distribution.

`draft_probs` is `(K, V)` and `target_probs` is `(K + 1, V)`, so the first `K` rows of the target line up with the draft row for row, and the last one is only used for the bonus.
The two probabilities to compare at each position are one entry per row, the entry at that row's draft token: the per-row lookup from topic 05, with advanced indexing `probs[torch.arange(K), tokens]` or with `gather`.
The returned count is a Python `int`, so convert it with `int(...)` if it comes out of a tensor.

**Why the result is exactly the target's distribution.**
The probability that the step emits token $t$ has two parts.
The draft proposes $t$ with probability $p_{\text{draft}}(t)$ and it is kept with probability $\min\big(1, p_{\text{target}}(t) / p_{\text{draft}}(t)\big)$, which multiplies out to $\min\big(p_{\text{draft}}(t), p_{\text{target}}(t)\big)$.
A rejection happens with the remaining probability $1 - \sum_s \min\big(p_{\text{draft}}(s), p_{\text{target}}(s)\big)$, and then $t$ is drawn from the residual.
The residual's normalizer is the same number, because $\sum_s \mathrm{relu}\big(p_{\text{target}}(s) - p_{\text{draft}}(s)\big) = 1 - \sum_s \min\big(p_{\text{draft}}(s), p_{\text{target}}(s)\big)$.
So the second part is $\mathrm{relu}\big(p_{\text{target}}(t) - p_{\text{draft}}(t)\big)$, and

```math
\min(p_{\text{draft}}, p_{\text{target}}) + \mathrm{relu}(p_{\text{target}} - p_{\text{draft}}) = p_{\text{target}}
```

holds for every token.
The output is not an approximation of the target model; it **is** the target model, sampled faster.
The `relu` is what makes the residual a distribution, non-negative and concentrated on exactly the tokens the target wants more than the draft gave them; without it the "distribution" has negative entries.

**How fast.**
If each draft token is accepted independently with probability $\alpha$, a cycle yields $1$ token with probability $1 - \alpha$, $2$ with $\alpha (1 - \alpha)$, and so on, up to $K + 1$ when all are accepted and the bonus is added.
The expected count is the geometric sum

```math
1 + \alpha + \alpha^2 + \dots + \alpha^K = \frac{1 - \alpha^{K+1}}{1 - \alpha}
```

At $\alpha = 1$ the closed form is $0 / 0$, while the sum is plainly $K + 1$, so that case is handled separately.
At $\alpha = 0$ it is 1: the target always overrules the draft, and every cycle yields just the target's own token.

| `alpha` | `K = 4` | `K = 8` |
| --- | --- | --- |
| 0.5 | 1.94 | 2.00 |
| 0.7 | 2.77 | 3.20 |
| 0.8 | 3.36 | 4.33 |
| 0.9 | 4.10 | 6.13 |

`expected_speedup` is this count of tokens per cycle, which is the speedup only if the draft were free.
A real cycle costs `K` draft steps plus one target pass, so the wall-clock gain is smaller, and a larger `K` only pays while $\alpha$ is high.

A small draft agrees with the target more often than you might expect, because most tokens in natural text are easy: punctuation, common words, the second half of a word the tokenizer split.
They disagree exactly where the interesting decisions are, which is where the target's verification takes over.
Acceptance rates with a well-matched draft are often between 0.6 and 0.9.
Variants drop the separate draft model: Medusa trains extra prediction heads on the target itself, and prompt-lookup drafting copies continuations out of the context, which works well for summarization and code editing.

### 19.5 Beam search

Greedy decoding takes the single best token at each step, which can walk into a sequence whose later tokens are all unlikely.
Beam search keeps `beam_size` partial sequences, each with its accumulated log probability, and at every step keeps the `beam_size` best one-token extensions **across all beams jointly**.

Scores live in log space.
A sequence's probability is the product of its tokens' probabilities, and the log of a product is a sum, so extending a beam adds the token's log probability to the beam's score; it also avoids underflow, since a product of a hundred probabilities is far below float32's range.

**One step.**
The inputs are `beam_scores` of `(num_beams,)` and `next_log_probs` of `(num_beams, V)`, the model's log probabilities for the next token on each beam.

1. **Add**: the score of extending beam `b` with token `v` is `beam_scores[b] + next_log_probs[b, v]`, which by broadcasting is a `(num_beams, V)` table of candidates.
2. **Flatten** it to `(num_beams * V,)`, so every candidate is one entry.
3. **`topk(beam_size)`**: the best candidates, with their scores sorted descending, and their flat indices.
4. **Recover** which beam and which token each flat index came from.

The last step is the inverse of the flatten.
A row-major `(R, C)` table flattened to `(R * C,)` puts entry $[r, c]$ at flat index $r \cdot C + c$, so the row is `flat // C` and the column is `flat % C`.
On int64 tensors, `//` is floor division and `%` is the matching remainder; for non-negative indices both are exact.
`torch.div(a, b, rounding_mode="floor")` is the spelling that names the rounding explicitly; older PyTorch versions warned on a bare `//`, current ones do not.

The flatten, topk, divmod pattern is the thing to remember.
A per-beam topk followed by a merge is more code and is easy to get wrong: the joint winners can all come from one beam, which happens constantly when one beam is far ahead, and only a joint topk over the whole table expresses that directly.

`num_beams` and `beam_size` need not be equal: the second test feeds in 4 beams and keeps 3.
All three returned tensors are `(beam_size,)`; the parent indices and token indices are int64.

`beam_idx` is the parent pointer.
After every step the KV cache and the token history have to be reordered by it, which is `gather_beams` from topic 05.
At the very first step all beams hold the same prompt, so implementations usually start with scores `[0, -inf, -inf, ...]`, which keeps the first topk from picking the same continuation once per identical beam.

**Why beam search lost for chat.**
Raw scores penalize length, because each extra token adds another negative log probability, so finished beams are compared after **length normalization**, dividing the score by the length or by $\text{length}^\alpha$.
Even normalized, beam search produces bland, repetitive text for open-ended generation, the finding that the nucleus sampling paper behind topic 17 opens with: the most probable continuation is not the most human one.
It remains the right tool for translation, summarization, and any task with a well-defined best answer.

## API you need

| Call | Result |
| --- | --- |
| `class F(torch.autograd.Function)` with static `forward(ctx, ...)`, `backward(ctx, grad_out)` | an operation with a hand-written backward (topic 08) |
| `F.apply(*inputs)` | call it; arguments in the order of `forward` |
| `ctx.save_for_backward(*tensors)`, `ctx.saved_tensors`, `ctx.name = value` | carry tensors and plain values from forward to backward |
| `torch.autograd.gradcheck(f, inputs)` | numeric versus analytic Jacobian; float64 inputs with `requires_grad=True` |
| `x.pow(2).mean(-1, keepdim=True)`, `torch.rsqrt(t)` | the RMS statistic, kept as `(..., 1)`; $1 / \sqrt{t}$ |
| `t.sum(dim=tuple(range(t.ndim - 1)))`, `t.reshape(-1, D).sum(0)` | sum over every axis but the last |
| `torch.arange(T)[:, None] - torch.arange(T)[None, :]` | the `(T, T)` table of $i - j$ |
| `a[:, None] == a[None, :]` | outer equality, `(T, T)` bool |
| `mask_a & mask_b` | allowed by both |
| `torch.repeat_interleave(values, repeats)` | each value repeated its own number of times; `repeats` a tensor or an int |
| `torch.topk(t, k, dim=-1)` | `(values, indices)`, sorted descending, indices int64 |
| `t.softmax(dim=-1)` | softmax along the last axis |
| `F.one_hot(idx, num_classes)` | `(..., C)` int64 one-hot |
| `t.cumsum(dim)` | running sum, inclusive |
| `t.gather(dim, idx)` | one entry per row, at the column `idx` names |
| `torch.bincount(idx, minlength=E)` | count of each value in $[0, E)$, int64 |
| `torch.where(cond)` | the index tensors of the `True` entries, one per axis |
| `out.index_add_(0, rows, values)` | `out[rows[k]] += values[k]`, accumulating repeats |
| `F.relu(t)` / `torch.relu(t)`, `t.clamp(min=0)` | negative entries to 0 |
| `probs[torch.arange(K), tokens]` | one entry per row by advanced indexing |
| `flat.topk(k)`, then `idx // C`, `idx % C` | best entries of a flattened table, back to `(row, column)` |
| `torch.div(idx, C, rounding_mode="floor")` | floor division with the rounding named explicitly |

### A worked example on neutral data

A complete custom function with a non-tensor argument and a broadcast weight, $y = w \cdot \tanh(c \cdot x)$.
It has no shared statistic, so no coupling term; what it shows is the three return values, the `None`, and the sum over the leading axes:

```python
>>> class ScaledTanh(torch.autograd.Function):
...     @staticmethod
...     def forward(ctx, x, w, c):
...         t = torch.tanh(c * x)
...         ctx.save_for_backward(t, w)
...         ctx.c = c                                    # a float, not a tensor
...         return w * t
...     @staticmethod
...     def backward(ctx, grad_out):
...         t, w = ctx.saved_tensors
...         grad_x = grad_out * w * ctx.c * (1 - t * t)
...         grad_w = (grad_out * t).sum(dim=tuple(range(grad_out.ndim - 1)))
...         return grad_x, grad_w, None                  # one per forward input
>>> x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
>>> w = torch.randn(4, dtype=torch.float64, requires_grad=True)
>>> torch.autograd.gradcheck(lambda a, b: ScaledTanh.apply(a, b, 0.5), (x, w))
True
```

Tables built from `arange` and from labels, none of them the masks of 19.2:

```python
>>> i = torch.arange(5)
>>> i[:, None] - i[None, :]                     # entry [i, j] is i - j
tensor([[ 0, -1, -2, -3, -4],
        [ 1,  0, -1, -2, -3],
        [ 2,  1,  0, -1, -2],
        [ 3,  2,  1,  0, -1],
        [ 4,  3,  2,  1,  0]])
>>> colors = torch.tensor([0, 1, 0])
>>> colors[:, None] == colors[None, :]          # same color?
tensor([[ True, False,  True],
        [False,  True, False],
        [ True, False,  True]])
>>> torch.repeat_interleave(torch.tensor([10, 20]), torch.tensor([1, 3]))
tensor([10, 20, 20, 20])
```

Ranking items within their category, and counting categories:

```python
>>> labels = torch.tensor([2, 0, 2, 1, 2])
>>> running = F.one_hot(labels, 3).cumsum(0)
>>> running
tensor([[0, 0, 1],
        [1, 0, 1],
        [1, 0, 2],
        [1, 1, 2],
        [1, 1, 3]])
>>> running.gather(1, labels[:, None]).squeeze(1)    # 1-based place in its own category
tensor([1, 1, 2, 1, 3])
>>> torch.bincount(labels, minlength=4)
tensor([1, 1, 3, 0])
>>> pairs = torch.tensor([[3, 1], [0, 2]])
>>> pairs.reshape(-1), pairs.T.reshape(-1)           # row-major versus column-major order
(tensor([3, 1, 0, 2]), tensor([3, 0, 1, 2]))
```

Finding entries, and adding rows into place:

```python
>>> table = torch.tensor([[1, 2], [2, 0]])
>>> torch.where(table == 2)                          # (rows, columns) of the matches
(tensor([0, 1]), tensor([1, 0]))
>>> out = torch.zeros(3, 2)
>>> out.index_add_(0, torch.tensor([0, 2, 0]), torch.ones(3, 2))    # row 0 receives twice
tensor([[2., 2.],
        [0., 0.],
        [1., 1.]])
```

Reading one entry per row, and the best entries of a whole matrix:

```python
>>> p = torch.tensor([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])
>>> p[torch.arange(2), torch.tensor([1, 0])]
tensor([0.6000, 0.5000])
>>> m = torch.tensor([[0.1, 0.9, 0.3], [0.8, 0.2, 0.7]])
>>> values, flat = m.flatten().topk(3)
>>> values, flat // 3, flat % 3                      # value, row, column
(tensor([0.9000, 0.8000, 0.7000]), tensor([0, 1, 1]), tensor([1, 0, 2]))
```

The speculative identity on two arbitrary distributions:

```python
>>> p = torch.tensor([0.5, 0.3, 0.2])               # plays the target
>>> q = torch.tensor([0.2, 0.3, 0.5])               # plays the draft
>>> torch.minimum(p, q) + torch.relu(p - q)
tensor([0.5000, 0.3000, 0.2000])
>>> torch.relu(p - q).sum(), 1 - torch.minimum(p, q).sum()
(tensor(0.3000), tensor(0.3000))
```

## Traps

- `eps` goes inside the square root, and is a plain attribute on `ctx`, not a saved tensor.
- `backward` returns one value per `forward` input, three here, with `None` for `eps`.
- Per-row statistics in the RMSNorm backward need `keepdim=True`, or they broadcast against the wrong axis.
- The coupling term is easy to drop: $x_j$ reaches the loss through $r$ too, not only through $y_j$.
- The weight gradient sums over **all** leading axes; autograd will silently do it for you if you forget, a real kernel will not.
- `gradcheck` in float32 fails even on a correct backward; use float64.
- A sliding window mask must equal the causal mask when `window >= T`, and row $i$ must have $\min(i + 1, \text{window})$ allowed keys.
- A document mask needs causality **and** the same-document condition; either alone is wrong.
- `repeat_interleave` takes a tensor of repeats, not a Python list.
- Softmaxing over all `E` experts and then selecting the top 2 gives weights that do not sum to 1.
- The capacity cumsum is inclusive: the first token in a queue has position 1, not 0.
- The capacity queue is slot-major: slot 0 of every token before slot 1 of any token; `reshape(-1)` on `(N, K)` is token-major.
- $P_e$ in the balancing loss is the mean of the full router softmax, $f_e$ a fraction of top-1 counts; return a 0-dim float tensor, not a Python float or an int64 count.
- Running every token through every expert defeats the purpose of MoE.
- `w1[e]` is `(D_ff, D)`; rows of tokens multiply its transpose.
- `index_add_` rather than indexed assignment, because a token receives a contribution from each of its two experts.
- A routing weight of shape `(n,)` does not broadcast against `(n, D)`; it needs a trailing axis.
- In speculative decoding, continuing past the first rejection breaks the distributional guarantee.
- The residual needs the `relu` before renormalizing, or it has negative mass.
- `target_probs` has `K + 1` rows; the last one is only for the bonus token after a full acceptance.
- `expected_speedup` divides by zero at `alpha == 1`.
- A per-beam topk instead of a flat one cannot select several winners from the same beam.
- Beam recovery divides by the vocabulary size `V`, not by the number of beams.

## Ready when

Without looking anything up you can:

- [ ] derive the RMSNorm backward on paper, including where the coupling term comes from
- [ ] state the two general rules a normalization backward illustrates
- [ ] write a `torch.autograd.Function` with a non-tensor argument, and say what `backward` returns for it
- [ ] say why `gradcheck` needs float64
- [ ] build a sliding-window and a document mask from `arange` comparisons, and turn document lengths into document ids
- [ ] explain top-2 routing, and why the weights are a softmax over the two chosen logits
- [ ] rank tokens within their expert's queue with a cumsum, slot 0 before slot 1, and say where the off-by-one sits
- [ ] say what the balancing loss reads under uniform and under collapsed routing, and which half of it carries the gradient
- [ ] describe the gather, compute, scatter forward of a MoE layer, and why it accumulates with `index_add_`
- [ ] state the speculative acceptance rule, and show why it reproduces the target distribution exactly
- [ ] derive the expected number of tokens per speculative cycle
- [ ] write the beam step as add, flatten, topk, divmod, and say why per-beam topk fails

## Going deeper

- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), Zhang and Sennrich 2019, which derives the RMSNorm gradient in its appendix.
- [Extending PyTorch](https://pytorch.org/docs/stable/notes/extending.html), the full contract of `torch.autograd.Function`.
- [Mistral 7B](https://arxiv.org/abs/2310.06825), for sliding window attention and its rolling KV cache.
- [Efficient Sequence Packing without Cross-contamination](https://arxiv.org/abs/2107.02027), Krell et al. 2021, for what packing saves and why the document mask is needed.
- [GShard](https://arxiv.org/abs/2006.16668), Lepikhin et al. 2020, the top-2 gating with expert capacity that later MoE models build on.
- [Switch Transformers](https://arxiv.org/abs/2101.03961), Fedus et al. 2021, for the load balancing loss and top-1 routing.
- [Mixtral of Experts](https://arxiv.org/abs/2401.04088), Jiang et al. 2024, for top-2 routing with renormalized weights at scale.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), Leviathan et al. 2023, whose appendix has the correctness proof in full.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), Chen et al. 2023, the same idea found independently.
- [Medusa](https://arxiv.org/abs/2401.10774), Cai et al. 2024, speculative decoding without a separate draft model.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751), Holtzman et al. 2020, on why beam search makes dull open-ended text.
