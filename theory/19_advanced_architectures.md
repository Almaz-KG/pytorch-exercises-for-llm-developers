# Topic 19 - Advanced mechanisms

> **Exercise** `test_19_advanced_architectures.py` · **Reference** `solutions/sol_19_advanced_architectures.py` · **Assumes** topics 01-18

## Where this fits

Parts of the model swapped for the variants modern models actually use, plus two ways of running the finished model faster.

Five storylines, each a paper that became a standard, and together they are most of what separates a 2019 transformer from a 2025 one.

- A hand-derived backward pass, because that is how custom kernels are written.
- Structured masks, because training on packed documents and serving long context both depend on them.
- Mixture-of-Experts, because it is how parameter count is decoupled from compute.
- Speculative decoding, because it is free speed with no quality cost.
- Beam search, because it is still the right answer for translation and anything with a scoring objective.

This is the hardest file in the repository, and the five sections are independent.
Take them one at a time.

What you need from earlier topics: `autograd.Function` (08), RMSNorm (11), arange masks (03), `cumsum` and `index_add_` (05), `topk` (04), the KV cache reorder (17).

## What you build here

Eleven components: ten functions and one `autograd.Function`.

### 19.1 A backward of your own

| Component | In -> out | What it must do |
| --- | --- | --- |
| `RMSNormFunction` | `forward(ctx, x, weight, eps)`, `backward(ctx, grad_out)` | RMSNorm with a hand-derived backward. `backward` returns **three** values, `None` for `eps` |

### 19.2 Harder masks

| Component | In -> out | What it must do |
| --- | --- | --- |
| `sliding_window_mask(seq_len, window)` | ints -> `(T, T)` bool | `i - window < j <= i`. At `window >= T` it must equal the plain causal mask |
| `document_mask(doc_ids)` | `(T,)` int64 -> `(T, T)` bool | Causal **and** same-document |
| `block_diagonal_lengths_to_doc_ids(lengths)` | `list[int] -> (T,)` | `[3, 2, 4]` becomes `[0,0,0,1,1,2,2,2,2]` |

### 19.3 Mixture of Experts

| Component | In -> out | What it must do |
| --- | --- | --- |
| `top2_routing(router_logits)` | `(N, E) -> ((N, 2)` int64, `(N, 2))` | Softmax over **the two chosen logits only**, so the weights sum to 1 per token. Indices descending by logit |
| `capacity_mask(expert_idx, num_experts, capacity)` | `(N, K)`, ints -> `(N, K)` bool | True when the token fits the expert's capacity. Tokens queue by index, slot 0 before slot 1. No loop over tokens |
| `load_balancing_loss(router_probs, top1_idx, num_experts)` | `(N, E)`, `(N,)`, int -> scalar | `E * sum_e f_e * P_e`. Exactly 1.0 under uniform routing, `E` under collapse |
| `moe_forward(x, router, w1, w2)` | `(N, D)`, `(E, D)`, `(E, F, D)`, `(E, D, F) -> (N, D)` | Top-2, `w2_e @ relu(w1_e @ x)` per expert, summed with the routing weights. Gather the tokens for each expert, compute only those, scatter back |

### 19.4 Speculative decoding

| Component | In -> out | What it must do |
| --- | --- | --- |
| `speculative_accept(draft_probs, target_probs, draft_tokens, u)` | `(K, V)`, `(K+1, V)`, `(K,)`, `(K,)` -> `(int, (V,))` | Accept while `u[i] <= p_target/p_draft`. On the first rejection return that index and `norm(relu(p_target - p_draft))`. On full acceptance return `(K, target_probs[K])` |
| `expected_speedup(acceptance_rate, k)` | float, int -> float | `(1 - alpha^(k+1)) / (1 - alpha)`, and `k + 1` when `alpha == 1` |

### 19.5 Beam search

| Component | In -> out | What it must do |
| --- | --- | --- |
| `beam_search_step(beam_scores, next_log_probs, beam_size)` | `(num_beams,)`, `(num_beams, V)`, int -> three tensors | New scores, parent beam indices, token indices. Pick the best `beam_size` across **all** `num_beams * V` candidates |

## The mental model

### 19.1 The RMSNorm backward, by hand

Forward, statistics over the last axis of width `D`:

```
r   = (mean(x^2) + eps)^(-1/2)
y_i = g_i * x_i * r
```

The derivation, which the exercise wants you to do on paper before reading.

```
dr/dx_j = -1/2 * (mean(x^2) + eps)^(-3/2) * (2 x_j / D) = -x_j r^3 / D
```

Write `a_i = grad_y_i * g_i`, folding the weight into the incoming gradient.
`x_i` affects `y` twice: directly through its own term, and indirectly through `r`, which every output depends on.
So

```
dL/dx_j = a_j * r  +  (sum_i a_i * x_i) * dr/dx_j
        = a_j * r  -  x_j * r^3 * sum_i(a_i * x_i) / D
```

And the weight gradient, where `g` is shared across every leading axis, so the gradient sums over all of them:

```
dL/dg_i = sum over batch and time of grad_y_i * x_i * r
```

Two structural facts worth extracting, because they generalize.
**A broadcast in the forward pass becomes a sum in the backward pass**, which is why `g` accumulates over all leading axes; the linear bias in topic 06 did the same.
And **a shared statistic produces a coupling term**: because `r` depends on every `x_i`, the gradient at `j` contains a sum over all `i`.
Any normalization's backward has exactly this shape.

Implementing it as a `torch.autograd.Function` follows topic 08's contract.
`forward` takes `(ctx, x, weight, eps)`, so `backward` returns **three** values with `None` for `eps`.
Save `x` and `weight` with `save_for_backward` and keep `eps` on `ctx` as a plain attribute, since it is not a tensor.

`gradcheck` against the autograd version, in float64, is how you confirm it.

### 19.2 Harder masks

**Sliding window** (Mistral, Longformer).
Position `i` may see `j` when `i - window < j <= i`: causal, plus a floor on how far back.
As a comparison of aranges that is two conditions ANDed together, and at `window >= T` it must reduce exactly to the plain causal mask, which is the test worth writing.

The point is cost: attention becomes `O(T * window)` instead of `O(T^2)`, and information still propagates across the full sequence because stacked layers compose their windows, giving an effective receptive field of `L * window`.

**Document masking**, also called sequence packing.
Training on short documents wastes compute on padding, so several documents are concatenated into one length-`T` example.
But attention must not cross document boundaries, or the model learns to condition on unrelated text.

Given `doc_ids` of `(T,)`, the mask is

```
causal  AND  doc_ids[i] == doc_ids[j]
```

which is `arange(T)[:, None] >= arange(T)[None, :]` combined with `doc_ids[:, None] == doc_ids[None, :]`.
The result is block-diagonal within the lower triangle.

Turning a list of lengths `[3, 2, 4]` into `[0,0,0,1,1,2,2,2,2]` is `repeat_interleave` over `arange(len(lengths))`.

### 19.3 Mixture of Experts

Replace one FFN with `E` FFNs and route each token to only `K` of them, usually `K = 2`.
Parameters scale with `E`, FLOPs per token scale with `K`.

**Top-2 routing**, Mixtral's rule.
A linear `router` of `(E, D)` produces logits `(N, E)`.
Take the top 2, and softmax **over those two logits only**, so the two weights sum to 1 per token.
Softmaxing over all `E` and then selecting would give weights summing to less than 1 and a token-dependent total scale.
`topk` already returns its results sorted descending.

**Capacity.**
On a real accelerator every expert gets a fixed-size buffer, so each accepts at most `capacity` tokens and the rest are **dropped**: their contribution through that expert is lost, and they pass through with only their other expert's output.

The allocation rule, without a loop: build a one-hot of `(N, E)` for a given slot, `cumsum` it along the token axis, and the value at each token's expert is that token's **position in the queue** for that expert.
Keep the token if the position is below `capacity`.
Because the cumsum is inclusive the first token gets position 1, so either compare with the appropriate off-by-one or subtract the one-hot to get an exclusive count.
Slot 0, the more confident choice, is processed before slot 1 and takes capacity first.

This is the third appearance of the same trick, after topic 05's EOS masking and topic 17's nucleus sampling.
It is worth recognizing as a pattern: **cumsum over a one-hot or a boolean gives you a rank without a loop.**

**Load balancing loss**, from the Switch Transformer:

```
loss = E * sum_e f_e * P_e
```

where `f_e` is the fraction of tokens whose top-1 expert is `e`, and `P_e` is the mean router probability assigned to `e` over all tokens.

Why this shape.
Both `f` and `P` sum to 1 over experts.
Their dot product is minimized when both are uniform, giving `E * E * (1/E)^2 = 1`, and maximized when both concentrate on one expert, giving `E`.
So the loss reads 1 for perfect balance and `E` for total collapse, which makes it directly interpretable when logged.
It uses `f`, discrete and non-differentiable, times `P`, continuous and differentiable, so the gradient flows through `P` while `f` acts as a fixed weight measuring where the tokens actually went.

Without this auxiliary loss routing collapses: a few experts get chosen, improve faster because they get more gradient, and get chosen even more.
It is added to the main loss with a small coefficient, typically 0.01.

**The sparse forward.**
The naive implementation runs every token through every expert and masks, which wastes exactly the compute MoE exists to save.
The real one is **gather, compute, scatter**:

```
for each expert e:
    idx = the tokens routed to e
    out[idx] += weight[idx] * expert_e(x[idx])
```

with `index_add_` from topic 05 for the accumulation, since a token routed to two experts receives two contributions.
The loop over `E` experts is fine and is what real implementations do; the forbidden thing is a loop over tokens.

### 19.4 Speculative decoding

Generation is bound by memory bandwidth, not compute: each decoded token requires reading the entire model's weights out of memory, and the arithmetic done with them is trivial.
So verifying `K` tokens in one forward pass costs almost exactly what verifying one costs.

The scheme: a small **draft** model proposes `K` tokens cheaply and autoregressively; the large **target** model scores all `K + 1` positions in a single pass; an acceptance rule decides how many to keep.

**The acceptance rule.**
For the token `t` the draft proposed at position `i`, with `u ~ Uniform[0, 1)`:

- Accept if `u <= p_target(t) / p_draft(t)`.
  This accepts unconditionally whenever the target likes the token at least as much as the draft did.
- On the first rejection at position `j`, resample from the **residual distribution** `norm(relu(p_target - p_draft))` and stop.
  Everything after `j` is discarded, because it was conditioned on a token that was not kept.
- If all `K` are accepted, the target's distribution at position `K` yields one extra **bonus** token for free, since that forward pass already computed it.

The guarantee is that the resulting token distribution is **exactly** the target model's, not an approximation.
The proof is a short calculation: the probability of emitting token `t` is `p_draft(t) * min(1, p_target(t)/p_draft(t))` from the acceptance branch, plus the rejection probability times the residual's mass at `t`, and the two sum to `p_target(t)`.
The `relu` and the renormalization are what make the residual a proper distribution over exactly the tokens the target wants more than the draft does.

**Expected tokens per cycle** at acceptance rate `alpha`:

```
E[tokens] = (1 - alpha^(K+1)) / (1 - alpha)
```

the sum of the geometric series `1 + alpha + ... + alpha^K`, counting the bonus token.
At `alpha == 1` the formula is `0/0` and the answer is `K + 1`, so that case is handled separately.

### 19.5 Beam search

Maintain `beam_size` partial sequences, each with an accumulated log probability, and at each step keep the `beam_size` best continuations across all beams jointly.

Given `beam_scores` of `(num_beams,)` and `next_log_probs` of `(num_beams, V)`:

1. Add: `total = beam_scores[:, None] + next_log_probs`, giving `(num_beams, V)`.
   Adding log probabilities is multiplying probabilities, which is why beam search works in log space.
2. Flatten to `(num_beams * V,)`.
3. `topk(beam_size)`.
4. Recover the pairs: `beam_idx = flat_idx // V`, `token_idx = flat_idx % V`.

The flatten, topk, divmod pattern is the thing to remember.
A per-beam topk followed by a merge is both more code and wrong, because it cannot express "all `beam_size` winners came from one beam", which happens constantly.

`beam_idx` is the parent pointer, and it is what the KV cache must be reordered by after every step, which is `gather_beams` from topic 05.

## API you need

| Call | Used for |
| --- | --- |
| `torch.autograd.Function`, `ctx.save_for_backward`, `ctx.saved_tensors` | 19.1 |
| `torch.autograd.gradcheck(f, inputs)` | verifying 19.1, in float64 |
| `x.rsqrt()`, `x.pow(2).mean(-1, keepdim=True)` | 19.1 |
| `arange(T)[:, None] >= arange(T)[None, :]` | causal comparison |
| `mask_a & mask_b` | combining masks |
| `torch.repeat_interleave(arange(n), lengths)` | lengths to doc ids |
| `torch.topk(x, k, dim=-1)` | routing and beam search |
| `F.softmax(top_logits, dim=-1)` | top-2 weights |
| `F.one_hot(idx, E)` | routing one-hots |
| `x.cumsum(dim)` | capacity queue positions |
| `torch.nonzero(mask, as_tuple=True)` | the token indices for one expert |
| `out.index_add_(0, idx, values)` | scatter expert outputs back |
| `torch.relu(a - b)` | the speculative residual |
| `t / t.sum()` | renormalizing a residual distribution |
| `flat.topk(k)` then `idx // V`, `idx % V` | the beam step |
| `torch.div(idx, V, rounding_mode="floor")` | integer division without a deprecation warning |

### A worked example on neutral data

```python
>>> lengths = [3, 2, 4]
>>> doc_ids = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
>>> doc_ids
tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
>>> same_doc = doc_ids[:, None] == doc_ids[None, :]
>>> causal = torch.arange(9)[:, None] >= torch.arange(9)[None, :]
>>> (same_doc & causal)[4]           # position 4 sees only positions 3 and 4
tensor([False, False, False,  True,  True, False, False, False, False])

>>> onehot = F.one_hot(torch.tensor([0, 1, 0, 0]), 2)
>>> onehot.cumsum(0)                  # each token's queue position per expert
tensor([[1, 0], [1, 1], [2, 1], [3, 1]])
```

## Traps

- `backward` must return one value per `forward` input, with `None` for `eps`.
- In the RMSNorm backward, the weight gradient sums over **all** leading axes, not just the batch.
- A sliding window mask must coincide with the causal mask when `window >= T`.
- Document masking needs causality **and** the same-document condition; either alone is wrong.
- Softmaxing over all `E` experts and then selecting the top 2 gives weights that do not sum to 1.
- The capacity cumsum is inclusive, so the first token's position is 1, not 0.
- Running every token through every expert defeats the purpose of MoE.
- `index_add_` rather than assignment, because a token contributes through two experts.
- Forgetting the load balancing loss lets routing collapse within a few thousand steps.
- In speculative decoding, accepting on `u <= p_t/p_d` and then *continuing* past a rejection breaks the distributional guarantee.
- The residual distribution needs the `relu` before renormalizing, or it has negative mass.
- `expected_speedup` divides by zero at `alpha == 1`.
- Per-beam topk instead of a flat one cannot select multiple winners from the same beam.
- `//` on a tensor warns about rounding mode in some versions; `torch.div(..., rounding_mode="floor")` is the explicit form.

## Ready when

Without looking anything up you can:

- [ ] derive the RMSNorm backward, including why there is a coupling term
- [ ] state the two general rules a normalization backward illustrates
- [ ] build a sliding-window and a document mask from arange comparisons
- [ ] explain top-2 routing, expert capacity, and how a cumsum assigns queue positions
- [ ] say what the load balancing loss reads under uniform and under collapsed routing
- [ ] explain why the speculative acceptance rule reproduces the target distribution exactly
- [ ] write the beam step as flatten, topk, divmod, and say why per-beam topk fails

## Context



**What MoE actually buys.**
Mixtral 8x7B holds 47B parameters and uses about 13B per token, so it costs roughly a 13B model to run and behaves closer to a much larger one.
The catch is memory: all 47B have to be resident even though only a fraction is used per token, so MoE trades memory capacity, which is cheap to add, for compute, which is not.
That is a good trade in a datacenter and a bad one on a laptop, which is why MoE models dominate served frontier systems and are rare among locally-run ones.

**Sequence packing is not a niche trick.**
It is how essentially all modern pretraining is done, and on a corpus with varied document lengths it is worth 20 to 40% of the compute that would otherwise go into padding.
The document mask is the part that makes it correct rather than merely cheap; without it, packing quietly teaches the model that unrelated documents predict each other.

**Why a draft model can be so much cheaper and still agree.**
Most tokens in natural text are easy: punctuation, common words, the second half of a word the tokenizer split.
A 1B draft agrees with a 70B target on those, and disagrees exactly where the interesting decisions are, which is where the target's verification pass takes over.
Real acceptance rates with a well-matched draft are 0.7 to 0.9, giving two to three tokens per cycle.
Variants avoid the second model entirely: Medusa trains extra prediction heads on the target itself, and n-gram or prompt-lookup drafting proposes continuations copied from the context, which works surprisingly well for summarization and code editing.

**Why beam search lost for chat.**
It needs **length normalization**, dividing the score by length or by `length^alpha`, because otherwise longer sequences are penalized simply for accumulating more negative log-probabilities.
And even normalized, it produces bland repetitive text for open-ended generation, which is the finding topic 17's nucleus sampling paper opens with: the most probable continuation is not the most human one.
It remains the right tool for translation, summarization and any task with a well-defined best answer.

## Going deeper

- [Switch Transformers](https://arxiv.org/abs/2101.03961) and [Mixtral of Experts](https://arxiv.org/abs/2401.04088).
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), Leviathan et al., whose appendix has the correctness proof in full.
- [Mistral 7B](https://arxiv.org/abs/2310.06825) for sliding window attention in practice.
- [Efficient Sequence Packing](https://arxiv.org/abs/2107.02027) for what document masking is worth.
