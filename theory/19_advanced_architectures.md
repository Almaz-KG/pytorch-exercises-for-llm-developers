# Topic 19 - Advanced mechanisms

> **Exercise** `test_19_advanced_architectures.py`

## Why this topic

The earlier topics built a plain transformer, trained it and generated text with it.
The open frontier models of 2025 and 2026, DeepSeek-V3 and V4, Qwen3 and Qwen3.5, Kimi K2 and Kimi Linear, gpt-oss, Gemma 3, Llama 4, are still that transformer underneath.
What separates them from it is a handful of pieces swapped in on top of the plain design, and each piece started as one paper that the next generation of models copied.
Topic 13 already took two of them, the sliding window and the attention sink, because they are a mask and a column of the softmax.
This topic drills five more, each one a change to the model's structure rather than to a single formula.

- **Multi-head Latent Attention**, MLA: how DeepSeek and Kimi cache one small latent per token instead of a key and a value per head, and the algebra that lets attention run on that latent directly.
- **Linear attention**: Gated DeltaNet, the recurrent layer that takes three of every four layers in Qwen3-Next and Qwen3.5, its two equivalent forms, and what a hybrid stack costs in memory.
- **Mixture of Experts** as DeepSeek-V3 and its descendants do it: many small experts, sigmoid routing, a shared expert, and balancing without an auxiliary loss.
- **N-gram memory**: a huge embedding table indexed by hashed n-grams rather than by single tokens, read in constant time and gated into the residual stream (DeepSeek's Engram, Qwen3.8-Flash-Next).
- **Speculative decoding**: free generation speed with a mathematical guarantee that the output distribution does not change.

Its five sections are independent.
Take them one at a time, in any order; nothing in 19.3 needs anything from 19.1.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| local, global layer | a layer with a sliding window mask, and a layer with the plain causal mask (topic 13) |
| latent, `Dc` | the compressed vector MLA caches per position, `Dc` wide, from which keys and values are rebuilt |
| absorption | folding an up-projection into the query or the output so the big per-head tensor is never built |
| state, `S` | the fixed-size memory of a linear-attention layer, one `(Dk, Dv)` matrix per head |
| decay gate $\alpha$, write strength $\beta$ | how much of the old state survives a step, and how strongly the new pair is written |
| hybrid stack | a model whose layers mix types: global with local, or linear attention with full attention |
| routed, shared expert | an expert a token reaches only when the router picks it, and one every token passes through |
| routing bias | a per-expert number added to the router scores for the choice only, never for the weights |
| slot | which of a token's `K` chosen experts we mean: slot 0 is its best expert, slot 1 its second best |
| capacity | the most tokens one expert accepts in a batch; the surplus is dropped |
| n-gram, hash head | the last `n` tokens ending at a position, and one of several independent hash functions that map it to a table row |
| draft, target | the small fast model that proposes tokens, and the large model whose distribution we want |
| acceptance rate, $\alpha$ | the probability that the target accepts one draft token |

$\alpha$ means a decay in 19.2 and an acceptance rate in 19.5; the two sections never meet, and both names are the ones their papers use.

### 19.1 Multi-head Latent Attention

**The problem it solves.**
During generation the KV cache, not the weights, is what limits the batch size at long context.
Topic 13 cut it with GQA, by sharing one key and value head among a group of query heads.
MLA, introduced in DeepSeek-V2 and used by DeepSeek-V3 and V4, Kimi K2 and Kimi Linear, cuts it further without sharing heads: it caches one small **latent** vector per position and rebuilds every head's key and value from it.

```math
c_s = W_{\mathrm{DKV}} \, x_s \in \mathbb{R}^{D_c} \qquad k_{s,h} = W_{\mathrm{UK}}^{(h)} c_s \qquad v_{s,h} = W_{\mathrm{UV}}^{(h)} c_s
```

`w_uk[h]` and `w_uv[h]` are `(Dh, Dc)`: each head has its own up-projection, so the heads stay different, but they all read the same `Dc` numbers.
DeepSeek-V3 has 128 heads of width 128 and a latent of 512.
Full multi-head attention would cache $2 \cdot 128 \cdot 128 = 32768$ numbers per token per layer; MLA caches $512 + 64 = 576$, the 64 being the rotary part explained below, about 57 times less.
The query is compressed the same way, through its own latent, but only to save training memory; nothing about queries is cached.

**The naive way back**, `mla_expand_kv`, rebuilds `k` and `v` of `(B, H, S, Dh)` from the cached `(B, S, Dc)` latents and runs ordinary attention.
It is correct, and it gives back at every step exactly the memory traffic MLA was meant to save: a decode step would read the small cache and then write out the large one.

**Absorption.**
Matrix products associate, and that lets the up-projections move to the other side of the dot product.
For one head:

```math
q^{\top} k_s = q^{\top} \big(W_{\mathrm{UK}} \, c_s\big) = \big(W_{\mathrm{UK}}^{\top} q\big)^{\top} c_s
```

Move every query into the latent space once, with $W_{\mathrm{UK}}^{\top} q$ of width `Dc`, and compare it with the cached latents directly.
The scale stays $1/\sqrt{D_h}$: the value of every score is unchanged, only the order of multiplication is, and the softmax still needs the scale it was trained with.

The values work the same way from the other end.
The output of a head is a weighted sum of values, and the up-projection is linear, so it commutes with the sum:

```math
\sum_s w_s \, v_s = \sum_s w_s \, W_{\mathrm{UV}} c_s = W_{\mathrm{UV}} \Big(\sum_s w_s \, c_s\Big)
```

Mix the latents with the attention weights first, a `(B, H, T, Dc)` result, and up-project once per query.
In a real model $W_{\mathrm{UV}}$ is absorbed one step further, into the output projection, so even the per-head output of width `Dh` is never built separately; the exercise stops at the head output so the result can be compared with ordinary attention.

Both directions are one `einsum` each, and naming the axes is the way to keep them straight (topic 07): the head axis `h` is shared between the query and the weight, the latent axis `c` is contracted, and the batch of latents has no head axis at all.

When is absorption worth it?
During decoding, `T = 1`, and the absorbed path does $O(S \cdot D_c)$ work per head on a cache that is never expanded, which is the whole point.
During prefill, with `T` as long as `S`, expanding `k` and `v` once and running a standard flash kernel is usually faster, and implementations switch between the two paths.

**Why RoPE needs a side door.**
Rotary embeddings (topic 12) rotate keys by an angle that depends on the position $s$: $k_s = R_s W_{\mathrm{UK}} c_s$.
Absorption now has to move $R_s$ into the query too, and $R_s$ differs for every key, so there is no single matrix left to absorb.
MLA's fix is **decoupled RoPE**: a small separate part of every query and key, 64 wide in DeepSeek-V3, carries the rotation, and only that part.
The key's rotary part is computed straight from $x_s$, shared by all heads, and cached next to the latent, which is the 64 in $512 + 64$; the score is the sum of two dot products, the absorbed one on the latent and an ordinary rotated one on the small part.
The exercise leaves the rotary part out so the algebra stays visible.

### 19.2 Linear attention: Gated DeltaNet and hybrid stacks

**From attention to a recurrence.**
Take causal attention and delete the softmax.
The output at position $t$ becomes $o_t = \sum_{s \le t} (q_t^{\top} k_s) \, v_s$, and the same associativity as in 19.1 regroups it:

```math
o_t = \Big(\sum_{s \le t} v_s k_s^{\top}\Big) q_t = S_t^{\top} q_t \qquad S_t = S_{t-1} + k_t v_t^{\top}
```

`S` is a `(Dk, Dv)` matrix per head, a running sum of outer products.
Generation now carries `S` from step to step instead of a cache that grows with every token: constant memory and constant work per token, at any context length.
This is the **recurrent form**, and the same `S` is called the **state** of the layer.

**What goes wrong, and the two fixes.**
The plain sum never forgets, and an associative memory that only adds fills up with interference: reading at a key returns the value written there plus a blur of everything else.
Two fixes, each a paper, combine into the layer that Qwen3-Next and Qwen3.5 use.

- **A decay gate** $\alpha_t \in (0, 1)$, computed from the token, scales the whole state down before each write: $S_t = \alpha_t S_{t-1} + k_t v_t^{\top}$.
  This is gated linear attention, and Mamba-2's state-space form is the same recurrence.
  The model learns to forget fast where the context changes and slowly where it does not.
- **The delta rule** replaces "add" with "overwrite".
  Before writing $v_t$ at key $k_t$, read what the memory already holds there, $S^{\top} k_t$, and write only the difference, scaled by a write strength $\beta_t \in (0, 1)$:

  ```math
  S \leftarrow S + \beta_t \, k_t \big(v_t - S^{\top} k_t\big)^{\top} = \big(I - \beta_t k_t k_t^{\top}\big) S + \beta_t \, k_t v_t^{\top}
  ```

  This is one step of gradient descent on $\frac{1}{2}\lVert S^{\top} k_t - v_t \rVert^2$, so the state is an online regression from keys to values, and $\beta$ is its learning rate.

Gated DeltaNet applies both, with the decay on the old state only:

```math
S_t = \alpha_t \big(I - \beta_t k_t k_t^{\top}\big) S_{t-1} + \beta_t \, k_t v_t^{\top} \qquad o_t = S_t^{\top} q_t
```

Keys are L2-normalized, so $k^{\top} k = 1$.
That is what makes $I - \beta k k^{\top}$ a clean erase: it scales the component of every column of `S` along $k$ by $1 - \beta$ and leaves everything orthogonal to $k$ alone.
At $\alpha = \beta = 1$ the step replaces whatever the memory held at $k$ with exactly $v$, and one test reads it back to check.
In the models, $\beta$ is a sigmoid of a projection of the token, and $\alpha$ is an exponential of a negative softplus, so both stay inside $(0, 1)$; the exercise takes them as inputs.

**The step without the big matrix.**
$\big(I - \beta k k^{\top}\big) S$ never needs the `(Dk, Dk)` matrix, because $k k^{\top} S = k \, (k^{\top} S)$: first contract `k` with the state over `Dk`, a `(B, H, Dv)` result, then take its outer product with `k` and subtract.
That is $O(D_k D_v)$ per head instead of $O(D_k^2 D_v)$.
In batch form every factor needs explicit trailing axes, `k[..., :, None]` against `x[..., None, :]`, and the scalars $\alpha$, $\beta$ of `(B, H)` need `[..., None, None]` to scale a `(B, H, Dk, Dv)` state; `einsum` over named axes saves most of that bookkeeping.

`gated_delta_recurrent` is the same step in a loop over `T`, collecting the outputs and returning the final state.
A Python loop over time is allowed here, and it is honest: this is what decoding runs, one token per step.
The test that splits a sequence into two calls and hands the first call's final state to the second checks that `initial_state` is actually used.

**The parallel form, for training.**
A loop over `T` is hopeless on a GPU at training length.
For gated linear attention without the delta rule, unroll the recurrence instead:

```math
o_t = \sum_{s \le t} \Big(\prod_{r = s+1}^{t} \alpha_r\Big) (q_t^{\top} k_s) \, v_s
```

That is attention with a decay table in place of the softmax: a `(T, T)` matrix $D_{ts} = \prod_{r \in (s, t]} \alpha_r$ for $s \le t$ and 0 above the diagonal, multiplied element-wise into $Q K^{\top}$ and then into $V$.
The products come from a cumulative sum in log space: with $c_t = \sum_{r \le t} \log \alpha_r$, the product over $(s, t]$ is $\exp(c_t - c_s)$.
That is why the exercise takes `log_alpha` rather than `alpha`.

The trap is the upper triangle.
For $s > t$, $c_t - c_s$ is a sum of positive numbers, and with strong decay it is large: $\exp(930)$ is `inf` in float32, and `inf * 0` is NaN, so masking the decay table **after** the `exp` poisons the output.
Mask in log space, with $-\infty$ above the diagonal, and let the `exp` turn it into an exact 0.
One test runs a decay of $\log \alpha = -30$ over 32 positions to make sure of it.

The delta rule has a parallel form too, but not a simple one, because the erase term couples every step to the state that came before it.
Its training kernels split the sequence into chunks of about 64, run a parallel form inside each chunk through a triangular solve, and pass the state between chunks with the recurrence; the papers in Going deeper derive it.
The two-form idea is the one to keep: one formula, a parallel shape for training, a recurrent shape for generation, with the same numbers coming out of both.

**The output gate.**
Qwen3-Next puts a sigmoid gate on the output of both of its token mixers, the Gated DeltaNet layers and the attention layers:

```math
\mathrm{out} = y \odot \sigma(W_{\mathrm{gate}} \, x)
```

`y` is the per-head output laid out as `(B, T, H, Dh)`, and `x` is the layer's input, so the gate is computed from the token itself and has one value per element of every head.
`w_gate` is `(H * Dh, D)`, so `x @ w_gate.T` comes out `(B, T, H * Dh)` and has to be reshaped to `(B, T, H, Dh)` before it lines up with `y`, which is the head split of topic 02.
In the papers the gate brought a small but consistent improvement and better training stability, and in attention layers it took away the need for an attention sink (topic 13).

**Hybrid stacks.**
Linear attention is cheap but lossy: a fixed-size state cannot recall an exact token from 100k positions back the way full attention can.
So current models mix layer types and let a minority of full-attention layers do exact retrieval.
Qwen3-Next and Qwen3.5 repeat a group of three Gated DeltaNet layers and one gated attention layer; Kimi Linear repeats three linear layers of its own variant, Kimi Delta Attention, and one MLA layer.
Gemma 3 and gpt-oss are hybrids of the same kind with local layers instead of linear ones (topic 13).

The cache of a hybrid is the sum over its layers, and each layer type has its own formula per sequence:

| Layer type | Cached numbers |
| --- | --- |
| full attention, GQA | $2 \cdot T \cdot G \cdot D_h$, keys and values for every position |
| sliding window | $2 \cdot \min(T, \text{window}) \cdot G \cdot D_h$ |
| linear attention | $G \cdot D_h \cdot D_h$, one square state per head, independent of $T$ |
| MLA | $T \cdot (D_c + D_{\mathrm{rope}})$, one latent and one shared rotary key per position |

`kv_cache_elems` takes the list of layer types and adds those up.
The first row is topic 17's `cache_memory_bytes` for one layer, counted in numbers rather than bytes; the other three are what this topic adds to it.
On an arbitrary configuration of 8 kv heads of width 128 and a context of 128k, one full-attention layer caches $2 \cdot 131072 \cdot 8 \cdot 128 \approx 268$M numbers, while one linear layer caches $8 \cdot 128 \cdot 128 = 131$k, two thousand times less.
A 3:1 hybrid therefore pays almost exactly a quarter of the full model's cache, and at long context nearly all of it sits in the few attention layers.

### 19.3 Mixture of Experts

Replace one FFN with `E` FFNs, the **experts**, and send each token to only `K` of them.
Parameters grow with `E`, while the compute per token grows only with `K`.
DeepSeek-V3 holds 671B parameters and activates about 37B of them per token; Kimi K2 holds about 1T and activates 32B.
The catch is memory: every expert must be resident, even though each token uses a fraction of them.
MoE trades memory, which is cheap to add in a datacenter, for compute, which is not; that is why nearly every open frontier model of 2025 and 2026 is a MoE.

The generation before, Mixtral and GShard, used 8 large experts and top-2 routing.
The current recipe, from DeepSeekMoE, uses **many small experts** and a larger `K`: DeepSeek-V3 routes each token to 8 of 256, Qwen3 to 8 of 128, Kimi K2 to 8 of 384.
Smaller experts can specialize more narrowly, and with 8 of them per token there are far more combinations to express a token with.

Five pieces make a MoE layer, and each is one stub.

#### Sigmoid top-k routing

A **router** is a `(E, D)` matrix, a bias-free linear map with no module around it; `x @ router.T` turns `(N, D)` tokens into `(N, E)` logits, one score per expert per token.

The classical rule takes a softmax over the logits.
DeepSeek-V3 takes a **sigmoid** per expert instead: every expert gets an independent score in $(0, 1)$ that does not shrink as `E` grows, which a softmax over 256 entries would do.
The token's `K` experts are the `K` largest scores, and their weights are those scores normalized to sum to 1:

```math
s_e = \sigma(z_e) \qquad \mathcal{T} = \mathrm{topk}_K(s_e + b_e) \qquad w_e = \frac{s_e}{\sum_{e' \in \mathcal{T}} s_{e'}} \quad (e \in \mathcal{T})
```

The $b_e$ is the **routing bias** of the balancing scheme below, and the asymmetry is deliberate: the bias takes part in **which** experts are chosen, never in **how much** they weigh.
With a zero bias the rule is plain top-k on sigmoid scores.

`torch.topk(t, k, dim=-1)` returns the values and the int64 indices, already sorted from largest to smallest, which is the order the contract asks for.
Here only the indices are wanted from it, because the values it returns are the biased ones; the unbiased scores at those indices come from `gather` on the sigmoid scores (topic 05).

#### The shared expert

Every token needs some of the same transformation: common syntax, frequent words.
With routed experts only, each expert has to learn that common part again.
A **shared expert** is an FFN every token passes through with weight 1, next to the routed ones, so the routed experts are left with what is specific.
DeepSeek-V3 and Kimi K2 have one shared expert, as do Llama 4 and Qwen3-Next; Qwen3 has none.
In code it is a dense FFN over all `N` tokens, added to the routed sum.

#### Expert capacity

On real hardware every expert gets a fixed-size buffer, sized ahead of time as roughly `capacity_factor * N * K / E` tokens, so that the computation has a static shape.
An expert accepts at most `capacity` tokens; the surplus is **dropped**: its contribution through that expert is lost, and the token keeps only what its other experts gave it (and the residual connection around the layer).
DeepSeek-V3 reports that its balancing is good enough to drop no tokens at all during training, but capacity is still how expert-parallel systems bound the buffers on each device, and dropping is what happens when the balance fails.
This exercise passes `capacity` in, and asks only for the boolean "did this assignment fit" for each of the `(N, K)` assignments.

**The queue order.**
Each expert fills its slots in a fixed order:

1. first every token's **slot-0** choice, in increasing token order;
2. then every token's **slot-1** choice, in increasing token order;
3. and so on, up to slot `K - 1`.

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

**Carrying the load from one slot to the next.**
A slot-1 assignment's position in the queue is its rank among the slot-1 assignments to the same expert **plus** the number of slot-0 assignments that expert already took.
Two ways to get there:

- lay the `(N, K)` assignments out in queue order, slot-major, as one long `(K * N,)` sequence, rank that, and lay the result back out as `(N, K)`; note that plain `reshape(-1)` on `(N, K)` is token-major, the wrong order, and the transpose has to come first;
- rank each slot's `(N,)` column on its own, and add to every later slot the per-expert totals of the slots before it, which are column sums of their one-hots.

A Python loop over the `K` slots is fine; a loop over the `N` tokens is what the cumsum replaces.
Keep the function general in `K`, read from `expert_idx.shape[1]`.

In this exercise `moe_forward` has no capacity: every token reaches all its experts.
`capacity_mask` stands alone.

#### Balancing with a loss

Without help, routing collapses.
A few experts get chosen a little more often early on, get more gradient, improve faster, get chosen even more, and the rest sit idle.
The Switch Transformer's auxiliary loss pushes back, and generalizes to `K` choices per token directly:

```math
\text{loss} = E \cdot \sum_e f_e \cdot P_e
```

- $f_e$ is the **fraction of all $N \cdot K$ assignments that went to expert $e$**: count them from `expert_idx` and divide by its number of entries.
  `torch.bincount(idx.reshape(-1), minlength=E)` counts directly; `bincount` takes only a 1-D input.
- $P_e$ is the **mean router probability of expert $e$** over all tokens: the column means of `router_probs`.

Both vectors sum to 1 over experts, and the loss reads directly when logged:

- perfectly balanced routing, $f_e = P_e = 1/E$, gives $E \cdot E \cdot (1/E)^2 = 1$;
- total collapse onto one expert, possible with $K = 1$, gives $E$;
- with $K > 1$ a token's choices are distinct experts, so the worst case is collapse onto `K` of them, which gives $E / K$.

$f$ comes from a top-k, so it is discrete and carries no gradient; $P$ is a mean of router outputs, so it does.
The gradient of the loss with respect to $P_e$ is $E \cdot f_e$, which pushes down hardest on the probability of the experts that are actually receiving the most tokens, which is exactly the feedback that undoes collapse.

The function returns a 0-dim float tensor, and the test compares it with `torch.tensor(1.0)`.
A Python float, a `(1,)` tensor or an int64 tensor all fail that comparison on type, shape or dtype; `bincount` and `one_hot` return int64 counts, so divide them into a float.

The trouble with the loss is that it is a second objective fighting the language-model loss: weigh it too little and routing collapses, too much and it costs quality.
Qwen3 kept it and changed what it averages over: $f$ is counted over the global batch across all devices rather than per micro-batch, which lets experts specialize by domain while staying balanced on average.

#### Balancing without a loss

DeepSeek-V3 drops the auxiliary loss as the main mechanism, and steers the routing directly through the bias $b_e$ of the routing rule.
After every batch, count the load of every expert, compare it with the mean load, and move each bias by a fixed step $\gamma$ against the imbalance:

```math
b_e \leftarrow b_e + \gamma \cdot \mathrm{sign}\big(\bar{c} - c_e\big) \qquad c_e = \text{assignments to } e \text{ in the batch}, \quad \bar{c} = \tfrac{1}{E} \textstyle\sum_e c_e
```

An overloaded expert gets its bias lowered and is chosen a little less often next time; an idle one gets it raised; one exactly at the mean keeps its bias, since $\mathrm{sign}(0) = 0$.
The bias is not a parameter: no gradient flows into it, and it is updated outside the optimizer, like a running statistic of BatchNorm.
Because it only touches the choice and not the weights (the rule above), it does not distort the layer's output scale, which was the cost of the loss.
DeepSeek-V3 used $\gamma = 0.001$ for most of its training, together with a sequence-level balancing loss of tiny weight as a safety net.

`update_routing_bias` returns the new bias and leaves the old one untouched: in-place updates of a tensor someone else holds are the classic source of a stateful bug that no shape check catches (topic 16).
One test runs the update for a few hundred batches on a deliberately skewed router and checks that the busiest expert ends up close to the mean load; it is the rule above, applied repeatedly, nothing more.

#### The sparse forward

In this exercise an expert is a two-layer ReLU network without biases, with the weights of all experts stacked along a leading `E` axis.
Real experts are SwiGLU blocks (topic 11); ReLU keeps the focus on the routing.

```
x:          (N, D)
router:     (E, D)
w1:         (E, D_ff, D)     w1[e] is (D_ff, D), the nn.Linear layout: (out, in)
w2:         (E, D, D_ff)     w2[e] is (D, D_ff)
shared_w1:  (D_ff, D)        one more expert, outside the stack
shared_w2:  (D, D_ff)
```

```math
\begin{aligned}
&\text{expert } e \text{ on one token:} && w_2[e] \, \mathrm{relu}(w_1[e] \, x_n) \qquad \text{a } (D,) \text{ vector} \\
&\text{layer output for token } n \text{:} && \text{shared}(x_n) + \sum_{\text{its } K \text{ slots}} \text{weight}[n, \text{slot}] \cdot \text{expert}_{\text{idx}[n, \text{slot}]}(x_n)
\end{aligned}
```

The formula is written for one column vector `x_n`.
For a batch of `n` tokens stored as rows, `(n, D)`, the same map is `rows @ W.T` with `W` in `(out, in)` layout, exactly as `nn.Linear` computes it (topic 08).

**Gather, compute, scatter.**
The naive implementation runs every token through every expert and multiplies the unchosen ones by zero, which spends exactly the compute MoE exists to save.
The real one, and the one the docstring asks for, works expert by expert:

- **gather**: find the tokens routed to expert `e`;
- **compute**: run only those rows through expert `e`;
- **scatter**: scale each result by its routing weight and add it into the output rows of its tokens.

Finding the tokens is a comparison of the `(N, K)` index table with `e`.
`torch.where(cond)` on a 2-D boolean table returns two index tensors, the rows and the columns of its `True` entries; here those are the token numbers **and** the slot through which each token reached `e`, which together pick out the matching routing weight.
A per-token weight is `(n,)` and has to become `(n, 1)` to scale `(n, D)` rows.
`out.index_add_(0, rows, values)` adds `values[k]` into `out[rows[k]]`, and adds rather than overwrites, which is what each token needs: it receives one contribution from each of its `K` experts, on top of the shared one.
An expert that no token chose gets an empty index tensor; every step above works on zero rows, so there is nothing to special-case.

A Python loop over the `E` experts is fine, and is what simple implementations do; a loop over tokens is not.
Production kernels go one step further, sorting the tokens by expert and running all experts as one grouped matrix multiplication, but the gather and the scatter are the same.

The tests route with a reference rule of their own, so `moe_forward` can be written before `topk_sigmoid_routing`.
Inside `moe_forward` you may call your own `topk_sigmoid_routing`, or route inline by the same rule; the dependent test `test_moe_routing_agrees_with_topk_sigmoid_routing` turns on once both exist and checks that the two agree.
Another test zeroes the weights of every expert no token chose and expects the output not to move, which holds automatically for a sparse forward.

### 19.4 N-gram memory

**The observation.**
A lot of what a language model predicts needs no reasoning, only recall: "Barack" is followed by "Obama", a function name is followed by its usual arguments, an idiom finishes itself.
A transformer recovers those facts by spending its first layers recombining the last few tokens, which is computation used as a lookup table.
The idea of n-gram memory is to give the model the lookup table itself.

**The table.**
Index embeddings by the **n-gram** ending at each position, the last `n` tokens, rather than by the single token.
The space of n-grams is far too large to enumerate, $V^n$ for vocabulary $V$, so the n-gram is **hashed** into a table of fixed size.
Collisions are unavoidable, and the remedy is the one Bloom filters use: several independent **hash heads**, each with its own table, so that two n-grams that collide in one head almost never collide in all of them.
The rows the heads return are concatenated into one memory vector per position.

DeepSeek's Engram module and Qwen3.8-Flash-Next, the open preview of the Qwen4 architecture, both build on this.
Qwen3.8-Flash-Next puts one such layer near the bottom of a 125B-parameter MoE and keeps 51B more parameters in its n-gram tables; the Engram paper reports beating a MoE with the same parameter count and the same compute.

**Why it is cheap.**
The table indices depend only on the token ids, not on any activation, so they are known **before** the forward pass begins.
The tables can then live in host memory, off the accelerator, with the needed rows prefetched while the first layers run.
That makes the memory a new axis of sparsity next to MoE: an expert costs a matrix multiplication per token that uses it, a table row costs one read.

**The hash.**
The exercise's hash, given in the docstring, is a multiplicative-XOR hash: multiply each token of the n-gram by its own large odd constant, XOR the products together, take the result modulo the table size.

```math
\mathrm{id}_h(t) = \Big(\, m_{h,0} \, x_t \;\oplus\; m_{h,1} \, x_{t-1} \;\oplus\; \dots \;\oplus\; m_{h,n-1} \, x_{t-n+1} \Big) \bmod P_h
```

The multipliers spread the bits of each token across the word; a different multiplier per offset makes the hash order-sensitive, so "dog bites man" and "man bites dog" land in different rows; prime table sizes $P_h$ keep the modulo from discarding structure.
Engram uses a hash of this family, with a separate prime-sized range for every pair of n-gram order and head; the exercise uses one order at a time.

In tensor form, "the token `j` positions back" is the token row shifted right by `j`, with the vacated places filled by the padding token 0: `F.pad(tokens, (j, 0))` pads `j` zeros on the left of the last axis, and slicing the result back to length `T` drops the overflow on the right.
`torch.bitwise_xor(a, b)`, or `a ^ b` on integer tensors, is the XOR.
The multiplications stay well inside int64, about $2^{17}$ for a token id times $2^{31}$ for a multiplier, and everything is non-negative, so `%` gives exactly the row.
The multipliers are `(Hn, n)` and the shifted tokens `(B, T)`, so the product for one offset is a `(B, T, Hn)` tensor, `shifted[..., None] * multipliers[:, j]`, and the XOR accumulates over the `n` offsets.
`table_sizes` of `(Hn,)` broadcasts against the last axis for the final modulo.

**The gate.**
A lookup ignores context, so it is sometimes wrong: a collision returns a stranger's row, and even the right n-gram can mean different things in different documents.
So the memory is written into the residual stream through a gate that compares it with what the model already believes:

```math
k = W_K \, e \qquad v = W_V \, e \qquad g = \sigma\!\left(\frac{\mathrm{rms}(h)^{\top} \, \mathrm{rms}(k)}{\sqrt{D}}\right) \qquad h \leftarrow h + g \cdot v
```

When the retrieved memory agrees with the hidden state, $g$ opens and the value is added; when it contradicts it, $g$ closes and the lookup is ignored.
`rms` is RMSNorm without a weight, so the gate measures the direction of the two vectors and not their size.
The gate is one number per position, `(B, T, 1)`, and multiplies the whole `(B, T, D)` value.

The lookup itself is advanced indexing: `table[ids + offsets]` with `ids` of `(B, T, Hn)` and `offsets` of `(Hn,)` gives `(B, T, Hn, De)`, one row per head, and merging the last two axes gives the concatenation.
All the heads' tables live in one tensor, head `j` owning the rows from `offsets[j]` on, which is how it is laid out in memory in practice.

### 19.5 Speculative decoding

**Why it works at all.**
Generating one token with a large model is bound by memory bandwidth, not by arithmetic: every step reads all the weights out of GPU memory to do a small amount of work with each.
So scoring `K + 1` positions in one forward pass costs almost exactly what scoring one costs, as long as `K` is small.

The scheme:

1. a small **draft** proposes `K` tokens, cheaply, one at a time, recording its distribution $p_{\text{draft}}$ at each position;
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

**Where the draft comes from now.**
A separate small model is the original form, and it has to share the target's tokenizer and roughly its taste.
Current systems mostly grow the draft out of the target instead.

- **Multi-token prediction.** DeepSeek-V3 and V4 train an extra module that predicts the token after next from the main model's hidden state, as an auxiliary training objective that also densifies the training signal.
  At inference that module is a ready-made draft: DeepSeek-V3 reports that its second-token prediction is accepted 85 to 90 percent of the time, for about 1.8 times the decoding speed.
- **Draft heads on the target's features.** EAGLE trains one small transformer layer that reads the target's last hidden states and proposes the next tokens; Medusa trains several heads that each guess one position ahead.
- **Prompt lookup.** Copy the continuation of the longest earlier match of the last few tokens out of the context itself, an n-gram lookup in the spirit of 19.4, which works well for code editing and summarization, where the output repeats the input.

The acceptance rule above stays exactly the same for all of them; only the source of $p_{\text{draft}}$ changes.

## API you need

| Call | Result |
| --- | --- |
| `t.masked_fill(mask, value)` | `value` where `mask` is `True`, `t` elsewhere |
| `einsum(a, b, "b h t d, h d c -> b h t c")` (einops) | contract named axes; shared names stay, missing names are summed |
| `F.normalize(t, dim=-1)` | divide by the L2 norm along an axis |
| `t.cumsum(dim)` | running sum, inclusive |
| `t.exp()`, `F.logsigmoid(t)` | $e^t$; $\log \sigma(t)$, computed stably |
| `torch.ones(T, T, dtype=torch.bool).tril()` | the causal `(T, T)` table |
| `torch.stack(tensors, dim)` | join along a new axis |
| `t.new_zeros(*sizes)` | zeros with `t`'s dtype and device |
| `torch.sigmoid(t)` | $1 / (1 + e^{-t})$ element-wise |
| `torch.topk(t, k, dim=-1)` | `(values, indices)`, sorted descending, indices int64 |
| `t.gather(dim, idx)` | one entry per row, at the column `idx` names |
| `F.one_hot(idx, num_classes)` | `(..., C)` int64 one-hot |
| `torch.bincount(idx, minlength=E)` | count of each value in $[0, E)$, int64; 1-D input only |
| `torch.sign(t)` | $-1$, $0$ or $+1$ element-wise |
| `torch.where(cond)` | the index tensors of the `True` entries, one per axis |
| `out.index_add_(0, rows, values)` | `out[rows[k]] += values[k]`, accumulating repeats |
| `F.relu(t)` / `torch.relu(t)`, `t.clamp(min=0)` | negative entries to 0 |
| `F.pad(t, (left, right))` | pad the last axis; integer tensors are padded with 0 |
| `torch.bitwise_xor(a, b)`, `a ^ b` | bitwise XOR of integer tensors |
| `a % b` | remainder; exact and non-negative for non-negative integers |
| `table[idx]` with an integer tensor `idx` | one row of `table` per entry of `idx`, shape `idx.shape + (De,)` |
| `t.flatten(-2)` | merge the last two axes |
| `probs[torch.arange(K), tokens]` | one entry per row by advanced indexing |

### A worked example on neutral data

Moving a matrix across a dot product, which is all absorption is:

```python
>>> a, b = torch.randn(3), torch.randn(5)
>>> W = torch.randn(3, 5)
>>> torch.allclose(a @ (W @ b), (W.T @ a) @ b)
True
```

Products over segments from a cumsum of logs, and why the masking has to happen before the `exp`:

```python
>>> c = torch.log(torch.tensor([0.5, 0.5, 0.5])).cumsum(0)
>>> (c[:, None] - c[None, :]).exp()             # entry [t, s] is the product over (s, t]
tensor([[1.0000, 2.0000, 4.0000],
        [0.5000, 1.0000, 2.0000],
        [0.2500, 0.5000, 1.0000]])
>>> torch.tensor(1000.0).exp() * 0              # the upper triangle at strong decay
tensor(nan)
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
>>> torch.sign(torch.tensor([-2.0, 0.0, 3.0]))
tensor([-1.,  0.,  1.])
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

Shifting an integer row, XOR, and reading rows by a table of indices:

```python
>>> t = torch.tensor([[4, 5, 6, 7]])
>>> F.pad(t, (2, 0))[..., :4]                        # every entry two places to the right
tensor([[0, 0, 4, 5]])
>>> torch.tensor([12, 10]) ^ torch.tensor([10, 10])
tensor([6, 0])
>>> rows = torch.arange(12.0).reshape(6, 2)
>>> rows[torch.tensor([[0, 5], [2, 2]])].shape       # one row per index
torch.Size([2, 2, 2])
```

Reading one entry per row:

```python
>>> p = torch.tensor([[0.1, 0.6, 0.3], [0.5, 0.2, 0.3]])
>>> p[torch.arange(2), torch.tensor([1, 0])]
tensor([0.6000, 0.5000])
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

- MLA's absorbed scores keep the scale $1/\sqrt{D_h}$; dividing by $\sqrt{D_c}$ changes every softmax.
- The latent has no head axis; the up-projections do. Contract `Dc`, keep `h`.
- In the delta rule, the decay $\alpha$ scales the erased old state, not the newly written $\beta k v^{\top}$.
- Building `I - beta k k^T` as a `(Dk, Dk)` matrix works and wastes a factor of `Dk`; read, then erase, then write.
- `gated_delta_recurrent` must start from `initial_state` when one is given.
- The parallel decay table is masked in log space before the `exp`; masking after it turns `inf * 0` into NaN.
- The decay over $(s, t]$ excludes $\alpha_s$: position $s$ wrote its pair after its own decay was applied.
- The output gate `(B, T, H * Dh)` has to be reshaped to `(B, T, H, Dh)` before it multiplies `y`.
- A linear-attention layer's cache does not grow with the sequence; a sliding one stops growing at `window`.
- The routing bias picks the experts but must not enter the weights; the weights are the unbiased sigmoid scores, renormalized.
- `topk` on the biased scores returns biased values; take only its indices and `gather` the unbiased scores.
- The capacity cumsum is inclusive: the first token in a queue has position 1, not 0.
- The capacity queue is slot-major: slot 0 of every token before slot 1 of any token; `reshape(-1)` on `(N, K)` is token-major.
- $f_e$ in the balancing loss divides by all $N \cdot K$ assignments, not by `N`; return a 0-dim float tensor, not a Python float or an int64 count.
- `bincount` takes a 1-D tensor; flatten the `(N, K)` indices first.
- The bias update moves against the load: an overloaded expert's bias goes **down**. And it returns a new tensor rather than modifying its argument.
- The shared expert is added for every token with weight 1, outside the routing weights.
- Running every token through every expert defeats the purpose of MoE.
- `w1[e]` is `(D_ff, D)`; rows of tokens multiply its transpose.
- `index_add_` rather than indexed assignment, because a token receives a contribution from each of its `K` experts.
- A routing weight of shape `(n,)` does not broadcast against `(n, D)`; it needs a trailing axis.
- The n-gram shift fills with token 0 on the left and drops the overflow on the right; `torch.roll` wraps around and brings the end of the sequence to its start.
- XOR first, modulo last, and each head with its own table size.
- The n-gram gate is one number per position, `(B, T, 1)`, not one per channel.
- Every head's ids are local to its table; add the head's offset before indexing the stacked table.
- In speculative decoding, continuing past the first rejection breaks the distributional guarantee.
- The residual needs the `relu` before renormalizing, or it has negative mass.
- `target_probs` has `K + 1` rows; the last one is only for the bonus token after a full acceptance.
- `expected_speedup` divides by zero at `alpha == 1`.

## Ready when

Without looking anything up you can:

- [ ] say what MLA caches per token, and compute it next to GQA's for a given configuration
- [ ] show on paper why absorption gives the same scores and outputs, and why RoPE needs a decoupled part
- [ ] write the gated delta rule, and say what $\alpha$, $\beta$ and the normalized key each do
- [ ] turn gated linear attention from a recurrence into a decay-masked attention, and keep it finite
- [ ] add up the cache of a hybrid stack of full, sliding, linear and MLA layers
- [ ] route with sigmoid top-k and a bias, and say why the bias stays out of the weights
- [ ] rank tokens within their expert's queue with a cumsum, slot 0 before slot 1, and say where the off-by-one sits
- [ ] say what the balancing loss reads under uniform and under collapsed routing, and which half of it carries the gradient
- [ ] state the bias update of auxiliary-loss-free balancing, and its sign
- [ ] describe the gather, compute, scatter forward of a MoE layer with a shared expert, and why it accumulates with `index_add_`
- [ ] hash the n-grams of a batch without a loop over positions, and say why the indices are known before the forward pass
- [ ] state the speculative acceptance rule, and show why it reproduces the target distribution exactly
- [ ] derive the expected number of tokens per speculative cycle, and name three sources of a draft

## Going deeper

- [Gemma 3 Technical Report](https://arxiv.org/abs/2503.19786), Gemma Team 2025, for the 5:1 interleaving of local and global layers and what it saves in KV cache.
- [Gated Attention for Large Language Models](https://arxiv.org/abs/2505.06708), Qiu et al. 2025, the output gate and its effect on sinks.
- [DeepSeek-V2](https://arxiv.org/abs/2405.04434), DeepSeek-AI 2024, which introduces MLA, absorption and decoupled RoPE.
- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437), DeepSeek-AI 2024, for MLA at scale, sigmoid routing, auxiliary-loss-free balancing and multi-token prediction.
- [DeepSeek-V4](https://arxiv.org/abs/2606.19348), DeepSeek-AI 2026, compressed sparse attention for million-token contexts on top of the same MoE.
- [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635), Yang et al. 2023, for the decay gate and the chunked training form.
- [Transformers are SSMs](https://arxiv.org/abs/2405.21060), Dao and Gu 2024, Mamba-2 and the decay-masked attention it is equivalent to.
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length](https://arxiv.org/abs/2406.06484), Yang et al. 2024, the chunked form of the delta rule.
- [Gated Delta Networks](https://arxiv.org/abs/2412.06464), Yang, Kautz and Hatamizadeh 2024, the layer Qwen3-Next and Qwen3.5 are built on.
- [Kimi Linear](https://arxiv.org/abs/2510.26692), Moonshot AI 2025, Kimi Delta Attention in a 3:1 hybrid with MLA.
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388), Qwen Team 2025, for fine-grained MoE without a shared expert and global-batch balancing.
- [DeepSeekMoE](https://arxiv.org/abs/2401.06066), Dai et al. 2024, fine-grained and shared experts.
- [Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664), Wang et al. 2024, the bias update in full.
- [Kimi K2](https://arxiv.org/abs/2507.20534), Moonshot AI 2025, the DeepSeek-V3 architecture scaled to 384 experts.
- [Switch Transformers](https://arxiv.org/abs/2101.03961), Fedus et al. 2021, for the load balancing loss and expert capacity.
- [Engram](https://github.com/deepseek-ai/Engram), DeepSeek-AI 2026, conditional memory through hashed n-gram lookup, with the paper in the repository.
- [On the Design of Qwen3.8-Next Architecture](https://arxiv.org/abs/2608.30320), Qwen Team 2026, n-gram embeddings next to a Gated DeltaNet hybrid MoE.
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192), Leviathan et al. 2023, whose appendix has the correctness proof in full.
- [Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318), Chen et al. 2023, the same idea found independently.
- [EAGLE](https://arxiv.org/abs/2401.15077), Li et al. 2024, drafting from the target's own features.
- [Medusa](https://arxiv.org/abs/2401.10774), Cai et al. 2024, speculative decoding without a separate draft model.
