# Topic 15 - The transformer block

> **Exercise** `test_15_transformer_block.py`

## Why this topic

Topics 10 to 14 built the parts of a language model one at a time: the loss, the normalization, the feed-forward network, the embeddings, attention.
This topic snaps them together into a model that takes token ids and returns, for every position, a score for every possible next token.
It is the same architecture as Llama and as the CS336 assignment, minus RoPE, which is left out for brevity.

Two things make it more than gluing.

First, there is one idea that makes the whole architecture readable, the **residual stream**.
Once you see the model as a stream of vectors that every layer reads from and adds to, the order of the lines in a block, the reason for the final norm and the reason a deep model trains at all stop being things to memorize.

Second, an assembled model has failure modes that none of its parts has on its own.
Everything can have the right shape while the residual is missing, a norm is shared by accident, or the initialization is off by a factor of fifty.
So this topic also teaches the checks that catch that: the loss of an untrained model must be `log(V)`, a model must be able to memorize one batch, and changing a later token must never change an earlier prediction.

The three bricks at the top of the exercise, RMSNorm, SwiGLU and attention, are deliberately written once more from scratch.
The point of the repository is that a transformer block comes out of your hands without thinking, and repetition is how that happens.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| token id | an integer in `[0, V)` standing for one piece of text; the model's input is a `(B, T)` tensor of them |
| logits | the model's output, `(B, T, V)`: for every position, one unnormalized score per vocabulary entry for the **next** token |
| residual stream | the `(B, T, D)` tensor that flows from the embeddings to the output, which every sublayer reads and adds to |
| sublayer | one of the two halves of a block: attention or the feed-forward network |
| block, layer | one `ln1, attn, ln2, ffn` unit; the model is `L` of them in a row |
| pre-norm | the norm is applied to a sublayer's **input**, not to its output |
| `V`, `D`, `d_ff`, `L` | vocabulary size, model width, FFN hidden width, number of layers |

### A refresher: the bricks

You built each of these before; here is what each one does in one line.

- **RMSNorm** (topic 11): divides every vector by its root mean square, then multiplies by a learned per-channel `weight` that starts at 1: `x / sqrt(mean(x²) + eps) * weight`, with the mean over the last axis.
- **SwiGLU** (topic 12): the feed-forward network, `w2(silu(w1(x)) * w3(x))`, three bias-free linear maps; `w1` and `w3` go from `D` up to `d_ff`, `w2` comes back down to `D`.
- **Causal multi-head self-attention** (topic 14): projections `q_proj`, `k_proj`, `v_proj`, a head split, scaled dot-product attention under a causal mask, a head merge, `output_proj`.
- **Embedding** (topic 13): a `(V, D)` table; looking up `(B, T)` ids gives `(B, T, D)` vectors.

All of them except attention act on each position on its own.

### The architecture, in full

```
ids                                    (B, T) int64
  -> token_embeddings                  (B, T, D)
  -> for each of the L blocks:
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
  -> ln_final                          (B, T, D)
  -> lm_head                           (B, T, V)
```

Pre-norm, RMSNorm, SwiGLU, and no biases anywhere.

The output at position `t` is the model's guess for the token at position `t + 1`.
Training compares it with the actual next token through cross-entropy (topic 10), which is why inputs and targets are the same sequence shifted by one.

### The residual stream

The most useful way to read a transformer is as a **residual stream**: a `(B, T, D)` tensor that runs from the embeddings straight to the output.
Every sublayer reads a copy of it, computes something, and **adds** the result back.
Nothing ever overwrites the stream.

Read the two lines of a block that way:

```
x = x + attn(ln1(x))     # read a normalized copy, gather information from earlier positions, add it back
x = x + ffn(ln2(x))      # read a normalized copy, transform each position on its own, add it back
```

Three consequences follow, and each of them is something you would otherwise have to memorize.

**A deep model can train.**
The derivative of `x + f(x)` with respect to `x` is `1 + f'(x)`: the gradient always has a direct path back through the `+`, untouched by any sublayer (topic 08).
So a signal from the loss reaches layer 0 of a 100-layer model as easily as layer 99.
Without the `x +`, every layer multiplies the gradient by its own derivative on the way back, and a deep stack trains badly or not at all.

**A block with silent sublayers is the identity.**
If both sublayers output zero, each line becomes `x = x + 0`, and the block passes its input through unchanged.
That is a test in the exercise: zero the last weight matrix of each sublayer (`output_proj` and `w2`) and the block's output must equal its input exactly.
It also means a freshly initialized model starts close to "embeddings straight into the head", and each layer learns a small correction on top.

**Only attention moves information between positions.**
Embeddings, norms, the FFN and the head all act on each position independently.
Attention, under its causal mask, is the only place where position `t` reads anything from another position, and only from earlier ones.
That single fact is why causality needs to be enforced in one place only, and why a KV cache (topic 17) is possible at all.

### Pre-norm, and why there is a final norm

**Pre-norm** means each sublayer sees a normalized copy of the stream, `attn(ln1(x))`, while the stream itself is never normalized.
The original 2017 transformer did it the other way round, **post-norm**: `x = norm(x + attn(x))`, which normalizes the stream after every addition (topic 11 compares the two).
Pre-norm keeps the residual path clean, which makes training stable without a careful warm-up, and every modern LLM uses it.

The price is that nothing ever normalizes the stream itself.
After `L` blocks have each added their contribution, its scale can be anything, so one last norm, `ln_final`, is applied before the head.
Forget it and the logits inherit whatever scale the stream drifted to.

### 15.1 The bricks

The exercise asks for each brick again, with the same contracts as before.

**RMSNorm.**
One parameter, `weight`, of shape `(D,)`, initialized to ones, and nothing else in its `state_dict`: no bias, and `eps` stored as a plain attribute, not a parameter or buffer.
The test compares against `x * rsqrt(mean(x²) + eps) * weight` with `eps = 1e-5`, `eps` inside the square root.

**SwiGLU.**
Three `nn.Linear` layers without biases, named `w1`, `w2`, `w3`, so its parameters are `3 * D * d_ff`.
`w1` and `w3` are `D -> d_ff`, `w2` is `d_ff -> D`.
The test uses an odd `d_ff = 21` to make sure nothing assumes a round number.

**CausalSelfAttention.**
The multi-head attention module from topic 14, but in a single class: there are no `split_heads` or `causal_attention` helpers in this file to call, so the head split, the causal mask and the merge happen inside the class.
Private helper methods are fine; what the tests see is the four submodules `q_proj`, `k_proj`, `v_proj`, `output_proj`, each `nn.Linear(D, D, bias=False)`, and the output.
The test compares that output against a reference built from your module's own weights with `F.scaled_dot_product_attention(..., is_causal=True)`, for one, two and four heads.

### 15.2 The block

`TransformerBlock(d_model, num_heads, d_ff)` owns four submodules with fixed names: `ln1`, `attn`, `ln2`, `ffn`.
`ln1` normalizes the input to attention, `ln2` the input to the FFN.
The names match the reference `state_dict` of CS336, and the tests ask for them by name.

`forward` is literally the two lines from the architecture.
The test rebuilds those two lines from your block's own submodules and compares.

`ln1` and `ln2` must be **two separate instances**.
They have the same shape, so assigning one `RMSNorm` object to both names raises nothing, but then a single `weight` serves both places and the two learn as one.
The same goes for every submodule: each attribute assignment should construct a new object.

Three mistakes that keep every shape correct:

- normalizing the sublayer's **output** instead of its input, which is post-norm;
- writing `x = attn(ln1(x))` without the `x +`, which drops the residual;
- feeding `ffn` the block's original input instead of the stream after the attention line: the second line reads the **updated** `x`.

### 15.3 The whole model

`TransformerLM(vocab_size, d_model, num_heads, d_ff, num_layers)` has exactly four children:

| Name | Type |
| --- | --- |
| `token_embeddings` | `nn.Embedding(vocab_size, d_model)` |
| `layers` | `nn.ModuleList` of `num_layers` `TransformerBlock`s |
| `ln_final` | `RMSNorm(d_model)` |
| `lm_head` | `nn.Linear(d_model, vocab_size, bias=False)` |

which gives `state_dict` keys like `token_embeddings.weight`, `layers.3.attn.q_proj.weight`, `layers.3.ln1.weight` and `lm_head.weight`.
The `3` in those keys is the position of the block in the `ModuleList`.

`nn.Embedding(V, D)` is a module holding a `(V, D)` `weight`, and calling it on ids is the table lookup from topic 05, `weight[ids]`.

**`nn.ModuleList`, not a Python list** (topic 09).
`nn.Module` finds submodules only among attributes that are themselves modules.
A plain list of blocks is invisible to it: the blocks' parameters are missing from `parameters()`, so the optimizer never trains them, `.to(device)` never moves them, and `state_dict()` never saves them.
A `ModuleList` registers each element, and iterating over it in `forward` works like iterating over a list.

Likewise, every layer must be its own block.
Building one block and putting the same object into the list `L` times gives `L` references to one set of weights: the model is as deep as intended but has the parameters of one layer.
Construct a new block for each position, for example inside a list comprehension.

**Initialization matters here specifically.**
The exercise asks for `N(0, 0.02)`, a normal distribution with mean 0 and standard deviation 0.02, on the embedding table and on every linear weight.
The norm weights stay at their initial ones.
The usual way is `model.apply(fn)`, which calls `fn` once on every module in the tree, and inside `fn` an `isinstance` check picks out the `nn.Linear` and `nn.Embedding` modules and calls `nn.init.normal_` on their weights (topic 09).
This has to happen after the submodules exist, so at the end of `__init__`.

The reason is the next test.

#### The `log(V)` check

Before training, the model knows nothing, so the best it can honestly do is spread its prediction evenly: probability `1/V` for every token.
Cross-entropy then gives `-log(1/V) = log(V)` (topic 10).

| `V` | `log(V)` |
| --- | --- |
| 512 | 6.24 |
| 10000 | 9.21 |
| 50257 (GPT-2) | 10.82 |
| 128256 (Llama 3) | 11.76 |

A model only lands there if its initial logits are close to equal across the vocabulary, that is, small.
With the final norm, the vector entering `lm_head` has root mean square 1, so each logit is a sum of `D` terms whose spread is set by `lm_head`'s weights.
For `D = 64` that gives:

| Initialization of `lm_head` | std of the logits | initial loss above `log(V)` by about |
| --- | --- | --- |
| `N(0, 0.02)` | 0.16 | 0.01 |
| `nn.Linear` default (uniform, std about `1/sqrt(3D)` = 0.072) | 0.58 | 0.17 |

The excess is roughly half the variance of the logits, `std² / 2`.
The test requires the initial loss within 0.1 of `log(V)`, and a separate test checks the standard deviation of every embedding and linear weight directly.
`nn.Embedding`'s default initialization is `N(0, 1)`, fifty times larger than asked, so leaving the defaults in place is caught there.

The check is worth running on every real model you build, because each way of failing it points somewhere specific:

- loss noticeably **above** `log(V)`: the initialization is too large, or the head is badly scaled;
- loss noticeably **below** `log(V)` before any training: the model can see the answer, which almost always means a broken causal mask or inputs and targets not shifted by one;
- loss **at** `log(V)`: the plumbing is right.

#### Two more checks on the whole model

**Causality, end to end.**
Change the token at position 3 and run the model again.
The logits at positions 0, 1, 2 must not change at all, because they are not allowed to see position 3; the logits from position 3 on must change.
This tests the real `forward`, embeddings to head, rather than the mask in isolation.

**Memorizing one batch.**
Train on one small batch for 60 steps with AdamW: the loss must fall by more than a factor of ten.
A model that cannot even memorize its training batch has a broken gradient path, typically a detached tensor, a missing residual, or parameters that never reached the optimizer.

### 15.4 The arithmetic

**Parameter count**, ignoring the norm weights:

```
embeddings:      V * D
per layer:       4 * D²             attention: q, k, v, output
               + 3 * D * d_ff       SwiGLU: w1, w2, w3
lm_head:         V * D              (0 if tied to the embeddings)

total = V*D + L * (4*D² + 3*D*d_ff) + (0 if tied else V*D)
```

`num_heads` appears nowhere: splitting `D` into heads reinterprets the same `D x D` matrices, it does not add any (topic 14).
The test checks the formula on a GPT-2-small-sized model with a SwiGLU FFN: `V = 50257`, `D = 768`, `d_ff = 3072`, `L = 12`, about 152M parameters tied.
A second test builds a real model and compares the formula with `sum(p.numel() for p in model.parameters())`: the only difference must be the norm weights, `D` per norm, two norms per layer plus `ln_final`.

With the conventional `d_ff = 8/3 * D` (topic 12), the per-layer term is `4D² + 8D² = 12D²`, the familiar rule of thumb that a transformer has about `12 * L * D²` non-embedding parameters.

**Where the parameters are.**
The embedding and head terms grow with `V` and not with depth.
For a small model with a big vocabulary, they dominate: with `L = 6`, `D = 512`, `V = 50257`, the embedding table alone is 25.7M parameters against 18.9M in all six layers together.
At 70B parameters the same `V * D` is a rounding error.

That is also the argument about **weight tying**, using the embedding table as `lm_head`'s weight too.
Tied, the two terms cost `V * D` instead of `2 * V * D`, which is a large saving for a small model.
At large scale the saving is negligible and the input and output roles of a token genuinely differ, so several large models do not tie.

**Activation memory.**
To compute gradients, the backward pass needs the activations of the forward pass.
The residual stream alone is one `(B, T, D)` tensor per layer, and all `L` of them are alive at the same time:

```
bytes ≈ B * T * D * L * bytes_per_element
```

For `B = 8`, `T = 2048`, `D = 4096`, `L = 32` in bf16, that is 4.3 GB.
`bytes_per_element` comes from the dtype exactly as in topic 14: `dtype.itemsize`.

The formula is a deliberate underestimate.
Every sublayer saves its own intermediates as well: the `(B, H, T, T)` attention weights from topic 14, the `(B, T, d_ff)` FFN hidden states, the norm statistics.
The real figure is somewhere between 5 and 20 times larger, depending on what is fused and what is recomputed.

The simple formula still gets the **scaling** right: linear in batch, sequence length, width and depth.
That tells you that halving the batch halves activation memory while doing nothing to the weights or the optimizer state, which is the first lever anyone reaches for when a run does not fit.

**What training costs in memory, all in.**
Per parameter, a standard bf16 run with Adam holds roughly 2 bytes of weights, 2 of gradients, 8 of Adam state (two float32 moments) and often 4 of float32 master weights: about 16 bytes, before a single activation.
So "a 7B model needs about 14 GB" is true for inference in bf16 and badly wrong for training, which needs over 100 GB.
Topic 16 does this arithmetic properly, and topic 18 is a tour of ways to cut each line of it.

## API you need

| Call | Result |
| --- | --- |
| `nn.Embedding(V, D)` | a module with a `(V, D)` `weight`; `emb(ids)` gives `ids.shape + (D,)` |
| `nn.Linear(d_in, d_out, bias=False)` | weight stored as `(d_out, d_in)` |
| `nn.Parameter(torch.ones(d))` | a trainable tensor, registered when assigned as an attribute |
| `nn.ModuleList([...])` | a list whose modules are registered; iterate it in `forward` |
| `nn.functional.silu(x)` / `F.silu(x)` | `x * sigmoid(x)` |
| `torch.rsqrt(t)` | `1 / sqrt(t)` elementwise |
| `t.masked_fill(cond, float("-inf"))`, `t.softmax(dim=-1)` | the attention mask and softmax (topic 14) |
| `model.apply(fn)` | call `fn(module)` on every module in the tree, children first |
| `isinstance(m, nn.Linear \| nn.Embedding)` | type check against several classes at once |
| `nn.init.normal_(t, mean=0.0, std=0.02)` | fill a tensor in place from a normal distribution |
| `model.named_children()`, `model.named_parameters()` | direct children; every parameter with its dotted name |
| `model.get_submodule("layers.0")`, `model.get_parameter("lm_head.weight")` | reach a submodule or parameter by its dotted name |
| `sum(p.numel() for p in model.parameters())` | count the parameters |
| `dtype.itemsize` | bytes per element of a dtype |

### A worked example on neutral data

A plain list hides its modules from the parent; a `ModuleList` does not; one object added twice is counted once:

```python
>>> class Plain(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.parts = [nn.Linear(4, 4) for _ in range(2)]
>>> class Listed(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.parts = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])
>>> sum(p.numel() for p in Plain().parameters()), sum(p.numel() for p in Listed().parameters())
(0, 40)
>>> one = nn.Linear(4, 4)
>>> sum(p.numel() for p in nn.ModuleList([one, one]).parameters())     # shared: 20, not 40
20
```

What `apply` and `isinstance` see, and how dotted names reach inside a model:

```python
>>> net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
>>> _ = net.apply(lambda m: print(type(m).__name__))     # apply returns the model itself
Linear
ReLU
Linear
Sequential
>>> isinstance(nn.Embedding(3, 2), nn.Linear | nn.Embedding)
True
>>> [name for name, _ in net.named_children()]
['0', '1', '2']
>>> net.get_parameter("2.weight").shape
torch.Size([2, 8])
```

## Traps

- The submodule names are the contract: `token_embeddings`, `layers`, `ln_final`, `lm_head`; `ln1`, `attn`, `ln2`, `ffn`; `q_proj`, `k_proj`, `v_proj`, `output_proj`; `w1`, `w2`, `w3`.
- A Python list of blocks instead of `nn.ModuleList` leaves those blocks untrained, unsaved and on the wrong device.
- The same block object appended `L` times, or one `RMSNorm` used as both `ln1` and `ln2`, shares weights by accident.
- Missing `ln_final` in a pre-norm model.
- Normalizing after the sublayer instead of before it.
- Dropping the `x +` in a residual line, or feeding the FFN the block's original input instead of the updated stream.
- Leaving the default initialization: `nn.Embedding` starts at `N(0, 1)` and `nn.Linear` at a uniform with std `1/sqrt(3 * d_in)`, neither of which is `N(0, 0.02)`.
- Initializing before the submodules exist; `apply` must run at the end of `__init__`.
- A bias anywhere, including `lm_head`.
- Counting heads in the parameter formula.
- An initial loss below `log(V)` means a leak, not a good model.

## Ready when

Without looking anything up you can:

- [ ] write the whole model, block included, from memory with the canonical names
- [ ] explain the residual stream, and why a block whose sublayers output zero is the identity
- [ ] say why the gradient reaches the first layer of a deep model through the residual adds
- [ ] say which single operation in the model moves information between positions
- [ ] explain pre-norm versus post-norm, and why pre-norm needs `ln_final`
- [ ] say why a Python list of blocks breaks training, and how to give every layer its own weights
- [ ] predict the initial loss of an untrained model, and say what a deviation in each direction means
- [ ] test causality of a whole model by changing one token
- [ ] write the parameter-count formula, say why `num_heads` is absent, and when the embeddings dominate
- [ ] estimate activation memory, and say why the simple formula is an underestimate

## Going deeper

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), Jay Alammar, if you want the pictures.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), Elhage et al. 2021, where the residual stream framing comes from: heads read from and write to the stream, and that view predicts real behaviour.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), Xiong et al. 2020, on why pre-norm trains without a warm-up.
- [The Llama paper](https://arxiv.org/abs/2302.13971), section 2.2, which is this exact architecture in four sentences.
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), `model.py`, a compact real implementation to read once you have written your own.
