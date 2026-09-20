# Topic 15 - The transformer block

> **Exercise** `test_15_transformer_block.py` · **Reference** `solutions/sol_15_transformer_block.py` · **Assumes** topics 01-14

## Where this fits

The whole model, assembled.

Topics 10 to 14 built the pieces separately.
This file puts them together into a working language model and then checks the one thing that only works when everything is simultaneously correct: an untrained model's loss must equal `log(V)`.

The bricks are deliberately rewritten from scratch rather than imported.
The goal of the repository is that a transformer block comes out of your hands without thinking, and rewriting is the only way that happens.

RoPE is omitted here for brevity, which is the only simplification against the real architecture.

What you need from earlier topics: essentially all of 09 to 14, and the `log(V)` fact from 10.

## What you build here

Seven components: five modules and two functions.
The first three modules are topics 11, 12 and 14 written again from memory.

### 15.1 The bricks

| Component | In -> out | What it must do |
| --- | --- | --- |
| `RMSNorm(d_model, eps=1e-5)` | `(..., D) -> (..., D)` | One `weight` of ones. No bias, and `state_dict` must contain only `weight` |
| `SwiGLU(d_model, d_ff)` | `(..., D) -> (..., D)` | Submodules `w1`, `w2`, `w3`, all `nn.Linear` without bias. `w2(silu(w1(x)) * w3(x))` |
| `CausalSelfAttention(d_model, num_heads)` | `(B, T, D) -> (B, T, D)` | Submodules `q_proj`, `k_proj`, `v_proj`, `output_proj`, each `nn.Linear(D, D, bias=False)`. Causal |

### 15.2 The block

| Component | In -> out | What it must do |
| --- | --- | --- |
| `TransformerBlock(d_model, num_heads, d_ff)` | `(B, T, D) -> (B, T, D)` | Submodules named `ln1`, `attn`, `ln2`, `ffn`. `x = x + attn(ln1(x))` then `x = x + ffn(ln2(x))` |

### 15.3 The whole model

| Component | In -> out | What it must do |
| --- | --- | --- |
| `TransformerLM(vocab_size, d_model, num_heads, d_ff, num_layers)` | `(B, T)` int64 -> `(B, T, V)` | Submodules `token_embeddings`, `layers` (an `nn.ModuleList`), `ln_final`, `lm_head` (no bias). Embeddings and every linear weight initialized `N(0, 0.02)`, which the `log(V)` test depends on |

### 15.4 The arithmetic

| Component | In -> out | What it must do |
| --- | --- | --- |
| `count_lm_params(vocab_size, d_model, d_ff, num_layers, tied_embeddings)` | ints, bool -> `int` | By formula, without building the model. Embeddings, `4*D^2` attention and `3*D*d_ff` SwiGLU per layer, plus `lm_head` unless tied. Norms and biases ignored |
| `activation_memory_bytes(batch, seq_len, d_model, num_layers, dtype)` | ints, dtype -> `int` | One `(B, T, D)` residual tensor per layer. A deliberate underestimate of the order of magnitude |

## The mental model

### The architecture, in full

```
ids                                    (B, T) int64
  -> token_embeddings                  (B, T, D)
  -> for each block:
        x = x + attn(ln1(x))
        x = x + ffn(ln2(x))
  -> ln_final                          (B, T, D)
  -> lm_head                           (B, T, V)
```

Pre-norm, RMSNorm, SwiGLU, no biases anywhere.

### The residual stream

The most useful way to think about a transformer is as a **residual stream**: a `(B, T, D)` tensor that flows from the embeddings straight to the output, with every sublayer *reading* from it, computing something, and *adding* the result back.

Nothing ever overwrites the stream.
Each block's two sublayers are increments, which is why the model degrades gracefully when a layer is ablated and why the gradient reaches layer 0 undiminished, per topic 11.

Read the two lines that way:

```
x = x + attn(ln1(x))     # read a normalized copy, move information between positions, add back
x = x + ffn(ln2(x))      # read a normalized copy, transform each position on its own, add back
```

Attention is the only operation in the whole model that moves information **between token positions**.
Everything else, the FFN, the norms, the embeddings, the head, acts on each position independently.
That single sentence explains why causality is enforced only inside attention, and why a KV cache is possible at all.

### 15.2 The block

The submodule names are fixed: `ln1`, `attn`, `ln2`, `ffn`.
`ln1` normalizes the input to attention, `ln2` the input to the FFN.
Those names are asserted on by the tests and match the reference `state_dict` of CS336.

`forward` is literally the two lines above.

Two mistakes to avoid.
Normalizing the *output* of the sublayer rather than its input is post-norm, discussed in topic 11.
And writing `x = attn(ln1(x))` without the `x +` drops the residual entirely, which trains very badly and is easy to do while typing quickly.

### 15.3 The whole model

| Name | Type |
| --- | --- |
| `token_embeddings` | `nn.Embedding(vocab_size, d_model)` |
| `layers` | `nn.ModuleList` of `TransformerBlock` |
| `ln_final` | `RMSNorm(d_model)` |
| `lm_head` | `nn.Linear(d_model, vocab_size, bias=False)` |

giving `state_dict` keys like `layers.3.attn.q_proj.weight` and `layers.3.ln1.weight`.

`nn.ModuleList`, not a Python list, per topic 09.
`ln_final` is required because pre-norm leaves the stream unnormalized on the way out.

**Initialization matters here specifically.**
The exercise asks for `N(0, 0.02)` on the embeddings and on every linear weight, and the reason is the `log(V)` test.
With small random weights the logits at initialization are near zero and near identical across the vocabulary, so the softmax is near uniform and the loss is `-log(1/V) = log(V)`.
Leave the default `nn.Linear` initialization in place and the logits come out larger and more varied, the initial loss lands above `log(V)`, and the test fails while the model still trains fine.

For `V = 10000` the expected initial loss is about 9.21; for GPT-2's 50257 it is 10.82.

**The `log(V)` check as a debugging tool.**

- Loss noticeably **above** `log(V)`: the initialization is too large, or the output head is badly scaled.
- Loss noticeably **below** `log(V)` on the very first step: information is leaking from the targets, which almost always means a broken causal mask or an off-by-one in the input/target shift.
- Loss **exactly** `log(V)`: the plumbing is right.

### 15.4 The arithmetic

**Parameter count**, ignoring norms and biases:

```
embeddings:      V * D
per layer:       4 * D^2            attention: q, k, v, output
               + 3 * D * d_ff       SwiGLU: w1, w2, w3
lm_head:         V * D              (0 if tied to the embeddings)
```

so

```
total = V*D + L * (4*D^2 + 3*D*d_ff) + (0 if tied else V*D)
```

With the conventional `d_ff = 8/3 * D`, the per-layer term is `4*D^2 + 8*D^2 = 12*D^2`, which is the familiar `12 * L * D^2` rule of thumb for the non-embedding parameters of a transformer.

Two things worth noticing in that formula.
The embedding and head terms scale with `V`, not with depth, so for small models with a large vocabulary they dominate: a 6-layer model with `D=512` and `V=50257` is more than half embeddings.
And `num_heads` appears nowhere, for the reason given in topic 06.

**Activation memory.**
The residual stream alone is `B * T * D * element_size` per layer, and the backward pass needs all `L` of them alive simultaneously:

```
bytes ~= B * T * D * L * element_size
```

This is a deliberate underestimate.
Reality is several times worse, because every sublayer saves its own intermediates: the `(B, H, T, T)` attention matrix from topic 14, the `(B, T, d_ff)` FFN hidden state, the normalization statistics.
A realistic multiplier is somewhere between 5 and 20 depending on what is fused.

The simple formula still gives the right order of magnitude and, more importantly, the right *scaling*: linear in batch, sequence, width and depth.
That is what tells you halving the batch halves activation memory while doing nothing to the weights or the optimizer state, which is the first lever anyone reaches for when a run does not fit.
Topic 18 is about the other levers.

## API you need

Everything in this file has already appeared:

| Call | From |
| --- | --- |
| `nn.Embedding(V, D)`, or a `nn.Parameter` table indexed directly | topics 05, 09 |
| `nn.Linear(d_in, d_out, bias=False)` | topics 06, 09 |
| `nn.ModuleList([...])` and iterating it in `forward` | topic 09 |
| `nn.init.normal_(t, mean=0.0, std=0.02)` | topic 09 |
| `model.apply(fn)` | topic 09 |
| RMSNorm, with the float32 upcast | topic 11 |
| SiLU and the `w1`/`w2`/`w3` gating | topic 12 |
| the head split, causal mask and four-line attention | topic 14 |
| `x.element_size()` | topic 01 |

### A worked example on neutral data

Checking the `state_dict` shape of a model you have just written:

```python
>>> [k for k in model.state_dict() if k.startswith("layers.0")]
['layers.0.ln1.weight',
 'layers.0.attn.q_proj.weight',
 'layers.0.attn.k_proj.weight',
 'layers.0.attn.v_proj.weight',
 'layers.0.attn.output_proj.weight',
 'layers.0.ln2.weight',
 'layers.0.ffn.w1.weight',
 'layers.0.ffn.w2.weight',
 'layers.0.ffn.w3.weight']
>>> import math
>>> F.cross_entropy(model(ids).flatten(0, 1), targets.flatten()), math.log(vocab_size)
(tensor(9.2107, ...), 9.210340371976184)
```

## Traps

- Submodule names are the contract.
  `ln1`, `attn`, `ln2`, `ffn`, `token_embeddings`, `layers`, `ln_final`, `lm_head`, and inside them `q_proj`, `k_proj`, `v_proj`, `output_proj`, `w1`, `w2`, `w3`.
- A Python list of blocks instead of `nn.ModuleList` makes most of the model untrainable.
- Missing `ln_final` in a pre-norm model.
- Normalizing after the sublayer instead of before it.
- Dropping the `x +` in a residual line.
- Leaving the default `nn.Linear` initialization and then wondering why the initial loss is not `log(V)`.
- Reusing one `RMSNorm` instance for both `ln1` and `ln2`, which ties their weights by accident.
  Each needs its own instance.
- Building one block and appending the same object `L` times to the `ModuleList`, which ties every layer together.
  Construct inside the comprehension.
- `lm_head` with a bias.
- Counting heads in the parameter formula.

## Ready when

Without looking anything up you can:

- [ ] write the whole model, block included, from memory with the canonical names
- [ ] say which single operation in the model moves information between positions
- [ ] explain what the residual stream is and why pre-norm never normalizes it
- [ ] predict the initial loss of an untrained model and say what a deviation in each direction means
- [ ] write the parameter-count formula and say why `num_heads` is absent from it
- [ ] say how activation memory scales in `B`, `T`, `D` and `L`, and why the simple formula is an underestimate

## Context



**What training actually costs in memory.**
Per parameter `P`, a standard bfloat16 run with Adam holds roughly:

```
weights          P * 2 bytes   (bf16)
gradients        P * 2 bytes
Adam state       P * 8 bytes   (two fp32 moments)
master weights   P * 4 bytes   (if kept)
activations      the formula above
```

which is why "a 7B model needs about 16 GB" is true for inference and badly wrong for training, where the same model needs well over 100 GB before a single activation exists.
Topic 16 does this arithmetic properly, and everything in topic 18 is an attack on one of these lines.

**Where weight tying pays.**
The embedding and head terms are `2 * V * D` untied and `V * D` tied, and neither depends on depth.
For a small model with a big vocabulary that is most of the parameter count, so tying is close to free capacity.
At 70B parameters the same `V * D` is a rounding error and the argument goes the other way: the input and output roles of a token genuinely differ, and there is capacity to spare, so several large models do not tie.

**The residual stream framing is not just pedagogy.**
It comes from interpretability work, where treating the stream as a shared communication channel that each head reads from and writes to turned out to predict real behaviour: heads that compose across layers, features that live in identifiable directions, and circuits that can be traced end to end.
If the framing feels unusually useful, that is why, and the paper under "Going deeper" is where it comes from.

## Going deeper

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) if you want the pictures.
- [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html), which is where the residual stream framing comes from.
- [The Llama paper](https://arxiv.org/abs/2302.13971), section 2.2, which is this exact architecture in four sentences.
