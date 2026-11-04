# Topic 11 - Normalization

> **Exercise** `test_11_normalization.py`

## Why this topic

A transformer is a tall stack of layers, and every layer takes a tensor of numbers and hands a new tensor of numbers to the next one.
Nothing in that arrangement keeps the numbers a sensible size.
If each layer makes its output a little larger than its input, say by a third, then after eight layers everything is about twelve times larger, and after forty-eight layers it has overflowed.
If each layer makes it a little smaller, the signal fades into rounding noise instead.
Training makes this worse, because the weights keep changing, so the size of what each layer receives keeps drifting too.

A **normalization layer** is the fix.
It looks at one token's vector of `D` numbers, measures how big the numbers in it are, and rescales the vector to a standard size.
Whatever the previous layer produced, the next layer receives something of a predictable scale.
That one guarantee is what makes a 48-layer model trainable at all.

The code is three lines, which is why it is easy to get subtly wrong.
Using the wrong kind of variance, computing it in low precision, or putting the normalization on the wrong side of the residual connection all give a model that trains, just not as well.
None of them raise an error.

This topic covers the two normalization layers you will meet, **LayerNorm** and **RMSNorm**, and the question of **where** in a block to put them.
Why RMSNorm replaced LayerNorm, and why "pre-norm" replaced "post-norm", is also the shortest summary of what changed between the original 2017 transformer and a modern LLM.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| activations | the numbers flowing between layers; for a batch of sequences, a `(B, T, D)` tensor |
| a token's vector | one `(D,)` slice of that tensor: everything the model holds about one position |
| channel | one of the `D` positions inside that vector |
| residual stream | the running `(B, T, D)` tensor that every block reads from and adds its result back into |
| sublayer | the thing inside a block that computes something: attention (topic 14) or the feed-forward network (topic 12) |

### A refresher on mean, variance and RMS

All three are summaries of a list of numbers, and topic 04 computed them as reductions.
Take the list `[1, 2, 3, 4]`.

The **mean** is the average: $(1 + 2 + 3 + 4) / 4 = 2.5$.
It says where the numbers sit.

The **variance** is the average squared distance from the mean.
The distances are `[-1.5, -0.5, 0.5, 1.5]`, their squares are `[2.25, 0.25, 0.25, 2.25]`, and the average is `1.25`.
It says how spread out the numbers are.
The **standard deviation** is its square root, `1.118`, back in the same units as the numbers.

The **root mean square**, RMS, skips the mean entirely: square every number, average, take the square root.
$(1 + 4 + 9 + 16) / 4 = 7.5$, and $\sqrt{7.5} = 2.739$.
It says how large the numbers are, measured from zero rather than from their average.
When the mean happens to be 0, RMS and standard deviation are the same thing.

**Biased and unbiased variance.**
The average above divided by `D`, the number of values.
Statistics textbooks often divide by `D - 1` instead, which corrects a bias when the list is a small sample from some larger population.
That is the **unbiased** variance, and it is what `x.var()` computes by default:

```python
>>> a = torch.tensor([1., 2., 3., 4.])
>>> a.var()                  # divides by D - 1 = 3
tensor(1.6667)
>>> a.var(correction=0)      # divides by D = 4
tensor(1.2500)
```

Normalization layers are not estimating anything about a population.
They describe the exact vector in front of them, so they use the plain average, dividing by `D`, the **biased** variance.
`correction=0` asks for it, and so does the older spelling `unbiased=False`.
The same applies to `x.std()`.

### 11.1 The two formulas

**LayerNorm** works on one token's vector at a time, along the last axis:

```math
\begin{aligned}
\mu &= \text{mean of } x \qquad \text{over the last axis} \\
\mathrm{var} &= \text{mean of } (x - \mu)^2 \qquad \text{biased, divided by } D \\
y &= \frac{x - \mu}{\sqrt{\mathrm{var} + \varepsilon}} \cdot \text{weight} + \text{bias}
\end{aligned}
```

Read the last line in two halves.
$(x - \mu) / \sqrt{\mathrm{var} + \varepsilon}$ is the normalization itself: subtract the mean so the vector is centered on zero, then divide by the standard deviation so its spread is 1.
`* weight + bias` then lets the model undo some of that if it wants: `weight` and `bias` are learned, one number per channel, both of shape `(D,)`.

On `[1, 2, 3, 4]`: subtract 2.5 to get `[-1.5, -0.5, 0.5, 1.5]`, divide by 1.118 to get `[-1.342, -0.447, 0.447, 1.342]`.
That has mean 0 and standard deviation 1, whatever the input looked like.

**RMSNorm** is LayerNorm with two pieces removed:

```math
\begin{aligned}
\mathrm{rms} &= \sqrt{\text{mean of } x^2 + \varepsilon} \qquad \text{over the last axis} \\
y &= x / \mathrm{rms} \cdot \text{weight}
\end{aligned}
```

No mean is subtracted, and there is no `bias`.
On `[1, 2, 3, 4]`: divide by 2.739 to get `[0.365, 0.730, 1.095, 1.461]`.
That has RMS 1, but its mean is 0.913, not 0.

**What the two pieces were doing.**
LayerNorm was introduced with two jobs in mind, re-centering and re-scaling, and RMSNorm (2019) was the experiment of dropping the first one.
Quality held up.
The usual explanation is that the mean of a long vector of activations carries almost no information, so throwing away the step that removes it costs almost nothing.
What RMSNorm saves is real: one reduction instead of two, no subtraction, and `D` fewer parameters per layer.
LLaMA, Mistral, Qwen, CS336's assignment and essentially every model since 2022 use it.

**Shapes and broadcasting.**
The mean, the variance and the RMS are reductions over the last axis, one number per token.
For `x` of shape `(B, T, D)` they have shape `(B, T, 1)` with `keepdim=True`, which is what lets them broadcast back against `x` in the subtraction and the division (topics 03 and 04).
Without `keepdim` they are `(B, T)`, and the broadcast either fails or, worse, lines up the wrong axes.
`weight` and `bias` are `(D,)`, which broadcasts against the last axis of `(B, T, D)` on its own.
Writing the reduction over `-1` rather than over a fixed axis number makes the same function work for an input of shape `(D,)`, `(T, D)` or `(B, T, D)`, and the tests try all three.

**Only the last axis.**
The statistics are taken separately for every token of every example.
Nothing is ever averaged across the batch or across positions.
That is what makes it *layer* normalization rather than *batch* normalization, the older technique from image models that averages over the batch.
It is also why LayerNorm behaves identically at batch size 1 and needs no stored statistics at inference time.

**Where `eps` goes.**
`eps` is a small number, `1e-5` here, added to protect against dividing by zero.
A vector of all zeros has variance 0 and RMS 0, and without `eps` that is `0 / 0 = nan`; the tests check that an all-zero input comes out finite.

It goes **inside** the square root: $\sqrt{\mathrm{var} + \varepsilon}$, never $\sqrt{\mathrm{var}} + \varepsilon$.
The two agree when $\mathrm{var}$ is large and disagree exactly when $\varepsilon$ matters.
With $\mathrm{var} = 10^{-8}$, $\sqrt{\mathrm{var} + \varepsilon}$ is about $0.0032$, while $\sqrt{\mathrm{var}} + \varepsilon$ is about $0.00011$, thirty times smaller, so the output would be thirty times larger.

**`rsqrt`.**
`torch.rsqrt(t)` computes $1 / \sqrt{t}$ in one operation, so dividing by a square root can be written as multiplying by an `rsqrt`.
Real implementations do that, because multiplication is cheaper than division.
The two spellings agree to float32 rounding.

**Gradients.**
The test for RMSNorm runs `torch.autograd.gradcheck` (topic 08), which feeds in float64 tensors and compares autograd's gradient with a numerical estimate.
That only works if your function keeps the dtype it was given.
Write it in ordinary tensor operations and do not force a dtype inside it.

### Precision, and why every real implementation upcasts

This part is not in the formula, and it is in every production codebase.

Large models usually run in **bfloat16**, a 16-bit float format with the same range as float32 but far less precision.
It keeps only about 3 significant decimal digits:

```python
>>> torch.tensor(1 / 3, dtype=torch.bfloat16).item()
0.333984375
>>> torch.tensor(1000.3, dtype=torch.bfloat16).item()
1000.0
>>> torch.tensor(257.0, dtype=torch.bfloat16).item()
256.0
```

Now think about what normalization computes.
It squares `D` numbers, adds them up, and the whole output depends on that sum.
Every square is rounded to bfloat16, and if the running total is kept in bfloat16 too, then once it grows large, small additions stop registering at all, exactly as `257` became `256` above.

The extreme case makes the point.
Adding up 4096 squared values one at a time with the total held in bfloat16 gives a mean square of `4.19`, when the true answer is `8.88`.
PyTorch's own reduction kernels are smarter than that loop and accumulate in higher precision internally, but the inputs, every square, the mean that comes out, the `rsqrt` and the final product are still each rounded to 3 digits.
That is a small error added in every normalization layer of every block, on every step.

So the pattern is universal: **upcast, normalize, cast back**.
Note the input's dtype, convert the input to float32, do the whole computation there, and convert the result back to the dtype you noted.
`t.dtype` gives a tensor's dtype, and `t.to(dtype)` converts.

The cast back is not optional.
The rest of the model is running in bfloat16, and a float32 tensor handed to it silently promotes every later operation to float32 (topic 01's promotion rules), doubling memory with no error.

`torch.autocast`, which topic 18 covers, keeps PyTorch's own `F.layer_norm` in float32 on purpose, a strong hint about how much this matters.
That protection only covers the library's own layer.
A hand-written norm gets nothing for free and has to upcast itself.

### 11.2 The modules

Both modules are small `nn.Module` subclasses in the shape topic 09 taught:

- `__init__` calls `super().__init__()`, stores `eps`, and creates the learned parameters;
- `forward` calls the function from 11.1 with those parameters.

The parameters, and their starting values:

| Module | `weight` | `bias` |
| --- | --- | --- |
| RMSNorm | `(d_model,)`, ones | none |
| LayerNorm | `(d_model,)`, ones | `(d_model,)`, zeros |

Ones and zeros are chosen so that a freshly built norm does pure normalization: multiplying by 1 and adding 0 change nothing.
Training then adjusts them.

The names `weight` and `bias` matter, because the tests check the `state_dict` keys, and later topics load whole models by them.
RMSNorm must have **no** `bias` at all, not a bias of zeros: the test expects exactly one key, `weight`.

`eps` is a plain Python float stored as an ordinary attribute.
It is a setting, not a tensor, so by topic 09's rules it is neither a parameter nor a buffer, and it correctly does not appear in the `state_dict`.
There are no buffers in either module.

### 11.3 Pre-norm versus post-norm

**The residual connection.**
Every block in a transformer has the same outline: run a sublayer, then **add** its result back onto its input.
$x + \mathrm{sublayer}(\dots)$ is called a residual connection.
It means each block only has to learn a correction to the vector, not rebuild the vector from scratch, and the running sum of all those corrections is the residual stream.

The question is where the normalization goes in that outline, and there are two answers.

**Post-norm**, the original 2017 transformer, normalizes **after** adding:

```math
x = \mathrm{norm}(x + \mathrm{sublayer}(x))
```

**Pre-norm**, used by almost everything since about 2019, normalizes only **what goes into the sublayer**, and adds onto the raw stream:

```math
x = x + \mathrm{sublayer}(\mathrm{norm}(x))
```

They look like a reshuffle of the same three pieces.
They behave very differently.

**Follow the stream.**
In pre-norm, the residual stream itself is never normalized.
Each block reads a normalized copy, computes something, and adds that onto the stream; the stream is never rescaled.
So from the input of the network to its output there is a path that passes through nothing but additions.

That path matters on the backward pass.
The derivative of $x + \text{something}$ with respect to $x$ includes a plain 1 (topic 08), so the gradient travels back along the stream unchanged, and it arrives at layer 1 with the same size it had at layer 48.
Deep pre-norm models train stably from scratch.

In post-norm, every addition is immediately followed by a normalization, so the stream itself gets rescaled at every block, and so does the gradient on its way back.
Those rescalings multiply together across depth, and past a couple of dozen layers training becomes fragile.
The 2020 paper that analysed this showed that in post-norm the gradients at initialization depend on depth, and that the long **learning rate warmup** that post-norm models needed (starting training with tiny steps and slowly increasing them, topic 16) existed mostly to survive the first steps where that is worst.
Pre-norm can train with little or no warmup; large models keep a short one anyway, for reasons that have more to do with the optimizer than with the norms.

**A quick test of which one you wrote.**
Imagine a sublayer that outputs all zeros.
Pre-norm then returns $x$ unchanged, exactly, because the stream is never touched.
Post-norm returns $\mathrm{norm}(x)$, which is different.
The exercise has a test for exactly this.

**The extra norm at the end.**
Because the pre-norm stream is never normalized, the vector reaching the end of the model may have any size.
So a pre-norm model adds one last normalization after the final block, before the output layer: `ln_final` in this repository.
A model with $L$ blocks therefore has $2L + 1$ norms: two per block, one before attention and one before the feed-forward network, plus the final one.

**The two stubs take different arguments.**
Read the signatures in 11.3 carefully.
The post-norm function receives the sublayer's **output**, already computed.
The pre-norm function receives the sublayer **module** itself, because it has to decide what to call it on.

### 11.4 Properties worth feeling

**RMSNorm ignores scale.**
Multiply the input by any positive number $c$.
Every value gets $c$ times bigger, the RMS gets $c$ times bigger, and the division cancels them:

```math
\mathrm{rms\_norm}(c x) = \mathrm{rms\_norm}(x) \qquad \text{for } c > 0
```

That is the stabilizing guarantee from the start of this file, in one line: however large the previous layer's output was, the next layer sees the same thing.
It also means the vector's **direction** carries all the information and its length carries none.

`eps` breaks the identity slightly, since it is added to $c^2 \cdot \mathrm{mean}(x^2)$ and not scaled with it, so the measured difference is small but not exactly zero.
"Measured difference" means the same thing as in topic 10: subtract the two outputs, take the absolute value, take the largest entry.
`weight` should be all ones for this, and `torch.ones` needs the right length, `x.shape[-1]`, to broadcast.

**RMSNorm does not ignore shifts, LayerNorm does.**
Add the same number to every entry.
LayerNorm subtracts the mean first, so the shift vanishes immediately, and the output is unchanged.
RMSNorm has no such step, so the shift changes the output:

```python
>>> x = torch.tensor([[1., 2., 3., 4.]])
>>> F.layer_norm(x + 10, (4,))
tensor([[-1.3416, -0.4472,  0.4472,  1.3416]])     # same as for x
>>> F.rms_norm(x + 10, (4,), eps=1e-5)
tensor([[0.8765, 0.9562, 1.0359, 1.1155]])         # different from x
```

So LayerNorm ignores both scale and shift, and RMSNorm ignores scale only.

**What comes out of LayerNorm.**
With `weight = 1` and `bias = 0`, every token's vector has mean exactly 0 and biased standard deviation exactly 1, up to the tiny effect of `eps`.
That is the definition of the layer, and the exercise asks you to measure it.
Measure the standard deviation with the same `correction=0` the layer itself uses: with the default `D - 1`, a 64-channel vector reads as about `1.008`, not 1.
RMSNorm's output has an RMS of 1 but, in general, a nonzero mean, which is the observable difference between them.

## API you need

| Call | Result |
| --- | --- |
| `x.mean(-1, keepdim=True)` | mean over the last axis, kept as a size-1 axis |
| `x.var(-1, keepdim=True, correction=0)` | biased variance; the default `correction=1` divides by `D - 1` |
| `x.std(-1, correction=0)` | biased standard deviation |
| `torch.var_mean(x, -1, correction=0, keepdim=True)` | `(var, mean)` in one pass |
| `x.pow(2)` / `x ** 2` / `x.square()` | elementwise square |
| `torch.sqrt(t)` / `torch.rsqrt(t)` | square root / $1 / \sqrt{t}$ in one op |
| `x.dtype`, `x.to(dtype)`, `x.float()` | read the dtype, convert, convert to float32 |
| `torch.ones(n)` / `torch.zeros(n)` | starting values for `weight` and `bias`; also take `dtype=` |
| `nn.Parameter(t)` | a learned tensor, registered on assignment (topic 09) |
| `F.layer_norm(x, (D,), weight=None, bias=None, eps=1e-5)` | PyTorch's LayerNorm, for checking yours |
| `F.rms_norm(x, (D,), weight=None, eps=None)` | PyTorch's RMSNorm, 2.4+; pass `eps` explicitly to match |
| `nn.LayerNorm(D)` / `nn.RMSNorm(D)` | the library modules |

`normalized_shape` in `F.layer_norm` and `F.rms_norm` is a tuple of the trailing axes to normalize over, `(D,)` here.

### A worked example on neutral data

The library versions on one short vector, to compare your own against:

```python
>>> x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
>>> y = F.layer_norm(x, (4,))
>>> y
tensor([[-1.3416, -0.4472,  0.4472,  1.3416]])
>>> y.mean(-1), y.std(-1, correction=0)
(tensor([0.]), tensor([1.0000]))
>>> r = F.rms_norm(x, (4,), eps=1e-5)
>>> r
tensor([[0.3651, 0.7303, 1.0954, 1.4606]])
>>> r.pow(2).mean(-1).sqrt(), r.mean(-1)       # RMS is 1, mean is not 0
(tensor([1.0000]), tensor([0.9129]))
>>> F.rms_norm(x * 100, (4,), eps=1e-5)        # scaled input, same output
tensor([[0.3651, 0.7303, 1.0954, 1.4606]])
```

## Traps

- `x.var()` and `x.std()` default to dividing by `D - 1`.
  Normalization divides by `D`: pass `correction=0`.
- $\sqrt{\mathrm{var}} + \varepsilon$ is not $\sqrt{\mathrm{var} + \varepsilon}$.
- Without `eps`, an all-zero vector gives `0 / 0 = nan`.
- Forgetting `keepdim=True` breaks the broadcast in the subtraction or the division.
- Reducing over a fixed axis number instead of `-1` breaks the same function for inputs of other ranks.
- Computing the statistics in bfloat16 adds a little noise to every layer.
  Upcast to float32, then cast the result back.
- Returning float32 from a norm inside a bfloat16 model promotes everything after it and silently doubles memory.
- Forcing float32 inside the plain `rms_norm` breaks the float64 gradient check.
- RMSNorm has no bias.
  A bias of zeros is still a `bias` key in the `state_dict`, and fails the test.
- `eps` is a plain attribute, not a parameter or a buffer.
- Post-norm and pre-norm in this exercise take different arguments: an output for one, a module for the other.
- A pre-norm block must return its input unchanged when the sublayer outputs zeros; if it does not, the norm is on the stream.
- A pre-norm model without a final `ln_final` hands an unnormalized stream to the output layer.

## Ready when

Without looking anything up you can:

- [ ] explain in two sentences why a deep stack of layers needs normalization at all
- [ ] compute the mean, biased variance and RMS of `[1, 2, 3, 4]` by hand
- [ ] write the LayerNorm and RMSNorm formulas from memory, with `eps` in the right place
- [ ] say what RMSNorm removes from LayerNorm and why that costs almost nothing
- [ ] say which variance normalization uses and why `x.var()`'s default is wrong for it
- [ ] explain the upcast, normalize, cast back pattern, and what goes wrong without each half of it
- [ ] build both modules with the right parameter names, shapes and starting values
- [ ] write both residual placements and argue, in terms of the gradient's path, why pre-norm trains more stably
- [ ] say why a pre-norm model needs one extra norm at the end, and how many norms a 12-block model has
- [ ] state what RMSNorm is invariant to, and what LayerNorm is additionally invariant to

## Going deeper

- [Layer Normalization](https://arxiv.org/abs/1607.06450), Ba, Kiros and Hinton 2016, the original paper, short and readable.
- [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467), Zhang and Sennrich 2019, the experiment of dropping the mean.
- [On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745), Xiong et al. 2020, the pre-norm argument in full.
- [`F.layer_norm`](https://pytorch.org/docs/stable/generated/torch.nn.functional.layer_norm.html) and [`nn.RMSNorm`](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html), for the exact signatures you test against.
