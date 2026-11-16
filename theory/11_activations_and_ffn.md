# Topic 11 - Activations and the FFN

> **Exercise** `test_11_activations_and_ffn.py`

## Why this topic

Every transformer block has two parts.
**Attention** (topic 13) is where the tokens of a sequence look at each other and exchange information.
The **feed-forward network**, FFN for short, is the other part: it takes each token's vector on its own, runs it through a small two-layer network, and adds the result back into the residual stream (topic 10).
Attention moves information between positions; the FFN processes it at each position.

The FFN is easy to overlook, because it is just two or three matrix multiplications with a simple function in between.
But it holds about two thirds of every block's parameters, so most of what a model "knows" is stored in FFN weights, and most of the compute of a forward pass is spent there.

The simple function in between is called an **activation**, and it is the reason the network can learn anything interesting at all.
This topic writes the activations you will meet (ReLU, GELU, SiLU), derives one of their derivatives by hand, builds the two kinds of FFN (the classic one and the gated **SwiGLU** that every modern LLM uses), and does the arithmetic of their sizes.

That arithmetic explains a number that looks arbitrary in every LLaMA config, `d_ff = 11008` for `d_model = 4096`.
By the end of this file you will be able to derive it.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| `d_model`, `D` | the width of a token's vector in the residual stream |
| `d_ff`, `F` | the width of the FFN's hidden layer, usually several times `d_model` |
| linear layer | a matrix multiplication, `x @ W.T`, with the weight in the `(out, in)` convention from topic 08 |
| activation | a fixed function applied to every number separately, with no learned parameters |
| elementwise | applied to each entry on its own; the output has the same shape as the input |
| gate | a multiplier between 0-ish and 1-ish that decides how much of something passes through |

### Why there must be an activation at all

Topic 06 showed that matrix multiplication is associative: $B (A v)$ is the same as $(B A) v$.
So two linear layers in a row are secretly one linear layer, with the single matrix $B A$.
Ten linear layers in a row are still one.
However many you stack, the whole thing can only compute linear functions: stretch, rotate, project.
It could never learn "if this, then that".

An activation breaks the chain.
Put a nonlinear function between the two multiplications, and $B \mathrm{act}(A v)$ no longer collapses into one matrix.
With a wide enough hidden layer, that combination can approximate essentially any function, which is why every FFN is shaped "linear, activation, linear".

The activation is applied elementwise, to every one of the `d_ff` hidden numbers separately.
That makes it cheap, and it means its derivative is elementwise too.

### 11.1 The activations

Here are the three activations of this topic on the same handful of inputs:

| $x$ | -3 | -1 | -0.5 | 0 | 0.5 | 1 | 3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ReLU | 0 | 0 | 0 | 0 | 0.5 | 1 | 3 |
| GELU | -0.004 | -0.159 | -0.154 | 0 | 0.346 | 0.841 | 2.996 |
| SiLU | -0.142 | -0.269 | -0.189 | 0 | 0.311 | 0.731 | 2.858 |

All three agree on the big picture: large positive inputs pass through almost unchanged, large negative inputs are switched off.
They differ in what happens near zero.

**ReLU**, the rectified linear unit, is $\max(x, 0)$: negative becomes 0, positive stays.
It is the simplest possible nonlinearity and was the standard for a decade.

Its weakness is the flat half.
For a negative input the output is a constant 0, so the derivative there is 0, and no gradient flows back through it.
A hidden unit whose input is negative for every example never receives a gradient again, and so never changes: a "dead" neuron.
At exactly $x = 0$ the derivative is undefined mathematically; PyTorch uses 0 there by convention.

**GELU**, the Gaussian error linear unit, is a smooth version of ReLU:

```math
\mathrm{gelu}(x) = x \cdot \Phi(x)
```

$\Phi(x)$ is the probability that a random number from a standard normal distribution comes out below $x$.
It rises smoothly from 0 (for very negative $x$) through 0.5 (at $x = 0$) to 1 (for very positive $x$).
So GELU multiplies $x$ by a soft switch: far positive, the switch is on and $x$ passes; far negative, the switch is off; near zero, $x$ is let through partly.

Unlike ReLU, slightly negative inputs give slightly negative outputs, down to a minimum of about $-0.17$ near $x = -0.75$, and the derivative is never stuck at exactly 0.

$\Phi$ is computed with the **error function** $\mathrm{erf}$, a standard special function available as `torch.erf`:

```math
\begin{aligned}
\Phi(x) &= 0.5 \cdot (1 + \mathrm{erf}(x / \sqrt{2})) \\
\mathrm{gelu}(x) &= x \cdot 0.5 \cdot (1 + \mathrm{erf}(x / \sqrt{2}))
\end{aligned}
```

You do not need to know how $\mathrm{erf}$ is defined to use it.
It is an S-shaped curve from -1 to 1, with $\mathrm{erf}(0) = 0$, $\mathrm{erf}(1) \approx 0.843$ and $\mathrm{erf}(2) \approx 0.995$.

**The tanh approximation.**
When GPT-2 was built, $\mathrm{erf}$ was slow on GPUs, so it used a formula that approximates GELU with $\tanh$, the hyperbolic tangent (another S-shaped curve from -1 to 1, which was fast):

```math
\mathrm{gelu\_tanh}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\left(\sqrt{2 / \pi} \cdot (x + 0.044715 \cdot x^3)\right)\right)
```

The constants are just the ones that make the curve fit: $\sqrt{2 / \pi} \approx 0.798$ and $0.044715$.
On inputs between -5 and 5 the two forms differ by at most about `5e-4`.

Hardware caught up long ago, but a model trained with one form behaves very slightly differently under the other.
The difference is big enough to move a benchmark score and small enough that nobody wants to retrain, so the approximation lives on as a compatibility switch: `F.gelu(x, approximate="tanh")`, and `gelu_new` in HuggingFace, which means "the old one".
`F.gelu`'s default is the exact form, `approximate="none"`.

**SiLU**, the sigmoid linear unit, also called swish, has the same idea with a different soft switch:

```math
\mathrm{silu}(x) = x \cdot \mathrm{sigmoid}(x) \qquad \text{where } \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}
```

$\mathrm{sigmoid}$ is the S-curve from 0 to 1 that you may know from logistic regression: $\mathrm{sigmoid}(0) = 0.5$, $\mathrm{sigmoid}(3) \approx 0.95$, $\mathrm{sigmoid}(-3) \approx 0.05$.
SiLU looks almost exactly like GELU, dips a little deeper (minimum about $-0.28$ near $x = -1.28$), and is cheaper to compute.
It is the activation inside SwiGLU, so it is the one in LLaMA and its descendants, and in CS336.

**The derivative of SiLU, by hand.**
Topic 15 asks you to write a whole backward pass by hand, for RMSNorm, and this is the warm-up.
Two facts are enough.

The product rule: the derivative of $f(x) \cdot g(x)$ is $f' \cdot g + f \cdot g'$.

The derivative of sigmoid is a famously tidy expression of sigmoid itself.
Write $s = \mathrm{sigmoid}(x)$; then

```math
s' = s \cdot (1 - s)
```

Apply the product rule to $x \cdot s$, with $f = x$ (derivative 1) and $g = s$ (derivative $s \cdot (1 - s)$):

```math
\begin{aligned}
\frac{d}{dx} [x \cdot s] &= 1 \cdot s + x \cdot s \cdot (1 - s) \\
&= s \cdot (1 + x \cdot (1 - s))
\end{aligned}
```

That is the formula in the stub's docstring, and now you know where it comes from.
Read it for a moment, because it says something surprising.
At $x = 0$ it is $0.5$.
For large positive $x$, $s$ approaches 1 and the derivative approaches 1, like ReLU's.
But for every $x$ below about $-1.28$ it is **negative**, bottoming out near $-0.1$.
A sigmoid's derivative is always positive; SiLU's is not, because the curve dips down before it comes back up to 0.
GELU behaves the same way, and ReLU, with its derivative of exactly 0 or 1, does not.

To check a derivative you wrote by hand, compare it with PyTorch's automatic differentiation, autograd, which topic 15 explains in full: put `requires_grad=True` on an input, run the library function, call `.sum().backward()`, and read `.grad`.
`.sum()` is needed because `backward()` wants a single number, and since the function is elementwise, the gradient of the sum at each position is exactly the derivative at that position.

### 11.2 The FFN blocks

**The classic FFN**, used from the 2017 transformer through GPT-3:

```math
\mathrm{FFN}(x) = W_2 \mathrm{act}(W_1 x)
```

Project up, apply the activation, project back down.
With the weights stored in the `(out, in)` convention from topic 08, and `x` of shape `(..., d_model)`:

| Step | Weight shape | Result shape |
| --- | --- | --- |
| up: `x @ w1.T` | `w1: (d_ff, d_model)` | `(..., d_ff)` |
| activation | none | `(..., d_ff)` |
| down: `... @ w2.T` | `w2: (d_model, d_ff)` | `(..., d_model)` |

The output has the same shape as the input, which is what lets the block add it back onto the residual stream.

Traditionally $d_{\text{ff}} = 4 \cdot d_{\text{model}}$.
The wide middle is where the capacity is: it is a large set of "features" that each look for some pattern in the token's vector, and the activation decides which ones fire.

From topic 06, `@` treats every axis except the last two as batch axes, so the same code works for `x` of shape `(N, d_model)` and `(B, T, d_model)` alike.
No reshaping should appear in an FFN.

**SwiGLU**, used by PaLM, LLaMA and almost every model since 2022, and the convention throughout this repository:

```math
\mathrm{FFN}(x) = W_2 \left( \mathrm{SiLU}(W_1 x) \odot (W_3 x) \right)
```

There are now **two** up-projections, and they are combined with an elementwise multiplication:

| Name | Weight shape | Role |
| --- | --- | --- |
| `w1` | `(d_ff, d_model)` | the **gate** branch: up-projection, then SiLU |
| `w3` | `(d_ff, d_model)` | the **value** branch: up-projection, no activation |
| `w2` | `(d_model, d_ff)` | the down-projection, applied to the product of the two |

Read the elementwise product as a gate.
The value branch proposes `d_ff` numbers, and the gate branch decides, separately for each of them, how much to let through: near 0 blocks it, around 1 passes it, and SiLU's negative dip can even flip its sign.
In the classic FFN the activation's decision and the value are the same number; here they are learned separately, which gives the network more freedom per parameter.

**Why it works is not known.**
The paper that introduced SwiGLU for transformers (Shazeer 2020) is two pages long.
It tries eight gated variants, reports that the gated ones beat the classic FFN, and ends by saying it offers no explanation and attributes the success to divine benevolence.
That is an honest summary of the evidence even today: gating was adopted because the improvement replicated, not because anyone has a mechanism.

**The names are confusing, and they are fixed.**
`w1` is the gated branch, `w3` the plain one, and `w2` the way down.
The numbering is inherited from LLaMA's reference code and from CS336, and the tests check the `state_dict` keys, so they are not a matter of taste.
They are also easy to mix up, so when reading someone else's implementation, check which branch gets the activation rather than assuming.

The SiLU goes on the `w1` branch **only**.
Applying it to both branches, or to the product, gives a different function that trains worse, and nothing crashes.

**No biases.**
Modern FFNs have no bias vectors anywhere.

**The module.**
`SwiGLUFFN` is a topic 08 module holding three weights named `w1`, `w2`, `w3`.
`nn.Linear(d_in, d_out, bias=False)` stores its weight with shape `(d_out, d_in)`, the same convention as above, and its `state_dict` key becomes `w1.weight` when it is assigned to `self.w1`.
A plain `nn.Parameter` assigned to `self.w1` gives the key `w1`; the test accepts either.
Either way, the module should hold exactly $3 \cdot d_{\text{model}} \cdot d_{\text{ff}}$ numbers.

### 11.3 The arithmetic

**Parameters.**
Each FFN matrix has $d_{\text{model}} \cdot d_{\text{ff}}$ entries:

```math
\begin{aligned}
&\text{classic:} && \text{2 matrices} && \to 2 \cdot d_{\text{model}} \cdot d_{\text{ff}} \\
&\text{gated:} && \text{3 matrices} && \to 3 \cdot d_{\text{model}} \cdot d_{\text{ff}}
\end{aligned}
```

**Where $8/3$ comes from.**
To compare SwiGLU fairly with the classic FFN, give it the same number of parameters.
It has three matrices instead of two, so its `d_ff` must shrink by a factor of $2/3$:

```math
\begin{aligned}
3 \cdot d_{\text{model}} \cdot d_{\text{ff\_new}} &= 2 \cdot d_{\text{model}} \cdot (4 \cdot d_{\text{model}}) \\
d_{\text{ff\_new}} &= (8 / 3) \cdot d_{\text{model}}
\end{aligned}
```

That is the origin of the strange $8/3$ in every LLaMA-style config.
It is not a tuned hyperparameter; it is the number that keeps the parameter count the same.

**Rounding up to a multiple.**
$8/3 \cdot d_{\text{model}}$ is usually not a whole number: for `d_model = 512` it is `1365.33`.
And GPUs compute matrix multiplications in fixed-size tiles, so a dimension that is not a multiple of the tile size wastes part of the last tile.
So `d_ff` is rounded **up** to a multiple of 64, or of 128 or 256.

Up, not to the nearest, so the model is never smaller than the target.

| `d_model` | $8/3 \cdot d_{\text{model}}$ | up to a multiple of 64 | up to a multiple of 256 |
| --- | --- | --- | --- |
| 512 | 1365.33 | 1408 | 1536 |
| 768 | 2048 | 2048 | 2048 |
| 4096 | 10922.67 | 10944 | **11008** |

And there is LLaMA-7B's `11008`: $8/3$ of 4096, rounded up to a multiple of 256.
A config number that looks arbitrary usually is a rounding of something simple.

**Rounding up with integers.**
Python's `//` rounds **down**: `8 * 512 // 3` is `1365`, and rounding a number down never gets you to "at least".
To round $a$ up to a multiple of $m$, round $a / m$ up to a whole number, then multiply by $m$ again.
For whole numbers, the ceiling of $a / m$ has an exact integer spelling, `(a + m - 1) // m`, or `-(-a // m)`, which avoids floats entirely.
`math.ceil` on a float works too, but think about where the division by 3 happens: `8 * d_model` is a whole number, and dividing it by 3 is where the fraction appears.
Note that with `multiple_of=1` the answer is still rounded up: $8/3 \cdot 64 = 170.67$ becomes `171`.

The function returns a Python `int`.

**FLOPs.**
Topic 06's rule: multiplying an `(m, k)` matrix by a `(k, n)` one costs $2 \cdot m \cdot n \cdot k$ floating-point operations, one multiply and one add per term.
With `N` tokens flattened into the rows, each FFN matrix multiplication is between `(N, d_model)` and `(d_model, d_ff)`, or the reverse for the way down, and all of them cost the same:

```math
\begin{array}{lll}
\text{one matmul:} & & 2 \cdot N \cdot d_{\text{model}} \cdot d_{\text{ff}} \\
\text{classic:} & \text{2 of them} & \to 4 \cdot N \cdot d_{\text{model}} \cdot d_{\text{ff}} \\
\text{gated:} & \text{3 of them} & \to 6 \cdot N \cdot d_{\text{model}} \cdot d_{\text{ff}}
\end{array}
```

The activation and the gating product touch each of the `N * d_ff` hidden numbers only a few times, which is tiny next to the matmuls, so nobody counts them.

At the same `d_ff`, the gated FFN costs 50% more.
At the `8/3` sizing it costs the same as the classic one with $4 \cdot d_{\text{model}}$, because the smaller `d_ff` exactly compensates.
That is the whole point of $8/3$: same parameters, same compute, better model.

**Where the parameters live.**
With $d_{\text{ff}} = 4 \cdot d_{\text{model}}$ the FFN holds $8 \cdot d_{\text{model}}^2$ numbers, while attention's four `d_model x d_model` projections (topic 13) hold $4 \cdot d_{\text{model}}^2$.
So the FFN is two thirds of every block.
That is why making the FFN cheaper is where the leverage is, and why Mixture-of-Experts (topic 19) replaces exactly this part of the block with several FFNs and a router.

## API you need

### Activations and their pieces

| Call | Result |
| --- | --- |
| `torch.maximum(a, b)` | elementwise larger of two tensors |
| `x.clamp(min=0)` | values below 0 replaced by 0 |
| `torch.zeros_like(x)` | zeros of the same shape and dtype |
| `torch.erf(x)` | the error function, for the exact GELU |
| `torch.tanh(x)` | the hyperbolic tangent, for the approximation |
| `torch.sigmoid(x)` / `x.sigmoid()` | $1 / (1 + e^{-x})$ |
| `x.pow(3)` / `x ** 3` | elementwise cube |
| `math.sqrt(2.0)`, `math.sqrt(2.0 / math.pi)`, `math.pi` | the constants in the GELU formulas, as Python floats |
| `F.relu(x)`, `F.gelu(x, approximate="none")`, `F.gelu(x, approximate="tanh")`, `F.silu(x)` | PyTorch's versions, for checking yours |
| `nn.ReLU()`, `nn.GELU()`, `nn.SiLU()` | the same as modules |

### Building the block

| Call | Result |
| --- | --- |
| `x @ w.T` / `F.linear(x, w)` | a linear layer with an `(out, in)` weight |
| `nn.Linear(d_in, d_out, bias=False)` | the module; its weight is `(d_out, d_in)` |
| `a * b` | the elementwise gate |
| `p.numel()` | number of entries in one tensor |

### Arithmetic

| Call | Result |
| --- | --- |
| `a // b` | integer division, rounds **down** |
| `(a + b - 1) // b`, `-(-a // b)` | integer division rounding **up**, exact |
| `math.ceil(f)` | smallest whole number `>= f`, as an `int` |

### A worked example on neutral data

The library versions, to compare your own against:

```python
>>> x = torch.tensor([-1.0, 0.0, 1.0])
>>> F.relu(x)
tensor([0., 0., 1.])
>>> F.gelu(x)
tensor([-0.1587,  0.0000,  0.8413])
>>> F.gelu(x, approximate="tanh")
tensor([-0.1588,  0.0000,  0.8412])
>>> F.silu(x)
tensor([-0.2689,  0.0000,  0.7311])
>>> torch.sigmoid(x)
tensor([0.2689, 0.5000, 0.7311])
>>> torch.erf(torch.tensor([0.0, 1.0, 2.0]))
tensor([0.0000, 0.8427, 0.9953])

>>> t = torch.tensor([-1.0, 0.0, 1.0], requires_grad=True)
>>> F.relu(t).sum().backward()
>>> t.grad                                  # ReLU's derivative, 0 at x = 0 by convention
tensor([0., 0., 1.])
```

## Traps

- `F.gelu`'s default is the exact form, while GPT-2 checkpoints were trained with `approximate="tanh"`.
  Mixing them changes the numbers slightly, without an error.
- In the tanh formula the cube is `x ** 3`, and `0.044715` multiplies only the cube, inside the parentheses.
- The weights are stored `(out, in)`, so every multiplication is against `w.T`.
- In SwiGLU the SiLU goes on the `w1` branch only, not on `w3` and not on the product.
- `w1` is the gate, `w3` the value and `w2` the down-projection; the names do not follow the order they are used in.
- No biases anywhere in a modern FFN.
- `8 * d_model // 3` rounds down; `d_ff` must be rounded **up** to the multiple.
- A gated FFN has three matrices: counting two is off by a third, in parameters and in FLOPs.
- The activation and the gate are not counted in FLOPs; only the matmuls are.
- SiLU's derivative is negative for inputs below about `-1.28`; a hand-derived gradient that is never negative is wrong.
- ReLU's derivative at exactly 0 is a convention, 0 in PyTorch, not a mathematical fact.

## Ready when

Without looking anything up you can:

- [ ] explain why two linear layers without an activation between them are no better than one
- [ ] say what ReLU, GELU and SiLU do with a large positive, a large negative and a small negative input
- [ ] write the exact GELU, say what $\Phi$ is, and say why the tanh approximation still exists
- [ ] derive SiLU's derivative from the product rule and $s' = s(1 - s)$, and say where it is negative
- [ ] write the classic FFN and SwiGLU in the `(out, in)` convention, with the activation on the right branch
- [ ] name what `w1`, `w2` and `w3` each do
- [ ] derive $8/3$ from equal parameter counts, and get from `d_model = 4096` to `11008`
- [ ] round a number up to a multiple using only integer arithmetic
- [ ] compute an FFN's parameters and forward FLOPs for given `N`, `d_model` and `d_ff`
- [ ] say what fraction of a block's parameters the FFN holds, and why that matters for Mixture-of-Experts

## Going deeper

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), Shazeer 2020, two pages and worth all of them.
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), Hendrycks and Gimpel, for where the $x \cdot \Phi(x)$ formulation comes from.
- [Searching for Activation Functions](https://arxiv.org/abs/1710.05941), Ramachandran et al., the automated search that found swish.
- [LLaMA](https://arxiv.org/abs/2302.13971), Touvron et al. 2023, section 2.2, for SwiGLU with the $2/3 \cdot 4d$ sizing in a real model.
