# Topic 12 - Activations and the FFN

> **Exercise** `test_12_activations_and_ffn.py`

The feed-forward branch, the second of the two sublayers in every transformer block.

Attention is where tokens read each other.
The feed-forward block is where each token is transformed on its own, and it is where two thirds of the model's parameters and essentially all of its nonlinearity live.

The block has changed exactly twice in eight years: ReLU became GELU, and GELU became gated SwiGLU.
Each change came with an arithmetic adjustment to `d_ff` that keeps the parameter count constant, and understanding that adjustment is the difference between copying a config and knowing what it says.

This file also drills deriving an activation's gradient by hand, which is the warm-up for topic 19.

What you need from topic 06: the `(out, in)` weight convention, and `2mnk` per matmul.

## The mental model

### 12.1 The activations

**ReLU**: `max(x, 0)`.
Cheap, sparse, and dead below zero: an input that spends its life negative gets exactly zero gradient forever.

**GELU**, the Gaussian Error Linear Unit: `x * Phi(x)`, where `Phi` is the standard normal CDF.
The intuition is a smooth, stochastic gate: instead of a hard on/off at zero, the input is scaled by the probability that a standard normal is below it.

```
gelu(x) = x * 0.5 * (1 + erf(x / sqrt(2)))
```

The **tanh approximation**, which is what GPT-2 shipped:

```
gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```

The two agree to a few parts in a thousand.
`F.gelu(x, approximate="tanh")` selects it, and `approximate="none"` is the default exact form.

**SiLU**, also called swish: `x * sigmoid(x)`.
Smooth like GELU, almost indistinguishable in practice, and cheaper because `sigmoid` is cheaper than `erf`.
It is the activation inside SwiGLU and therefore the one in LLaMA and its descendants.

**The derivative of SiLU** is worth deriving rather than memorizing.
With `s = sigmoid(x)` and the standard fact `s' = s(1 - s)`:

```
d/dx [x * s]  =  s + x * s * (1 - s)  =  s * (1 + x * (1 - s))
```

Note what this says: the derivative is not bounded in `[0, 1]` like a sigmoid's, and it goes slightly negative for moderately negative `x`.
That non-monotonicity is shared by GELU and absent from ReLU.

All three activations are elementwise, so their gradients are elementwise too and there is nothing to contract.

### 12.2 The FFN blocks

**The classic FFN**, 2017 through GPT-3:

```
FFN(x) = W2 @ act(W1 @ x)
```

In PyTorch's `(out, in)` weight convention with `x` of `(..., d_model)`:

```
w1: (d_ff, d_model)      h   = act(x @ w1.T)      (..., d_ff)
w2: (d_model, d_ff)      out = h @ w2.T           (..., d_model)
```

Two matrices, and `d_ff = 4 * d_model` by convention.
The block projects up into a wider space, applies the nonlinearity there, and projects back.
The width is where the capacity is; the nonlinearity is what makes the whole thing more than a single linear map.

**SwiGLU**, PaLM and LLaMA onwards, and the convention used throughout this repository:

```
FFN(x) = W2 @ (SiLU(W1 @ x) * (W3 @ x))
```

that is,

```
w1: (d_ff, d_model)      gate  = silu(x @ w1.T)       (..., d_ff)
w3: (d_ff, d_model)      value = x @ w3.T             (..., d_ff)
w2: (d_model, d_ff)      out   = (gate * value) @ w2.T
```

The elementwise product is the "gating": one branch decides, multiplicatively, how much of the other branch to let through.

The naming matters.
`w1` is the gate branch, the one that gets the SiLU, `w3` is the linear branch, and `w2` is the down projection.
That ordering is inherited from CS336 and from LLaMA's reference code, and tests assert on it.
It is also an objectively confusing set of names, so check rather than assume when reading somebody else's implementation.

There are no biases anywhere in a modern FFN.

### 12.3 The arithmetic

**Parameters.**

```
ungated:  2 * d_model * d_ff
gated:    3 * d_model * d_ff
```

**Why `8/3`.**
SwiGLU has three matrices where the classic FFN has two, so keeping the parameter count fixed requires shrinking `d_ff` by a factor of `2/3`:

```
3 * d_model * d_ff_new  =  2 * d_model * (4 * d_model)
d_ff_new = (8/3) * d_model
```

That is where the otherwise bizarre `8/3` in every LLaMA config comes from.
It is not a tuned hyperparameter; it is the number that makes the comparison against the old FFN fair.

In practice the result is **rounded up to a multiple of 64**, sometimes 128 or 256.
GPU tensor cores operate on fixed tile sizes, and a matmul dimension that is not a multiple of the tile size wastes a fraction of every tile.
Rounding *up* rather than to the nearest keeps the model at least as large as the target.

**FLOPs.**
Using `2 * m * n * k` per matmul from topic 06, with `N` tokens:

```
ungated:  2*N*d_model*d_ff  +  2*N*d_ff*d_model  =  4 * N * d_model * d_ff
gated:    three matmuls of the same size          =  6 * N * d_model * d_ff
```

The elementwise activation and the gating product are `O(N * d_ff)` and vanish next to the matmuls, which is why nobody counts them.

A gated FFN at the same `d_ff` costs 50% more FLOPs, but at the `8/3` sizing it costs the same as the ungated one, because the smaller `d_ff` exactly compensates.
That is the whole point of the `8/3`.

**Where the parameters live.**
With `d_ff = 4 * d_model`, the FFN holds `8 * d_model^2` and attention holds `4 * d_model^2`, so the FFN is two thirds of every block.
This is why FFN efficiency, and Mixture-of-Experts in topic 19, is where the leverage is.

## API you need

### Activations

| Call | Result |
| --- | --- |
| `F.relu(x)` / `x.relu()` / `x.clamp(min=0)` | ReLU |
| `F.gelu(x, approximate="none")` | exact GELU |
| `F.gelu(x, approximate="tanh")` | the GPT-2 approximation |
| `F.silu(x)` | `x * sigmoid(x)` |
| `torch.sigmoid(x)` / `x.sigmoid()` | the logistic function |
| `torch.erf(x)` | the error function, for the exact GELU |
| `torch.tanh(x)` | for the approximation |
| `nn.ReLU()`, `nn.GELU()`, `nn.SiLU()` | the module forms |

`math.sqrt(2.0)`, `math.sqrt(2.0 / math.pi)` and the constant `0.044715` are the numbers in the two GELU formulas.

### Building the block

| Call | Result |
| --- | --- |
| `F.linear(x, w)` | `x @ w.T`, the `(out, in)` convention |
| `nn.Linear(d_in, d_out, bias=False)` | the module, no bias |
| `a * b` | the elementwise gate |
| `-(-a // b) * b`, or `((a + b - 1) // b) * b` | round `a` up to a multiple of `b` |

Integer ceiling division in Python is `-(-a // b)` or `(a + b - 1) // b`; both are exact and neither involves a float.

### A worked example on neutral data

```python
>>> x = torch.tensor([-2.0, 0.0, 2.0])
>>> F.gelu(x)
tensor([-0.0455,  0.0000,  1.9545])
>>> x * 0.5 * (1 + torch.erf(x / math.sqrt(2)))
tensor([-0.0455,  0.0000,  1.9545])
>>> F.silu(x)
tensor([-0.2384,  0.0000,  1.7616])

>>> s = torch.sigmoid(x)
>>> s * (1 + x * (1 - s))                      # the SiLU derivative
tensor([-0.0908,  0.5000,  1.0908])
```

## Traps

- `F.gelu`'s default is the exact form, but GPT-2 checkpoints were trained with `approximate="tanh"`.
  Mixing them changes the numbers slightly.
- The weight convention is `(out, in)`, so every multiplication is against `w.T`.
- In SwiGLU the SiLU goes on the `w1` branch only.
  Applying it to both branches, or to the product, is a different and worse function.
- `d_ff = 8/3 * d_model` must be rounded up to a hardware-friendly multiple, and the rounding is *up*, not nearest.
- Integer division truncates: `8 * d_model // 3` is not the same as rounding up.
- A gated FFN has three matrices, so counting two is off by a third.
- No biases in a modern FFN.
- ReLU's derivative at exactly 0 is a convention, PyTorch picks 0, not a mathematical fact; it comes up in gradient checks.

## Ready when

Without looking anything up you can:

- [ ] write the exact GELU and say what `Phi` is
- [ ] write SiLU and derive its gradient from `s' = s(1 - s)`
- [ ] write the SwiGLU FFN with the right branch getting the activation, in the `(out, in)` convention
- [ ] derive `8/3` from the requirement that a gated FFN match an ungated one's parameter count
- [ ] compute FFN parameters and forward FLOPs for a given `d_model` and `d_ff`
- [ ] say what fraction of a transformer block's parameters the FFN holds

## Context



**Nobody knows why gating helps.**
The paper that introduced the transformer GLU variants (Shazeer 2020) is two pages long, tries eight combinations, reports that the gated ones win, and closes with a line to the effect that it offers no explanation and attributes the success to divine benevolence.
That is not a joke at the paper's expense: it is an honest description of the state of the evidence, and the variants were adopted because the numbers replicated, not because anyone has a mechanism.

**Why the tanh approximation outlived its reason.**
`erf` was slow on 2018 hardware, so GPT-2 shipped an approximation.
Hardware caught up, but checkpoints trained under the approximation behave slightly differently under the exact form, and the difference is large enough to shift a benchmark and small enough that nobody wants to retrain.
So the approximation survives as a compatibility flag, and `gelu_new` in HuggingFace means "the old one".

**Where LLaMA's 11008 comes from.**
For `d_model = 4096`, `8/3 * 4096` is 10922.67.
Round up to a multiple of 64 and you get 10944, which is not what LLaMA uses.
Its actual rule multiplies by `2/3`, rounds to a multiple of 256, and lands on 11008.
The lesson is not the specific number but that these constants are all rounding artifacts of the same `8/3`, and a config that looks arbitrary usually is not.

## Going deeper

- [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202), Shazeer 2020, two pages and worth all of them.
- [Gaussian Error Linear Units](https://arxiv.org/abs/1606.08415), Hendrycks and Gimpel, for where the CDF formulation comes from.
