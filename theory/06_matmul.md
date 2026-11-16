# Topic 06 - Matrix multiplication

> **Exercise** `test_06_matmul.py`


Well over 99% of the floating point work in training an LLM happens inside `matmul`.
Normalization, activations and softmax are a rounding error by comparison, which is why the whole field reasons about cost in matmul FLOPs and nothing else.

Three things get drilled here: predicting the output shape of a batched matmul without running it, writing the forward and backward of a linear layer from memory, and doing the arithmetic that says how big a model is and what it costs to train.

## The mental model

### The shape rules of `matmul`

`torch.matmul` is one function with several behaviours depending on rank:

| `a` | `b` | result | what happens |
| --- | --- | --- | --- |
| `(n,)` | `(n,)` | `()` | dot product |
| `(m, k)` | `(k, n)` | `(m, n)` | plain matrix product |
| `(m, k)` | `(k,)` | `(m,)` | matrix times vector |
| `(k,)` | `(k, n)` | `(n,)` | vector times matrix |
| `(..., m, k)` | `(..., k, n)` | `(..., m, n)` | batched |

For the batched case the **last two axes** are the matrix, and every leading axis is a batch axis following the broadcasting rules of topic 03.
So `(B, H, T, Dh) @ (B, H, Dh, S)` gives `(B, H, T, S)`, and `(B, 1, T, Dh) @ (1, H, Dh, S)` also works, giving `(B, H, T, S)`.
The inner dimensions must match exactly: no broadcasting happens on `k`.

When a 1-D argument is involved, PyTorch temporarily prepends or appends a 1 to make it a matrix and removes that axis from the result.

`@` is the operator form and is what you should be writing.
`torch.bmm` requires exactly 3-D inputs and does no broadcasting, `torch.mm` requires exactly 2-D, and both exist mostly for historical reasons.

### 6.1 Building shapes by hand

Peel off the last two axes of each operand, check that `a[-1] == b[-2]`, broadcast the remaining leading shapes against each other, and append `(a[-2], b[-1])`.
Doing this on paper makes the attention shapes in topic 13 read themselves.

A **Gram matrix**, every pairwise dot product of the rows of `(N, D)`, is `x @ x.transpose(-2, -1)`, giving `(N, N)`.
That is the same computation as attention scores with `q == k`, which is not a coincidence: attention scores *are* a Gram matrix between the query and key projections.

### 6.2 The linear layer, forward and backward

`nn.Linear(in, out)` stores its weight as `(out, in)`, not `(in, out)`.
Its forward is `x @ weight.T + bias`, equivalently `F.linear(x, weight, bias)`.
The `(out, in)` layout exists so each output neuron's weights are one contiguous row, which suits the memory access pattern of the backward pass.
It is a constant source of transposed-weight bugs when writing layers by hand, so check `weight.shape` rather than assuming.

Leading axes need no special handling: `x` of `(B, T, in)` against `weight.T` of `(in, out)` broadcasts to `(B, T, out)`.

The backward pass is three formulas.
With $y = x W^\top$, `x` of `(N, in)`, `W` of `(out, in)`, `y` and `grad_out` of `(N, out)`:

```math
\begin{aligned}
\text{grad\_x} &= \text{grad\_out} \, W && \texttt{(N, out) @ (out, in) -> (N, in)} \\
\text{grad\_W} &= \text{grad\_out}^\top x && \texttt{(out, N) @ (N, in) -> (out, in)} \\
\text{grad\_b} &= \textstyle\sum_{n} \text{grad\_out}_{n} && \texttt{(N, out) -> (out,)}
\end{aligned}
```

The mnemonic that removes the need to remember them: **each gradient has the shape of the thing it is a gradient of**, and there is exactly one way to contract the available tensors into that shape.
`grad_x` must be `(N, in)`, and the only contraction of `grad_out` `(N, out)` with `W` `(out, in)` giving that is the plain product.
`grad_W` must be `(out, in)`, which forces the transpose.
`grad_b` must be `(out,)`, so the batch axis is summed, which is the general rule that a broadcast in the forward pass becomes a sum in the backward pass.

This is the same derivation you redo for RMSNorm in topic 19, so the habit matters more than the result.

### 6.3 Attention as two matmuls

```
scores  = q @ k.transpose(-2, -1) / sqrt(Dh)      (B,H,T,Dh) @ (B,H,Dh,S) -> (B,H,T,S)
weights = softmax(scores, dim=-1)
out     = weights @ v                             (B,H,T,S) @ (B,H,S,Dh) -> (B,H,T,Dh)
```

The scale is $1 / \sqrt{d_{\text{head}}}$, the head dimension, never `d_model`.
The reason is variance.
If `q` and `k` have unit-variance independent entries, their dot product over `Dh` terms has variance $d_{\text{head}}$, so the scores grow like $\sqrt{d_{\text{head}}}$, the softmax saturates into a near one-hot distribution and the gradients vanish.
Dividing by $\sqrt{d_{\text{head}}}$ restores unit variance regardless of head size.

The softmax is over the **last** axis, the keys, so each query's weights sum to 1 across the keys it can see.

### 6.4 The arithmetic of cost

One `(m, k) @ (k, n)` matmul does $m n k$ multiplies and $m n k$ adds, so by the standard convention it costs **$2mnk$ FLOPs**.
Everyone in the field uses this convention, including the scaling-law papers, so use it too.

**Associativity changes the cost, not the answer.**
For `A (m,k) @ B (k,n) @ C (n,p)`:

```math
\begin{aligned}
(AB)C &\quad \text{costs} \quad 2mnk + 2mpn \\
A(BC) &\quad \text{costs} \quad 2knp + 2mkp
\end{aligned}
```

This is precisely why **LoRA** computes `(x @ A) @ B` and not `x @ (A @ B)`.
With `x` of `(N, in)`, `A` of `(in, r)`, `B` of `(r, out)` and $r$ tiny, the left association costs $2 N r \cdot \text{in} + 2 N \cdot \text{out} \cdot r$, linear in $r$.
The right association first builds the full `(in, out)` matrix at a cost of $2 \cdot \text{in} \cdot r \cdot \text{out}$ and then does a full-size matmul, throwing away the entire point of the low rank.

The LoRA update is $\delta = (x A) B \cdot (\alpha / r)$.

**Parameter counts** for one pre-norm transformer block, ignoring biases and norms:

```
attention:  4 * d_model^2          q_proj, k_proj, v_proj, output_proj
FFN:        2 * d_model * d_ff     w1, w2          (ungated)
            3 * d_model * d_ff     w1, w2, w3      (gated, SwiGLU)
```

With the usual $d_{\text{ff}} = 4 d_{\text{model}}$ for an ungated FFN, a block is $12 d_{\text{model}}^2$, and a whole model is roughly $12 L d_{\text{model}}^2$ plus embeddings.

The number of heads does **not** appear in that count.
Heads are a reshape of the same `d_model` channels, so `H` changes how the projections are interpreted, not how many parameters they contain.

### 6.5 Numerics

`matmul` in low precision is not done the naive way.
Real GPU kernels read bfloat16 or float16 inputs, accumulate the running sum in **float32**, and write the result back in whatever precision was asked for.
A dot product over thousands of terms accumulates rounding error, and bfloat16 has only 7 mantissa bits, so a naive bfloat16 accumulator loses small contributions entirely once the partial sum grows.

Reproducing that by hand means upcasting the inputs to float32 before the matmul and leaving the result in float32.
The cast is what puts the accumulation in the higher precision.

## API you need

### Products

| Call | Result |
| --- | --- |
| `a @ b`, `torch.matmul(a, b)` | the general form, broadcasting leading axes |
| `torch.mm(a, b)` | 2-D only |
| `torch.bmm(a, b)` | 3-D only, no broadcasting |
| `torch.mv(a, v)` | matrix times vector |
| `torch.dot(a, b)` | 1-D only |
| `torch.outer(a, b)` | `(n,)` and `(m,)` to `(n, m)` |
| `torch.addmm(bias, a, b, beta=1, alpha=1)` | fused `beta*bias + alpha*(a@b)` |
| `torch.einsum(spec, *ops)` | everything above, spelled explicitly (topic 07) |

### Linear layers

| Call | Result |
| --- | --- |
| `F.linear(x, weight, bias=None)` | `x @ weight.T + bias`, weight is `(out, in)` |
| `nn.Linear(in, out, bias=True)` | the module; `.weight` is `(out, in)` |
| `x.transpose(-2, -1)`, `x.mT` | the transpose that works at any rank |

### Precision

| Call | Result |
| --- | --- |
| `x.to(torch.float32)` | upcast before accumulating |
| `torch.finfo(dtype).eps` | the spacing at 1.0, around 1e-7 fp32, around 8e-3 bf16 |
| `torch.backends.cuda.matmul.allow_tf32` | whether fp32 matmuls use TF32 on GPU |

### A worked example on neutral data

```python
>>> a = torch.randn(2, 3, 4, 5)
>>> b = torch.randn(2, 3, 5, 6)
>>> (a @ b).shape
torch.Size([2, 3, 4, 6])
>>> (a @ torch.randn(1, 1, 5, 6)).shape        # leading axes broadcast
torch.Size([2, 3, 4, 6])

>>> w = torch.randn(7, 5)                       # nn.Linear(5, 7) layout
>>> F.linear(torch.randn(2, 3, 5), w).shape
torch.Size([2, 3, 7])
```

## Traps

- `nn.Linear.weight` is `(out_features, in_features)`.
  Writing `x @ weight` instead of `x @ weight.T` gives a shape error when the dimensions differ and a silently wrong answer when they happen to match.
- The inner dimension never broadcasts.
  `(4, 3) @ (1, 5)` raises, it does not stretch the 1.
- `bmm` refuses anything but 3-D and does not broadcast batch axes.
- Scaling attention by `1/sqrt(d_model)` instead of `1/sqrt(d_head)` is a real bug that still trains, just worse.
- Computing LoRA as `x @ (A @ B)` gives the right numbers and destroys the efficiency.
- Counting FLOPs as $mnk$ instead of $2mnk$ puts you a factor of two off every published number.
- `matmul` on two bfloat16 tensors returns bfloat16; the float32 accumulation inside the kernel is invisible from Python, so reproducing it requires casting the inputs yourself.
- Head count does not change parameter count.

## Ready when

Without looking anything up you can:

- [ ] predict the shape of `(B, 1, T, Dh) @ (B, H, Dh, S)`
- [ ] write `grad_x`, `grad_W` and `grad_b` for a linear layer and justify each by shape alone
- [ ] say why `nn.Linear` stores `(out, in)`
- [ ] compute the FLOPs of one transformer block's attention for given `B`, `T`, `D`
- [ ] explain in one sentence why LoRA associates left
- [ ] say why the attention scale is `sqrt(d_head)`

## Context



**Why $\alpha / r$ is in the LoRA formula at all.**
The scaling exists so that changing the rank does not change the effective magnitude of the update.
Without it, doubling $r$ roughly doubles the size of the correction and you have to retune the learning rate every time you touch the rank.
With it, $\alpha$ is the knob you tune and $r$ is the capacity you choose, independently.

**TF32, and why two machines disagree.**
`torch.backends.cuda.matmul.allow_tf32` controls whether float32 matmuls on Ampere and later silently run in TF32, which has float32's range but only 10 mantissa bits.
It is on by default for convolutions and has flipped defaults for matmuls across PyTorch versions.
When the same float32 code gives slightly different numbers on two GPUs, or between CPU and GPU, this flag is the usual reason.

**Where the 12 comes from.**
$12 L d_{\text{model}}^2$ is the parameter count people quote for a transformer, and it is worth being able to rebuild: $4 d_{\text{model}}^2$ of attention plus $2 d_{\text{model}} d_{\text{ff}}$ of FFN, with $d_{\text{ff}} = 4 d_{\text{model}}$, is $4 + 8 = 12$.
A model using SwiGLU has three FFN matrices instead of two and usually shrinks $d_{\text{ff}}$ to $\frac{8}{3} d_{\text{model}}$ to keep the same total, which is why odd numbers like 11008 appear in real configs.
Topic 11 does that arithmetic.

## Going deeper

- [`torch.matmul`](https://pytorch.org/docs/stable/generated/torch.matmul.html), whose doc page enumerates all the rank cases.
- The [Transformer FLOPs](https://arxiv.org/abs/2001.08361) accounting in the Kaplan scaling-law paper, appendix B.
