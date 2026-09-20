# Topic 13 - Embeddings and RoPE

> **Exercise** `test_13_embeddings_and_rope.py`

The first thing the model does: turn token ids into the vectors everything else operates on, plus the positional information attention needs in order to know what order the tokens came in.

Attention is a set operation.
Shuffle the tokens and, with no positional information, the output is shuffled identically and nothing else changes: the model literally cannot tell "dog bites man" from "man bites dog".
Positional encoding is the entire fix, and how it is done has changed three times, with RoPE winning decisively.

RoPE is also the piece of a modern transformer that people most often implement almost right, because the convention for which channels get paired differs between codebases and a wrong pairing still trains.

## The mental model

### 13.1 Embeddings

**Token embeddings** are a `(V, D)` lookup table, indexed as in topic 05.
Nothing more.

**The `sqrt(d_model)` scaling** in the original transformer multiplies the looked-up embeddings by `sqrt(D)`.
The embedding table is initialized with variance around `1/D`, giving vectors of norm around 1, while the sinusoidal encodings added to them have entries in `[-1, 1]` and norm around `sqrt(D/2)`.
Without the scaling the positional signal would swamp the token signal.
Models using RoPE add nothing to the embeddings and generally do not need this scaling.

**Learned absolute positions**, the GPT-2 approach, are a second table `(max_len, D)` indexed by position and added to the token embeddings.
It works, it costs `max_len * D` parameters, and it fails completely past `max_len`, since there is no row to look up.

The **offset** argument matters during generation.
At decoding step `t` you feed a single token, but its position is `t`, not 0.
Any positional scheme therefore has to accept "where in the sequence does this chunk start", and forgetting to thread that offset through is the classic bug that makes cached generation disagree with a full forward pass.
It comes back in topic 17.

**Sinusoidal encodings**, 2017, are computed rather than learned:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/D))
PE[pos, 2i+1] = cos(pos / 10000^(2i/D))
```

Channel pair `i` is a point moving around a circle with its own frequency.
The frequencies form a geometric series from 1 down to `1/10000`, so the first pair completes a rotation every few tokens and the last takes tens of thousands of tokens.
Together they encode position the way the hands of a clock encode time: periodic readings at different scales that jointly identify the position without repeating.

`PE[0]` is `[sin(0), cos(0), sin(0), cos(0), ...]`, so `[0, 1, 0, 1, ...]`, which is a useful thing to check an implementation against.

### 13.2 RoPE

RoPE does not add anything.
It **rotates** the query and key vectors by an angle proportional to their position.

Split a head vector of even length `Dh` into `Dh/2` pairs of adjacent channels.
**In this repository, and in CS336, the pairs are `(x0, x1), (x2, x3), ...`**, adjacent channels, the "interleaved" convention.
Pair `i` gets its own frequency:

```
inv_freq[i] = theta^(-2i / Dh)          for i = 0 .. Dh/2 - 1,  theta = 10000
```

The first pair has frequency 1 and spins fastest; the last is close to `1/theta` and barely moves.
Same geometric spread as the sinusoids.

At position `pos` the angle for pair `i` is `pos * inv_freq[i]`, and the pair is rotated by the standard 2-D rotation:

```
out_even = x_even * cos - x_odd * sin
out_odd  = x_even * sin + x_odd * cos
```

Cache `cos` and `sin` as `(T, Dh/2)` tensors, computed once, and apply them by slicing the even and odd channels with `x[..., 0::2]` and `x[..., 1::2]`.
Written that way the function works for `(T, Dh)` and `(B, H, T, Dh)` alike, because the cache broadcasts against the leading axes.

Two things RoPE is not applied to.
Not `v`: only `q` and `k` are rotated, because the point is to shape the *dot product*, and `v` is only ever summed with weights.
And not the embeddings: RoPE lives inside attention, applied per layer, per head, right before the score computation.

**A convention warning.**
The other common convention splits the vector into halves and pairs `x[i]` with `x[i + Dh/2]`.
HuggingFace's `rotate_half` uses that one, and so do checkpoints trained with it.
The two are related by a permutation of the channels, so a model trained under one and run under the other produces garbage.
When porting weights, this is the first thing to check.

### 13.3 The property that makes RoPE work

Rotate `q` by angle `pos_q * w` and `k` by angle `pos_k * w`, and take their dot product.
For a single pair, writing the pair as a complex number, the rotation is multiplication by `exp(i * pos * w)`, and the inner product involves the conjugate:

```
<R(pos_q) q, R(pos_k) k>  =  Re( q * conj(k) * exp(i * (pos_q - pos_k) * w) )
```

The absolute positions appear **only through their difference**.

Three consequences, and they are what made RoPE the standard:

- The model gets **relative** position information with no relative-position parameters, no `(T, T)` bias table, and zero extra parameters of any kind.
- Shifting an entire sequence by a constant offset leaves all attention scores unchanged, so the model is translation invariant by construction.
- Positions beyond the training length are still well-defined, because the formula is just arithmetic.

**A rotation preserves norms.**
`|R(pos) x| == |x|` exactly, for every position.
That is the cheapest possible test of whether your rotation is actually a rotation: if the norm changes you have a shear or a scaling, which usually means the even/odd pairing or a sign is wrong.
Checking that `cos^2 + sin^2 == 1` across the cache tests the same thing from the other side.

### 13.4 The module

A `RotaryEmbedding` module precomputes the `cos` and `sin` caches up to `max_len` and stores them as **buffers**, not parameters: they are constants, they must travel to the GPU with the model, and they are not trained.

`forward(x, offset)` slices the caches at `[offset : offset + T]` and applies the rotation.
That slice is the offset plumbing from earlier, and getting it right is what makes a KV cache produce the same numbers as a full forward pass.

Whether the buffers should be `persistent=False` is a real design question from topic 09: they are fully recomputable from `d_head`, `max_len` and `theta`, so keeping them out of the checkpoint is defensible and is what several production implementations do.

## API you need

| Call | Result |
| --- | --- |
| `table[ids]` | the embedding lookup |
| `math.sqrt(d_model)` | the 2017 scaling |
| `torch.arange(offset, offset + T)` | positions with an offset |
| `torch.arange(0, d_head, 2)` | the even channel indices, for building `inv_freq` |
| `theta ** (-arange / d_head)` | the inverse frequencies |
| `pos[:, None] * inv_freq[None, :]` | the outer product giving `(T, Dh/2)` angles |
| `torch.outer(pos, inv_freq)` | the same, spelled as a function |
| `angles.cos()`, `angles.sin()` | the caches |
| `x[..., 0::2]`, `x[..., 1::2]` | the even and odd channels, as views |
| `torch.stack((a, b), dim=-1).flatten(-2)` | interleave two half-width tensors back together |
| `torch.empty_like(x)` plus slice assignment | the other way to interleave |
| `self.register_buffer("cos_cached", t, persistent=False)` | the module storage |
| `torch.linalg.vector_norm(x, dim=-1)` | the norm check |

Building `inv_freq` with `dtype=torch.float32` explicitly matters: `torch.arange(0, d_head, 2)` is int64, and raising a float to an integer-tensor power gives results you did not intend in some PyTorch versions.
Cast, then exponentiate.

### A worked example on neutral data

```python
>>> d_head, T = 8, 4
>>> inv_freq = 10000.0 ** (-torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)
>>> inv_freq
tensor([1.0000e+00, 1.0000e-01, 1.0000e-02, 1.0000e-03])
>>> angles = torch.outer(torch.arange(T, dtype=torch.float32), inv_freq)
>>> angles.shape
torch.Size([4, 4])
>>> cos, sin = angles.cos(), angles.sin()
>>> cos[0], sin[0]                       # position 0 is the identity rotation
(tensor([1., 1., 1., 1.]), tensor([0., 0., 0., 0.]))
>>> (cos**2 + sin**2).allclose(torch.ones(T, d_head // 2))
True
```

## Traps

- The pairing convention.
  Adjacent pairs `(0,1), (2,3)` here; HuggingFace's `rotate_half` pairs `i` with `i + Dh/2`.
  A model is not portable across the two.
- RoPE goes on `q` and `k` only, never on `v`.
- RoPE is applied inside every attention layer, not once to the embeddings.
- The position offset must be threaded through, or cached generation diverges from a full forward pass.
- `inv_freq` must **decrease**, so a positive exponent somewhere is a sign error that shows up as the last channels spinning fastest.
- `cos` and `sin` are `(T, Dh/2)`, half the head width, because each pair shares one angle.
  Building them at `(T, Dh)` is a different convention and will not match these tests.
- The caches are buffers, not parameters.
  As parameters they would be trained, and as plain attributes they would stay on CPU.
- `d_head` must be even.
- If the norm of a rotated vector changes, the rotation is wrong.
  Check that before debugging anything downstream.

## Ready when

Without looking anything up you can:

- [ ] say why attention without positional information is permutation equivariant
- [ ] write the sinusoidal formula and say what `PE[0]` looks like
- [ ] write `inv_freq` for RoPE and say which end of it spins fastest
- [ ] write the 2-D rotation applied to a channel pair, with the signs in the right places
- [ ] explain in one sentence why the rotated dot product depends only on `i - j`
- [ ] name two numerical invariants that catch a wrong RoPE implementation
- [ ] say why the caches are buffers and why `forward` needs an offset

## Context



**How a 4k model gets a 128k context.**
RoPE extrapolates partially, not fully: run a model far past its training length and the fast-spinning pairs have wrapped around many times into angles the model never saw, and quality collapses.
The fixes all work on the frequencies rather than on the weights.
Position interpolation divides the positions by a factor so that 32k tokens occupy the angular range 4k used to.
NTK-aware scaling raises `theta` instead, which stretches the slow pairs more than the fast ones.
YaRN combines both with a per-frequency ramp and a temperature correction on the attention.
All three need only a short fine-tune, or sometimes none, which is why context windows grew so fast in 2023 without anyone retraining from scratch.

**Why RoPE beat the alternatives.**
Learned absolute positions cost parameters and stop at `max_len`.
Sinusoidal ones are cheap but the model has to infer relative distance from the sum of two absolute encodings, which it does imperfectly.
ALiBi from topic 03 gives relative distance directly but only as a monotone decay, the same shape for every pair of tokens.
RoPE gives full relative position, costs nothing, and leaves the model free to learn whatever function of distance it wants per head.
The one real cost is that it has to be applied inside every attention layer rather than once at the input.

## Going deeper

- [RoFormer](https://arxiv.org/abs/2104.09864), Su et al., the RoPE paper, whose section 3.4 is the relative-position proof.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.5, for the original sinusoids.
- [YaRN](https://arxiv.org/abs/2309.00071) for what people do to RoPE when they want a longer context than they trained on.
