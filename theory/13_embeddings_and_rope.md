# Topic 13 - Embeddings and RoPE

> **Exercise** `test_13_embeddings_and_rope.py`

## Why this topic

A language model receives text as a row of integers, the token ids.
The very first thing it does is turn each id into a vector of `D` numbers, because everything after that, normalization, attention, the feed-forward network, works on vectors.
That step is the **embedding**, and it is one line of indexing you already wrote in topic 05.

The harder half of this topic is **position**.
Attention, the heart of the model (topic 14), compares every token with every other token, and on its own it has no idea which token came first.
To attention, "dog bites man" and "man bites dog" contain the same three tokens and look exactly alike.
Something has to tell the model where each token sits.

There have been three main answers to that, and you will build all three:

- **sinusoidal positions** (2017): a fixed pattern of sines and cosines added to each token's vector;
- **learned positions** (GPT-2): a second lookup table, one row per position, added the same way;
- **RoPE**, rotary position embedding (LLaMA, Qwen, Mistral, CS336): instead of adding anything, rotate parts of the vectors inside attention by an angle that grows with the position.

RoPE won, decisively, and by the end of this file you will know why: after the rotation, the comparison between two tokens depends only on **how far apart** they are, not on where they are.
It costs zero parameters and keeps working on positions the model never saw in training.

It is also the part of a modern transformer that people most often implement almost right.
There are two conventions for which numbers get rotated together, and a model trained with one and run with the other still produces output, just garbage.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| token id | an integer in `[0, V)` standing for one piece of text |
| embedding table | a `(V, D)` tensor, one learned row per token id |
| position | the index of a token in its sequence: 0, 1, 2, ... |
| query, key | the two vectors attention computes for every token and compares with a dot product (topic 14) |
| head, `Dh` | attention works in several independent "heads"; each one sees a slice of width `Dh`, so `q` and `k` have shape `(B, H, T, Dh)` |
| offset | the position at which the current chunk of tokens starts |

### Why attention needs to be told about order

You will build attention in topic 14, but its core is simple enough to describe now.
Every token produces a query vector and a key vector.
Token `i` decides how much to pay attention to token `j` by taking the dot product of its query with `j`'s key: a large dot product means "relevant to me".

Notice that nothing in that recipe mentions positions.
If you shuffle the tokens, every query and key is shuffled along with them, every dot product is still computed between the same pairs, and the output comes out shuffled the same way, otherwise unchanged.
So position has to be put into the numbers somehow, either into the token vectors before attention, or into the queries and keys themselves.

### A refresher on sine, cosine and rotation

Picture a point on a circle of radius 1, starting at the rightmost point and moving counter-clockwise.
After it has turned through an angle $a$ (measured in radians, where a full turn is $2\pi \approx 6.28$), its coordinates are $(\cos a, \sin a)$.
That is all sine and cosine are: the two coordinates of a point going around a circle.

Three facts follow from that picture:

- $\cos 0 = 1$ and $\sin 0 = 0$: no turn, still at the start.
- $\cos^2 a + \sin^2 a = 1$ for every $a$: the point is always on the circle.
- Both repeat every full turn: $\sin(a + 2\pi) = \sin a$.

**Rotating a 2-D vector.**
To turn a vector $(x, y)$ by an angle $a$ around the origin, keeping its length:

```math
\begin{aligned}
x_{\text{new}} &= x \cos a - y \sin a \\
y_{\text{new}} &= x \sin a + y \cos a
\end{aligned}
```

Check it on $(1, 0)$ turned by a quarter turn, $a = \pi / 2$: $\cos = 0$, $\sin = 1$, so the result is $(0, 1)$, pointing straight up.

**A rotation keeps lengths, and keeps angles between vectors.**
Turn two vectors by the same angle and the angle between them does not change, nor do their lengths.
And the dot product of two vectors is $\lvert q \rvert \cdot \lvert k \rvert \cdot \cos(\text{angle between them})$, built from exactly those things.
So rotating both vectors by the same angle leaves their dot product unchanged.
This one observation is the whole reason RoPE works, and 13.3 comes back to it.

### 13.1 Embeddings

**Token embeddings.**
An embedding table is a `(V, D)` tensor with one row per token id, and looking up ids is indexing, exactly as in topic 05: ids of shape `(B, T)` give vectors of shape `(B, T, D)`.
The rows start out random and are learned like any other weight, so that tokens used in similar ways end up with similar vectors.

**The $\sqrt{D}$ scaling.**
The original 2017 transformer multiplied every looked-up embedding by $\sqrt{D}$, where `D` is the width of the table, its last axis.
The reason was a size mismatch.
The embedding table was initialized with small numbers, giving vectors whose length is around 1.
The sinusoidal position vectors added to them have entries between -1 and 1, so their length is around $\sqrt{D / 2}$, much bigger for a wide model.
Without the scaling, the position signal would drown out the token signal.
Models that use RoPE add nothing to the embeddings, so they generally skip this.

**Learned absolute positions.**
GPT-2's approach is a second table of shape `(max_len, D)`: row `p` is a learned vector meaning "I am at position `p`".
For a sequence of `T` tokens you take rows `0` to `T - 1` and add them to the token embeddings.
That slice is `(T, D)`, and it broadcasts against `(B, T, D)` across the batch (topic 03): every sequence in the batch gets the same position vectors.

It works, it costs `max_len * D` parameters, and it simply cannot go beyond `max_len`: there is no row for position `max_len` to look up.

**The offset.**
When a model generates text, it produces one token at a time.
At step 100, it feeds in a single token, but that token is at **position 100**, not position 0.
So every position scheme needs to be told where the current chunk starts, the offset, and use positions `offset, offset + 1, ..., offset + T - 1`.
For a learned table that means taking a slice that starts at the offset instead of at 0.

Forgetting to pass the offset along is the classic bug that makes step-by-step generation disagree with running the whole sequence at once.
Topic 17 depends on getting it right.

**Sinusoidal positions.**
The 2017 transformer computed its position vectors instead of learning them:

```math
\begin{aligned}
\text{PE}[\text{pos}, 2i] &= \sin\left(\text{pos} / 10000^{2i/D}\right) \\
\text{PE}[\text{pos}, 2i+1] &= \cos\left(\text{pos} / 10000^{2i/D}\right)
\end{aligned}
```

Read it in pairs of channels.
Channel pair $i$ (channels $2i$ and $2i + 1$) holds the $(\sin, \cos)$ coordinates of a point going around a circle, at angle $\text{pos} \cdot f_i$, where $f_i = 10000^{-2i/D}$ is that pair's own speed, or frequency.
Each pair is a clock hand turning at a different speed as the position increases.

The speeds form a **geometric series**: each pair is slower than the one before by the same factor.
For `D = 8` the four frequencies are `1, 0.1, 0.01, 0.001`.
The first pair makes a full turn every $2\pi \approx 6.3$ positions; the last one needs about `6283`.
Fast hands tell nearby positions apart, slow hands tell distant ones apart, and together they identify every position without repeating, like the second, minute and hour hands of a clock.

The $2i$ in the exponent is the index of the **even** channel of the pair: `0, 2, 4, ...`.
`torch.arange(0, D, 2)` produces exactly those.

Two sanity checks.
At position 0 every angle is 0, so `PE[0]` is `[sin 0, cos 0, sin 0, cos 0, ...] = [0, 1, 0, 1, ...]`.
And the last pair turns so slowly that at position 1 its cosine is still almost exactly 1.

Writing a result into "every other channel" is slice assignment with a step: `pe[:, 0::2] = ...` fills the even channels and `pe[:, 1::2] = ...` the odd ones.

### 13.2 RoPE

**The idea.**
RoPE does not add a position vector to anything.
Inside attention, right before queries and keys are compared, it **rotates** each query and key by an angle that depends on its position.
By the property from the refresher, the comparison between two rotated vectors then depends only on the **difference** of their angles, that is, on how far apart the two tokens are.

A rotation is a 2-D thing, so the head vector of width `Dh` (which must be even) is cut into `Dh / 2` pairs, and each pair is rotated as a little 2-D vector on its own.

**Which channels form a pair.**
In this repository, and in CS336, the pairs are **adjacent** channels: $(x_0, x_1), (x_2, x_3), (x_4, x_5), \dots$.
This is called the interleaved convention.

The other common convention cuts the vector into a first and a second half and pairs `x[i]` with `x[i + Dh/2]`.
HuggingFace's `rotate_half` does that, and so do checkpoints trained with it.
The two are the same idea with the channels in a different order, so a model trained under one and run under the other produces garbage without any error.
When moving weights between codebases, check this first.

**Frequencies.**
Each pair gets its own speed, with the same geometric spread as the sinusoids:

```math
\text{inv\_freq}[i] = \theta^{-2i / d_{\text{head}}} \qquad \text{for } i = 0, 1, \dots, d_{\text{head}}/2 - 1, \quad \theta = 10000
```

Pair 0 has speed 1 and turns fastest, the last pair has a speed close to $1 / \theta$ and barely moves.
They must **decrease** along the pairs; if the last pair spins fastest, a sign is wrong.
The name `inv_freq`, inverse frequency, is historical: it is the number you multiply the position by to get the angle.

Build it in floating point from the start.
`torch.arange` gives `int64` by default, and an integer power of an integer stays an integer, silently:

```python
>>> i = torch.arange(0, 8, 2)                 # int64
>>> 10000 ** (-i)                              # integer arithmetic, rounded to 0
tensor([1, 0, 0, 0])
>>> 10000.0 ** (-i / 8)                        # true division makes it float
tensor([1.0000, 0.1000, 0.0100, 0.0010])
```

Passing `dtype=torch.float32` to `arange`, and dividing with `/` rather than `//`, keeps you out of that.

**The cache.**
At position `pos`, pair `i` is rotated by the angle `pos * inv_freq[i]`.
For `T` positions and `Dh / 2` pairs that is a `(T, Dh/2)` table of angles, every position times every frequency, an **outer product**.
`torch.outer(a, b)`, or `a[:, None] * b[None, :]` by broadcasting (topic 03), builds it.
With an offset, the positions are `offset` to `offset + T - 1` instead of `0` to `T - 1`.

Then take the cosine and the sine of every angle.
These two `(T, Dh/2)` tables are all the rotation needs, and they depend only on the positions and the frequencies, never on the data, so a real model computes them once and reuses them.
They are **half** the head width, because both channels of a pair share one angle.
At position 0 the cosines are all 1 and the sines all 0: no rotation at all.

**Applying the rotation.**
For every pair, call its first channel "even" and its second "odd", and apply the 2-D rotation from the refresher, with the pair's own angle:

```math
\begin{aligned}
\text{out}_{\text{even}} &= x_{\text{even}} \cdot \cos - x_{\text{odd}} \cdot \sin \\
\text{out}_{\text{odd}} &= x_{\text{even}} \cdot \sin + x_{\text{odd}} \cdot \cos
\end{aligned}
```

All pairs at all positions can be done at once.
`x[..., 0::2]` is every even channel and `x[..., 1::2]` every odd one, each of shape `(..., T, Dh/2)`, and both are views (topic 02):

```python
>>> v = torch.arange(6.0)
>>> v[0::2], v[1::2]
(tensor([0., 2., 4.]), tensor([1., 3., 5.]))
```

The `(T, Dh/2)` cache broadcasts against `(..., T, Dh/2)` from the right, so the same few lines work for an input of shape `(T, Dh)` and of shape `(B, H, T, Dh)`.

That leaves putting the two rotated halves back in their **interleaved** places: new even channels at `0, 2, 4, ...`, new odd ones at `1, 3, 5, ...`.
Concatenating them would put all evens first, which is the other convention.
Two ways to interleave, shown on neutral data:

```python
>>> a = torch.tensor([1., 2.])        # goes to the even slots
>>> b = torch.tensor([3., 4.])        # goes to the odd slots
>>> torch.stack((a, b), dim=-1)       # (2, 2): pairs side by side
tensor([[1., 3.],
        [2., 4.]])
>>> torch.stack((a, b), dim=-1).flatten(-2)
tensor([1., 3., 2., 4.])
>>> torch.cat((a, b))                 # not interleaved
tensor([1., 2., 3., 4.])
```

The other way is to make an empty tensor shaped like the input with `torch.empty_like` and fill its even and odd slots by slice assignment, as `PE` was filled above.

Do not compute `out_odd` from an `x_even` you have already overwritten: since the slices are views, writing into the input in place would change it under your feet.
Build the output as a new tensor.

**Where RoPE is applied.**
To queries and keys only, never to the values `v`: the point is to shape the dot product between `q` and `k`, and `v` never takes part in one.
And it is applied inside every attention layer, for every head, right before the comparison, not once to the embeddings at the start.

For `q` and `k` of shape `(B, H, T, Dh)`, `T` is `q.shape[-2]` and `Dh` is `q.shape[-1]`.
Both get the same cache.

**dtypes.**
The cache is float32.
If the queries are bfloat16, multiplying them by a float32 cache promotes the result to float32 (topic 01).
Real implementations convert the cache slice to the input's dtype first, with `cos.to(x.dtype)`.

### 13.3 The property that makes RoPE work

**Only the difference matters.**
Take one channel pair, and think of the query's pair and the key's pair as two arrows in the plane.
The query sits at position $m$, so its arrow is turned by $m w$; the key at position $n$ is turned by $n w$.

Their dot product depends only on their lengths and on the angle between them.
Rotations do not change lengths.
And the angle between them changes by $m w - n w = (m - n)\, w$: only the difference of the positions survives.

The full dot product of two head vectors is the sum of the dot products of their pairs, and every pair obeys the same argument with its own $w$.
So the whole score between a query at $m$ and a key at $n$ depends on $m - n$ and nothing else.
Positions `(5, 3)`, `(10, 8)` and `(101, 99)` all give the same score, and `(5, 0)` gives a different one.

On a single pair with $w = 0.5$:

```
q = (1, 2),  k = (3, -1)
no rotation:                       q . k = 1
both at position 1:                       1        same angle, nothing changes
q at 3, k at 1:                          -5.35
q at 10, k at 8:                         -5.35     same distance, same score
```

The original paper writes this with complex numbers, where turning by an angle is multiplying by $\exp(i \cdot \text{angle})$; the picture above is the same argument without them.

**Why that is so valuable.**

- The model gets **relative** position, which is what language mostly cares about ("the word two places back"), with no extra parameters at all.
- Shifting a whole sequence by a constant changes no score, so the model behaves the same wherever a passage sits in the document.
- Every position has a well-defined rotation, so the formula still works beyond the training length; there is no table to run out of.

Compare the alternatives.
Learned positions cost parameters and stop at `max_len`.
Sinusoidal positions are free, but the model has to work out the distance between two tokens from the sum of two absolute encodings, which it does imperfectly.
ALiBi, from topic 03, gives the distance directly but only as a fixed penalty that grows linearly with distance; heads differ only in the slope.
RoPE gives exact relative position for free and leaves each head free to learn what to do with it.
Its one cost is that it is applied inside every attention layer rather than once at the input.

**Longer contexts.**
"Still works" past the training length is only partly true.
Far beyond it, the fast pairs have turned through angle combinations the model never saw, and quality drops.
The standard fixes all change the frequencies rather than the weights.
Position interpolation divides the positions by a factor, so 32k tokens use the angles 4k used to.
NTK-aware scaling raises $\theta$, which slows the slow pairs more than the fast ones.
YaRN combines the two with a per-frequency ramp.
All three need only a short fine-tune, sometimes none, which is how context windows grew from 4k to 128k so fast in 2023.

**A rotation keeps lengths.**
$\lvert \text{rotated } x \rvert = \lvert x \rvert$ at every position.
That is the cheapest test of whether your rotation really is one: if the length changes, you have a stretch or a shear, which nearly always means the pairing or a sign is wrong.
$\cos^2 + \sin^2 = 1$ across the whole cache tests the same thing from the cache's side.
`torch.linalg.vector_norm(x, dim=-1)`, or `x.norm(dim=-1)`, gives the length.

**One vector at one position.**
The helpers in 13.3 take a single vector of shape `(Dh,)` and one position.
The rotation machinery from 13.2 expects a position axis `T`, so treat the vector as a sequence of length 1: `x[None]` gives `(1, Dh)`, a cache for one position starting at the offset `pos` is `(1, Dh/2)`, and `[0]` takes the single row back out.
The "change in norm" is a signed difference, a 0-d tensor, which should be zero up to rounding.

### 13.4 The module

A `RotaryEmbedding` module computes the `cos` and `sin` tables once, in `__init__`, for every position up to `max_len`, and stores them.
Topic 09 decides how:

- they are constants, so they must **not** be parameters: the optimizer would train them;
- they must move to the GPU with the model, so they cannot be plain attributes;
- so they are **buffers**, registered with `register_buffer` under the names `cos_cached` and `sin_cached`.

Topic 09 also raised the question of `persistent=False`: these tables can always be rebuilt from `d_head`, `max_len` and `theta`, so several production implementations keep them out of checkpoints.
In this exercise they stay **persistent**: the test expects exactly the two keys `cos_cached` and `sin_cached` in the `state_dict`, and no parameters at all.

`forward(x, offset)` takes `x` of shape `(B, H, T, Dh)`, slices rows `offset` to `offset + T - 1` out of both tables, and applies the rotation.
Basic slicing is a view (topic 02), so taking the slice costs nothing.
That slice is the offset plumbing from 13.1 again, and it is what makes generating one token at a time produce exactly the numbers a full forward pass would.

## API you need

| Call | Result |
| --- | --- |
| `table[ids]` | the embedding lookup, `ids.shape + (D,)` |
| `table.shape[-1]`, `x.shape[-2]` | the width `D`; the sequence length `T` of `(..., T, Dh)` |
| `math.sqrt(d)` | square root of a Python number |
| `t[a : a + n]` | `n` rows starting at `a`, a view |
| `torch.arange(start, end, dtype=torch.float32)` | positions as floats |
| `torch.arange(0, d, 2, dtype=torch.float32)` | the even channel indices `0, 2, 4, ...` |
| `base ** t` / `torch.pow(base, t)` | elementwise power; keep `t` float |
| `torch.outer(a, b)` / `a[:, None] * b[None, :]` | every `a_i` times every `b_j` |
| `t.cos()`, `t.sin()` / `torch.cos(t)`, `torch.sin(t)` | elementwise cosine and sine |
| `x[..., 0::2]`, `x[..., 1::2]` | even and odd channels of the last axis, as views |
| `torch.stack((a, b), dim=-1).flatten(-2)` | interleave two tensors along the last axis |
| `torch.zeros(...)`, `torch.empty_like(x)` + `t[..., 0::2] = ...` | the other way to interleave |
| `x[None]`, `out[0]` | add a leading axis of size 1, take it back off |
| `t.to(x.dtype)` | match a cache to the input's dtype |
| `self.register_buffer("name", t)` | a saved, moved, untrained tensor (topic 09) |
| `torch.linalg.vector_norm(x, dim=-1)` / `x.norm(dim=-1)` | length of each vector |
| `(a * b).sum()` / `a @ b` | the dot product of two 1-D vectors |

### A worked example on neutral data

Rotating one 2-D arrow by hand, and the length check:

```python
>>> angle = torch.tensor(math.pi / 2)              # a quarter turn
>>> x, y = 1.0, 0.0
>>> (x * angle.cos() - y * angle.sin(), x * angle.sin() + y * angle.cos())
(tensor(-4.3711e-08), tensor(1.))                  # (0, 1) up to rounding
>>> v = torch.tensor([3.0, 4.0])
>>> torch.linalg.vector_norm(v)
tensor(5.)
>>> a = torch.tensor([0.0, 1.0, 2.0])
>>> torch.outer(a, torch.tensor([1.0, 0.1]))        # every row times every column
tensor([[0.0000, 0.0000],
        [1.0000, 0.1000],
        [2.0000, 0.2000]])
```

## Traps

- The pairing convention: adjacent channels `(0, 1), (2, 3)` here; HuggingFace's `rotate_half` pairs `i` with `i + Dh/2`.
  A model is not portable between the two.
- Concatenating the rotated evens and odds instead of interleaving them is the other convention, and fails the tests.
- `cos` and `sin` are `(T, Dh/2)`, half the head width; building them at `(T, Dh)` is a different convention.
- `inv_freq` must **decrease**; a missing minus sign makes the last pair spin fastest.
- An integer `arange` raised to integer powers stays an integer and silently rounds to 0; keep it float.
- RoPE goes on `q` and `k` only, never on `v`, and inside every attention layer, not once on the embeddings.
- Forgetting the offset makes token-by-token generation disagree with a full forward pass.
- Overwriting the input in place while you still need its old even or odd channels corrupts the rotation.
- The caches are buffers, not parameters, and in this exercise they must be persistent.
- A float32 cache times a bfloat16 input promotes the output to float32.
- If a rotated vector's length changes, the rotation is wrong; check that before debugging anything downstream.
- The $\sqrt{D}$ uses the width of the table, its last axis, not the vocabulary size.
- `d_head` must be even.

## Ready when

Without looking anything up you can:

- [ ] explain why attention with no position information cannot tell "dog bites man" from "man bites dog"
- [ ] rotate a 2-D vector by an angle, with the signs in the right places, and check it on a quarter turn
- [ ] say why rotating two vectors by the same angle leaves their dot product unchanged
- [ ] add learned position vectors with an offset, and say why generation needs the offset
- [ ] write the sinusoidal formula, say what `PE[0]` looks like, and which pair turns fastest
- [ ] write `inv_freq` for RoPE in floating point, and build the `(T, Dh/2)` cos and sin tables with an offset
- [ ] apply the rotation to adjacent pairs and interleave the result back, for `(T, Dh)` and `(B, H, T, Dh)` alike
- [ ] explain in two sentences why the rotated dot product depends only on the distance between positions
- [ ] name two numerical checks that catch a wrong RoPE implementation
- [ ] say why the caches are buffers, and what `forward` does with the offset

## Going deeper

- [RoFormer](https://arxiv.org/abs/2104.09864), Su et al., the RoPE paper; section 3.4 is the relative-position proof with complex numbers.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762), section 3.5, for the original sinusoids and the $\sqrt{d_{\text{model}}}$ scaling.
- [Rotary Embeddings: A Relative Revolution](https://blog.eleuther.ai/rotary-embeddings/), EleutherAI's blog post, the friendliest walk-through of RoPE with pictures.
- [YaRN](https://arxiv.org/abs/2309.00071) for what people do to RoPE when they want a longer context than they trained on.
