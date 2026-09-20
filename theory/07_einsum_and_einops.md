# Topic 07 - einsum and einops

> **Exercise** `test_07_einsum_and_einops.py`


`transpose(-2, -1).matmul(...).permute(0, 2, 1, 3).reshape(...)` is correct code that nobody can read, including the person who wrote it last week.
`einsum` and `einops` replace it with a statement of *what the axes mean*, which turns a class of silent shape bugs into either a working line or a loud error.
CS336 uses `einops` as its primary tool, and so does every modern codebase worth reading, so the muscle memory being built here is typing `"b t (h d) -> b h t d"` without thinking.

Everything you do from topic 08 onward can be written either way.
From here on this repository uses whichever is clearer in the moment, and expects you to read both.

The exercise forbids `transpose`, `permute`, `view` and `reshape` wherever it asks for `einsum` or `einops`.
That constraint is the whole point.

## The mental model

### einsum in one sentence

Name the axes of every input, name the axes of the output, and every named axis that does **not** appear in the output is summed over.

```python
torch.einsum("ij,jk->ik", a, b)  # matmul: j is in both inputs and not in the output, so it is summed
```

Three rules follow:

1. **A letter repeated across inputs** is a contraction axis: multiplied elementwise, then summed if it is absent from the output.
2. **A letter present in the output** is kept, and if it appears in more than one input it is a batch axis: multiplied elementwise, not summed.
3. **A letter repeated within a single input** takes the diagonal of that input along those axes.

That third rule is how `"ii->"` is the trace and `"ii->i"` is the diagonal.

| Spec | Meaning |
| --- | --- |
| `"ij->ji"` | transpose |
| `"ii->i"` | the diagonal |
| `"ii->"` | the trace |
| `"ij->i"` | row sums |
| `"ij->"` | sum of everything |
| `"i,j->ij"` | outer product |
| `"i,i->"` | dot product |
| `"ij,jk->ik"` | matrix product |
| `"bij,bjk->bik"` | batched matrix product, `b` is a batch axis |
| `"ni,no->oi"` | the weight gradient of a linear layer, summing over the batch `n` |

The last one is worth staring at.
`x` is `(N, in)` and `grad_out` is `(N, out)`, the output is `(out, in)`, and `n` is absent from the output so it is summed.
That is exactly `grad_out.T @ x` from topic 06, except that here you never decided which side to transpose.
You only said which axes survive.

### 7.2 einsum where it actually earns its keep

**Attention scores.**
`q` is `(B, H, T, Dh)`, `k` is `(B, H, S, Dh)`, and you want `(B, H, T, S)` by summing over `Dh`.
That is `"bhtd,bhsd->bhts"`: `b` and `h` are batch axes present everywhere, `t` and `s` survive, `d` is summed.
No transpose is written anywhere, because the contraction is named rather than arranged.

**Attention output** is `"bhts,bhsd->bhtd"`, summing over the keys `s`.

**Per-head projections** are the case that makes the multi-head picture click.
Instead of one `(D, D)` matrix reshaped afterwards, imagine `H` separate matrices of shape `(D, Dh)` stacked into `(H, D, Dh)`.
Then `x` of `(B, T, D)` against `w` of `(H, D, Dh)` giving `(B, H, T, Dh)` is `"btd,hde->bhte"`: `d` is contracted, `h` comes only from the weight and survives, `b` and `t` come only from the input and survive.
An axis that appears in exactly one input and in the output is simply carried along, which is how einsum says "do this once per head" with no loop and no broadcast juggling.

**Bilinear forms.**
`x` of `(B, D1)`, `W` of `(D1, D2)`, `y` of `(B, D2)`, result `(B,)`, is `"bi,ij,bj->b"`.
Three operands: `b` appears in two inputs and in the output so it is a batch axis, `i` and `j` are both contracted.
einsum takes any number of operands and chooses a contraction order itself.

Two practical notes.
`...` is allowed and stands for any number of leading axes, so `"...ij,...jk->...ik"` is a matmul that works at any rank.
And einsum never materializes the outer product of all the axes; it dispatches to BLAS calls, so it is not slower than the hand-written equivalent at the shapes that matter here.

### 7.3 einops rearrange

`rearrange(tensor, "input_pattern -> output_pattern", **sizes)` does permutation, merging and splitting in one readable string.

- **Space-separated names** are the axes, in order.
- **Parentheses on the left** split one physical axis into several named ones: `(h d)` means "this axis is `h * d`, treat it as two".
- **Parentheses on the right** merge several named axes into one physical axis.
- **Keyword arguments** supply the sizes that cannot be inferred: in `(h d)` you must say `h=...` or `d=...`, and einops derives the other.

So the head split from topic 02 is one line:

```python
rearrange(x, "b t (h d) -> b h t d", h=num_heads)
```

and the merge is the same string read backwards:

```python
rearrange(x, "b h t d -> b t (h d)")
```

**Order inside the parentheses is significant.**
`(h d)` means the axis is laid out head-major: all of head 0's `d` channels, then all of head 1's.
`(d h)` means the opposite interleaving.
This is precisely the ordering question from topic 02, now explicit in the notation instead of implied by the order of `reshape` and `transpose`.
That explicitness is the reason einops prevents the bug.

Flattening tokens is `"b t d -> (b t) d"`.
Cutting a sequence into blocks is `"b (n g) d -> b n g d", g=group`, and again `(n g)` rather than `(g n)` is what makes each block a contiguous stretch of time.

A fused QKV tensor of `(B, T, 3*H*Dh)` becomes `(3, B, H, T, Dh)` with `"b t (three h d) -> three b h t d", three=3, h=num_heads`.
Reading the left pattern tells you the memory layout the projection produced: q first, then k, then v, with heads inside each and channels innermost.

`rearrange` returns a view when it can and copies when it must, exactly like `reshape`.

### 7.4 einops reduce and repeat

`reduce(tensor, "pattern -> pattern", reduction)` drops axes with a named reduction.
Any axis in the input pattern missing from the output pattern is reduced.
The reductions are `"mean"`, `"sum"`, `"max"`, `"min"`, `"prod"`, `"any"`, `"all"`.

```python
reduce(x, "b t d -> b d", "mean")     # mean-pool over time
reduce(x, "b h t d -> b t d", "max")  # max over heads
```

The advantage over `x.mean(1)` is that the axis being reduced is named rather than numbered, so the line survives a change of rank.

`repeat(tensor, "pattern -> pattern", **sizes)` adds axes or tiles existing ones.
A new letter on the right that is not on the left is a new axis, sized by keyword.

```python
repeat(mask, "b t s -> b h t s", h=num_heads)
```

For grouped-query attention the pattern makes the ordering unambiguous:

```python
repeat(kv, "b hkv t d -> b (hkv g) t d", g=num_groups)
```

`(hkv g)` puts the `g` copies of each KV head next to each other, which is the head order GQA requires.
`(g hkv)` would give the tiled order instead.
In topic 02 that distinction needed a paragraph; here it is two characters, which is the argument for einops in a nutshell.

`repeat` copies.
When you want the zero-stride view, `expand` is still the tool.

## API you need

### einsum

| Call | Result |
| --- | --- |
| `torch.einsum(spec, *operands)` | contraction per the spec |
| `"..."` in a spec | any number of leading axes |
| repeated letter across operands, absent from output | summed |
| repeated letter across operands, present in output | batch axis |
| repeated letter within one operand | diagonal |

### einops

| Call | Result |
| --- | --- |
| `rearrange(x, "a b -> b a")` | permute |
| `rearrange(x, "b (h d) -> b h d", h=H)` | split an axis |
| `rearrange(x, "b h d -> b (h d)")` | merge axes |
| `reduce(x, "b t d -> b d", "mean")` | reduce the missing axes |
| `repeat(x, "b t -> b h t", h=H)` | add or tile axes |
| `einops.pack` / `einops.unpack` | variadic concatenation, occasionally useful |
| `einops.layers.torch.Rearrange` | the same as an `nn.Module`, usable inside `nn.Sequential` |

All einops functions raise a descriptive error when the sizes do not multiply out, which is the main reason to prefer them over `reshape`.

### A worked example on neutral data

```python
>>> a = torch.randn(3, 4)
>>> torch.einsum("ij->j", a).shape          # column sums: i is summed
torch.Size([4])
>>> torch.einsum("ij,ij->i", a, a).shape    # per-row squared norm
torch.Size([3])

>>> x = torch.arange(24).reshape(1, 2, 12)  # (b=1, t=2, h*d=12)
>>> rearrange(x, "b t (h d) -> b h t d", h=3).shape
torch.Size([1, 3, 2, 4])
>>> rearrange(x, "b t (h d) -> b h t d", h=3)[0, 0, 0]
tensor([0, 1, 2, 3])                         # head 0 gets the FIRST four channels
```

That last line is the check worth running whenever you doubt an ordering: head 0 must own channels `0..Dh-1`.

## Traps

- An axis name in the output that appears in no input is an error in einsum, but a new tiled axis in einops `repeat`.
  The two libraries read the same-looking pattern differently.
- Forgetting the size keyword in a split pattern leaves einops an underdetermined system and it raises.
  That is a feature.
- `(h d)` and `(d h)` are different layouts, and only one of them matches how the projection weight was laid out.
- einsum letters are case sensitive and each letter is one axis; there is no way to name an axis with two letters.
- `reduce` with a reduction the library does not know raises; the strings are fixed.
- `rearrange` on a non-contiguous tensor will copy, silently, just like `reshape`.
- The exercise forbids `transpose`, `permute`, `view` and `reshape` in this file.
  If you find yourself reaching for one, the pattern string is what you are missing.

## Ready when

Without looking anything up you can:

- [ ] state the einsum rule in one sentence
- [ ] write attention scores and attention output as einsum specs
- [ ] write the linear weight gradient as an einsum spec and say which axis gets summed
- [ ] write the head split and merge as `rearrange` patterns, with the size keyword in the right place
- [ ] say what `(h d)` versus `(d h)` changes, and how you would detect the difference on a toy tensor
- [ ] express GQA head expansion as a single `repeat` pattern

## Context



**When einsum is slower.**
`torch.einsum` picks a contraction path heuristically, and for two operands it lowers to the same BLAS call a hand-written matmul would.
With three or more operands the path matters, and the heuristic is usually right but not always; the `opt_einsum` package exists to search harder when it is not.
For the shapes in this repository the difference is invisible, and readability wins.
The place to be careful is a hot inner loop with an unusual operand count, where it is worth timing rather than assuming.

**Why this notation spread through the field.**
Named axes are a shape type system that costs nothing.
`rearrange(x, "b t (h d) -> b h t d", h=8)` fails loudly when `x` is not divisible by 8, when the rank is wrong, or when the sizes do not multiply out, and it says which of those happened.
The equivalent `reshape` chain fails silently in two of those three cases.
That is the entire argument, and it is why the pattern strings are worth typing even where a `transpose` would be shorter.

## Going deeper

- [einops documentation and tutorial](https://einops.rocks/), especially the "writing better code with einops" page.
- [Tim Rocktäschel's einsum tutorial](https://rockt.ai/2018/04/30/einsum), the clearest short explanation of the notation.
