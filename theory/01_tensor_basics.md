# Topic 01 - Tensor basics

> **Exercise** `test_01_tensor_basics.py`·

A tensor is three things bolted together: a flat block of memory, a dtype that says how to read that memory, and a shape that says how to index into it.
Almost every confusing bug in an LLM codebase is one of those three quietly not being what you assumed.
A model that trains in float32 when you meant bfloat16 costs twice the memory.
An in-place write into a view corrupts a tensor you thought you had copied.
An integer division you did not notice turns a learning rate into zero.

This topic is where checking those three becomes reflex.
Topic 02 then starts moving shapes around, and everything after it assumes you can tell a copy from a view without thinking.

## The mental model

### A tensor is a view over storage

```
x = torch.arange(6).reshape(2, 3)

storage (flat, 6 int64 values):   [0][1][2][3][4][5]
shape:                            (2, 3)
strides:                          (3, 1)      # how far to step per axis
x[1, 2]  ->  storage[1*3 + 2]  =  storage[5]
```

Two tensors can point at the same storage with different shapes and strides.
That is what makes `reshape`, `transpose`, slicing and `expand` free: no data moves, only the metadata changes.
It is also why writing into one of them changes the other.

"Are these the same memory?" is answered by comparing `a.untyped_storage().data_ptr()` with `b.untyped_storage().data_ptr()`.
`x.data_ptr()` is a different thing: it is the address of element zero of *this* view, so a slice starting partway into the storage has a different `data_ptr()` while sharing the same storage.

### 1.1 Creating tensors

Every constructor takes `dtype=` and `device=`, and both have defaults that will surprise you eventually.

```
torch.arange(5)          int64
torch.arange(5.0)        float32
torch.tensor([1, 2])     int64
torch.tensor([1.0, 2.0]) float32
```

The `*_like` family copies dtype, device and layout from an existing tensor, which is the habit worth building.
`torch.zeros_like(x)` never gets the dtype wrong; `torch.zeros(x.shape)` frequently does.
There is no `torch.eye_like`, so an identity matching some `x` has to be assembled from `x.shape[0]`, `x.dtype` and `x.device` by hand.

### 1.2 Metadata

| Attribute | Type | Meaning |
| --- | --- | --- |
| `x.shape` | `torch.Size`, a `tuple` subclass | size along each axis |
| `x.ndim` | `int` | number of axes, same as `len(x.shape)` |
| `x.numel()` | `int` | total element count, the product of the shape |
| `x.dtype` | `torch.dtype` | how the bytes are interpreted |
| `x.device` | `torch.device` | where the bytes live |
| `x.element_size()` | `int` | bytes per element |
| `x.requires_grad` | `bool` | whether autograd tracks it |

The memory of the data itself is `x.numel() * x.element_size()`.
That product is the whole of `model_memory_bytes`, and the arithmetic behind every "will this fit on the GPU" question.

| dtype | bytes | notes |
| --- | --- | --- |
| `torch.float32` | 4 | the default, around 7 decimal digits |
| `torch.float16` | 2 | 10 mantissa bits, max around 65504, overflows easily |
| `torch.bfloat16` | 2 | 7 mantissa bits, the same exponent range as float32, the LLM default |
| `torch.float64` | 8 | almost never used in deep learning |
| `torch.int64` | 8 | the default integer, and what token ids are |
| `torch.int32` | 4 | |
| `torch.bool` | 1 | masks |

### 1.3 Arithmetic and type promotion

Mixing dtypes in one expression triggers promotion, by rules close to NumPy's.
int64 with float32 gives float32, float32 with float64 gives float64, anything with bool gives the other type.
You can ask for the answer without doing the work: `torch.result_type(a, b)` takes tensors or scalars, `torch.promote_types(dt1, dt2)` takes two dtypes.

Some reductions refuse integers outright.
`torch.mean` on an int64 tensor raises, because the mean of integers is not an integer and PyTorch will not guess.
The fix is to cast before the reduction, not after: casting after has already thrown the information away.

Casting itself is `x.to(torch.float32)`, with `x.float()` and `x.long()` as shorthands.
`.to()` is a no-op when the dtype already matches and returns the very same object rather than a copy, which is a property `to_float32` is built on.

### 1.4 Copies, views and in-place

Three operations that sound similar and are not:

| Call | New storage? | In the autograd graph? |
| --- | --- | --- |
| `x.clone()` | yes | yes, gradients flow back to `x` |
| `x.detach()` | no, shares storage | no, cuts the graph |
| `x.detach().clone()` | yes | no |
| `x.contiguous()` | only if `x` was not already contiguous | yes |

An in-place operation is spelled with a trailing underscore: `add_`, `mul_`, `zero_`, `clamp_`, `copy_`.
It writes into the existing storage and returns the same tensor object, so both `id(x)` and `x.data_ptr()` survive unchanged.
Optimizer steps rely on exactly that: `p.data.add_(grad, alpha=-lr)` updates the parameter without reallocating it, which is why the optimizer's reference to `p` stays valid.

Assignment into a slice, `x[i] = 0` or `x[:, 2:] = value`, is also an in-place write and also preserves the storage.
Rebinding, `x = x + 1`, is not: it allocates a new tensor and points the local name at it, leaving the original untouched.

### 1.5 Small things LLM code needs constantly

Counting elements that satisfy a condition is a comparison followed by a sum.
A comparison yields a bool tensor, and summing a bool tensor gives an int64 count.
`.item()` pulls a zero-dimensional tensor out into a Python number, and is the only way to get a real `int` or `float` rather than a tensor.

`torch.clamp(x, min=lo, max=hi)` saturates in both directions, which is how you keep values inside a range before dividing by their spread.

## API you need

### Construction

| Call | Result |
| --- | --- |
| `torch.tensor(data, dtype=..., device=...)` | copies Python data into a new tensor |
| `torch.arange(start, end, step, dtype=...)` | `[start, end)` with the given step |
| `torch.linspace(a, b, steps)` | `steps` evenly spaced points including both ends |
| `torch.zeros(*shape)` / `torch.ones(*shape)` | filled with 0 / 1 |
| `torch.full(shape, fill_value, dtype=...)` | filled with a constant, shape passed as a tuple |
| `torch.empty(*shape)` | uninitialized, contains whatever was in memory |
| `torch.eye(n, m=None, dtype=..., device=...)` | identity, `m` defaults to `n` |
| `torch.randn(*shape)` / `torch.rand(*shape)` | standard normal / uniform on `[0, 1)` |
| `torch.randint(low, high, shape)` | random integers, `high` exclusive |
| `torch.zeros_like(x)`, `torch.ones_like(x)`, `torch.full_like(x, v)`, `torch.empty_like(x)`, `torch.randn_like(x)` | same shape, dtype and device as `x` |

Note the shape convention: `torch.zeros(2, 3)` takes varargs, `torch.full((2, 3), 7.0)` takes a tuple.

### Metadata and dtype

| Call | Result |
| --- | --- |
| `x.shape`, `x.ndim`, `x.numel()`, `x.dtype`, `x.device` | see the table above |
| `x.element_size()` | bytes per element |
| `x.stride()` | tuple of strides, in elements not bytes |
| `x.is_contiguous()` | whether the strides are the row-major ones for this shape |
| `torch.result_type(a, b)` | the dtype `a op b` would produce |
| `torch.promote_types(dt1, dt2)` | the same, for two dtypes |
| `x.to(dtype)`, `x.float()`, `x.long()`, `x.bool()` | cast |
| `x.type_as(other)` | cast to `other.dtype` |
| `torch.finfo(dtype)` / `torch.iinfo(dtype)` | `.min`, `.max`, `.eps`, `.tiny`, `.bits` |

### Arithmetic

| Call | Result |
| --- | --- |
| `a + b`, `a * b`, `a / b`, `a ** n` | elementwise, with broadcasting |
| `torch.add(a, b, alpha=k)` | `a + k * b`, one fused call |
| `x.sum()`, `x.mean()`, `x.max()` | reductions over everything when no `dim` is given |
| `torch.clamp(x, min=..., max=...)` | saturate, either bound optional |
| `torch.where(cond, a, b)` | elementwise select |
| `x.item()` | a 0-d tensor as a Python scalar |

### Copies and memory

| Call | Result |
| --- | --- |
| `x.clone()` | a copy with its own storage, still differentiable |
| `x.detach()` | the same storage, cut out of the autograd graph |
| `x.contiguous()` | a row-major copy if needed, otherwise `x` itself |
| `x.untyped_storage().data_ptr()` | the address of the shared storage |
| `x.data_ptr()` | the address of element zero of this view |
| `x.add_(v)`, `x.mul_(v)`, `x.zero_()`, `x.clamp_(lo, hi)`, `x.copy_(src)` | in-place, returns `x` |
| `x[i] = v`, `x[:, j] = v` | in-place slice assignment |

### A worked example on neutral data

```python
>>> x = torch.arange(4, dtype=torch.float32)
>>> y = x[1:3]                       # a view
>>> y.add_(100)                      # in-place into the view
>>> x
tensor([  0., 101., 102.,   3.])     # x changed too
>>> x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
True
>>> z = x.clone()
>>> z.zero_()
>>> x
tensor([  0., 101., 102.,   3.])     # untouched this time
```

## Traps

- `torch.mean` on an integer tensor raises `RuntimeError: mean(): could not infer output dtype`.
  Cast to float first.
- Integer division: `torch.arange(5) / 2` promotes to float32, but `//` on int64 stays integer and truncates.
- `torch.full((2, 2), 1)` gives int64, `torch.full((2, 2), 1.0)` gives float32.
  When a specific dtype is required, pass `dtype=` rather than hoping the fill value implies it.
- Summing a bool tensor gives int64, not bool.
  `mask.sum()` is a count, `mask.any()` and `mask.all()` are the boolean questions.
- `x.T` on a tensor with more than two dimensions is deprecated and warns.
  Use `x.transpose(-2, -1)` or `x.mT`.
- `torch.eye_like` does not exist.
- `.item()` on anything with more than one element raises.
- A tensor created inside a function with `torch.zeros(...)` lands on CPU regardless of where its neighbours live.
  Use the `*_like` form or pass `device=`.

## Ready when

Without looking anything up you can:

- [ ] say what dtype `torch.arange(5) + 0.5` has, and why
- [ ] compute how many bytes a `(4096, 4096)` bfloat16 tensor occupies
- [ ] name three ways to get a tensor that does *not* share memory with another
- [ ] explain why `x = x + 1` and `x += 1` differ for a caller holding a reference to `x`
- [ ] check whether two tensors alias the same storage

## Context

**Why bfloat16 won.**
float16 and bfloat16 both take two bytes, and they spend those bytes differently.
float16 keeps more mantissa bits and a small exponent range, so it is more precise but overflows above roughly 65504, which happens constantly in the middle of a large model and forces loss scaling to work around it.
bfloat16 keeps float32's exponent range and gives up mantissa bits instead, so it almost never overflows and is simply less precise.
Training turned out to tolerate the lost precision far better than the overflows, which is why every large model today trains in bfloat16.

**Why the memory arithmetic matters early.**
A 7B parameter model in bfloat16 is `7e9 * 2` bytes, so 14 GB of weights before a single activation exists.
Training it also needs gradients, usually in the same dtype, plus optimizer state, which for Adam is two more numbers per parameter in float32.
That is where the rule of thumb "training needs roughly ten to twenty times the weights in memory" comes from, and topic 16 does the arithmetic properly.

## Going deeper

- [PyTorch internals, by Edward Yang](https://blog.ezyang.com/2019/05/pytorch-internals/), the canonical explanation of storage, strides and views.
- [Tensor attributes](https://pytorch.org/docs/stable/tensor_attributes.html) in the docs, which contains the full type promotion table.
