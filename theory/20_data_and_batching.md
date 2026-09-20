# Topic 20 - Data and checkpoints

> **Exercise** `test_20_data_and_batching.py` · **Reference** `solutions/sol_20_data_and_batching.py` · **Assumes** topics 01-19

## Where this fits

The two ends of the pipeline that are not part of the model itself: what feeds token ids in at one end, and what writes the trained weights to disk at the other.

This is the plumbing without which training never starts, and it is where two of the least glamorous and most expensive bugs live: an off-by-one in the window bounds, and a checkpoint that cannot be resumed from.

None of it is conceptually hard.
All of it is the kind of thing that costs a day when it is wrong, and these functions are exactly the ones CS336 asks for.

What you need from earlier topics: the input/target shift (05), masked reductions and `ignore_index` (04, 10), `state_dict` and `BytesIO` (09), the optimizer state and the step-indexed schedule (16).

## What you build here

Eight functions.
Every one of them is short, and at least three have an off-by-one waiting in them.

### 20.1 Cutting batches out of a token stream

| Component | In -> out | What it must do |
| --- | --- | --- |
| `get_batch(data, batch_size, context_length, generator=None)` | 1-D array, ints -> two `(B, T)` int64 | Random windows, `targets` shifted one token forward. Start index drawn from `[0, len(data) - context_length - 1]` |
| `max_start_index(num_tokens, context_length)` | ints -> `int` | The largest admissible start, **inclusive** |
| `steps_per_epoch(num_tokens, batch_size, context_length)` | ints -> `int` | Steps to see every token once, rounding down |
| `sequential_windows(data, context_length)` | array, int -> `(num_windows, T)` int64 | Non-overlapping, deterministic, tail discarded. This is how validation is cut |

### 20.2 Padding and labels

| Component | In -> out | What it must do |
| --- | --- | --- |
| `collate_with_padding(sequences, pad_id)` | `list[list[int]]`, int -> `((B, T_max)` int64, `(B, T_max)` bool) | Pad to the longest, mask True on real tokens |
| `make_labels(ids, mask, ignore_index=-100)` | `(B, T)`, `(B, T)` -> `(B, T)` | Shift one token left; padding positions and **the last position of every row** become `ignore_index` |

### 20.3 Checkpoints

| Component | In -> out | What it must do |
| --- | --- | --- |
| `save_checkpoint(model, optimizer, iteration, out)` | module, optimizer, int, buffer -> `None` | Model state, optimizer state and the iteration number |
| `load_checkpoint(src, model, optimizer)` | buffer, module, optimizer -> `int` | Load both states **in place**, return the iteration |

## The mental model

### The shape of LM training data

There is no dataset of examples.
There is **one long array of token ids** on disk, produced by running the tokenizer over the whole corpus and concatenating, with a document separator between texts.
Training examples are windows cut out of that stream at random offsets.

Three consequences, and they are unlike most machine learning:

- An "epoch" is a fiction; you sample windows until the token budget is spent.
- Examples overlap and cross document boundaries, which is fine at scale and is what document masking in topic 19 addresses when it is not.
- The array is stored as `uint16` when the vocabulary fits in 65536 ids, halving the disk footprint.
  It must be cast to `int64` before it becomes token ids for PyTorch, since indexing and `cross_entropy` want `int64`.

**`np.memmap`** is how the array is read.
It maps the file into virtual memory and the OS pages in only the parts actually touched, so a 200 GB corpus works on a machine with 32 GB of RAM and no explicit streaming logic.

`torch.from_numpy` shares memory with the NumPy array, which for a memmap slice is usually not what you want, so materialize the slice before it goes into a batch.
It also refuses `uint16`, which is another reason the cast happens on the way in.

### 20.1 Cutting batches

**The off-by-one.**
A window must yield both inputs and targets, so it needs `context_length + 1` tokens: `data[i : i + context_length]` as the input and `data[i + 1 : i + context_length + 1]` as the target.

Therefore the largest valid start index is

```
max_start = num_tokens - context_length - 1
```

inclusive.
Check it against the degenerate case: with `num_tokens = context_length + 1` there is exactly one valid window, at index 0, and the formula gives 0.
Sampling from `[0, max_start]` inclusive is `rng.integers(0, max_start + 1)`, and NumPy's `integers` has an **exclusive** high bound, which is exactly where the off-by-one usually enters.

**Determinism.**
Pass a `np.random.Generator` in rather than using the global NumPy random state, so tests are reproducible and two workers do not draw the same windows.
`conftest.py` reseeds `torch`, `numpy` and `random` before each test, so a function using the global state would pass and still be the wrong design.

**Steps per epoch** is `num_tokens // (batch_size * context_length)`, rounding down and discarding the incomplete final batch.

**Sequential windows** for validation are non-overlapping and deterministic: truncate the stream to a multiple of `context_length`, then reshape to `(num_windows, context_length)`.
Random windows are right for training, where you want decorrelated batches, and wrong for evaluation, where you want a fixed, repeatable measurement that is comparable across runs.

### 20.2 Padding and labels

Fixed-length windows from a stream need no padding, which is why pretraining mostly avoids the topic entirely.
Fine-tuning on variable-length examples does not.

**Collating** takes a list of variable-length sequences, finds `T_max`, and produces `(B, T_max)` ids padded with `pad_id` plus a `(B, T_max)` bool mask that is True on real tokens.
That mask is what every masked reduction in topic 04 and every key padding mask in topics 03 and 14 consumes.

Padding on the right is the convention for training.
Padding on the **left** is the convention for batched generation, so that every sequence's last real token sits at the same position and one `logits[:, -1]` works for the whole batch.

**Labels** are the shift-by-one from topic 05, plus two masking rules:

- Positions that are padding get `ignore_index`.
- The **last position of every row** gets `ignore_index`, because there is no next token to predict.

`-100` is the convention, and `F.cross_entropy` excludes those positions from both the numerator and the denominator, per topic 10.

The second rule is the one people forget.
Without it the last real token is trained to predict either padding or garbage, which injects a small constant amount of nonsense into every batch.

### 20.3 Checkpoints

A checkpoint that only holds the model weights is not resumable.
It must hold three things:

```python
{
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "iteration": step,
}
```

**Why the optimizer state.**
Adam's `m` and `v` are exponential moving averages built up over thousands of steps, plus the step counter `t` that drives the bias correction.
Resuming without them restarts every moment at zero, so the bias correction restarts too and the first steps after the resume are effectively at a much smaller learning rate, producing a visible bump in the loss curve.

**Why the iteration.**
The learning rate schedule is a function of the step number, per topic 16, so a resume that restarts the counter at zero re-runs the warmup and then decays from a point the model has already passed.

Save with `torch.save(obj, path_or_buffer)`, load with `torch.load`.
For an `io.BytesIO`, **`buf.seek(0)` before loading** is mandatory, since `torch.save` leaves the position at the end.

`load_state_dict` on both the model and the optimizer works **in place**, so `load_checkpoint` mutates the objects it was given and returns only the iteration number.
That is the right design: the caller has already constructed the model and the optimizer, and only their contents need replacing.

## API you need

### Data

| Call | Result |
| --- | --- |
| `np.memmap(path, dtype=np.uint16, mode="r")` | lazy on-disk array |
| `np.random.default_rng(seed)` | a `Generator` |
| `rng.integers(low, high, size)` | random ints, `high` exclusive |
| `data[i : i + n]` | a NumPy slice |
| `np.stack(list_of_arrays)` | build the batch |
| `torch.from_numpy(arr)` | shares memory, refuses `uint16` |
| `torch.tensor(arr, dtype=torch.int64)` | copies and casts |
| `arr.astype(np.int64)` | cast on the NumPy side |
| `x.reshape(-1, context_length)` | non-overlapping windows |
| `x.unfold(0, size, step)` | overlapping windows, topic 02 |

### Padding

| Call | Result |
| --- | --- |
| `max(len(s) for s in seqs)` | `T_max` |
| `torch.full((B, T), pad_id, dtype=torch.int64)` | the padded buffer |
| `F.pad(t, (0, n), value=pad_id)` | pad one sequence on the right |
| `torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=p)` | the library version |
| `ids[:, 1:]`, slice assignment | the label shift |

### Checkpoints

| Call | Result |
| --- | --- |
| `torch.save(obj, buf)` | serialize |
| `buf.seek(0)` | rewind before reading |
| `torch.load(buf, weights_only=True)` | deserialize |
| `model.load_state_dict(sd)` | in place |
| `optimizer.load_state_dict(sd)` | in place, restores `m`, `v` and `t` |
| `io.BytesIO()` | an in-memory file |

### A worked example on neutral data

```python
>>> data = np.arange(10, dtype=np.uint16)
>>> ctx = 4
>>> max_start = len(data) - ctx - 1          # 5, inclusive
>>> i = 5
>>> data[i : i + ctx], data[i + 1 : i + ctx + 1]
(array([5, 6, 7, 8], dtype=uint16), array([6, 7, 8, 9], dtype=uint16))
```

Start index 5 is valid and reads exactly to the end; start index 6 would run off the array.
That is the whole off-by-one, and writing it out on a length-10 array once is faster than reasoning about it.

## Traps

- `max_start = num_tokens - context_length - 1`, inclusive.
  Forgetting the `- 1` reads one token past the end on the target slice.
- NumPy's `rng.integers(low, high)` has an exclusive `high`; sampling inclusive of `max_start` needs `max_start + 1`.
- `torch.from_numpy` does not accept `uint16`.
  Cast first.
- `torch.from_numpy` shares memory with the NumPy array, including a memmap.
- Token ids must be `int64` for indexing and for `cross_entropy`.
- Using the global NumPy random state instead of a passed-in generator makes batches non-reproducible.
- Forgetting to set the last position of each row to `ignore_index`.
- Padding on the right for batched generation, where left padding is what you want.
- Saving the model but not the optimizer, which costs a visible loss bump on every resume.
- Saving the optimizer but not the iteration, which replays the learning rate schedule from the start.
- `torch.load` on a `BytesIO` without `seek(0)`.
- Random windows for validation, which makes evaluation numbers noisy and incomparable between runs.

## Ready when

Without looking anything up you can:

- [ ] state the largest valid window start index and verify it on a length-10 array
- [ ] explain why pretraining data is one flat token array rather than a list of examples
- [ ] say what `np.memmap` buys and why `uint16` is used on disk
- [ ] build a padded batch with its mask, and construct the labels with both `ignore_index` rules
- [ ] say why generation pads on the left
- [ ] list the three things a resumable checkpoint must contain, and what breaks if each is missing
- [ ] explain why random windows are right for training and wrong for validation

## Context



**Why the optimizer state dict is fragile.**
It references parameters by **index into `param_groups`**, not by name, so it can only be loaded into an optimizer constructed over the same parameters in the same order.
Change the architecture, reorder the parameter groups, or construct the optimizer before the model is fully built, and the load either raises or silently applies the wrong moments to the wrong tensors.
This is why the conventional order is: build the model, build the optimizer over its parameters, then load both states.

**Why `torch.load` got stricter.**
A checkpoint is a pickle and unpickling executes arbitrary code, so recent PyTorch defaults `torch.load` to `weights_only=True`, which accepts tensors, plain containers and primitives and refuses everything else.
A checkpoint of the shape above loads fine; one that pickled a whole optimizer object or a custom config class will need an explicit opt-in.

**What a real training data pipeline adds.**
On top of this: shuffling at the document level before concatenation, so that the stream is not ordered by source; a separate held-out shard that the training stream never touches; and, increasingly, packing with document masks from topic 19 rather than letting windows straddle boundaries.
Distributed training also needs each rank to draw different windows, which is usually done by seeding the generator with the rank, and checkpointing the data loader's position so a resume does not replay the same tokens.

## Going deeper

- [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).
- [Saving and loading a general checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) in the PyTorch recipes.
- The nanoGPT `get_batch`, which is about eight lines and is the canonical version of everything in section 20.1.
