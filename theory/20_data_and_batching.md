# Topic 20 - Data and checkpoints

> **Exercise** `test_20_data_and_batching.py`

## Why this topic

[Topics 15](./15_transformer_block.md) and [16](./16_training_loop.md) built a model and a loop that trains it.
Neither of them says where the token ids come from, or what happens to the weights when the run stops.
This topic is those two ends of the pipeline: what feeds `(B, T)` batches of ids in at one end, and what writes a resumable state to disk at the other.

None of it is deep, and all of it is load-bearing.
It is where two of the least glamorous and most expensive bugs live: an off-by-one in the window bounds that reads one token past the end of the data, and a checkpoint that saves the weights but cannot actually be resumed from.
Both run without an error, and both cost you a training run before anyone notices.


## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| token stream | the whole tokenized corpus as one flat 1-D array of ids |
| window | a contiguous slice of the stream; one training example |
| `context_length` | the length `T` of every input window |
| inputs, targets | two `(B, T)` tensors; the targets are the inputs shifted one token forward |
| memmap | a NumPy array backed by a file on disk, read lazily |
| `Generator` | a NumPy random number generator object with its own private state |
| padding, `pad_id` | filler ids appended so that sequences of different lengths fit one rectangular tensor |
| mask | a `(B, T)` bool tensor, `True` on real tokens and `False` on padding |
| labels | the targets for the loss, with `ignore_index` wherever there is nothing to predict |
| `ignore_index` | `-100`: a label value that `F.cross_entropy` skips (topic 10) |
| checkpoint | everything needed to continue a run: model weights, optimizer state, step number |

### The shape of LM training data

There is no dataset of examples.
There is **one long array of token ids** on disk, produced by running the tokenizer over the whole corpus and concatenating the results, with a document separator token between texts.
Training examples are windows cut out of that stream at random offsets.

Three consequences, and they are unlike most of machine learning:

- An "epoch" is mostly a fiction; you sample windows until a token budget is spent.
- Windows overlap each other and cross document boundaries, which is fine at scale; document masks in topic 19 (section 19.2) handle the case where it is not.
- Each window yields `T` training signals at once, one per position, which is why the targets are a whole shifted window rather than a single label.

**The dtype on disk.**
The array is stored as `uint16` when the vocabulary fits in 65536 ids (`0` to `65535`), which halves the footprint compared with `int32`.
GPT-2's vocabulary of 50257 fits; Llama 3's 128256 does not, and needs `int32` or `uint32`.

**`np.memmap`** is how the array is read.
`np.memmap(path, dtype=np.uint16, mode="r")` maps the file into virtual memory and returns something that behaves like an ordinary 1-D NumPy array.
The operating system reads in only the pages actually touched, so a 200 GB corpus works on a machine with 32 GB of RAM, with no explicit streaming logic.
Slicing it, or indexing it with an integer array, reads just those tokens.

**Getting from NumPy to PyTorch.**
Token ids must be `int64` by the time they reach the model: `nn.Embedding` accepts only `int64` or `int32` indices, and `F.cross_entropy` wants `int64` targets.
Recent PyTorch will wrap a `uint16` array with `torch.from_numpy` and hand you a `torch.uint16` tensor, but almost nothing accepts it afterwards, not even `+ 1`:

```python
>>> t = torch.from_numpy(np.array([3, 1, 4], dtype=np.uint16))
>>> t.dtype
torch.uint16
>>> nn.Embedding(8, 2)(t)
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; ...
```

So the cast to `int64` is part of the job, not an optimization, and the exercise asserts on it with a memmap of `uint16` tokens.
Cast on the NumPy side with `.astype(np.int64)` or on the torch side with `.long()` / `.to(torch.int64)`; either is fine as long as the result is `int64`.

Two more facts about `torch.from_numpy`.
It **shares memory** with the array rather than copying it, so a tensor made straight from a memmap slice is a view of the file.
And for a read-only memmap it warns that the array is not writable.
Both go away once the tokens have been copied into a fresh array, which `astype` and `np.stack` both do.
`torch.tensor(arr)` and `torch.as_tensor(arr, dtype=torch.int64)` are the alternatives: the first always copies, and with a `dtype` both cast.

### 20.1 Cutting batches

**The off-by-one.**
A window must yield both its inputs and its targets, so it needs `context_length + 1` tokens: `data[i : i + context_length]` is the input and `data[i + 1 : i + context_length + 1]` is the target.
The target slice ends one token later, and that last token must exist.

Therefore the largest valid start index is

```math
\text{max\_start} = \text{num\_tokens} - \text{context\_length} - 1
```

inclusive.
Check it against the degenerate case: with `num_tokens = context_length + 1` there is exactly one window, starting at 0, and the formula gives 0.
With `num_tokens = 100` and `context_length = 10`, the last window starts at 89 and its target reads index 99, the last element.

**Sampling start indices.**
The starts are drawn uniformly from `[0, max_start]`, inclusive on both ends.
NumPy's `rng.integers(low, high, size=n)` draws `n` integers from `[low, high)`, with an **exclusive** `high`, so covering `max_start` needs `high = max_start + 1`.
This is exactly where the off-by-one usually enters, in one direction or the other: too small a `high` and the last window is never drawn, too large and the target slice of the last draw runs off the end of the array.
Running off the end does not raise in NumPy: a slice past the end is just shorter, and the error surfaces later as a shape mismatch, or not at all.

**From starts to a batch.**
`B` start indices give `B` windows, each read twice (inputs and targets), and the two batches are `(B, T)` each.
Two standard ways to get there:

- slice once per start and `np.stack` the slices, which is a loop over the batch, not over tokens, and is what nanoGPT does;
- build a `(B, T)` grid of indices by broadcasting a column of starts against a row of offsets (topic 03), and index the array with it in one go (topic 05).

Either way, the targets are the same windows moved one token forward, so `targets[:, :-1]` equals `inputs[:, 1:]`, and on data that is `arange`, `targets == inputs + 1` everywhere.
`context_length = 1` must work too: then each row has one input token and one target token.

**Determinism.**
`get_batch` takes an optional `np.random.Generator` and must draw its starts from it when one is given.
Two generators made with the same seed, `np.random.default_rng(0)`, produce the same sequence of numbers, so two calls with fresh same-seed generators must return the same batch, and a different seed should give a different one:

```python
>>> np.random.default_rng(42).integers(0, 100, size=3), np.random.default_rng(42).integers(0, 100, size=3)
(array([ 8, 77, 65]), array([ 8, 77, 65]))
```

Passing the generator in rather than reaching for the global NumPy state is the right design: a caller can reproduce a run, and two data-loading workers given differently seeded generators do not draw the same windows.
When no generator is passed, the function still needs one.
`np.random.default_rng()` with no seed is a fresh generator seeded from OS entropy, so it gives a different batch on every call; the global functions such as `np.random.randint` use the shared global state, which `conftest.py` reseeds before every test.
The tests that pass `None` only check shapes and bounds, so either choice passes them.

**Steps per epoch.**
One optimizer step consumes `batch_size * context_length` tokens, so seeing every token once takes

```math
\text{steps} = \left\lfloor \frac{\text{num\_tokens}}{\text{batch\_size} \cdot \text{context\_length}} \right\rfloor
```

rounding down: an incomplete final batch is not taken.
For one million tokens with `B = 8` and `T = 1024` that is 122 steps, since 8192 tokens per step goes into 1,000,000 a little more than 122 times.
With random windows this is only an accounting figure, since nothing guarantees that 122 random batches actually cover every token; it is how a token budget is converted into a step count.

**Sequential windows for evaluation.**
Validation wants the opposite of training: a fixed, repeatable measurement that is comparable between runs, so windows are cut deterministically and **without overlap**.
The stream is truncated to a whole number of windows, `num_windows = num_tokens // context_length`, and the remaining tail is discarded; the kept prefix is then reshaped to `(num_windows, context_length)` (topic 02).
With 11 tokens and `context_length = 4` there are two windows, `[0..3]` and `[4..7]`, and the last three tokens are dropped.
The result is returned as an `int64` tensor, whatever dtype the data had on disk.

Random windows are right for training, where you want decorrelated batches and do not care about exact coverage.
They are wrong for evaluation, where a different random draw on every run turns the validation loss into a noisy number that cannot be compared from one checkpoint to the next.

### 20.2 Padding and masks

Fixed-length windows from a stream need no padding at all, which is why pretraining mostly avoids the topic.
Fine-tuning on separate examples of different lengths, a set of chat conversations for instance, does not.

**Collating** takes a list of variable-length sequences and turns it into one rectangle:

- `T_max` is the length of the longest sequence;
- `ids` is `(B, T_max)` `int64`: each sequence written into the start of its row, the rest of the row filled with `pad_id`;
- `mask` is `(B, T_max)` `bool`: `True` on the real tokens, `False` on the padding.

`torch.tensor` cannot build a tensor from a ragged list of lists, it raises `ValueError`, which is why the padding has to happen first.
Walking over the Python list of sequences is fine here; there is no tensor to vectorize over until the rectangle exists.
The mask can come from the same loop, or from the lengths alone by broadcasting a row of positions against a column of lengths:

```python
>>> lengths = torch.tensor([2, 4, 1])
>>> torch.arange(4)[None, :] < lengths[:, None]
tensor([[ True,  True, False, False],
        [ True,  True,  True,  True],
        [ True, False, False, False]])
```

Its dtype must be `torch.bool`, not `int64` ones and zeros: the test compares with `assert_close`, which checks dtypes as well as values, and every consumer of the mask expects a bool.
This mask is what the masked reductions of topic 04 and the key padding masks of topics 03 and 14 consume.

Padding on the **right**, as here, is the convention for training.
Padding on the **left** is the convention for batched generation (topic 17), so that every sequence's last real token sits in the last column and one `logits[:, -1]` serves the whole batch.

**Labels** turn padded ids into the targets for the loss.
They start from the shift-by-one of topic 05: the label at position `t` is the id at position `t + 1`, because the logits at `t` predict the next token.
Within a single padded tensor, though, there is no extra column to shift in, so the labels have the same `(B, T)` shape as the ids, and some positions get `ignore_index` instead of a token:

- **the last position of every row**, because there is no position `t + 1` to take a label from;
- **every position whose next token is padding**, because predicting a pad id is not something the model should learn.

The second rule is about the token being predicted, position `t + 1`, not the token at `t`.
On a right-padded row the two coincide except at the last real token, and that one matters: its label would be the first pad id, and it must be `ignore_index` instead.
Spelled out on one neutral row of length 6 with four real tokens:

| position | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| ids | `a` | `b` | `c` | `d` | pad | pad |
| mask | T | T | T | T | F | F |
| label | `b` | `c` | `d` | ign | ign | ign |

Position 3 is a real token, but its next token is padding, so its label is ignored; positions 4 and 5 have padding next as well; position 5 is also the last column.
A row with no padding at all still loses its last label.
On a right-padded row a padding position is always followed by padding, so it is covered by the same rule; with left padding a pad position can be followed by a real token, which is why general-purpose code ignores a label unless **both** position `t` and position `t + 1` are real.

`-100` is the conventional `ignore_index`, and `F.cross_entropy(..., ignore_index=-100)` drops those positions from both the sum and the count it divides by (topic 10).
The labels keep the dtype of the ids, `int64`.

The last-column rule is the one people forget.
Without it, the last real token of each row is trained to predict a pad id, or whatever happened to be in the shifted-in slot, which injects a small, constant amount of nonsense into every batch.

**Do not write into `ids`.**
`ids[:, 1:]` is a view (topic 01), so assigning into it, or `masked_fill_` on it, changes the caller's tensor.
Build the labels in a new tensor, for example one created with `torch.full_like(ids, ignore_index)` and then filled by slice assignment, or with `torch.where`.

### 20.3 Checkpoints

A checkpoint that holds only the model weights is good for inference and useless for resuming training.
A resumable one must hold three things: the model's `state_dict`, the optimizer's `state_dict`, and the iteration number.
`torch.save` serializes any nesting of dicts, lists, tensors and plain Python values, so the natural container is one dict with those three entries.

**The key names are a contract.**
In this exercise the dict uses the keys `"model"`, `"optimizer"` and `"iteration"`.
`test_load_checkpoint` builds a checkpoint with exactly those keys by hand, with `torch.save`, so that loading can be written before saving; `load_checkpoint` has to read them, and `save_checkpoint` has to write them for the roundtrip test to pass.
Store the iteration as a plain Python `int`, which is what `load_checkpoint` returns.

**What is inside an optimizer's `state_dict`.**
Two entries, `"state"` and `"param_groups"`.
`"param_groups"` holds the hyperparameters (`lr`, `betas`, `weight_decay`, ...) and, under `"params"`, the parameters as integer indices.
`"state"` maps each of those indices to that parameter's buffers; for Adam and AdamW those are `step`, `exp_avg` (the first moment `m`) and `exp_avg_sq` (the second moment `v`), from topic 16.
A freshly constructed optimizer has an empty `"state"`: the buffers are created lazily, on the first `step()`.

```python
>>> net = nn.Linear(2, 1, bias=False)
>>> opt = torch.optim.Adam(net.parameters(), lr=0.1)
>>> opt.state_dict()["state"]
{}
>>> net(torch.ones(1, 2)).sum().backward(); opt.step()
>>> sd = opt.state_dict()
>>> list(sd), list(sd["state"]), list(sd["state"][0]), sd["param_groups"][0]["params"]
(['state', 'param_groups'], [0], ['step', 'exp_avg', 'exp_avg_sq'], [0])
```

**Why the optimizer state.**
Adam's `m` and `v` are moving averages built up over thousands of steps, and `v` in particular is what gives every parameter its own step size.
Resume without them and the optimizer starts over from zeros: after bias correction, the first update moves every parameter by about `lr` in the direction of the sign of its gradient, whatever that parameter's usual gradient scale is.
At the start of a run the warmup keeps `lr` small while the moments fill up; a resume lands in the middle of the schedule, at full `lr`, with no warmup, and the loss curve shows a visible spike.

**Why the iteration.**
The learning rate schedule is a function of the step number (topic 16), so a resume that restarts the counter at zero replays the warmup and then decays from a point the model has already passed.
The step number is also what a training loop uses to decide when to log, evaluate and stop.

**Saving and loading.**
`torch.save(obj, f)` writes to a path or to any file-like object, such as an `io.BytesIO`; `torch.load(f)` reads it back.
`torch.save` leaves a buffer's position at its end, and `torch.load` reads from the current position, so a buffer must be rewound with `buf.seek(0)` in between, or `torch.load` fails with an `EOFError`.
In the tests the caller does the rewinding: `save_checkpoint` writes into the buffer it was given, and `load_checkpoint` receives a buffer already at position 0.

`model.load_state_dict(sd)` and `optimizer.load_state_dict(sd)` both work **in place**: they copy the saved values into the objects they are called on.
So `load_checkpoint` changes the model and optimizer it was given and returns only the iteration number.
That is the right design: the caller has already constructed the model and the optimizer, with the right architecture and the right parameters, and only their contents need replacing.
Note that `model.load_state_dict` returns a report of missing and unexpected keys, not the model, and `optimizer.load_state_dict` returns `None`; neither return value is the thing to return from `load_checkpoint`.

**Why `torch.load` got stricter.**
A checkpoint is a pickle, and unpickling can execute arbitrary code, so since PyTorch 2.6 `torch.load` defaults to `weights_only=True`.
That mode accepts tensors, plain containers and primitive values and refuses anything else.
A checkpoint of model and optimizer `state_dict`s plus an `int` loads fine under it; one that pickled a whole optimizer object or a custom config class needs `weights_only=False`, and should only be loaded that way from a source you trust.
The test of `save_checkpoint` reads the blob back with `weights_only=False`, so it does not constrain what you store, but staying within what `weights_only=True` accepts is the habit to build.

**Why the optimizer `state_dict` is fragile.**
It refers to parameters by their **position** in `param_groups`, not by name.
It can only be loaded into an optimizer constructed over the same parameters in the same order, and the roundtrip test relies on that when it compares `state[0]["exp_avg"]` of the original and the restored optimizer.
Change the architecture or reorder the parameter groups, and the load either raises or silently applies the wrong moments to the wrong tensors.
Hence the conventional order: build the model, build the optimizer over its parameters, then load both states.

**`state_dict` returns references.**
`model.state_dict()` gives tensors that share memory with the live parameters, not copies.
`torch.save` serializes them at the moment of the call, so writing to a buffer or a file is safe, but keeping a `state_dict` in a Python variable as a "snapshot" and training on is not: the snapshot changes along with the model.

## API you need

### Data

| Call | Result |
| --- | --- |
| `np.memmap(path, dtype=np.uint16, mode="r")` | a lazy, read-only, on-disk 1-D array |
| `np.random.default_rng(seed)` | a `Generator`; same seed, same numbers |
| `np.random.default_rng()` | a `Generator` seeded from OS entropy |
| `rng.integers(low, high, size=n)` | `n` random ints from `[low, high)`, `high` exclusive |
| `data[i : i + n]` | a slice of `n` elements, shorter if it runs past the end |
| `starts[:, None] + np.arange(n)` | a `(len(starts), n)` grid of indices, topic 03 |
| `data[idx]` with an integer array `idx` | gathers `data` at every index, result has `idx`'s shape |
| `np.stack(list_of_arrays)` | stack equal-length 1-D arrays into a 2-D one, a copy |
| `arr.astype(np.int64)` | cast on the NumPy side, a copy |
| `torch.from_numpy(arr)` | a tensor sharing memory with `arr`, same dtype |
| `torch.as_tensor(arr, dtype=torch.int64)`, `torch.tensor(arr)` | convert, casting if needed; `torch.tensor` always copies |
| `t.long()`, `t.to(torch.int64)` | cast on the torch side |
| `n // k` | integer division, rounding down |
| `t.reshape(-1, k)`, `arr.reshape(-1, k)` | rows of `k`; the size must be divisible by `k` |

### Padding and labels

| Call | Result |
| --- | --- |
| `max(len(s) for s in seqs)` | `T_max` |
| `torch.full((B, T), value, dtype=torch.int64)` | a buffer filled with `value` |
| `torch.zeros((B, T), dtype=torch.bool)` | an all-`False` mask |
| `t[b, :n] = torch.tensor(seq)` | write a sequence into the start of row `b` |
| `torch.arange(T)[None, :] < lengths[:, None]` | `(B, T)` bool mask from lengths |
| `torch.nn.utils.rnn.pad_sequence(list_of_tensors, batch_first=True, padding_value=p)` | the library version of right padding |
| `torch.full_like(t, value)` | a new tensor of `t`'s shape and dtype, filled with `value` |
| `t[:, 1:]`, `t[:, :-1]` | drop the first, the last column; views |
| `torch.where(cond, a, b)` | elementwise choice, broadcasting `a` and `b` |
| `~mask`, `mask_a & mask_b` | invert, combine bool masks |

### Checkpoints

| Call | Result |
| --- | --- |
| `io.BytesIO()` | an in-memory binary file |
| `torch.save(obj, f)` | serialize `obj` into a path or file-like `f` |
| `buf.seek(0)` | rewind a buffer before reading |
| `torch.load(f, weights_only=True)` | deserialize; `True` is the default since 2.6 |
| `torch.load(f, map_location="cpu")` | load tensors saved on a GPU onto the CPU |
| `model.state_dict()` | an ordered dict, parameter name to tensor, sharing memory |
| `optimizer.state_dict()` | `{"state": {idx: {...}}, "param_groups": [{..., "params": [idx, ...]}]}` |
| `model.load_state_dict(sd)` | in place; returns missing/unexpected keys |
| `optimizer.load_state_dict(sd)` | in place, restores `m`, `v` and `step`; returns `None` |

### A worked example on neutral data

The off-by-one, on a ten-token stream with windows of four:

```python
>>> data = np.arange(10, dtype=np.uint16)
>>> ctx = 4
>>> len(data) - ctx - 1                       # the largest start, inclusive
5
>>> i = 5
>>> data[i : i + ctx], data[i + 1 : i + ctx + 1]
(array([5, 6, 7, 8], dtype=uint16), array([6, 7, 8, 9], dtype=uint16))
```

Start 5 is valid and its target reads exactly to the end; start 6 would ask for a target ending at index 10, which does not exist.
Writing this out on a length-10 array once is faster than reasoning about it.

`rng.integers` never returns its `high`:

```python
>>> np.random.default_rng(0).integers(0, 3, size=8)
array([2, 1, 1, 0, 0, 0, 0, 0])
```

Labels built in a new tensor, leaving the source alone:

```python
>>> row = torch.tensor([7, 8, 9])
>>> keep = torch.tensor([True, False, True])
>>> torch.where(keep, row, -1)
tensor([ 7, -1,  9])
>>> torch.full_like(row, -1)
tensor([-1, -1, -1])
```

A buffer roundtrip:

```python
>>> buf = io.BytesIO()
>>> torch.save({"answer": 42}, buf)
>>> buf.seek(0)
0
>>> torch.load(buf)
{'answer': 42}
```

## Traps

- `max_start = num_tokens - context_length - 1`, inclusive; forgetting the `- 1` reads one token past the end on the target slice.
- `rng.integers(low, high)` excludes `high`; drawing starts up to and including `max_start` needs `max_start + 1`.
- A slice past the end of a NumPy array is silently shorter, not an error.
- Ids left as `uint16` (or `int32`) after the memmap: `nn.Embedding` refuses `uint16`, `cross_entropy` wants `int64`, and the test asserts `int64`.
- `torch.from_numpy` shares memory with its source, including a memmap, and warns on a read-only one.
- Ignoring the `generator` argument and drawing from global state, which makes the batch independent of the seed passed in.
- Truncating with `data[:-(len(data) % k)]`: when the remainder is 0 that is `data[:0]`, an empty array; truncate to `num_windows * k` instead.
- Random windows for validation, which makes evaluation noisy and incomparable between runs.
- A mask of `0`/`1` integers instead of `torch.bool`.
- Masking labels by "is position `t` padding" instead of "is position `t + 1` padding", which trains the last real token to predict the pad id.
- Forgetting that the last column of every row has no label.
- Writing the labels into a view of `ids` and corrupting the caller's tensor.
- Padding on the right for batched generation, where left padding is what you want.
- Saving the model but not the optimizer, which costs a loss spike on every resume.
- Saving the optimizer but not the iteration, which replays the learning rate schedule from the start.
- Using different key names in `save_checkpoint` and `load_checkpoint`, or names other than `"model"`, `"optimizer"`, `"iteration"`.
- `torch.load` on a `BytesIO` that was not rewound with `seek(0)`.
- Returning the result of `load_state_dict` instead of the iteration number.

## Ready when

Without looking anything up you can:

- [ ] explain why pretraining data is one flat token array rather than a list of examples
- [ ] say what `np.memmap` buys, why `uint16` is used on disk, and when it is too small
- [ ] state the largest valid window start index and verify it on a length-10 array
- [ ] draw start indices with `rng.integers` without an off-by-one in either direction
- [ ] say why `get_batch` takes a generator, and what two same-seed generators guarantee
- [ ] compute steps per epoch, and cut a stream into non-overlapping windows without an empty-slice bug
- [ ] explain why random windows are right for training and wrong for validation
- [ ] build a padded batch with its bool mask, and say why generation pads on the left
- [ ] construct labels with both `ignore_index` rules, and say which position's padding decides
- [ ] list the three things a resumable checkpoint must contain, and what breaks if each is missing
- [ ] describe what is inside an optimizer `state_dict`, and why it only loads into the same parameter order
- [ ] say why a `BytesIO` needs `seek(0)` and why `torch.load` defaults to `weights_only=True`

## Going deeper

A real training data pipeline adds a few things on top of this topic.
Documents are shuffled before concatenation, so the stream is not ordered by source.
A separate held-out shard is kept that the training stream never touches.
Increasingly, several documents are packed into each window with document masks (topic 19) rather than letting attention straddle boundaries.
Distributed training also needs every rank to draw different windows, usually by seeding each rank's generator differently, and checkpoints the data loader's position so that a resume does not replay the same tokens.
Real checkpoints also carry the RNG states and the run configuration, and large runs shard them across files, one per rank.

- [`numpy.memmap`](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html), and the [`Generator`](https://numpy.org/doc/stable/reference/random/generator.html) page for `default_rng` and `integers`.
- [Saving and loading a general checkpoint](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html) in the PyTorch recipes.
- [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html), whose doc page explains `weights_only` and the allow-list behind it.
- [`torch.nn.utils.rnn.pad_sequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html).
- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT), `train.py`: its `get_batch` is about eight lines and is the canonical version of section 20.1, and the checkpoint code around it is section 20.3.
