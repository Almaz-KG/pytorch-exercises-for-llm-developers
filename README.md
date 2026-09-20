# PyTorch exercises

Drills on PyTorch and tensor algebra, tied to how all of it actually works inside an LLM.
The format is borrowed from [rustlings](https://rustlings.rust-lang.org/) and [100-exercises-to-learn-rust](https://github.com/mainmatter/100-exercises-to-learn-rust): one file per topic, holding function stubs with docstrings and the tests that check those stubs.
The job is to replace every `raise NotImplementedError` with an implementation and drive the file to green.

The goal is not to "learn about PyTorch" but to build muscle memory.
After twenty files, a transformer block, cross-entropy, AdamW and a KV cache should come out of the fingers without looking anything up.

## Theory

`theory/` holds one file per topic, and it is meant to be read before the matching exercise.
The files are plain Markdown, and they are also the source of a small static site, which is the nicer way to read them while solving:
search across all twenty topics, a sidebar, prev/next links and a dark theme.

```sh
uv run mkdocs serve -o           # serve on localhost:8000 and open a browser
uv run mkdocs build              # render to site/ and read it offline

open theory/14_attention.md      # or just read the Markdown, then solve test_14_attention.py
```

`mkdocs serve` watches `theory/` and rebuilds on save.
The site is [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/), configured in `mkdocs.yml`; `site/` is generated and git-ignored.

The point of that directory is that this repository is self-contained: after reading `theory/NN_<topic>.md` you should be able to
solve every stub in `test_NN_<topic>.py` in one sitting, with no search engine and no chat window open next to it.
Each file gives the concepts and the formulas, every `torch` / `nn` / `F` / `einops` call the topic uses, the traps that topic is
built to provoke, and a short readiness checklist.
What it deliberately does not give is a solution to any stub: the examples in it run on neutral data, not on the exercise's own task.
When you want the answer outright, that is what `solutions/` is for.

## Running

```sh
uv sync                                   # once

uv run pytest test_01_tensor_basics.py -x # the current file, stop at the first failure
uv run pytest -x                          # everything, in order
uv run pytest -k rope                     # a single test by name
```

The first run of an untouched file is nothing but `NotImplementedError`.
That is the intended starting state, not a broken environment.

Each test covers one stub, so a failure names the single function to go and write, and the stubs can be
done in any order. The few tests that check that several finished pieces agree with each other - a
split/merge roundtrip, a KV cache against a full forward pass - are skipped until every piece they need
is implemented, and turn themselves on once the last one lands:

```sh
uv run pytest test_02_shapes_and_views.py -rs   # -rs spells out what each skip is still waiting for
```

If you are stuck, `solutions/` holds a reference implementation of every topic.
You can peek at one spot, or run the whole suite against the reference:

```sh
SOLUTIONS=1 uv run pytest test_14_attention.py   # the tests run against solutions/sol_14_attention.py
```

`SOLUTIONS=1` swaps the functions and classes in the test module for the same-named ones from `solutions/`.
It doubles as the check that the tests themselves are correct.

## Rolling back your solutions

The original stubs for every topic live in `templates/`, byte-for-byte copies of the files as they started.
Rollback works without git, and your current version is always moved to `attempts/<date_time>/` before it gets overwritten, so nothing is lost.

```sh
uv run python reset.py                 # status: what is touched, how many stubs are left
uv run python reset.py 14              # roll topic 14 back (number, name or part of the name)
uv run python reset.py 14 15 16        # several topics at once
uv run python reset.py --all           # roll everything back and start over
uv run python reset.py --diff 14       # how my version differs from the original
uv run python reset.py --restore attempts/2026-08-25_12-56-10/test_14_attention.py
```

The status output shows, per topic, how many `NotImplementedError` stubs remain, which makes it easy to come back after a break.
Redoing a topic a month later is the best way to check whether the muscle memory stuck; that is what rollback is for.

## Rules of the game

- No loops over tensor elements unless a task says otherwise. Everything vectorized.
- If a task says "without `torch.softmax`" or "via einsum", that is part of the exercise, not nitpicking.
- The tests are off limits. A test bent to fit an implementation teaches nothing.
- Write the formula on paper first, then the code. Especially in topics 08, 18, 19.
- Module and parameter names are fixed by the docstrings (`q_proj`, `w1`, `ln1`, `lm_head` and so on, no biases anywhere). Several tests assert on `state_dict` keys, so the names are load-bearing.

## Order of topics

Difficulty grows top to bottom, and later topics lean on earlier ones.

| File | Theory | Topic | What it drills |
| --- | --- | --- | --- |
| 01 | [read](theory/01_tensor_basics.md) | Tensor basics | dtype, shape, memory, in-place, copy vs view |
| 02 | [read](theory/02_shapes_and_views.md) | Shapes and views | reshape, permute, contiguous, splitting and merging heads, expand vs repeat |
| 03 | [read](theory/03_broadcasting.md) | Broadcasting | right-aligned shapes, masks, ALiBi, pairwise distances |
| 04 | [read](theory/04_reductions.md) | Reductions | dim and keepdim, masked means, logsumexp, gradient norms |
| 05 | [read](theory/05_indexing_and_gather.md) | Indexing and gather | embedding lookup, gathering logits, scatter, top-k mask |
| 06 | [read](theory/06_matmul.md) | Matrix multiplication | shapes, linear and its backward by hand, LoRA, FLOPs |
| 07 | [read](theory/07_einsum_and_einops.md) | einsum and einops | the whole attention algebra in one line |
| 08 | [read](theory/08_autograd.md) | Autograd | backward, gradient accumulation, vjp, custom autograd.Function, STE |
| 09 | [read](theory/09_nn_module.md) | nn.Module | Parameter vs buffer, state_dict, initialization, weight tying, hooks |
| 10 | [read](theory/10_softmax_and_loss.md) | Softmax and loss | stable softmax, cross-entropy from scratch, KL, the gradient of CE |
| 11 | [read](theory/11_normalization.md) | Normalization | LayerNorm, RMSNorm, upcasting to float32, pre-norm vs post-norm |
| 12 | [read](theory/12_activations_and_ffn.md) | Activations and FFN | GELU, SiLU, SwiGLU, the arithmetic of d_ff |
| 13 | [read](theory/13_embeddings_and_rope.md) | Embeddings and RoPE | sinusoids, rotations, relative positions |
| 14 | [read](theory/14_attention.md) | Attention | scaled dot-product, causality, MHA, GQA, cost |
| 15 | [read](theory/15_transformer_block.md) | Transformer block | the pre-norm block, the whole model, initial loss equal to log(V) |
| 16 | [read](theory/16_training_loop.md) | Training loop | AdamW from scratch, cosine schedule, clipping, gradient accumulation |
| 17 | [read](theory/17_sampling_and_kv_cache.md) | Generation | temperature, top-k, top-p, the KV cache and its equivalence to a full pass |
| 18 | [read](theory/18_flash_and_memory.md) | Flash and memory | online softmax, blockwise attention, chunked CE, checkpointing, loss scaling |
| 19 | [read](theory/19_advanced_architectures.md) | Advanced mechanisms | a hand-written RMSNorm backward, window and document masks, MoE, speculative decoding, beam search |
| 20 | [read](theory/20_data_and_batching.md) | Data and checkpoints | slicing batches out of a memmap, padding, save/load checkpoint |

### Topics 01-07 are tensor mechanics.
08-09 are how PyTorch computes gradients and stores a model.
10-15 are the transformer, brick by brick and then whole.
16-17 are training and inference.
18-19 are what all of it is for: memory, speed and modern architectural tricks.
20 is the plumbing without which training never starts.
