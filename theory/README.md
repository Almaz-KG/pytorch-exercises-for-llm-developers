# Theory

One file per topic, meant to be read straight through before you open the matching `test_NN_*.py`.

The promise of this directory is that the repository is self-contained.
After reading `theory/NN_<topic>.md` you should be able to solve every stub in `test_NN_<topic>.py` in one sitting.

## The machine you are assembling

Twenty topics, one machine.
A language model takes a row of token ids and returns, for each position, a score for every word that could come next.
Between those two ends it does the same handful of things, and each of them is a topic here.

An embedding table turns the ids into one vector per token (13).
Those vectors are the residual stream, and every layer reads from it and adds its result back into it.
A layer normalizes (11), lets the tokens read each other through attention (14), adds that back, normalizes again, transforms each token on its own through a feed-forward branch (12), and adds that back too; stacking that layer is the whole model (15).
After the last layer, a final normalization and one more matrix (09) turn each vector into a score per vocabulary entry.
Cross-entropy (10) collapses those scores into a single loss, autograd (08) turns the loss into a gradient for every weight, and the optimizer (16) turns the gradients into new weights.

Topics 01 to 07 are none of those steps.
They are the tools every step is built from: a tensor and its dtype, moving axes around, broadcasting, reductions, indexing, matrix multiplication, einsum.
Six of the twenty files are spent there on purpose, because a shape bug inside attention is a shape bug, not an attention bug.

The last four topics surround the model rather than sit inside it.
Topic 17 runs the forward pass one token at a time to generate text, and caches what it can.
Topic 18 measures what attention costs and makes it cheaper.
Topic 19 swaps individual pieces for the variants modern models use.
Topic 20 is what feeds tokens in at one end and writes weights to disk at the other.

## Topics

| Theory | Exercise | What you build in it |
| --- | --- | --- |
| [01](01_tensor_basics.md) | `test_01_tensor_basics.py` | A tensor's three properties: storage, dtype, shape. Copies versus views |
| [02](02_shapes_and_views.md) | `test_02_shapes_and_views.py` | Moving axes without moving data: reshape, transpose, the head split |
| [03](03_broadcasting.md) | `test_03_broadcasting.py` | The rule that lets a `(B, S)` mask line up against `(B, H, T, S)` scores |
| [04](04_reductions.md) | `test_04_reductions.py` | Sum, mean, max and `logsumexp` along a chosen axis |
| [05](05_indexing_and_gather.md) | `test_05_indexing_and_gather.py` | Pulling one value per row out of a batch: the embedding lookup and the loss both need it |
| [06](06_matmul.md) | `test_06_matmul.py` | Batched matrix multiplication, its shape rules and its FLOP count |
| [07](07_einsum_and_einops.md) | `test_07_einsum_and_einops.py` | Naming axes instead of counting them |
| [08](08_autograd.md) | `test_08_autograd.py` | The backward pass: what a gradient is and where PyTorch keeps it |
| [09](09_nn_module.md) | `test_09_nn_module.py` | Parameters, `state_dict`, and layers of your own |
| [10](10_softmax_and_loss.md) | `test_10_softmax_and_loss.py` | Scores into probabilities, and probabilities into one loss number |
| [11](11_normalization.md) | `test_11_normalization.py` | LayerNorm and RMSNorm, and where in the block they go |
| [12](12_activations_and_ffn.md) | `test_12_activations_and_ffn.py` | The feed-forward branch: GELU, SiLU and the SwiGLU block |
| [13](13_embeddings_and_rope.md) | `test_13_embeddings_and_rope.py` | The token table, and rotary position encoding |
| [14](14_attention.md) | `test_14_attention.py` | Scaled dot-product attention, masks, multi-head and grouped-query |
| [15](15_transformer_block.md) | `test_15_transformer_block.py` | The pre-norm block, and the full model end to end |
| [16](16_training_loop.md) | `test_16_training_loop.py` | Optimizer step, schedule, clipping, accumulation |
| [17](17_sampling_and_kv_cache.md) | `test_17_sampling_and_kv_cache.py` | Sampling from logits, and the KV cache that makes generation linear |
| [18](18_flash_and_memory.md) | `test_18_flash_and_memory.py` | Online softmax, flash attention, checkpointing, mixed precision |
| [19](19_advanced_architectures.md) | `test_19_advanced_architectures.py` | A hand-written backward, mixture of experts, speculative decoding, beam search |
| [20](20_data_and_batching.md) | `test_20_data_and_batching.py` | Batches out of a token stream, padding, checkpoints |

Read them top to bottom.
Each file assumes every file above it and nothing below it.

## Conventions used throughout

Shapes are written as tuples of single letters, and the letters mean the same thing in every file.

| Letter | Meaning |
| --- | --- |
| `B` | batch size |
| `T` | query sequence length, or just sequence length |
| `S` | key/value sequence length, when it differs from `T` |
| `H` | number of attention heads |
| `G` | number of key/value groups, in grouped-query attention |
| `D` | `d_model`, the width of the residual stream |
| `Dh` | `d_head`, equal to `D // H` |
| `F` | `d_ff`, the hidden width of the feed-forward block |
| `V` | vocabulary size |
| `N` | a flattened token count, usually `B * T` |
| `E` | number of experts |
| `L` | number of layers |

`(B, T, D) -> (B, T, H, Dh)` means a function taking the first shape and returning the second.
`...` in a shape means "any number of leading axes", which is how most of these functions are written.

Mathematics is written in LaTeX: `$...$` inline, and a fenced block tagged `math` for a formula on its own line.
Those are the two forms both GitHub and the site render, so avoid `$$...$$` and `\(...\)`.
Code stays in backticks: `torch.exp(x)` is an API call, $e^{x}$ is the function it computes.
GitHub renders math with an allowlist of macros, stricter than the site's KaTeX: write named functions as `\mathrm{softmax}`, never `\operatorname`, and do not use `\phantom`.
Inside inline `$...$` GitHub first strips Markdown escapes, so a backslash before punctuation or a space (`\,`, `\;`, `\ `, `\{`, `\_`) silently disappears; keep those to `math` fences.

## Two ways to read it

These are plain Markdown files, so they render on GitHub and in any editor.
They are also the source of a small static site, which is the nicer way to read them while working: it has search across all twenty topics, a sidebar, prev/next links and a dark theme.

```sh
uv run mkdocs serve -o    # build, serve on localhost:8000 and open a browser
uv run mkdocs build       # render to site/ and read it offline
```

`mkdocs serve` watches `theory/` and rebuilds on save, so fixing a sentence updates the page you are looking at.
