# Topic 09 - nn.Module

> **Exercise** `test_09_nn_module.py` · **Reference** `solutions/sol_09_nn_module.py` · **Assumes** topics 01-08

## Where this fits

Every layer of the model is an `nn.Module`, and `lm_head` is one you build here.

`nn.Module` is the container that decides which tensors get trained, which get saved, and which move to the GPU when you call `.to("cuda")`.
All three are opt-in, and the way you opt in is by assigning the tensor to the right kind of attribute.

Get it wrong and the failures are silent: a mask that stays on CPU and crashes only on a multi-GPU run, a layer in a plain Python list whose parameters the optimizer never sees, a checkpoint that will not load because someone wrapped the model in DDP.

This file is also where the naming contract the rest of the repository depends on gets established.
From topic 11 onward every module you write has parameter names that the tests assert on, because they match CS336's, and topic 15 loads a whole model by those names.

What you need from topic 08: `requires_grad`, `parameters()`, and what freezing does.

## What you build here

Fifteen components: four modules and eleven functions.

### 9.1 Layers of your own

| Component | In -> out | What it must do |
| --- | --- | --- |
| `MyLinear(in_features, out_features, bias=True)` | `(..., in) -> (..., out)` | Parameters named `weight` `(out, in)` and `bias`. With `bias=False` the attribute is `None` and the key is absent from the `state_dict`. Any number of leading axes |
| `MyEmbedding(num_embeddings, embedding_dim)` | `(...)` int64 -> `(..., D)` | One parameter named `weight`, shaped `(V, D)` |
| `CausalSelfAttentionSkeleton(max_len)` | `int -> (T, T)` bool | Stores a causal mask as a **buffer** named `mask`, built once for `max_len`. `forward` returns the slice for the current length |
| `BlockStack(d_model, n_layers)` | `(..., D) -> (..., D)` | `n_layers` linear layers applied in order, living in an `nn.ModuleList` |

### 9.2 Inspecting a model

| Component | In -> out | What it must do |
| --- | --- | --- |
| `count_parameters(model, trainable_only=True)` | module -> `int` | Elements, not tensors |
| `parameter_names(model)` | module -> `list[str]` | Sorted full dotted names, as the `state_dict` has them |
| `total_bytes(model)` | module -> `int` | Parameters **and** buffers |

### 9.3 Initialization

| Component | In -> out | What it must do |
| --- | --- | --- |
| `init_linear_(layer, std)` | `nn.Linear`, float -> the same layer | Truncated normal `N(0, std)` clipped to `[-3*std, 3*std]`, bias zeroed, in place |
| `init_all_linears_(model, std)` | module, float -> the same module | Every `nn.Linear` in the tree, via one traversal |
| `scaled_residual_std(base_std, n_layers)` | float, int -> float | `std / sqrt(2 * n_layers)`, the GPT-2 residual scaling |

### 9.4 state_dict

| Component | In -> out | What it must do |
| --- | --- | --- |
| `save_load_roundtrip(model)` | `MyLinear` -> a new `MyLinear` | Save to an `io.BytesIO` with `torch.save`, build a fresh model, load the weights in |
| `strip_prefix(state_dict, prefix)` | dict, str -> dict | Remove the prefix where present, leave other keys alone |
| `tie_weights(embedding, head)` | two modules -> `None` | Make them share one tensor. A write to either must be visible in the other, and the unique parameter count must drop |

### 9.5 Modes and hooks

| Component | In -> out | What it must do |
| --- | --- | --- |
| `dropout_behaviour(p, x)` | float, tensor -> `(tensor, tensor)` | The output in train mode and in eval mode, from one `nn.Dropout(p)` |
| `capture_activations(model, x)` | `nn.Sequential`, tensor -> `dict[str, Tensor]` | Forward hooks on every submodule, keyed by module name. Remove the handles afterwards |

## The mental model

### The three kinds of tensor attribute

| Assigned as | In `parameters()` | In `state_dict()` | Moves with `.to(device)` | Trained |
| --- | --- | --- | --- | --- |
| `nn.Parameter` | yes | yes | yes | yes |
| registered buffer | no | yes, unless `persistent=False` | yes | no |
| plain tensor attribute | no | no | **no** | no |

`nn.Parameter(t)` is a `Tensor` subclass with `requires_grad=True` by default, and `nn.Module.__setattr__` recognizes it and registers it.
Buffers must be registered explicitly with `self.register_buffer("name", tensor)`.

The plain tensor row is the bug.
`self.mask = torch.ones(...)` looks identical to the buffer version and works perfectly on CPU, then stays on CPU when the model moves, or gets silently omitted from the checkpoint.
Anything that is a constant the module needs at forward time, such as a causal mask or a RoPE cache, is a buffer.

`persistent=False` keeps a buffer out of the `state_dict` while still moving it with the module, which is right for anything recomputable from the config.

### Containers

The same rule applies to submodules.
A module assigned to a plain attribute is registered; a module sitting inside a plain Python `list` or `dict` is not, and its parameters are invisible to `parameters()`, to `state_dict()`, and therefore to the optimizer.
The model trains, the loss goes down a bit, and a third of the network never updates.

| Container | Use |
| --- | --- |
| `nn.ModuleList` | a list of submodules you call yourself in `forward` |
| `nn.Sequential` | a list that is called in order automatically |
| `nn.ModuleDict` | a dict of submodules |
| `nn.ParameterList` / `nn.ParameterDict` | the same for bare parameters |

A stack of transformer blocks is an `nn.ModuleList` iterated in `forward`, because real blocks take more than one argument and `Sequential` cannot express that.

### 9.1 Writing a layer

The skeleton is always the same:

```python
class MyLayer(nn.Module):
    def __init__(self, ...):
        super().__init__()          # first line, always
        self.weight = nn.Parameter(torch.empty(out, in))
        self.register_buffer("something", torch.ones(n))

    def forward(self, x):
        return ...
```

Forgetting `super().__init__()` gives `AttributeError: cannot assign module before Module.__init__() call`, which is at least a loud failure.

Call the module, do not call `forward` directly: `layer(x)` runs `__call__`, which fires hooks and does bookkeeping that `layer.forward(x)` skips.

**Optional parameters.**
When `bias=False`, the right thing is `self.bias = None`, which keeps the attribute present and keeps the key out of the `state_dict`.
`register_parameter("bias", None)` is the explicit spelling of the same thing.
`F.linear` accepts `None` for the bias, so `forward` needs no branch.

**Leading axes.**
A layer written as `x @ self.weight.T` already works for `(B, T, in)` as well as `(N, in)`, per topic 06.
No reshaping should appear in a well-written linear layer.

**Buffers for masks.**
Computing a causal mask once at construction up to `max_len` and slicing `self.mask[:T, :T]` in `forward` is what every real implementation does, because building a `(T, T)` tensor every step is pure waste.
The slice is a view, so it costs nothing.

### 9.2 Inspecting a model

`model.parameters()` and `model.named_parameters()` walk the tree recursively.
Names are dotted paths: `blocks.0.attn.q_proj.weight`.
`model.buffers()` and `model.named_buffers()` do the same for buffers, and `model.modules()` walks the submodules themselves, including the root.

Counting parameters is `sum(p.numel() for p in model.parameters())`, optionally filtered by `p.requires_grad`.
Counting bytes adds `p.element_size()` and should include buffers, because they occupy memory too.

Two subtleties.
`named_parameters()` deduplicates shared tensors by default, so a model with tied weights reports the true count of unique parameters, which is also how you can *detect* tying.
`state_dict()` does the opposite and keeps both keys pointing at the same tensor, because loading needs both names to exist.

### 9.3 Initialization

`nn.Linear` defaults to Kaiming uniform, which is tuned for ReLU networks and is not what transformers use.
The GPT-2 recipe, which this repository follows, is a truncated normal with `std = 0.02`, clipped at plus or minus three standard deviations, and zeros for biases.

`nn.init` functions all end in an underscore, are in-place, and handle the `no_grad` context themselves:

| Call | Effect |
| --- | --- |
| `nn.init.trunc_normal_(t, mean=0.0, std=0.02, a=-0.06, b=0.06)` | normal, resampled outside `[a, b]` |
| `nn.init.normal_(t, mean, std)` | plain normal |
| `nn.init.zeros_(t)` / `nn.init.ones_(t)` | constants |
| `nn.init.xavier_uniform_(t)` / `nn.init.kaiming_normal_(t)` | the classical schemes |

`model.apply(fn)` calls `fn` on every submodule, depth first, including the root.
The idiom is a function that checks `isinstance(m, nn.Linear)` and initializes it, which handles a whole tree in one line.

**Residual scaling** is worth understanding rather than copying.
Each transformer layer adds its output into the residual stream.
If every layer's contribution has variance `s`, then after `L` layers the stream's variance has grown by roughly `L * s`, and with two residual writes per block it is `2L * s`.
GPT-2 compensates by initializing *the projections that write into the residual stream*, `output_proj` in attention and `w2` in the FFN, with `std / sqrt(2 * n_layers)`, so the accumulated variance stays order 1 regardless of depth.

### 9.4 state_dict

`state_dict()` returns an `OrderedDict` mapping dotted names to tensors.
Those tensors are **references**, not copies, so mutating the model mutates a previously captured `state_dict`.

`load_state_dict(sd, strict=True)` requires the key sets to match exactly and returns a named tuple of `missing_keys` and `unexpected_keys`.
With `strict=False` it loads what it can and reports the rest, which is how you load a pretrained backbone into a model with a new head.

Saving and loading through a byte buffer is `torch.save(obj, buf)`, then `buf.seek(0)`, then `torch.load(buf)`.
The `seek(0)` is mandatory and is the step people forget.

**Key prefixes** accumulate as a model gets wrapped.
`DistributedDataParallel` prepends `module.`, and `torch.compile` prepends `_orig_mod.`.
A checkpoint saved from a wrapped model will not load into an unwrapped one, and the fix is to strip the prefix from every key that has it, leaving the others alone.
`str.removeprefix` does exactly that.

**Weight tying** makes the embedding table and the output projection the same tensor.
`head.weight = embedding.weight` is the whole operation, and it works because both are `(V, D)`: the embedding reads row `i` for token `i`, and `F.linear` computes `x @ weight.T`, which scores every row against `x`.
After tying, `named_parameters()` reports one tensor where it used to report two, and writing into one is visible through the other because they are literally the same object.

Assign the `nn.Parameter`, not a tensor.
`head.weight = embedding.weight.data` would quietly turn it into a plain tensor and unregister it.

### 9.5 Modes and hooks

`model.train()` and `model.eval()` set a `training` flag on every submodule recursively.
Only two standard layers care: dropout, which becomes the identity in eval, and batch norm, which switches from batch statistics to running statistics.
Everything else ignores the flag.

Dropout in training mode does two things: it zeros elements with probability `p`, and it scales the survivors by `1 / (1 - p)` so the expected value is preserved.
That scaling is why eval mode can simply be the identity, with no compensation needed.

`model.eval()` does **not** disable gradients.
It is orthogonal to `torch.no_grad()`, and an evaluation loop needs both.

**Hooks** attach callbacks to a module's forward or backward:

| Call | Fires |
| --- | --- |
| `m.register_forward_hook(fn)` | after `forward`, as `fn(module, args, output)` |
| `m.register_forward_pre_hook(fn)` | before `forward`, as `fn(module, args)` |
| `m.register_full_backward_hook(fn)` | after gradients with respect to the inputs are computed |

Each returns a handle with a `.remove()` method, and leaking handles is a memory leak, so a capture utility should remove what it registered.
Returning a non-`None` value from a forward hook *replaces* the output, which is both a useful feature and an easy accident.

## API you need

### Defining

| Call | Result |
| --- | --- |
| `nn.Parameter(tensor, requires_grad=True)` | a trained tensor |
| `self.register_buffer(name, tensor, persistent=True)` | a non-trained, travelling tensor |
| `self.register_parameter(name, None)` | declare an absent optional parameter |
| `nn.ModuleList`, `nn.Sequential`, `nn.ModuleDict` | containers that register their contents |
| `super().__init__()` | the mandatory first line of `__init__` |

### Inspecting

| Call | Result |
| --- | --- |
| `model.parameters()` / `named_parameters()` | recursive, deduplicated |
| `model.buffers()` / `named_buffers()` | the same for buffers |
| `model.modules()` / `named_modules()` | submodules, root included |
| `model.children()` / `named_children()` | direct children only |
| `model.apply(fn)` | call `fn` on every submodule |
| `p.numel()`, `p.element_size()` | size arithmetic |

### Persisting

| Call | Result |
| --- | --- |
| `model.state_dict()` | `OrderedDict` of references |
| `model.load_state_dict(sd, strict=True)` | returns `(missing_keys, unexpected_keys)` |
| `torch.save(obj, path_or_buffer)` | pickle to disk or `io.BytesIO` |
| `torch.load(path_or_buffer, weights_only=True)` | read back, remember `buf.seek(0)` |
| `str.removeprefix(p)` | strip `module.` or `_orig_mod.` |

### Runtime

| Call | Result |
| --- | --- |
| `model.train()` / `model.eval()` | set the `training` flag recursively |
| `model.to(device_or_dtype)` | move parameters and buffers, in-place for modules |
| `nn.Dropout(p)`, `F.dropout(x, p, training=...)` | inverted dropout |
| `m.register_forward_hook(fn)` -> `handle.remove()` | observe or replace outputs |

### A worked example on neutral data

```python
>>> class M(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.w = nn.Parameter(torch.zeros(2, 2))
...         self.register_buffer("b", torch.ones(2))
...         self.plain = torch.ones(2)          # the bug
...         self.layers = [nn.Linear(2, 2)]     # also the bug
>>> m = M()
>>> list(dict(m.named_parameters()))
['w']
>>> list(m.state_dict())
['w', 'b']
```

Neither `plain` nor the linear layer inside the list appears anywhere.

## Traps

- A tensor assigned as a plain attribute does not move with `.to(device)` and is not in the `state_dict`.
- Submodules in a plain Python list are invisible to the optimizer.
- Missing `super().__init__()` raises on the first assignment.
- Calling `model.forward(x)` instead of `model(x)` skips hooks.
- `model.eval()` does not turn off autograd; you still need `torch.no_grad()`.
- `state_dict()` returns references, so it aliases the live model.
  Deep-copy it if you intend to keep a snapshot.
- `torch.load` on a `BytesIO` without `seek(0)` reads nothing.
- A checkpoint from a DDP or `torch.compile` wrapped model has prefixed keys and fails `strict=True`.
- Tying with `head.weight = emb.weight.data` unregisters the parameter.
  Assign the `Parameter` itself.
- `nn.Linear`'s default init is not the transformer init; if you do not initialize explicitly, you are training a differently initialized model than you think.
- Forward hooks that return a value replace the module's output.
- Leaked hook handles keep modules alive and fire on every later forward pass.

## Ready when

Without looking anything up you can:

- [ ] name the three kinds of tensor attribute and say what each does on `.to(device)` and in the `state_dict`
- [ ] explain why a causal mask should be a buffer and a RoPE cache should probably be a non-persistent one
- [ ] write a linear layer from scratch with an optional bias that stays out of the `state_dict`
- [ ] say what `model.apply` is for and initialize every `nn.Linear` in a tree with one call
- [ ] justify the `1 / sqrt(2 * n_layers)` residual init scaling
- [ ] tie an embedding to an output head and say how you would verify the tie
- [ ] say exactly what `model.eval()` changes and what it does not

## Context



**What weight tying buys.**
For a 50k vocabulary and `d_model = 768` the embedding table is 38M parameters, and the output projection is another 38M.
Tying them halves that, which in GPT-2 small is a substantial share of the whole model.
The other argument is a prior: a token's input representation and the direction that predicts it should be related, and forcing them to be identical is a cheap way to say so.
Most modern models tie; some large ones do not, on the grounds that at scale the two roles genuinely differ.

**Why `torch.load` got stricter.**
A checkpoint is a pickle, and unpickling executes arbitrary code, so loading a checkpoint from an untrusted source was a remote code execution waiting to happen.
Since PyTorch 2.6 `torch.load` defaults to `weights_only=True`, which accepts plain tensors and containers and refuses everything else.
A `state_dict` of tensors loads fine; a checkpoint that pickled an optimizer object or a custom class will need an explicit opt-in and a reason to trust the file.

**What hooks are really for.**
Beyond debugging, hooks are the extension point behind several standard tools: activation checkpointing registers them, quantization observers collect statistics through them, and interpretability work extracts or patches intermediate activations with them without touching the model's code.
The ability to replace an output from a forward hook is what makes activation patching possible.

## Going deeper

- [`nn.Module` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), which lists every registration method in one place.
- [Serialization semantics](https://pytorch.org/docs/stable/notes/serialization.html) for the `state_dict` contract and the `weights_only` change.
- The GPT-2 code release, where the `1/sqrt(2*n_layer)` residual scaling first appears as a two-line comment.
