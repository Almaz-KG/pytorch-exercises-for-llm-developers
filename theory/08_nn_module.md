# Topic 08 - nn.Module

> **Exercise** `test_08_nn_module.py`

## Why this topic

In the topics so far the weights were loose tensors that you created by hand and passed around yourself, like the weight of the linear layer in topic 06.
That works for three weights.
A real language model has hundreds of weight tensors, spread over dozens of layers, and you need to do the same few things to all of them at once: hand them to an optimizer, save them to disk, load them back, move them to the GPU.

`nn.Module` is the PyTorch class that keeps track of all of that for you.
Every layer is a module, every block of layers is a module, and the whole model is a module.
Each one knows which tensors it owns and which smaller modules live inside it, so asking the top-level model for "all the weights" walks the whole tree and finds every single one.

The catch is how a module finds out what it owns.
It watches what you assign to `self` inside `__init__`, and it only notices the things it recognizes.
Assign a tensor the wrong way and the module never learns it exists.
Nothing crashes.
The tensor just quietly stays behind on the CPU when the model moves to the GPU, or is missing from the checkpoint, or is never trained.

Those silent failures are what this topic is about.
You will write a few layers of your own, look inside a model, initialize it, save and load it, and switch it between training and evaluation.

There is one more reason this topic matters for the rest of the repository.
From topic 10 on, every module you write has parameter names that the tests check, because they match the names in CS336's first assignment, and topic 14 loads a whole model by those names.
This is where you learn how those names are built.

## The mental model

### A two-minute refresher on `self.x = ...`

If you have not written Python classes in a while, this is the one mechanism you need.

When you write `self.weight = something` inside a method, Python calls a special method on the object, `__setattr__`, with the name `"weight"` and the value.
For an ordinary class that method just stores the value.
A class can override `__setattr__` to do extra work first, and `nn.Module` does exactly that.

Its version looks at the value you assigned and asks what kind of thing it is:

- an `nn.Parameter`: write it down in the module's list of parameters;
- another `nn.Module`: write it down in the module's list of submodules, which PyTorch calls children;
- anything else, including an ordinary tensor: store it like any plain Python attribute and forget about it.

Those private lists are what `parameters()`, `state_dict()`, `.to(device)` and everything else read from.
So the rule of the whole topic is: **a module only manages what it was told about at assignment time.**

This is also why every module's `__init__` starts with `super().__init__()`.
That call creates the empty lists.
Assign a parameter before it, and `__setattr__` has nowhere to write it down, so it raises `AttributeError: cannot assign parameters before Module.__init__() call`.

### The three kinds of tensor attribute

A module can hold a tensor in one of three ways, and they behave very differently:

| Assigned as | In `parameters()` | In `state_dict()` | Moves with `.to(device)` | Trained |
| --- | --- | --- | --- | --- |
| `nn.Parameter` | yes | yes | yes | yes |
| registered buffer | no | yes, unless `persistent=False` | yes | no |
| plain tensor attribute | no | no | **no** | no |

**A parameter** is a weight the model learns.
`nn.Parameter(t)` wraps a tensor and switches on `requires_grad`, so PyTorch's automatic differentiation, autograd, computes a gradient for it whenever a loss is differentiated; topic 15 opens that machinery up.
It is still a tensor in every other way: you can index it, multiply it, and pass it to any `torch` function.

**A buffer** is a tensor the module needs but does not learn.
Think of a constant table computed once from the config: a mask, or the cos and sin tables for rotary position embeddings in topic 12.
You still want it to move to the GPU with the rest of the model, and usually you want it saved, so you tell the module about it explicitly with `self.register_buffer("name", tensor)`.
After that it is available as `self.name` like any attribute.

If a buffer can always be rebuilt from the config, there is no point storing it in every checkpoint.
`persistent=False` keeps it out of the `state_dict` while it still moves with the module.

**A plain tensor attribute** is the bug.
`self.mask = torch.ones(...)` looks exactly like the buffer version and works perfectly while everything is on the CPU.
Then the model moves to the GPU, the mask stays behind, and the first forward pass fails with a device mismatch, often only on the training cluster and never on your laptop.

Here is all three side by side, on a throwaway module:

```python
>>> class Demo(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.w = nn.Parameter(torch.zeros(2, 2))
...         self.register_buffer("table", torch.ones(2))
...         self.register_buffer("scratch", torch.ones(2), persistent=False)
...         self.plain = torch.ones(2)  # the bug
>>> d = Demo()
>>> [name for name, _ in d.named_parameters()]
['w']
>>> [name for name, _ in d.named_buffers()]
['table', 'scratch']
>>> list(d.state_dict())
['w', 'table']
```

`plain` appears nowhere, and `scratch` is a buffer that is not saved.

### Containers

The same rule applies to layers inside layers.
A module assigned directly to `self.something` is registered.
A module sitting inside an ordinary Python `list` or `dict` is not, because `__setattr__` sees a list, not a module, and stores it as a plain attribute.

```python
>>> class Forgetful(nn.Module):
...     def __init__(self):
...         super().__init__()
...         self.layers = [nn.Linear(2, 2), nn.Linear(2, 2)]  # a plain list
>>> list(Forgetful().parameters())
[]
```

The forward pass still runs, because Python can call the layers out of the list just fine.
But the optimizer is built from `model.parameters()`, and that list is empty, so those layers keep their random starting weights forever.
In a bigger model it is worse: the loss still goes down a little, because the other layers learn, and a third of the network silently never updates.

PyTorch has container classes that look like a list or a dict but register what you put in them:

| Container | Use |
| --- | --- |
| `nn.ModuleList` | a list of submodules; you call them yourself in `forward` |
| `nn.Sequential` | a list that calls its layers in order for you |
| `nn.ModuleDict` | a dict of submodules, keyed by string |
| `nn.ParameterList` / `nn.ParameterDict` | the same, for bare parameters |

`nn.ModuleList` behaves like a list in every way you would expect: it takes a list in its constructor, and you can index it, iterate over it, take its `len`, and `append` to it.
A stack of transformer blocks is an `nn.ModuleList` that `forward` loops over, rather than an `nn.Sequential`, because a real block takes more than one argument and `Sequential` can only pass a single value from one layer to the next.

### How the names are built

Every parameter and buffer gets a name, and the name is the path to it through the tree of modules, joined by dots.
The attribute name you chose becomes one piece of the path, and a position inside a `ModuleList` or `Sequential` becomes its index as a string.

So a model with `self.blocks = nn.ModuleList(...)`, where each block has `self.attn`, which has `self.q_proj`, which has `self.weight`, owns a parameter called `blocks.0.attn.q_proj.weight`.
Those dotted names are the keys of the `state_dict`, and they are why the attribute names in topics 10 to 14 are not a matter of taste.
Rename `q_proj` to `query`, and a checkpoint saved under the old name no longer loads.

### 8.1 Writing a layer

Every module has the same shape:

```python
class MyLayer(nn.Module):
    def __init__(self, n: int):
        super().__init__()  # always the first line
        self.scale = nn.Parameter(torch.ones(n))  # a weight it learns
        self.register_buffer("offset", torch.zeros(n))  # a constant it carries

    def forward(self, x):
        return x * self.scale + self.offset
```

`__init__` creates and registers everything the layer owns.
`forward` describes the computation, using those attributes.

**Call the module, not `forward`.**
You run a layer with `layer(x)`, not `layer.forward(x)`.
`layer(x)` goes through `nn.Module.__call__`, which runs the hooks from 8.5 and then calls your `forward`.
Calling `forward` directly works, but skips the hooks, so tools that rely on them silently see nothing.

**Where the starting values come from.**
A new parameter needs some starting numbers.
`torch.empty(shape)` is the fastest way to allocate a tensor, but it does not fill it: you get whatever bytes happened to be in that memory, which can be zeros, huge numbers, or `nan`.
That is fine when an initialization step overwrites every value right after (8.3 shows how), and dangerous when nothing does.
If you want something usable straight away, start from `torch.randn(shape)`, or from `torch.empty` followed by an `nn.init` function.

**The linear layer.**
A linear layer computes $y = x W^{\top} + b$.
By PyTorch convention the weight $W$ is stored with shape `(out_features, in_features)`: one row per output, which is why the formula has a transpose.
The bias $b$ has shape `(out_features,)` and is added to every output row by broadcasting (topic 03).
`F.linear(x, weight, bias)` computes exactly this, and `bias=None` simply skips the addition.

From topic 06 you know that `@` treats every axis except the last two as batch axes.
So a layer written with `x @ weight.T` already works for an input of shape `(N, in)` and for `(B, T, in)` alike.
No reshaping should appear in a well-written linear layer.

**An optional parameter.**
Many layers have a switch like `bias=False`.
When it is off, the right thing is to set the attribute to `None`.
The attribute still exists, so code that reads `self.bias` keeps working, but `None` is not a parameter, so nothing is registered and the key does not appear in the `state_dict`.
`self.register_parameter("bias", None)` is the explicit spelling of the same thing, and it is what `nn.Linear` itself does.

**The embedding table.**
It is a parameter of shape `(V, D)`, one row of learned numbers per token id.
Looking up a batch of ids is plain integer indexing, as in topic 05, so ids of shape `(B, T)` give vectors of shape `(B, T, D)`.

**A buffer for a causal mask.**
A language model predicts each token from the tokens before it, so position `t` is allowed to look at positions `0` to `t` and nothing after.
A causal mask records that as a square boolean table: row `i`, column `j` is `True` when position `i` may look at position `j`.
That is everything on or below the diagonal, which `torch.tril` builds from a square of ones:

```python
>>> torch.tril(torch.ones(4, 4, dtype=torch.bool))
tensor([[ True, False, False, False],
        [ True,  True, False, False],
        [ True,  True,  True, False],
        [ True,  True,  True,  True]])
```

The mask for a sequence of length `T` is the top-left `T x T` corner of the mask for any longer length.
So real implementations build it once in `__init__`, at the longest length the model will ever see, register it as a buffer, and in `forward` slice off the corner they need.
Basic slicing returns a view (topic 02), so the slice costs nothing, while rebuilding the table at every step would be pure waste.
Topic 13 uses this mask to stop attention from looking into the future.

### 8.2 Inspecting a model

Every module can list what it owns, and every list walks the whole tree, not just the top level:

| Call | Yields |
| --- | --- |
| `model.parameters()` | every parameter tensor |
| `model.named_parameters()` | `(dotted_name, parameter)` pairs |
| `model.buffers()` / `model.named_buffers()` | the same for buffers |
| `model.modules()` / `model.named_modules()` | every submodule, **including the model itself** |
| `model.children()` / `model.named_children()` | only the direct children, one level down |

In `named_modules()` the model itself comes first, under the empty name `""`.
Keep that in mind whenever you loop over it and only want the layers inside.

```python
>>> net = nn.Sequential(nn.Linear(2, 3), nn.ReLU())
>>> [name for name, _ in net.named_modules()]
['', '0', '1']
>>> [name for name, _ in net.named_parameters()]
['0.weight', '0.bias']
```

**Counting parameters.**
"How big is this model?" means how many numbers, not how many tensors.
`p.numel()` is the number of elements in one tensor, so the model's size is the sum of `numel()` over its parameters.
To count only what an optimizer will update, skip the frozen parameters, the ones with `requires_grad=False`.

**Counting bytes.**
Memory depends on the dtype as well as the count.
`t.element_size()` is the number of bytes one element takes: 4 for float32, 2 for bfloat16, 8 for float64.
A model's memory is the elements times the bytes per element, summed over everything it holds.
That includes buffers, because a buffer sits in GPU memory just like a weight does, even though it is never trained.

**Shared tensors.**
Two attributes can point at the very same parameter object (weight tying, in 8.4).
`parameters()` and `named_parameters()` skip duplicates by default, so each tensor is reported once, under the first name it was found at.
That is what you want for counting, and it is also how you can tell that two weights are tied.
`state_dict()` does the opposite and keeps both names, both pointing at the same tensor, because loading needs both names to exist.

### 8.3 Initialization

Before training, every weight needs a starting value, and the choice matters more than it looks.
Start too large, and the numbers grow from layer to layer until they overflow.
Start too small, and they shrink until the signal disappears.

**A one-paragraph refresher on variance.**
The variance of a random number measures how widely it spreads around its average; the standard deviation, `std`, is its square root, in the same units as the number itself.
Two facts are all this section needs.
Multiplying a random number by $c$ multiplies its `std` by $c$, and so its variance by $c^2$.
Adding up independent random numbers adds up their variances.

**The defaults are not what transformers use.**
`nn.Linear` initializes its own weights when you create it, with a scheme called Kaiming uniform that was tuned for networks with ReLU activations.
GPT-2 and most transformers since then use something simpler: a normal distribution with a small `std`, classically `0.02`, truncated so that no value lies further than three standard deviations from zero, and zeros for every bias.
"Truncated" means a value that lands outside the range is drawn again, so the result has no rare large outliers.
If you never initialize explicitly, you are training a differently initialized model than the paper you are copying.

**The `nn.init` functions.**
They all end in an underscore, because they fill the tensor you give them in place.
They also switch off autograd recording internally, so you can call them on a parameter without wrapping anything in `torch.no_grad()`.

| Call | Fills `t` with |
| --- | --- |
| `nn.init.trunc_normal_(t, mean=0.0, std=1.0, a=-2.0, b=2.0)` | normal values, redrawn while outside `[a, b]` |
| `nn.init.normal_(t, mean=0.0, std=1.0)` | plain normal values |
| `nn.init.zeros_(t)` / `nn.init.ones_(t)` | constants |
| `nn.init.xavier_uniform_(t)` / `nn.init.kaiming_normal_(t)` | the classical schemes, for reference |

Read the `trunc_normal_` signature carefully: `a` and `b` are the cut-off values themselves, not a number of standard deviations.
With `std=0.02` and the default `a=-2.0, b=2.0`, nothing is ever cut off, because almost no sample gets anywhere near 2.
Three standard deviations has to be turned into actual numbers before you pass them.

**Initializing a whole tree at once.**
`model.apply(fn)` calls `fn(m)` on every submodule `m`, children first, and on the model itself last.
The usual idiom is a small function that checks `isinstance(m, SomeLayerType)` and initializes `m` when it matches, and ignores everything else.
One `apply` call then reaches every layer of that type, however deeply it is nested.

**Residual scaling.**
This one is worth understanding rather than copying.
In a transformer, every block reads a running vector called the residual stream, computes something, and adds its result back into that stream (topic 14).
Each block adds twice: once from attention and once from the feed-forward network.
So a model with $L$ blocks makes $2L$ additions to the stream.

Now use the variance facts.
If every addition has variance $s$ and they are roughly independent, after $2L$ additions the stream's variance has grown to about $2L \cdot s$.
For a 48-layer model that is almost a hundred times larger than after one layer, and the deeper the model, the worse it gets.

GPT-2's fix is to shrink only the layers that write into the stream: the last projection of attention (`output_proj` in this repository) and the last projection of the feed-forward network (`w2`).
Their `std` is divided by $\sqrt{2L}$, which divides each addition's variance by $2L$, so $2L$ of them add up to about $s$ again, however deep the model is.
Every other layer keeps the base `std`.

**Finding layers by name.**
`apply` hands `fn` the module alone, and a type check cannot tell `output_proj` from `q_proj`: both are `nn.Linear`.
What tells them apart is the name, and `named_modules()` gives you the full dotted path of every submodule.
The layer's own name is the last piece of that path:

```python
>>> net = nn.ModuleDict({"enc": nn.ModuleDict({"head": nn.Linear(2, 2)})})
>>> [name for name, _ in net.named_modules()]
['', 'enc', 'enc.head']
>>> "enc.head".rsplit(".", 1)[-1]
'head'
```

Compare that last piece, not the whole path with `endswith`: a layer called `w2` should match, and one called `lw2` should not.

### 8.4 state_dict

A trained model is, as far as the computer is concerned, nothing but its numbers.
`model.state_dict()` collects them into an ordered dictionary: the keys are the dotted names from earlier, and the values are the parameter and persistent buffer tensors.
Saving a model means saving that dictionary, and loading means pouring a dictionary back into a freshly built model of the same shape.

**The values are references, not copies.**
The tensors in a `state_dict` are the model's own tensors.
If the model trains on, the `state_dict` you took earlier changes along with it.
To keep a real snapshot, copy it: `copy.deepcopy(model.state_dict())`, or `{k: v.clone() for k, v in ...}`.

**Loading.**
`model.load_state_dict(sd)` copies each tensor in `sd` into the model's tensor of the same name.
It does not create anything, so the model has to already exist with the right shapes, which means building a new one with the same constructor arguments first.
By default (`strict=True`) the two sets of names must match exactly, or it raises an error listing what is missing and what is unexpected.
With `strict=False` it loads whatever matches and returns the leftovers as `missing_keys` and `unexpected_keys`, which is how you load a pretrained model into a new one with a different head.

**Saving to a file, or to memory.**
`torch.save(obj, f)` writes any object, and `torch.load(f)` reads it back.
`f` can be a path, or an in-memory file such as `io.BytesIO()`, which behaves like a file opened for reading and writing.
Like a real file, it has a current position.
After `torch.save` writes into it, the position sits at the end, so a `torch.load` straight away reads nothing and fails.
Calling `buf.seek(0)` first rewinds it to the start.

```python
>>> buf = io.BytesIO()
>>> torch.save({"x": torch.arange(3)}, buf)
>>> buf.seek(0)
0
>>> torch.load(buf)
{'x': tensor([0, 1, 2])}
```

`torch.load` only accepts plain data by default: tensors, numbers, strings and the usual containers.
A checkpoint is a Python pickle, and unpickling a file can run arbitrary code, so since PyTorch 2.6 anything more exotic needs an explicit `weights_only=False` and a good reason to trust the file.
A `state_dict` is plain data and always loads.

**Key prefixes.**
Wrapping a model in another object often changes its names.
`DistributedDataParallel`, which trains one model on several GPUs, keeps your model in an attribute called `module`, so every key gains a `module.` prefix.
`torch.compile` does the same with `_orig_mod.`.
A checkpoint saved from a wrapped model then fails to load into an unwrapped one, because none of the names match.
The fix is to rename the keys: remove the prefix from every key that starts with it, and leave every other key as it is.
Python's `str.removeprefix(p)` does exactly that to one string, and returns it unchanged when it does not start with `p`.

**Weight tying.**
A language model has two tables of shape `(V, D)`: the embedding at the start, which turns token ids into vectors, and the output head at the end, which turns the final vectors into one score per vocabulary entry.
The head is a linear layer from `D` to `V`, so its weight has shape `(V, D)` too, by the `(out, in)` convention from 8.1.
Row `i` of the embedding is the vector for token `i`, and row `i` of the head is the direction whose dot product with the final vector gives token `i`'s score.

Weight tying makes those two tables one and the same tensor.
It is not a copy, and it is not an update of one from the other: both modules point at a single `nn.Parameter` object, so a change through either is visible through the other, and the optimizer sees and updates it once.
In plain Python terms, two attributes on two different objects end up holding the same object.

Two reasons to do it.
It saves memory: for a 50k vocabulary and `D = 768` each table is 38M numbers, which in GPT-2 small is about a third of the model.
And it encodes a reasonable belief, that the vector a token reads in as and the direction that predicts it should be related.
Most small and medium models tie; some very large ones do not.

What you assign must be the `nn.Parameter` object itself.
Assigning its `.data`, a plain tensor, to a name that already holds a parameter fails loudly with a `TypeError`.
The quiet mistake is wrapping it again, `nn.Parameter(emb.weight.data)`: the new parameter shares the same memory, but it is a second `Parameter` object.
`parameters()` then reports both, the optimizer keeps two sets of state for them, and the tie survives only until something replaces one of the tensors.

`t.data_ptr()`, the memory address of a tensor's first element, is equal for tied tensors, but it is equal after that mistake too.
The real check is object identity: `head.weight is emb.weight`.

### 8.5 Modes and hooks

**Training mode and evaluation mode.**
Every module has a boolean flag, `training`.
`model.train()` sets it to `True` on every submodule, and `model.eval()` sets it to `False`, recursively, in one call.

The flag itself does nothing.
Each layer decides whether to look at it, and almost none do.
Of the standard layers, only two care: dropout, explained next, and batch normalization, which transformers do not use.

**Dropout** is a regularizer: during training it randomly sets each element of its input to zero with probability $p$, so the network cannot lean too heavily on any single value.
Removing elements makes the output smaller on average, which would mean the network sees bigger numbers at evaluation time than it ever saw while training.
So dropout also multiplies the surviving elements by $1 / (1 - p)$, which keeps the average the same.
With $p = 0.5$, about half the elements become `0` and the rest are doubled:

```python
>>> drop = nn.Dropout(0.5)
>>> drop(torch.ones(8))  # training mode, the default; which ones are zeroed is random
tensor([0., 0., 2., 0., 0., 0., 2., 2.])
>>> _ = drop.eval()
>>> drop(torch.ones(8))
tensor([1., 1., 1., 1., 1., 1., 1., 1.])
```

Because the rescaling already happened during training, evaluation needs no correction at all: dropout in eval mode returns its input unchanged.
This scheme is called inverted dropout.

**`eval()` is not `no_grad()`.**
`model.eval()` only flips the `training` flag.
It does not stop autograd from building a graph, so an evaluation loop that forgets `torch.no_grad()` still stores every intermediate value for a backward pass that never comes, and runs out of memory.
An evaluation loop needs both: `eval()` for the layers' behaviour and `no_grad()` for the memory.

**Hooks.**
A hook is a function you attach to a module so that PyTorch calls it every time the module runs, without you changing the module's code.

| Call | Calls your `fn` |
| --- | --- |
| `m.register_forward_hook(fn)` | after `forward`, as `fn(module, args, output)` |
| `m.register_forward_pre_hook(fn)` | before `forward`, as `fn(module, args)` |
| `m.register_full_backward_hook(fn)` | when the gradients for that module's inputs have been computed |

`args` is the tuple of positional arguments the module was called with, and `output` is what `forward` returned.
Here is a forward hook that prints the shape coming out of one layer:

```python
>>> layer = nn.Linear(3, 5)
>>> handle = layer.register_forward_hook(lambda mod, args, out: print(out.shape))
>>> _ = layer(torch.randn(2, 3))
torch.Size([2, 5])
>>> handle.remove()
>>> _ = layer(torch.randn(2, 3))  # prints nothing now
```

Three things to know about hooks.

Each `register_*` call returns a handle, and `handle.remove()` detaches the hook.
A hook you forget to remove keeps firing on every later forward pass, and it keeps alive everything it refers to, which over a long run is a memory leak.
So a utility that adds hooks should remove them before it returns.

A forward hook that returns anything other than `None` **replaces** the module's output with that value.
That is useful on purpose and a nasty surprise by accident, so a hook that only observes should return nothing.

The hook is called with the module, not with its name.
If you need the name, the hook has to bring it from outside, for example by building one hook function per name inside another function, so that each hook remembers its own name.
Be careful with a `lambda` defined inside a loop: it looks up the loop variable when it runs, not when it was created, so every one of them sees the last value.
A function that takes the name as an argument and returns the hook avoids that.

Hooks are not only for debugging.
Activation checkpointing, quantization observers and interpretability tools all work through them, and the ability to replace an output is what lets interpretability research swap in activations from a different input to see what changes.

## API you need

### Defining

| Call | Result |
| --- | --- |
| `super().__init__()` | the mandatory first line of every module's `__init__` |
| `nn.Parameter(tensor)` | a trained tensor, `requires_grad=True` |
| `self.register_buffer(name, tensor, persistent=True)` | a tensor that travels and is saved but not trained |
| `self.register_parameter(name, None)` | declare an absent optional parameter |
| `nn.ModuleList(list_of_modules)`, `nn.Sequential(*modules)`, `nn.ModuleDict(dict)` | containers that register their contents |
| `F.linear(x, weight, bias=None)` | `x @ weight.T + bias` |
| `torch.empty(shape)` / `torch.randn(shape)` | uninitialized memory / standard normal values |
| `torch.tril(t)` | keep the lower triangle including the diagonal, zero (or `False`) above it |

### Inspecting

| Call | Result |
| --- | --- |
| `model.parameters()` / `named_parameters()` | recursive, each shared tensor once |
| `model.buffers()` / `named_buffers()` | the same for buffers |
| `model.modules()` / `named_modules()` | every submodule, the root first under the name `""` |
| `model.children()` / `named_children()` | direct children only |
| `p.requires_grad` | whether `p` is trainable |
| `t.numel()`, `t.element_size()` | number of elements, bytes per element |
| `t.data_ptr()` | memory address of the first element, equal for tensors sharing memory |
| `a is b` | whether two names hold the very same object, the check for a real tie |

### Initializing

| Call | Result |
| --- | --- |
| `nn.init.trunc_normal_(t, mean, std, a, b)` | in-place truncated normal; `a`, `b` are absolute bounds |
| `nn.init.normal_(t, mean, std)` | in-place normal |
| `nn.init.zeros_(t)` / `nn.init.ones_(t)` | in-place constants |
| `model.apply(fn)` | call `fn` on every submodule, the root last; returns `model` |
| `isinstance(m, nn.Linear)` | pick out one kind of layer |
| `model.named_modules()` | `(dotted_path, module)` for every submodule, the root first as `""` |
| `path.rsplit(".", 1)[-1]` | the last piece of a dotted path, the layer's own name |
| `math.sqrt(x)` / `x ** 0.5` | square root of a Python number |

### Persisting

| Call | Result |
| --- | --- |
| `model.state_dict()` | ordered dict of name to tensor, by reference |
| `model.load_state_dict(sd, strict=True)` | copy values in by name; reports `missing_keys`, `unexpected_keys` |
| `io.BytesIO()` | an in-memory file |
| `torch.save(obj, path_or_file)` | write `obj` |
| `buf.seek(0)` | rewind an in-memory file before reading it |
| `torch.load(path_or_file, weights_only=True)` | read it back |
| `str.removeprefix(p)` | strip `p` if the string starts with it |

### Runtime

| Call | Result |
| --- | --- |
| `model.train()` / `model.eval()` | set the `training` flag on every submodule |
| `model.training` | the current flag |
| `model.to(device_or_dtype)` | move or cast parameters and buffers; for a module, in place |
| `nn.Dropout(p)` | zero with probability `p` and rescale by `1 / (1 - p)` in training, identity in eval |
| `m.register_forward_hook(fn)` | returns a handle; `handle.remove()` detaches |

## Traps

- A tensor assigned as a plain attribute does not move with `.to(device)` and is not in the `state_dict`.
  Constants the module needs belong in a buffer.
- Layers in a plain Python `list` or `dict` are invisible to `parameters()`, so the optimizer never updates them.
- Assigning anything to `self` before `super().__init__()` raises an `AttributeError`.
- `model.forward(x)` instead of `model(x)` skips the hooks.
- `torch.empty` is uninitialized memory, so a parameter left that way can hold `nan` and garbage.
- `nn.init.trunc_normal_` takes absolute bounds; the defaults `-2.0, 2.0` do nothing at `std=0.02`.
- Leaving `nn.Linear`'s default initialization in place means training a differently initialized model than the one you meant.
- `named_modules()` and `modules()` include the model itself, under the name `""`.
- `state_dict()` returns references to the live tensors.
  Copy it if you want a snapshot.
- `torch.load` on a `BytesIO` without `seek(0)` reads from the end and fails.
- A checkpoint from a DDP or `torch.compile` wrapped model has prefixed keys and fails `strict=True`.
- Tying with `head.weight = nn.Parameter(emb.weight.data)` shares memory but creates a second parameter.
  Assign the `Parameter` itself, and check the tie with `is`, not with `data_ptr()`.
- `model.eval()` does not switch off autograd; evaluation also needs `torch.no_grad()`.
- A forward hook that returns a value replaces the module's output.
- A hook handle you never remove keeps firing and keeps memory alive.
- A `lambda` created in a loop sees the loop variable's last value, not the value at the time it was created.

## Ready when

Without looking anything up you can:

- [ ] explain what `nn.Module.__setattr__` does with a parameter, a module, and a plain tensor
- [ ] name the three kinds of tensor attribute and say what each does on `.to(device)` and in the `state_dict`
- [ ] explain why a causal mask should be a buffer and a RoPE cache probably a non-persistent one
- [ ] say why a plain list of layers trains worse without any error, and what to use instead
- [ ] read a dotted parameter name like `blocks.3.attn.q_proj.weight` and say where it lives in the model
- [ ] write a linear layer from scratch with an optional bias that stays out of the `state_dict`
- [ ] initialize every `nn.Linear` in a model with one call, using a truncated normal with the right bounds
- [ ] derive the $1 / \sqrt{2 \cdot n_{\text{layers}}}$ residual scaling from how variances add up, and apply it only to the layers named `output_proj` and `w2`
- [ ] save a model to an in-memory buffer and load it into a new one
- [ ] tie an embedding to an output head and check that the tie is real
- [ ] say exactly what `model.eval()` changes, and what it does not

## Going deeper

- [`nn.Module` documentation](https://pytorch.org/docs/stable/generated/torch.nn.Module.html), which lists every registration and inspection method in one place.
- [Saving and loading models](https://pytorch.org/tutorials/beginner/saving_loading_models.html), the official tutorial, if the `state_dict` section moved too fast.
- [Serialization semantics](https://pytorch.org/docs/stable/notes/serialization.html), for the full `state_dict` contract and the `weights_only` change.
- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), the GPT-2 paper, whose section 2.3 is the one-sentence origin of the residual scaling trick.
- CS336's first assignment uses its own variant of the truncated normal: linear weights get a `std` that depends on the layer's input and output sizes, rather than a fixed `0.02`.
  The mechanics are exactly the ones in 8.3; only the number changes.
