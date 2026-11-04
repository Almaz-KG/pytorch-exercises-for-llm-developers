# Topic 16 - The training loop

> **Exercise** `test_16_training_loop.py`

## Why this topic

Topic 15 produced a model and a loss, and topic 08 showed how a loss turns into a gradient for every weight.
This topic closes the loop: it is what turns those gradients into new weights, over and over, until the loss is low.

It is also where tensor exercises turn into training engineering.
AdamW, a warmup plus cosine schedule, gradient clipping and gradient accumulation are not four independent techniques.
They are the four things that stand between a correct model and a run that actually converges.

Every one of them has a detail that is easy to get wrong and hard to notice, because the run still trains, just worse: the bias correction, the decoupling of weight decay, the place of clipping in the step, the division by the number of micro-batches.
None of these bugs crashes anything.
That is why the exercise compares your optimizer and your clipping against PyTorch's own to six decimal places, rather than just checking that a loss goes down.

The last section is arithmetic: how many floating-point operations a training run costs, how many tokens a model of a given size deserves, and how many tokens one optimizer step consumes.
Those three numbers are what you reason with before you launch anything.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| parameter | a leaf tensor with `requires_grad=True` that the optimizer owns and updates, such as a weight matrix |
| gradient | `p.grad`, a tensor of the same shape as `p`, filled by `loss.backward()` |
| optimizer step | one update of every parameter from its current gradient |
| learning rate, `lr` | the scalar that sets how large a step is |
| EMA | exponential moving average: $\text{avg} = \beta \cdot \text{avg} + (1 - \beta) \cdot \text{new}$, a running mean that forgets old values at rate $\beta$ |
| moment | Adam's per-parameter running statistics: $m$, an EMA of the gradient, and $v$, an EMA of the squared gradient |
| weight decay | shrinking every weight a little toward zero on every step, a regularizer |
| schedule | the learning rate as a function of the step number |
| global gradient norm | the L2 norm of all gradients of the model flattened into one long vector |
| micro-batch | one slice of a batch, run forward and backward on its own so that the whole batch never has to fit in memory at once |
| FLOP | one floating-point operation, a multiply or an add |

### A refresher: where gradients live

Three facts from topic 08 carry this whole topic.

**`backward()` adds, it does not overwrite.**
Every call to `loss.backward()` adds its result into `p.grad`.
Two backward passes without clearing in between leave the sum of both gradients.
That is a bug in a plain loop and the whole mechanism of gradient accumulation in 16.4.

```python
>>> w = torch.tensor([1.0, -2.0], requires_grad=True)
>>> (w * w).sum().backward(); w.grad
tensor([ 2., -4.])
>>> (w * w).sum().backward(); w.grad      # the second call added to the first
tensor([ 4., -8.])
```

**Operations on a tensor that requires grad are recorded.**
Outside `torch.no_grad()`, `p - 0.1` builds a new tensor with a `grad_fn`, which is no longer a leaf, and an in-place `p -= ...` on a leaf that requires grad is refused outright:

```
RuntimeError: a leaf Variable that requires grad is being used in an in-place operation.
```

Inside `with torch.no_grad():`, nothing is recorded, in-place updates of a leaf are allowed, and the tensor stays the same leaf with the same identity.
An optimizer update is exactly that kind of operation: a change to the weights that is not part of the model's computation.

**`zero_grad` clears gradients.**
`opt.zero_grad()` and `model.zero_grad()` both take `set_to_none=True` by default, which replaces every `p.grad` by `None` instead of writing zeros into it.
That is faster and frees memory, and the cost is that any code reading gradients has to skip parameters whose `.grad` is `None`.

### The order of a step

One training step has six phases, and their order is fixed by what each one reads and writes.

| Phase | Reads | Writes |
| --- | --- | --- |
| forward | the parameters, the batch | the output, and the autograd graph behind it |
| loss | the output, the targets | one scalar |
| zero the gradients | - | `p.grad` of every parameter, cleared |
| backward | the graph | `p.grad` of every parameter, accumulated |
| clip | `p.grad` | `p.grad`, rescaled |
| optimizer step | `p.grad`, the optimizer state | the parameters and the optimizer state |

Two orderings matter.
Zeroing must come before backward, because backward accumulates and zeroing after it erases what it just computed.
Clipping must sit strictly between backward and the step: before backward there is nothing to clip, and after the step the gradients have already been consumed.

Zeroing before or after the forward pass makes no difference, because the forward pass does not touch `.grad`.
Putting it right before `backward` keeps the reads and writes of `.grad` together.

### 16.1 Optimization steps by hand

#### SGD

Plain stochastic gradient descent moves every parameter a small step against its gradient:

```math
p \leftarrow p - \operatorname{lr} \cdot \texttt{p.grad}
```

The arrow matters: the update must change the parameter tensor itself, the object the model and the optimizer hold, not compute a new tensor and bind it to a local name.
A new tensor bound to a local variable leaves the model's weight untouched.
The update must also happen under `torch.no_grad()`, for the reasons in the refresher: otherwise it is either refused or recorded in the graph, and a recorded update turns the parameter into a non-leaf and keeps every past step's graph alive.
The test checks both: the values change, and afterwards the parameter still `requires_grad` and is still a leaf.

#### AdamW, the formula

One AdamW step for one parameter tensor $p$, with gradient $g$, moments $m$ and $v$, and step number $t$:

```math
\begin{aligned}
m &= \beta_1 m + (1 - \beta_1)\, g && \text{first moment: an EMA of the gradient} \\
v &= \beta_2 v + (1 - \beta_2)\, g^2 && \text{second moment: an EMA of the squared gradient, elementwise} \\
\hat{m} &= m / (1 - \beta_1^t) && \text{bias correction} \\
\hat{v} &= v / (1 - \beta_2^t) \\
p &= p - \operatorname{lr} \cdot \text{wd} \cdot p && \text{decoupled weight decay, on the } p \text{ from before this step} \\
p &= p - \operatorname{lr} \cdot \hat{m} / \left(\sqrt{\hat{v}} + \varepsilon\right)
\end{aligned}
```

with $t$ counting from **1**, $m$ and $v$ starting as zeros of $p$'s shape, and every operation elementwise.
The defaults are $\beta_1, \beta_2 = 0.9, 0.999$, $\varepsilon = 10^{-8}$, and in this exercise $\text{wd} = 0.01$, which is also `torch.optim.AdamW`'s default.

Five things to understand rather than memorize.

**Why two moments.**
$m$ is a smoothed direction, which damps the noise of stochastic gradients.
$v$ is a smoothed magnitude, and dividing by $\sqrt{\hat{v}}$ makes the step size roughly invariant to the scale of the gradient, per parameter entry.
Every entry gets an individually adapted step size, which is why Adam works on transformers, where gradient scales differ by orders of magnitude between the embeddings and the last layer.

**Why the bias correction.**
$m$ and $v$ start at zero, so early on they are biased toward zero: after one step with $\beta_1 = 0.9$, $m$ is only $0.1 \cdot g$ rather than $g$.
Dividing by $1 - \beta_1^t$ corrects exactly that, and the correction decays to 1 as $t$ grows.
The consequence worth feeling: on step 1, $\hat{m} = g$ and $\hat{v} = g^2$, so the update is $\operatorname{lr} \cdot g / \lvert g \rvert$, a step of size $\operatorname{lr}$ in every entry, whatever the size of the gradient.
Without the correction the first few hundred steps are far too small, which is a slow silent start rather than a crash.
With $t$ starting at 0, $1 - \beta^0 = 0$ and the correction divides by zero.

**Why `eps` is where it is.**
It is added to $\sqrt{\hat{v}}$, after the square root, not inside it.
It only matters for entries whose gradients are tiny, where it keeps the denominator away from zero.
Inside the square root it would act on the scale of $\varepsilon$ instead of $\sqrt{\varepsilon}$, and for gradients around `1e-4` the step comes out about 30% smaller than PyTorch's.

**Why decoupled weight decay is the "W" in AdamW.**
The older approach, L2 regularization, added $\text{wd} \cdot p$ to the gradient before the moments were computed, so the decay then got divided by $\sqrt{\hat{v}}$ along with everything else.
Parameters with large gradients therefore got less decay, which is backwards.
AdamW instead shrinks $p$ by the factor $1 - \operatorname{lr} \cdot \text{wd}$ as a separate step, untouched by the moments, so every weight decays at the same relative rate.
The gradient $g$ that enters the moments is the plain gradient of the loss.
The test for this sets the gradient to zero: the moments and the Adam term are then zero, and the weight must still shrink by exactly $\operatorname{lr} \cdot \text{wd}$.
The decay is scaled by $\operatorname{lr}$, so it also follows the learning rate schedule.

**Why the order of the two $p$ lines matters.**
PyTorch applies the decay to the weights as they were before the step, then subtracts the Adam term.
The Adam term does not depend on $p$, so that is the same as computing both terms from the old $p$ and subtracting them together.
Applying the decay to the already-updated $p$ instead shrinks the Adam term as well, an extra $\operatorname{lr} \cdot \text{wd} \cdot \text{update}$ per step.
With $\operatorname{lr} = 0.1$ and $\text{wd} = 0.01$ that is about `1e-4` per step, a hundred times the `1e-6` tolerance of the comparison against `torch.optim.AdamW`.

#### `adamw_step`: the functional form

The exercise first asks for the formula as a pure function of tensors: it takes `p`, `grad`, `m`, `v` and `t`, and returns the new `p`, `m` and `v`.
"Pure" means that none of the inputs may change: the caller's `m` must still hold the old moment after the call.
Out-of-place arithmetic, `b1 * m + ...` rather than `m.mul_(b1)`, gives that for free, because every operation produces a new tensor.
The tensors here do not require grad, so no `no_grad` is needed.

#### `AdamW`: the same thing as a `torch.optim.Optimizer`

The second form is a real optimizer that a training loop can use like `torch.optim.AdamW`, and the test runs the two side by side on an `nn.Linear` for eight steps.
Three pieces of the base class do most of the work.

**`param_groups`.**
`super().__init__(params, defaults)` takes the parameters and a dict of hyperparameter defaults.
It stores them as `self.param_groups`, a list of dicts, each holding a `"params"` list and its own copy of every hyperparameter.
`model.parameters()` gives a single group; passing a list of dicts gives several, which is how per-group learning rates and weight decays work.
`step` reads the hyperparameters from the group, not from attributes saved in `__init__`, so that a scheduler that writes `group["lr"]` actually takes effect.
`params` is often a generator, which can be iterated only once: pass it straight to `super().__init__`, which turns it into a list.

**`state`.**
`self.state` is a `defaultdict(dict)` keyed by parameter tensor.
`self.state[p]` is an empty dict the first time it is looked up, which is the signal to initialize that parameter's state, for example with `torch.zeros_like(p)`.
Everything in it, tensors and plain numbers alike, is saved by `opt.state_dict()` and restored by `opt.load_state_dict()`.
The step counter belongs there too, one per parameter, because the base class has no notion of a global step.

**`step` and `zero_grad`.**
`zero_grad` is inherited and needs no code.
`step(closure=None)` is yours: it walks every group and every parameter, skips parameters whose `.grad` is `None`, and updates the rest in place.
The in-place part is the difference from `adamw_step`: here the model's own parameter tensors must change, and so must the moments stored in `self.state`.
Decorating `step` with `@torch.no_grad()` keeps the whole update out of the graph.
The optional `closure` is a function that re-runs forward and backward and returns the loss; some optimizers, such as L-BFGS, need to evaluate the loss several times per step.
If one is passed, call it under `torch.enable_grad()`, since the rest of `step` runs under `no_grad`, and return its loss; otherwise return `None`.

A neutral example of the whole interface: SGD with momentum, a different optimizer from AdamW, written against the base class.
It keeps one state tensor per parameter and matches `torch.optim.SGD(lr=0.1, momentum=0.9)` exactly.

```python
class MomentumSGD(torch.optim.Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9):
        super().__init__(params, dict(lr=lr, momentum=momentum))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "buf" not in state:
                    state["buf"] = p.grad.clone()
                else:
                    state["buf"].mul_(group["momentum"]).add_(p.grad)
                p.add_(state["buf"], alpha=-group["lr"])
        return loss
```

```python
>>> opt = MomentumSGD(nn.Linear(3, 2).parameters())
>>> # ... after a few steps:
>>> list(opt.state_dict()["state"][0].keys()), opt.state_dict()["param_groups"]
(['buf'], [{'lr': 0.1, 'momentum': 0.9, 'params': [0, 1]}])
```

#### Weight decay is not applied uniformly in practice

Real configurations exclude biases, normalization weights and often the embeddings from decay.
Shrinking a norm's gain or an embedding row toward zero regularizes nothing and just fights the model.
The mechanism is `param_groups`: one group of matrices with `weight_decay=0.1`, one group of everything else with `weight_decay=0.0`.
The exercise decays every parameter uniformly, including the bias of the `nn.Linear` in the test, because that is what `torch.optim.AdamW` does when given a single group.

### 16.2 The learning rate schedule

The schedule is a pure function from the step number to a learning rate, in three phases:

```math
\begin{aligned}
&\text{step} < \text{warmup\_steps} \text{, linear warmup from 0:} \\
&\qquad \operatorname{lr} = \operatorname{lr}_{\max} \cdot \text{step} / \text{warmup\_steps} \\[1ex]
&\text{warmup\_steps} \le \text{step} \le \text{cosine\_steps} \text{:} \\
&\qquad \operatorname{lr} = \operatorname{lr}_{\min} + 0.5 \cdot \left(1 + \cos(\pi \cdot \text{progress})\right) \cdot (\operatorname{lr}_{\max} - \operatorname{lr}_{\min}) \\[1ex]
&\text{step} > \text{cosine\_steps} \text{, constant tail:} \\
&\qquad \operatorname{lr} = \operatorname{lr}_{\min} \\[2ex]
&\text{progress} = \frac{\text{step} - \text{warmup\_steps}}{\text{cosine\_steps} - \text{warmup\_steps}}
\end{aligned}
```

`cosine_steps` is the **step number at which the cosine ends**, counted from 0 like `step` itself, not the length of the cosine phase.
With `warmup_steps = 4` and `cosine_steps = 20`, the cosine runs over steps 4 to 20, sixteen steps long.

`progress` runs from 0 to 1 across the cosine phase, and the factor $0.5 \cdot (1 + \cos(\pi \cdot \text{progress}))$ runs from 1 down to 0 along half a cosine wave:

| `progress` | 0 | 0.25 | 0.5 | 0.75 | 1 |
| --- | --- | --- | --- | --- | --- |
| $0.5 \cdot (1 + \cos(\pi \cdot \text{progress}))$ | 1 | 0.854 | 0.5 | 0.146 | 0 |

So the learning rate is exactly halfway between `lr_max` and `lr_min` in the middle of the cosine phase, falls fastest there, and flattens at both ends.

Check the boundaries, because the tests do:

- at `step = 0` the warmup gives exactly `0.0`;
- at `step == warmup_steps` the cosine branch takes over with `progress = 0` and gives exactly `lr_max`, which is also where the warmup line was heading, so the curve is continuous;
- at `step == cosine_steps`, `progress = 1` and `cos(pi) == -1.0` exactly, so the learning rate is `lr_min`;
- past `cosine_steps` the formula must not be used: `progress > 1` would send the cosine back up, so that branch returns `lr_min` as is.

Two degenerate configurations are tested as well, and both must give a learning rate in `[0, lr_max]` without crashing:

- `warmup_steps = 0`: the warmup branch never fires, and step 0 already starts the cosine at `lr_max`;
- `warmup_steps == cosine_steps`: the cosine phase is a single step, and `progress` at that step is `0 / 0`, which in plain Python floats raises `ZeroDivisionError`.
  Guard the denominator so that this step gives `lr_max`.

The function returns a Python float and needs no tensors at all; `math.cos` and `math.pi` are enough.

**Why warmup.**
Adam's `v` estimate is built from almost no samples on the first steps, so the adaptive denominator is unreliable, and full-size steps in that regime can throw the model somewhere it never recovers from.
Warmup keeps the steps small until the moment estimates have settled.
A typical warmup is 1% to 5% of the total steps.

**Why cosine.**
Large steps early to make progress, small steps late to settle into a minimum, with a smooth curve and no hyperparameters beyond the endpoints.
`lr_min` is usually around 10% of `lr_max`.

Writing the schedule as a pure function of the step is the cleanest form: no hidden state, trivially testable, and a resumed run gets the right learning rate just by passing the step number.
Applying it is one line per step: write the value into `group["lr"]` for every group of the optimizer before calling `step`.

### 16.3 Gradient clipping

```math
\text{total} = \sqrt{\sum_{\text{every parameter } p} \; \sum_{\text{every entry } k} \texttt{p.grad}[k]^2}
```

```
if total > max_norm:
    multiply every gradient by max_norm / (total + eps)
return total, measured before any rescaling
```

**Global, not per tensor.**
`total` is the L2 norm of all the model's gradients treated as one long vector, so it is the square root of the sum of squares over every entry of every tensor.
Equivalently, it is the norm of the vector of per-tensor norms:

```python
>>> a, b = torch.tensor([3.0, 4.0]), torch.tensor([[0.0, 12.0]])
>>> torch.linalg.vector_norm(torch.cat([a.flatten(), b.flatten()]))
tensor(13.)
>>> torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(a), torch.linalg.vector_norm(b)]))
tensor(13.)
```

Scaling every gradient by the same factor preserves the direction of the full update and only bounds its length.
Clipping each tensor to `max_norm` on its own would scale different tensors by different factors, which changes the direction, and it also bounds the total only to `sqrt(number of tensors) * max_norm`.

**The details the test checks.**

- The rescaling has to happen in place, on the tensors that live in `p.grad`, for example with `g.mul_(scale)`; computing a scaled copy changes nothing the optimizer will see.
- Parameters whose `.grad` is `None` take no part, neither in the norm nor in the scaling.
- The threshold is strict: a norm below `max_norm` leaves every gradient bit for bit as it was.
- `eps` in the denominator makes the clipped norm land just below `max_norm`, never above: a gradient of norm 50 clipped to 1 comes out at norm `0.99999994`.
- The return value is the norm **before** clipping, as a Python float: a 0-dim tensor becomes one with `.item()`.
  It is returned even when nothing was clipped.

`torch.nn.utils.clip_grad_norm_` does the same thing with the same `1e-6` and is what the test compares against, but the exercise asks for it without that function.

**Why clipping exists.**
Rare bad batches.
A single anomalous batch can produce a gradient many times the usual size, and one such step can undo thousands of good ones or produce a NaN that poisons the run.
On a healthy step the norm is below the threshold and nothing happens at all.
`max_norm = 1.0` is the near-universal default in LLM training.

Logging the returned pre-clip norm is one of the best training diagnostics there is: a steadily rising gradient norm often comes before a loss spike.

### 16.4 Gradient accumulation and the loop

#### Accumulation

When the batch you want does not fit in memory, run it as `N` micro-batches, forward and backward one at a time, and step once at the end.
Because `backward` adds into `.grad`, the gradients of all micro-batches pile up by themselves.

**The detail everybody gets wrong once: each micro-batch's loss must be divided by `N`.**
The loss you are targeting is the mean over the whole batch.
For `N` micro-batches of equal size, the mean of the whole batch equals the mean of the micro-batch means:

```math
\text{mean over the batch} = \frac{1}{N} \left(\text{mean}_1 + \text{mean}_2 + \dots + \text{mean}_N\right)
```

and since the gradient is linear, the same holds for gradients.
Accumulating the raw micro-batch losses gives the sum of the means, which is `N` times the full-batch value:

```python
>>> xs = torch.tensor([1., 2., 3., 4., 5., 6.])
>>> xs.mean(), xs[:3].mean() + xs[3:].mean(), (xs[:3].mean() + xs[3:].mean()) / 2
(tensor(3.5000), tensor(7.), tensor(3.5000))
```

Without the division the gradient is `N` times too large, so the effective learning rate is `N` times what you configured.
With Adam the damage is partly hidden, since dividing by $\sqrt{\hat{v}}$ cancels most of a constant factor.
But clipping sees the inflated norm and clips steps it should have left alone, and with SGD the run is simply wrong.
Nothing crashes and the loss still falls, so the bug survives until someone compares against a real large batch, which is exactly what the test does.

Three more details the test relies on:

- Zero the gradients first: the test runs a full-batch backward on the same model just before calling the function, so the parameters arrive with stale gradients in `.grad`.
- The return value is the loss of the whole batch, a Python float: with equal micro-batches it is the mean of the micro-batch losses, which is the sum of the divided losses.
  Collect it with `.item()`; adding up loss tensors would keep every micro-batch's graph alive.
- The loss is `F.mse_loss(model(x), y)`, the mean squared error averaged over every element, as stated in the docstring.

Dividing by `N` is exact only for micro-batches of equal size.
With ragged ones each micro-batch has to be weighted by its share of the total element or token count.

Accumulation gives exactly the gradient of the large batch, but it does not save compute, only peak memory, and it multiplies the time per optimizer step by `N`.

#### The training loop

`train_steps` puts 16.1 to 16.3 together into the loop from "The order of a step", run `steps` times on one fixed batch, with the exercise's own `AdamW` and its own `clip_grad_norm`.

- The optimizer is built **once, before the loop**.
  Building it inside the loop throws away the moments every step, so every step becomes a first step, of size `lr` in every entry and with no averaging at all.
- Clip with `max_norm = 1.0`, the standard value.
- `clip_grad_norm` takes a list of tensors, and `model.parameters()` is a generator: turn it into a list.
- Record each step's loss as a float with `.item()`.
  The test wants one loss per step and checks that the last one is under a fifth of the first.

### 16.5 The arithmetic of training

#### The 6ND rule

Training a model with `N` parameters on `D` tokens costs approximately

```math
\text{FLOPs} = 6 \cdot N \cdot D
```

The forward pass is about $2ND$: each parameter takes part in one multiply-accumulate per token, and the $2$ is the multiply plus the add.
The backward pass is about twice the forward, $4ND$, because for every weight matrix it computes two products of the same size as the forward one: the gradient with respect to the layer's input, to pass further back, and the gradient with respect to the weights.

The rule ignores attention's $T^2$ term, which is a correction rather than the main story below roughly 8k context (topic 14).
The `N` and `D` in it are counts, not this repository's shape letters.

A neutral example: a 125M-parameter model on 2.5B tokens costs $6 \cdot 125 \times 10^6 \cdot 2.5 \times 10^9 = 1.875 \times 10^{18}$ FLOPs.
At an effective $10^{14}$ FLOP/s, which is a realistic fraction of a modern GPU's peak, that is 18750 seconds, about 5.2 hours.

#### Chinchilla

For a fixed training compute budget, the loss is lowest when the model size and the number of tokens grow together, at about **20 tokens per parameter**.
The models of 2020 to 2022 were badly undertrained by that measure: GPT-3 was 175B parameters on 300B tokens, under 2 tokens per parameter.
Chinchilla itself was 70B parameters on 1.4T tokens, exactly 20 per parameter, and beat Gopher, four times its size and trained with the same compute.

**Why nobody trains at Chinchilla ratios any more.**
Chinchilla minimizes training compute and stops there.
If the model will be served to many users, inference cost dominates its lifetime budget, and a smaller model trained far past the compute-optimal point is cheaper overall, even though its training was formally wasteful.
LLaMA made that argument explicitly and trained its 7B model on a trillion tokens, about 140 per parameter.
Ratios of 100 to 1000 are now normal, and "undertrained" in the Chinchilla sense has become a design choice rather than a mistake.

#### Tokens per step

The number of tokens behind one optimizer step is

```math
\text{tokens per step} = \text{batch\_size} \cdot \text{seq\_len} \cdot \text{grad\_accum}
```

because each of the `grad_accum` micro-batches holds `batch_size` sequences of `seq_len` tokens.
For example, 16 sequences of 2048 tokens accumulated over 8 micro-batches is 262144 tokens per step.
This, not the per-device batch size, is the batch size a training recipe means, and it is usually quoted in tokens: large LLM runs use one to four million tokens per step.
Divide the total token budget by it to get the number of optimizer steps, which is what `warmup_steps` and `cosine_steps` are measured in.

#### What training costs in memory

Per parameter, a standard mixed-precision run with Adam holds about 16 bytes before a single activation:

| What | Bytes per parameter |
| --- | --- |
| weights in bf16 | 2 |
| gradients in bf16 | 2 |
| float32 master copy of the weights | 4 |
| Adam's `m` in float32 | 4 |
| Adam's `v` in float32 | 4 |
| total | 16 |

A pure float32 run lands on the same 16: 4 for the weights, 4 for the gradients, 8 for the two moments.
For a 7B model that is 112 GB, of which the optimizer state alone is 56 GB.
So "a 7B model needs about 14 GB" is true for inference in bf16 and badly wrong for training, which is why sharded optimizers such as ZeRO exist, and why topic 18 is a tour of ways to cut each line of this table.

## API you need

### Optimizers

| Call | Result |
| --- | --- |
| `torch.optim.SGD(params, lr, momentum=0)` | the baseline |
| `torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)` | the reference your AdamW must match |
| `class MyOpt(torch.optim.Optimizer)` | subclass; implement `__init__` and `step` |
| `super().__init__(params, defaults)` | registers the parameters and a dict of default hyperparameters |
| `self.param_groups` | list of dicts, each with `"params"` and the hyperparameters |
| `self.state[p]` | per-parameter state dict, empty on first access, saved by `state_dict()` |
| `@torch.no_grad()` on `step` | keeps the update out of the graph |
| `with torch.enable_grad():` | re-enables recording inside a `no_grad` region, for the closure |
| `opt.zero_grad()`, `model.zero_grad()` | clear gradients; `set_to_none=True` by default |
| `opt.state_dict()` / `opt.load_state_dict(sd)` | save and restore the optimizer, for resuming |

### Tensor updates

| Call | Result |
| --- | --- |
| `with torch.no_grad():` | a block where nothing is recorded and leaves may be changed in place |
| `p -= x`, `p.sub_(x)` | in-place subtraction, same tensor object |
| `p.add_(g, alpha=-lr)` | fused in-place `p -= lr * g` |
| `p.mul_(c)` | in-place scaling |
| `m.mul_(b1).add_(g, alpha=1 - b1)` | an EMA update, in place |
| `v.mul_(b2).addcmul_(g, g, value=1 - b2)` | the squared-gradient EMA, in place |
| `p.addcdiv_(a, b, value=-lr)` | fused in-place `p -= lr * a / b` |
| `torch.zeros_like(p)` | zeros of `p`'s shape, dtype and device, for lazy state |
| `t.sqrt()`, `t.pow(2)`, `t ** 2` | elementwise, out of place |

The fused in-place forms are what PyTorch's own optimizers use; plain arithmetic is equally correct and easier to read while learning.

### Schedule, clipping, loss

| Call | Result |
| --- | --- |
| `math.cos(x)`, `math.pi` | the cosine schedule, on plain floats |
| `torch.linalg.vector_norm(t)` | L2 norm of all entries, any shape, as a 0-dim tensor |
| `t.pow(2).sum()` | sum of squares of all entries |
| `torch.stack(list_of_scalars)` | 0-dim tensors into a 1-d tensor |
| `torch.nn.utils.clip_grad_norm_(params, max_norm)` | the reference; clips in place, returns the pre-clip norm as a tensor |
| `t.item()` | a one-element tensor as a Python number, detached from any graph |
| `F.mse_loss(pred, target)` | mean of `(pred - target)²` over all elements |
| `for g in opt.param_groups: g["lr"] = lr` | set the learning rate by hand |

### A worked example on neutral data

Adam's first step, for gradients of three very different sizes:

```python
>>> b1, b2, lr, eps, t = 0.9, 0.999, 0.1, 1e-8, 1
>>> for g in (torch.tensor([0.1]), torch.tensor([100.0]), torch.tensor([1e-3])):
...     m, v = (1 - b1) * g, (1 - b2) * g * g          # one EMA update from zeros
...     m_hat, v_hat = m / (1 - b1**t), v / (1 - b2**t)
...     print(lr * m_hat / (v_hat.sqrt() + eps))
tensor([0.1000])
tensor([0.1000])
tensor([0.1000])
```

After bias correction, the first step has magnitude $\operatorname{lr}$ no matter how large or small the gradient was.

Where the eps goes, for a gradient entry around `1e-4`, with the same one-step $\hat{v} = g^2$:

```python
>>> g = torch.tensor([1e-4]); v_hat = g * g
>>> g / (v_hat.sqrt() + 1e-8), g / (v_hat + 1e-8).sqrt()
(tensor([0.9999]), tensor([0.7071]))
```

## Traps

- `t` starting at 0 divides by zero in the bias correction; it starts at 1.
- Weight decay added into the gradient before the moments is Adam with L2, not AdamW.
- Decay applied to the already-updated `p` instead of the old one is off by about `lr * wd * update` per step, enough to fail the comparison with `torch.optim.AdamW`.
- `eps` inside the square root instead of outside gives visibly smaller steps for small gradients.
- Updating `m` or `v` in place inside `adamw_step` changes the caller's tensors; the functional form must not mutate its inputs.
- In the optimizer class, the opposite: the parameters and the stored moments must be updated in place, or the model never changes.
- Updating parameters outside `torch.no_grad()` either raises or builds a graph and leaks memory.
- `p = p - lr * p.grad` in a loop rebinds a local name and leaves the parameter untouched.
- Iterating the `params` generator before passing it to `super().__init__` leaves the optimizer with an empty parameter list.
- Reading `lr` from `self` instead of from the param group ignores per-group settings and any schedule.
- `p.grad` can be `None`; skip those parameters in the optimizer and in clipping.
- A step counter kept as a single attribute instead of per parameter in `self.state[p]` is not saved by `state_dict()`.
- `cosine_steps` is the step where the cosine ends, not the length of the cosine phase.
- Without the `step > cosine_steps` branch, `progress > 1` sends the learning rate back up.
- `warmup_steps == cosine_steps` makes `progress` a `0 / 0`.
- Clipping tensor by tensor instead of globally changes the update direction.
- Clipping a copy instead of `p.grad` itself, or clipping after `opt.step()`, does nothing.
- Returning the norm after clipping instead of before, or as a tensor instead of a float.
- Forgetting to divide by the number of micro-batches inflates the gradient by that factor.
- Not zeroing gradients before accumulating adds whatever was left in `.grad`.
- `zero_grad` between `backward` and `step` erases the gradients you just computed.
- Constructing the optimizer inside the training loop resets its state every step.
- Collecting losses as tensors instead of `.item()` floats keeps every step's graph alive.
- Resuming a run without the optimizer state restarts the moments from zero, which produces a visible loss bump.

## Ready when

Without looking anything up you can:

- [ ] write the AdamW update from memory, bias correction and decoupled decay included, in the order PyTorch applies them
- [ ] explain what each moment is for and why dividing by $\sqrt{\hat{v}}$ helps
- [ ] say why Adam's first step has size $\operatorname{lr}$ whatever the gradient
- [ ] say why weight decay must be decoupled from the adaptive scaling
- [ ] write a `torch.optim.Optimizer` subclass with `param_groups`, lazily initialized `state` and a closure
- [ ] write the three-phase learning rate schedule and check every boundary, including the degenerate ones
- [ ] implement global gradient clipping and say why it is global
- [ ] state the micro-batch division rule and explain what breaks without it
- [ ] write the six phases of a training step in order and say why each ordering constraint exists
- [ ] give the 6ND rule with its derivation, the Chinchilla ratio with its caveat, and the tokens per optimizer step
- [ ] say how many bytes per parameter training with Adam costs, and where they go

## Going deeper

- [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980), Kingma and Ba 2014, the original, with the derivation of the bias correction.
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), Loshchilov and Hutter, the AdamW paper.
- [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983), Loshchilov and Hutter, where the cosine schedule comes from.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), Hoffmann et al. 2022, the Chinchilla paper.
- [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054), Rajbhandari et al. 2019, on sharding the 16 bytes per parameter across devices.
- The source of [`torch.optim.AdamW`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py) and of [`torch.optim.Adam`](https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py), whose single-tensor path is the thing your implementation has to match.
