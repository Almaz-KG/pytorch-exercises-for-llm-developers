# Topic 16 - The training loop

> **Exercise** `test_16_training_loop.py`

The last step of a training iteration: topic 15 produced a loss, topic 08 turned it into gradients, and this is what turns gradients into new weights.

This is also where tensor exercises turn into training engineering.
AdamW, a warmup plus cosine schedule, gradient clipping and gradient accumulation are not four independent techniques; they are the four things that stand between a correct model and a run that actually converges.

Every one of them has a detail that is easy to get wrong and hard to notice: the bias correction, the decoupling of weight decay, the order of clipping relative to the optimizer step, and the division by the number of micro-batches.

## The mental model

### The order of a step

```
logits = model(x)                     forward
loss   = criterion(logits, y)
opt.zero_grad(set_to_none=True)       clear the previous step's gradients
loss.backward()                       accumulate
clip_grad_norm_(params, max_norm)     clip, AFTER backward and BEFORE step
opt.step()                            update
```

Two orderings matter: `zero_grad` before `backward`, because gradients accumulate, and clipping strictly between `backward` and `step`, because clipping reads `.grad` and the step consumes it.

`set_to_none=True` is the default in modern PyTorch and is faster, because it frees the gradient buffers instead of writing zeros over them.
The cost is that `p.grad` can be `None`, which any code reading gradients has to handle.

### 16.1 SGD and AdamW

**SGD** is `p -= lr * p.grad`, and it must happen **under `torch.no_grad()`**.
Without that the subtraction is recorded in the autograd graph, the parameter stops being a leaf, and memory grows every step.
Using the in-place `p.add_(...)` inside `no_grad` also avoids reallocating the parameter, which keeps the optimizer's references valid, per topic 01.

**AdamW**, one step, per parameter tensor:

```
m = b1 * m + (1 - b1) * g                  first moment, an EMA of the gradient
v = b2 * v + (1 - b2) * g^2                second moment, an EMA of the squared gradient
m_hat = m / (1 - b1^t)                     bias correction
v_hat = v / (1 - b2^t)
p = p - lr * m_hat / (sqrt(v_hat) + eps)   the Adam update
p = p - lr * wd * p                        decoupled weight decay
```

with `t` counting from **1**.

Four things to understand rather than memorize.

**Why two moments.**
`m` is a smoothed direction, which damps the noise of stochastic gradients.
`v` is a smoothed magnitude, and dividing by `sqrt(v_hat)` makes the step size roughly invariant to the scale of the gradient, per parameter.
Every parameter gets an individually adapted step size, which is why Adam works on transformers where the gradient scale varies enormously between the embeddings and the last layer.

**Why the bias correction.**
`m` and `v` start at zero, so early on they are biased toward zero: after one step with `b1 = 0.9`, `m` is only `0.1 * g` rather than `g`.
Dividing by `1 - b1^t` corrects exactly that, and the correction decays to 1 as `t` grows.
Without it the first few hundred steps are far too small, which is a slow silent start rather than a crash.

**Why `eps` is where it is.**
`sqrt(v_hat) + eps`, added *after* the square root, not inside it.
Its role is purely to avoid dividing by zero for a parameter whose gradient has been zero throughout.

**Why decoupled weight decay is the "W" in AdamW.**
The old approach, L2 regularization, added `wd * p` to the gradient *before* the moments were computed, so the decay then got divided by `sqrt(v_hat)` along with everything else.
Parameters with large gradients therefore got *less* decay, which is backwards.
AdamW applies `p -= lr * wd * p` as a separate multiplicative shrink, untouched by the adaptive scaling, so every parameter decays at the same relative rate.

The decay is scaled by `lr`, so it follows the learning rate schedule.

**Writing it as a `torch.optim.Optimizer`.**

```python
class MyOpt(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, ...):
        defaults = dict(lr=lr, ...)
        super().__init__(params, defaults)

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
                state = self.state[p]          # a defaultdict(dict), persisted in state_dict
                ...
        return loss
```

`self.param_groups` is a list of dicts, each with a `"params"` list and a copy of the hyperparameters, which is the mechanism behind per-group learning rates and per-group weight decay.
`self.state[p]` is where the moments live, and because the base class knows about it, `optimizer.state_dict()` saves and restores them.
Lazily initializing `m` and `v` on the first step with `torch.zeros_like(p)` is the standard pattern.

### 16.2 The learning rate schedule

```
step < warmup:                lr = lr_max * step / warmup                      linear warmup
warmup <= step <= cosine_end: lr = lr_min + 0.5*(1 + cos(pi * progress)) * (lr_max - lr_min)
step > cosine_end:            lr = lr_min                                      constant tail
```

with `progress = (step - warmup) / (cosine_end - warmup)`.

**Why warmup.**
Adam's `v` estimate is computed from almost no samples on the first few steps, so the adaptive denominator is unreliable and full-size steps in that regime can throw the model somewhere it never recovers from.
Warmup keeps the steps small until the moment estimates have settled.
A typical warmup is 1% to 5% of total steps.

**Why cosine.**
Large steps early to explore, small steps late to settle, with a smooth derivative and no hyperparameters to tune beyond the endpoints.
`lr_min` is usually around 10% of `lr_max`.

The schedule is a pure function of the step number, which is the cleanest way to write it: no hidden state, trivially testable, and resumable from a checkpoint by passing the step number in.

Check the boundaries: at `step == warmup` the two branches must agree on `lr_max`, and at `step == cosine_end` the cosine must reach exactly `lr_min`, since `cos(pi) == -1`.

### 16.3 Gradient clipping

```
total = sqrt(sum over all parameters of sum(g^2))
if total > max_norm:
    scale every gradient by max_norm / (total + eps)
```

The whole model's gradients are treated as one vector, so clipping preserves the *direction* of the update and only bounds its length.
Clipping each tensor separately would change the direction, which is a different and worse operation.

`max_norm = 1.0` is the near-universal default in LLM training.

Clipping exists because of rare bad batches.
A single anomalous example can produce a gradient many times the usual magnitude, and one such step can undo thousands of good ones or produce a NaN that poisons the run.
On a healthy step the norm is below the threshold and nothing happens at all.

Returning the pre-clip norm is standard and useful: logging it is one of the best training diagnostics there is, because a rising gradient norm precedes a loss spike.

### 16.4 Gradient accumulation

When the batch you want does not fit in memory, run it as `N` micro-batches and accumulate gradients before stepping.

**The detail everybody gets wrong once:** each micro-batch's loss must be divided by `N`.

`backward()` sums gradients across micro-batches, but the loss you are targeting is a *mean* over the full batch.
The sum of `N` means is `N` times the mean of the whole, so without the division the gradient is `N` times too large and the effective learning rate is `N` times what you configured.
The model may still train, worse, and the bug is invisible in the loss curve because the *logged* loss is fine.

```
opt.zero_grad()
for xb, yb in micro_batches:
    (criterion(model(xb), yb) / N).backward()
opt.step()
```

Dividing by `N` is exactly right only when the micro-batches are of equal size; with ragged ones you would weight by token count.

Gradient accumulation is mathematically identical to a large batch but does not reduce total compute, only peak memory.
It also multiplies wall-clock per step by `N`, which is the trade.

**Tokens per optimizer step** is `batch_size * seq_len * grad_accum`, and that is the number that actually matters for a training run.

### 16.5 The arithmetic of training

**The 6ND rule.**
Training a model with `N` parameters on `D` tokens costs approximately

```
FLOPs = 6 * N * D
```

The forward pass is about `2ND`, because each parameter participates in one multiply-accumulate per token and the `2` is the multiply plus the add.
The backward pass is about twice the forward, `4ND`, because it computes both the gradient with respect to the inputs and the gradient with respect to the weights, each costing about as much as the forward.

This ignores attention's quadratic term, which is a correction rather than the main story below roughly 8k context, per topic 14.

**Chinchilla.**
The compute-optimal ratio is about **20 tokens per parameter**.
For a fixed compute budget, `N` and `D` should scale together, and the models of 2020 to 2022 were badly undertrained: GPT-3 was 175B parameters on 300B tokens, under 2 tokens per parameter.
Chinchilla itself was 70B parameters on 1.4T tokens and beat a model two and a half times its size.

## API you need

### Optimizers

| Call | Result |
| --- | --- |
| `torch.optim.SGD(params, lr, momentum=0)` | the baseline |
| `torch.optim.AdamW(params, lr, betas, eps, weight_decay)` | the reference to match |
| `class MyOpt(torch.optim.Optimizer)` with `__init__(params, defaults)` and `step(closure=None)` | the interface |
| `self.param_groups` | list of dicts with `"params"` and the hyperparameters |
| `self.state[p]` | per-parameter state dict, saved by `state_dict()` |
| `opt.zero_grad(set_to_none=True)` | clear gradients |
| `opt.state_dict()` / `opt.load_state_dict(sd)` | for resuming |
| `@torch.no_grad()` on `step` | keep the update out of the graph |

### Tensor updates

| Call | Result |
| --- | --- |
| `p.add_(g, alpha=-lr)` | fused in-place `p -= lr * g` |
| `m.mul_(b1).add_(g, alpha=1 - b1)` | the EMA update, in place |
| `v.addcmul_(g, g, value=1 - b2)` | fused `v += (1-b2) * g * g` |
| `p.addcdiv_(m_hat, denom, value=-lr)` | fused `p -= lr * m_hat / denom` |
| `torch.zeros_like(p)` | lazy state initialization |

The fused forms are what the real implementation uses; the plain arithmetic is equally correct and easier to read while learning.

### Schedule and clipping

| Call | Result |
| --- | --- |
| `math.cos`, `math.pi` | the cosine schedule |
| `torch.nn.utils.clip_grad_norm_(params, max_norm)` | the reference, returns the pre-clip norm |
| `torch.linalg.vector_norm(g)` | per-tensor norms to combine |
| `g.mul_(scale)` | apply the clip in place |
| `for g in opt.param_groups: g["lr"] = new_lr` | set the learning rate by hand |

### A worked example on neutral data

```python
>>> p = torch.tensor([1.0]); g = torch.tensor([0.1])
>>> m = v = torch.zeros(1); b1, b2, lr, eps = 0.9, 0.999, 0.1, 1e-8
>>> m = b1*m + (1-b1)*g; v = b2*v + (1-b2)*g*g
>>> t = 1
>>> m_hat, v_hat = m/(1-b1**t), v/(1-b2**t)
>>> lr * m_hat / (v_hat.sqrt() + eps)        # the first step is ~lr, regardless of |g|
tensor([0.1000])
```

That last line is the thing to feel about Adam: after bias correction, the first step has magnitude `lr` no matter how large or small the gradient was.

## Traps

- Dividing by `1 - b1**t` with `t` starting at 0 gives a division by zero.
  `t` starts at 1.
- Weight decay added into the gradient is Adam, not AdamW, and is measurably worse.
- `eps` inside the square root instead of outside changes the numbers enough to fail a comparison against `torch.optim.AdamW`.
- Updating parameters outside `torch.no_grad()` builds a graph and leaks memory.
- Clipping after `opt.step()` does nothing at all.
- Clipping tensor by tensor instead of globally changes the update direction.
- Forgetting to divide by the number of micro-batches inflates the effective learning rate by that factor.
- `zero_grad` after `backward` erases the gradients you just computed.
- `set_to_none=True` means `p.grad` may be `None`; skip those parameters in a custom optimizer.
- Resuming a run without the optimizer state restarts the moments from zero, which produces a visible loss bump.
- The cosine branch must be clamped past `cosine_steps`, or `progress > 1` sends the learning rate back up.

## Ready when

Without looking anything up you can:

- [ ] write the AdamW update from memory, bias correction and decoupled decay included
- [ ] explain what each moment is for and why dividing by `sqrt(v_hat)` helps
- [ ] say why weight decay must be decoupled from the adaptive scaling
- [ ] write the three-phase learning rate schedule and check both boundaries
- [ ] implement global gradient clipping and say why it is global
- [ ] state the micro-batch division rule and explain what breaks without it
- [ ] give the 6ND rule with its derivation and the Chinchilla ratio with its caveat

## Context



**Why nobody trains at Chinchilla ratios any more.**
Chinchilla optimizes *training* compute and stops there.
If the model will be served to many users, inference cost dominates the lifetime budget, and a smaller model trained far past the compute-optimal point is cheaper overall even though its training was formally wasteful.
LLaMA made that argument explicitly and trained 7B on a trillion tokens, about 140 per parameter.
Ratios of 100 to 1000 are now normal, and "undertrained" in the Chinchilla sense has become a design choice rather than a mistake.

**Weight decay is not applied uniformly in practice.**
Real configurations exclude biases, normalization weights and often the embeddings from decay, on the grounds that shrinking a norm's gain or an embedding row toward zero regularizes nothing and just fights the model.
The mechanism is `param_groups`: one group with `weight_decay=0.01` for the matrices, one with `weight_decay=0.0` for everything else.
The exercise's uniform version is simpler and is what the tests check.

**What the optimizer state costs.**
Adam keeps two float32 moments per parameter, so 8 bytes per parameter on top of the weights and gradients.
For a 7B model that is 56 GB of optimizer state alone, which is why sharded optimizers such as ZeRO exist and why the memory table in topic 15 is dominated by lines that are not the weights.

## Going deeper

- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101), Loshchilov and Hutter, the AdamW paper.
- [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556), the Chinchilla paper.
- The source of [`torch.optim.AdamW`](https://pytorch.org/docs/stable/_modules/torch/optim/adamw.html), which is short and is the thing your implementation has to match.
