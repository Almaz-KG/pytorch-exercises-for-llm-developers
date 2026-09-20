# Topic 10 - Softmax and loss

> **Exercise** `test_10_softmax_and_loss.py`


This is the bottom of the forward pass: the step that turns `(B, T, V)` logits into one number, and the last thing that happens before topic 08's backward pass takes over.

The same softmax also sits in the middle of attention, so everything you learn here about stability applies twice.

Cross-entropy is the loss of every language model, and it is the one place where a subtle numerical mistake produces a model that trains to a slightly worse number instead of crashing.
Feeding probabilities into a function that expects logits, computing `log(softmax(x))` instead of `log_softmax(x)`, or averaging over padded positions each cost real quality and none of them raise.

Two payoffs at the end of this file: the prettiest formula in deep learning, `grad = (p - onehot) / N`, and the single most useful sanity check in LLM training, that the initial loss must be `log(V)`.

## The mental model

### 10.1 Softmax

```
softmax(x)_i = exp(x_i) / sum_j exp(x_j)
```

**Shift invariance.**
`softmax(x + c) == softmax(x)` for any constant `c`, because `exp(x_i + c) = exp(c) exp(x_i)` and the `exp(c)` cancels between numerator and denominator.
This is an identity, not an approximation, and it is the licence to subtract the maximum before exponentiating.

The stable form is therefore:

```
m = x.amax(dim, keepdim=True)
softmax(x) = exp(x - m) / sum(exp(x - m))
```

Every exponent is now at most 0, so the largest term is exactly 1 and nothing overflows.
Without the subtraction, logits around 100 already overflow float16 and logits around 800 overflow float32.

**log_softmax** is not `log(softmax(x))`.
Computing the softmax first pushes small probabilities to zero in floating point, and then the log is `-inf`.
The correct form goes straight to the answer:

```
log_softmax(x) = x - logsumexp(x, dim, keepdim=True)
```

Every quantity stays in log space, so a probability of `1e-40` is represented perfectly well as `-92.1`.
The rule to internalize: **stay in log space until the very last moment**.

**Temperature** divides the logits before the softmax: `softmax(logits / T)`.
As `T -> 0` the largest logit dominates and the distribution approaches the argmax one-hot.
As `T -> inf` all logits are squashed together and the distribution approaches uniform.
`T == 1` is no change.

### 10.2 Cross-entropy

For a target distribution `q` and a predicted `p`, cross-entropy is `-sum_i q_i log p_i`.
When the target is a single correct class, `q` is one-hot and this collapses to `-log p_target`.

Substituting the softmax and simplifying gives the form to implement directly:

```
loss_i = logsumexp(logits_i) - logits_i[target_i]
```

No probabilities, no logs of small numbers, one `logsumexp` and one `gather`.
This is why topics 04 and 05 came first.

**`F.cross_entropy` takes logits.**
It applies `log_softmax` internally.
Passing it a softmax output means it takes the softmax twice, which does not crash, does not look obviously wrong in the loss curve, and costs a visibly worse model.

Its expected shapes are `(N, C)` for the input and `(N,)` for the target, or `(N, C, d1, ...)` for higher-rank inputs where the **class axis is second**, not last.
For a language model with `(B, T, V)` logits the usual move is to flatten to `(B*T, V)` and `(B*T,)` rather than to transpose.

**`ignore_index`** excludes positions from both the numerator and the denominator.
Excluding from the numerator only, which is what a naive masked implementation does, divides by the wrong count and scales the loss.
With 30% padding, a loss computed over `T` instead of over the real token count is 30% too low.

**Label smoothing** replaces the one-hot target with `(1 - s)` on the correct class and `s / V` over all classes.
Expanding `-sum_i q_i log p_i` with that `q` gives

```
loss = (1 - s) * (-log p_target) + (s / V) * sum_i (-log p_i)
```

so it is the ordinary loss blended with the mean negative log probability over the whole vocabulary.

**Perplexity** is `exp(mean loss in nats)`.
It is the effective number of equally likely choices the model is deciding between at each token.
If a model's loss is `log(V)` its perplexity is `V`, meaning it has learned nothing.

**KL divergence** for distillation is

```
KL(teacher || student) = sum_i p_teacher_i * (log p_teacher_i - log p_student_i)
```

Compute both sides with `log_softmax` and take the teacher's probabilities as `exp` of its log-probabilities.
Never form the ratio `p_t / p_s` directly; the difference of logs is the numerically sound spelling of the same thing.
KL is not symmetric, and which argument goes first is a modelling decision.

### 10.3 The gradient

For the mean cross-entropy over `N` positions:

```
dL / d logits = (softmax(logits) - onehot(targets)) / N
```

The derivation composes the softmax Jacobian from topic 08 with the `-log p_target` term, and almost everything cancels.
Read the result structurally.
The gradient on the correct class is `p_target - 1`, negative, pushing that logit up.
The gradient on every other class is `p_j`, positive, pushing them down in proportion to the probability the model wrongly assigned.
A perfectly confident correct prediction gives zero gradient everywhere.

### 10.4 Properties worth feeling

**Shift invariance** can be measured: the maximum absolute difference between `softmax(x)` and `softmax(x + c)` should be at the level of floating point noise, not exactly zero, because the two computations round differently.

**The initial loss is `log(V)`.**
An untrained model with small random weights produces near-uniform logits, so every token has probability `1/V` and the loss is `-log(1/V) = log(V)`.
For GPT-2's 50257-token vocabulary that is about 10.82.
For a 128k vocabulary it is about 11.76.

This is the first thing to check on step zero of any training run.
A much higher initial loss means the initialization is too large or the output layer is badly scaled.
A much lower one means information is leaking from the targets into the inputs, usually an off-by-one in the causal mask or in the input/target shift.
Topic 15 checks this directly on a real model.

## API you need

### Softmax family

| Call | Result |
| --- | --- |
| `F.softmax(x, dim)` / `x.softmax(dim)` | stable softmax |
| `F.log_softmax(x, dim)` | stable log-softmax, not `log(softmax(x))` |
| `torch.logsumexp(x, dim, keepdim=False)` | the denominator in log space |
| `x.amax(dim, keepdim=True)` | the max for manual stabilization |

### Losses

| Call | Result |
| --- | --- |
| `F.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0, reduction="mean")` | takes **logits**, class axis is dim 1 |
| `F.nll_loss(log_probs, targets, ignore_index=-100)` | takes **log-probabilities** |
| `F.kl_div(input_log_probs, target_probs, reduction="batchmean", log_target=False)` | note the argument conventions |
| `F.one_hot(ids, num_classes)` | int64 one-hot, cast it before arithmetic |
| `torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)` | the target logits |

`reduction` is `"mean"`, `"sum"` or `"none"`, and `"none"` returning per-position losses is what you want when you are doing your own masking.

### A worked example on neutral data

```python
>>> logits = torch.tensor([[2.0, 1.0, 0.0]])
>>> targets = torch.tensor([0])
>>> F.cross_entropy(logits, targets)
tensor(0.4076)
>>> torch.logsumexp(logits, -1) - logits.gather(-1, targets[:, None]).squeeze(-1)
tensor([0.4076])

>>> p = logits.softmax(-1)
>>> (p - F.one_hot(targets, 3)) / 1        # the analytic gradient
tensor([[-0.3348,  0.2447,  0.0900]])
>>> logits.requires_grad_(True)
>>> F.cross_entropy(logits, targets).backward(); logits.grad
tensor([[-0.3348,  0.2447,  0.0900]])
```

## Traps

- `F.cross_entropy` takes logits.
  Softmaxing first is a silent quality regression.
- `log(softmax(x))` underflows where `log_softmax(x)` does not.
- `F.cross_entropy` puts the class axis at position 1 for rank > 2 inputs, not last.
  Flatten to `(N, V)` instead of guessing.
- Masking padding out of the numerator but dividing by the full count scales the loss.
- `F.kl_div` wants log-probabilities as input and probabilities as target, and `reduction="mean"` is not the mathematical mean.
- `F.one_hot` returns int64; multiplying it into a float tensor promotes, which is usually harmless but occasionally surprising.
- Temperature belongs on logits.
  Dividing probabilities by `T` and renormalizing is a different function.
- Perplexity is `exp` of the loss only when the loss is in nats.
  Base-2 losses need `2 ** loss`.
- An initial loss far from `log(V)` is a bug report, not noise.

## Ready when

Without looking anything up you can:

- [ ] write the stable softmax and say why subtracting the max is exact
- [ ] write cross-entropy as `logsumexp(logits) - logits[target]` and explain where it comes from
- [ ] state the gradient of mean CE with respect to the logits and interpret each term
- [ ] say what `log(50257)` is to one decimal place and why you would ever care
- [ ] explain what goes wrong when padding is excluded from the sum but not from the divisor
- [ ] describe what label smoothing changes about the target distribution

## Context



**Why cross-entropy is one fused kernel.**
The gradient `(p - onehot) / N` never needs the softmax Jacobian to be materialized, so `F.cross_entropy` computes the loss and stashes exactly what the backward needs, rather than running a softmax layer and a negative-log-likelihood layer back to back.
That fusion is also what makes chunked cross-entropy possible: since the loss decomposes per position and the gradient has a closed form, the `(B*T, V)` logit tensor can be processed a block at a time and never held whole.
Topic 18 builds that, and at a 128k vocabulary it is the difference between fitting and not fitting.

**Where label smoothing is and is not used.**
It stops the model driving the correct logit to infinity, which improves calibration and often generalization, and it has been standard in machine translation and image classification for years.
Modern LLM pretraining mostly does not use it, on the argument that at web scale the data already contains enough genuine ambiguity, and that a deliberately blurred target wastes capacity on a distribution nobody wants.
It reappears in fine-tuning, where overfitting a small dataset is the actual risk.

**Why `F.kl_div`'s signature is like that.**
It was written to pair with `log_softmax` output on the input side, which is the numerically sound thing to pass, and the asymmetry between "input in log space, target not" catches everyone once.
`reduction="mean"` averaging over every element, rather than per example, is a documented historical wart.
Computing KL by hand, as this exercise asks, avoids both.

## Going deeper

- [`F.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html), whose doc page spells out the shape conventions and the `ignore_index` semantics.
- [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567), section 7, where label smoothing was introduced.
- Anything by Andrej Karpathy on "the initial loss should be `-log(1/V)`", which is the single highest-value debugging habit in this file.
