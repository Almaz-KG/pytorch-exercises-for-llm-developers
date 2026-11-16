# Topic 09 - Softmax and loss

> **Exercise** `test_09_softmax_and_loss.py`

## Why this topic

A language model reads some text and, at every position, has to guess the next token.
It does not answer with a single word.
It answers with one number per entry of its vocabulary, `V` numbers in all, and a bigger number means "I think this one is more likely".
Those raw numbers are called **logits**.

Logits are awkward to work with directly.
They can be negative, they can be huge, and they do not add up to anything in particular.
So this topic does two things with them.

First, **softmax** turns a row of logits into probabilities: every value between 0 and 1, and the whole row adding up to 1.
Now "how likely is the word `cat`?" has an answer.

Second, **cross-entropy** grades the guess.
We know which token actually came next, so we look up the probability the model gave it and turn that into a single number, the **loss**: small when the model was confident and right, large when it was confident and wrong.
That number is where the backward pass starts (topic 15), and what the optimizer in topic 16 tries to push down.

Both steps look like one-line formulas, and on paper they are.
In a computer they are where numbers overflow to `inf`, quietly round to `0`, or turn into `nan`.
The nasty part is that the most common mistakes here do not crash.
Feeding probabilities where logits are expected, taking `log(softmax(x))`, or averaging over padding all give a loss that looks fine and a model that ends up slightly worse.

The same softmax also sits in the middle of attention (topic 13), so everything about stability you learn here pays off twice.
And the file ends with two rewards: the prettiest formula in deep learning, $\text{gradient} = (p - \text{onehot}) / N$, and the most useful one-second sanity check in LLM training, that an untrained model's loss is $\log V$.

## The mental model

### A few words you will meet

| Word | Meaning here |
| --- | --- |
| logits | the raw scores the model outputs, one per vocabulary entry, any real number |
| probability distribution | a row of numbers, each in $[0, 1]$, that add up to 1 |
| target | the id of the token that actually came next, an `int64` in $[0, V)$ |
| loss | one number that says how wrong the model is; training makes it smaller |
| nats | the unit of a loss computed with the natural log $\ln$; using $\log_2$ would give bits |

Shapes follow the alphabet from the theory index: a batch of `N` rows of `V` logits is `(N, V)`, and a batch of sequences is `(B, T, V)`.

### A two-minute refresher on `exp` and `log`

Everything in this topic is built from two functions, so it is worth remembering how they behave.

$\exp(x)$ is $e \approx 2.718$ raised to the power $x$.
It is always positive, even for very negative $x$: $e^{-5} \approx 0.0067$.
It keeps order: a bigger $x$ always gives a bigger $e^{x}$.
And it turns addition into multiplication: $e^{a+b} = e^{a} e^{b}$.

$\log x$ (always the natural log in this repository) undoes $\exp$: $\log(e^{x}) = x$.
It turns multiplication back into addition: $\log(a b) = \log a + \log b$, and $\log(a / b) = \log a - \log b$.
$\log 1 = 0$, the log of anything between 0 and 1 is negative, and as $x$ approaches 0, $\log x$ heads to minus infinity.

**Where the computer gives up.**
A float32 can hold numbers up to about $3.4 \times 10^{38}$.
$e^{88}$ is about $1.65 \times 10^{38}$ and still fits; $e^{89}$ does not, and becomes `inf`.
On the small side, $e^{-104}$ is too tiny to represent and becomes exactly `0`.
So $\exp$ of numbers a little outside $[-100, 88]$ breaks, and once you have an `inf`, the next division gives `inf / inf = nan`:

```python
>>> torch.exp(torch.tensor([88.0, 89.0, -104.0]))
tensor([1.6516e+38,        inf, 0.0000e+00])
>>> torch.exp(torch.tensor(1000.0)) / torch.exp(torch.tensor(1000.0))
tensor(nan)
```

In float16, which is common in training, the upper limit is only about $e^{11}$.
Logits of 20 or 30 are perfectly normal in a real model, so this is not an edge case.

You already met the cure in topic 04: `logsumexp`.
This topic is mostly about using it everywhere.

### 9.1 Softmax

**The formula.**
For a row of logits $x_1, \dots, x_V$:

```math
\mathrm{softmax}(x)_i = \frac{e^{x_i}}{e^{x_1} + e^{x_2} + \dots + e^{x_V}} = \frac{e^{x_i}}{\sum_{j=1}^{V} e^{x_j}}
```

Read it as two steps.
`exp` makes every score positive while keeping their order.
Dividing by the total makes the row add up to 1.

Here it is on a three-word vocabulary with logits `[2, 1, 0]`:

```
exp:          [7.389, 2.718, 1.000]      total = 11.107
softmax:      [0.665, 0.245, 0.090]      adds up to 1
```

Notice what `exp` does to the gaps.
The logits are 1 apart, but the first probability is $e \approx 2.7$ times the second, and the second is 2.7 times the third.
A difference of 1 in logits always means a factor of $e$ in probability, wherever on the number line it happens.
Only the differences between logits matter, never their absolute level.

**Shift invariance.**
That last sentence is exact, and it is the most important property of softmax.
Add the same constant $c$ to every logit: each $e^{x_i + c}$ becomes $e^{c} \cdot e^{x_i}$, the total gains the same factor $e^{c}$, and it cancels out of the fraction.

```math
\mathrm{softmax}(x + c) = \mathrm{softmax}(x) \qquad \text{for any number } c
```

So `[2, 1, 0]`, `[102, 101, 100]` and `[-3, -4, -5]` all give the same three probabilities.
This is an identity, not an approximation.

**The stable softmax.**
Shift invariance is a license to pick whatever $c$ is most convenient.
The convenient choice is to subtract the largest logit in the row.
After that, the largest value is exactly 0 and everything else is negative, so:

- the biggest `exp` is $e^{0} = 1$, and nothing can overflow;
- the total is at least 1, so the division never divides by zero.

Small terms may still round down to 0, but that is harmless: a probability of `1e-50` that becomes `0` was not going to change anything.
The softmax you write in this topic must do this, and the test feeds it logits of 1000 to check.

**Which axis.**
Softmax runs along one axis, `dim`, and every row along that axis is normalized on its own.
For logits of shape `(B, T, V)` that axis is the last one, but the function in 9.1 has to work for any `dim`, including `0`.
Both the maximum and the total are reductions along `dim` that must broadcast back against the input, which is exactly the `keepdim=True` habit from topic 04.
Note that `x.max(dim)` returns a pair of values and indices, while `x.amax(dim)` returns just the values.

**Masked entries.**
Attention (topic 13) hides some positions by setting their logit to `-inf`.
`exp(-inf)` is exactly 0, so a masked entry gets probability 0 and the rest of the row still adds up to 1:

```python
>>> torch.tensor([0.0, float("-inf")]).softmax(-1)
tensor([1., 0.])
```

A stable softmax handles this for free, as long as the row has at least one real value.
If every entry in a row is `-inf`, the maximum is `-inf` too, `-inf - (-inf)` is `nan`, and the whole row becomes `nan`.
That cannot happen in this topic, but remember it for topic 13.

**log_softmax.**
Cross-entropy needs the log of a probability, not the probability.
The obvious way, `log(softmax(x))`, is broken.
Look at logits `[0, -200]`: the second probability is $e^{-200} \approx 10^{-87}$, which float32 rounds to exactly `0`, and `log(0)` is `-inf`.

```python
>>> x = torch.tensor([0.0, -200.0])
>>> x.softmax(-1).log()
tensor([0., -inf])
>>> F.log_softmax(x, -1)
tensor([   0., -200.])
```

The true answer is about `-200`, a perfectly ordinary number.
The information was destroyed by rounding the probability before taking its log.

The fix comes from writing out the log of the softmax formula with the rules from the refresher:

```math
\begin{aligned}
\log \mathrm{softmax}(x)_i &= \log \frac{e^{x_i}}{\sum_j e^{x_j}} \\
&= x_i - \log \sum_j e^{x_j} \\
&= x_i - \mathrm{logsumexp}(x)
\end{aligned}
```

`log` and `exp` cancelled on top, and the bottom is exactly `logsumexp`, which topic 04 taught you to compute stably.
No probability is ever formed, so nothing is ever rounded to 0.

The rule to take away from this whole topic: **stay in log space until the very last moment.**
A probability of `1e-40` is hopeless in float32, but its log, `-92.1`, is an ordinary number.

**Temperature.**
Before the softmax, you can divide the logits by a positive number $T$, the temperature: $\mathrm{softmax}(\text{logits} / T)$.
Dividing by $T$ stretches or squeezes the gaps between logits, and the gaps are all softmax cares about.

Same logits `[2, 1, 0]` as before:

| $T$ | $\mathrm{softmax}(\text{logits} / T)$ | what happened |
| --- | --- | --- |
| 0.5 | `[0.867, 0.117, 0.016]` | gaps doubled, the winner takes almost everything |
| 1 | `[0.665, 0.245, 0.090]` | unchanged |
| 2 | `[0.507, 0.307, 0.186]` | gaps halved, flatter |
| 10 | `[0.367, 0.332, 0.301]` | almost uniform |

As $T$ goes towards 0, the distribution collapses onto the largest logit, like an argmax.
As $T$ grows, it flattens towards uniform, $1/V$ everywhere.
This is the knob that topic 17 turns when it samples text: low temperature for safe, repetitive output, high for creative, chaotic output.
In this topic the temperature softmax runs over the last axis, the vocabulary.

### 9.2 Cross-entropy

**Surprise.**
Say the model gave the correct next token probability $p$.
How should we grade that?
Cross-entropy uses $-\log p$, which you can read as "how surprised the model was by the right answer":

| $p$ given to the right token | loss $-\log p$ |
| --- | --- |
| 1.0 | 0 |
| 0.9 | 0.105 |
| 0.5 | 0.693 |
| 0.1 | 2.303 |
| 0.01 | 4.605 |
| 0.001 | 6.908 |
| 0 | infinite |

Certain and right costs nothing.
Each time the probability drops by a factor of 10, the loss grows by the same 2.3.
And giving the right answer probability 0 is infinitely bad, which is why the model learns never to rule anything out entirely.

The loss of a batch is the average of this number over all `N` positions in it.
That average, not the sum, is the convention, so that the loss does not change just because the batch got bigger.

**Where the name comes from.**
In general, cross-entropy compares a target distribution $q$ with a predicted one $p$:

```math
\mathrm{cross\_entropy}(q, p) = -\sum_i q_i \log p_i
```

For next-token prediction, the target puts all its weight on the one correct token: $q$ is 1 at the target and 0 everywhere else.
That kind of vector is called **one-hot**.
All terms but one vanish, and what is left is $-\log p_{\text{target}}$.
Label smoothing, below, is what you get when $q$ is not one-hot.

**The formula you actually compute.**
$-\log p_{\text{target}}$ is the negative of one entry of `log_softmax`.
Plug in the `log_softmax` formula from 9.1:

```math
\begin{aligned}
\text{loss}_i &= -\log \mathrm{softmax}(\text{logits}_i)[\text{target}_i] \\
&= \mathrm{logsumexp}(\text{logits}_i) - \text{logits}_i[\text{target}_i]
\end{aligned}
```

That is the form the exercise asks for: one `logsumexp` per row and one lookup of the target's logit per row, then the average over rows.
No probabilities and no logs of small numbers anywhere, so logits of `1e4` are as easy as logits of `1`.

Check it on the three-word example, with the first word as the target:

```
logsumexp([2, 1, 0]) = log(11.107) = 2.4076
logits[target]       = 2
loss                 = 0.4076            and indeed -log(0.665) = 0.4076
```

Looking up "the target's logit in each row" is the unsqueeze, gather, squeeze pattern from topic 05.
This is why topics 04 and 05 came first.

**`F.cross_entropy` takes logits.**
PyTorch's own function runs `log_softmax` internally.
If you pass it probabilities, it applies softmax a second time on top of your softmax.
Nothing crashes, the loss curve looks plausible, and the model trains to a noticeably worse result.

It also has its own shape convention.
The input is `(N, C)` with targets `(N,)`, where `C` is the number of classes, or for higher-rank inputs `(N, C, d1, ...)` with the **class axis second**, not last.
A language model's logits are `(B, T, V)` with the class axis last, so the usual move is to flatten them into `(B*T, V)` and the targets into `(B*T,)` with `reshape` (topic 02), rather than to transpose.

**Padding and `ignore_index`.**
Sentences in a batch have different lengths, but a tensor has to be a rectangle.
So short sequences are padded at the end with filler positions, and the targets at those positions are set to a special value that is not a real token id.
PyTorch's convention is `-100`, called the `ignore_index`.

A padded position must count for nothing: it must not appear in the sum of losses, and it must not appear in the count you divide by.
Forgetting the second half is the classic bug.
Say a batch has two real tokens with loss 2.0 each, and two padding positions:

```
correct:   (2.0 + 2.0) / 2 = 2.0      divide by the number of real tokens
wrong:     (2.0 + 2.0) / 4 = 1.0      divide by every position, padding included
```

The wrong version reports a loss that is too low by exactly the padding fraction.
It does not crash, and since the padding fraction changes from batch to batch, it also makes losses from different batches incomparable.

Two practical details for doing this by hand.
A boolean mask `targets != ignore_index` marks the real positions, and `mask.sum()` counts them.
Multiplying a float tensor by a boolean one treats `True` as 1 and `False` as 0.

The other detail bites everyone once.
`gather` needs every index to be a valid position, and `-100` is not one:

```python
>>> torch.randn(2, 3).gather(-1, torch.tensor([[0], [-100]]))
IndexError: index -100 is out of bounds for dimension 1 with size 3
```

The loss at a padded position is going to be thrown away anyway, so it does not matter which logit you read there.
What matters is that the index is some valid id before the lookup happens; `torch.where` or `clamp` will put one in.

**Label smoothing.**
With a one-hot target, the loss only reaches 0 when the correct probability is exactly 1, which needs the correct logit to be infinitely larger than all others.
The model keeps pushing in that direction forever and becomes overconfident.

Label smoothing softens the target.
With a smoothing amount $s$, the target distribution becomes $(1 - s)$ times the one-hot, plus $s / V$ spread over **every** class, the correct one included.
With $V = 4$, $s = 0.1$ and the target being the first class:

```
one-hot target:    [1,     0,     0,     0    ]
smoothed target:   [0.925, 0.025, 0.025, 0.025]     0.9 + 0.1/4 on the target, 0.1/4 elsewhere
```

Putting that $q$ into $-\sum_i q_i \log p_i$ and grouping the terms gives:

```math
\text{loss} = (1 - s) \cdot (-\log p_{\text{target}}) + \frac{s}{V} \sum_i (-\log p_i)
```

The first part is the ordinary cross-entropy, scaled down.
In the second part, $\frac{1}{V} \sum_i$ is just an average over the vocabulary, so it is $s$ times the mean of $-\log p_i$ over all classes.
Every $\log p_i$ there should come from a stable `log_softmax`, for the reasons in 9.1.
At $s = 0$ the formula is exactly the ordinary loss again.
This matches `F.cross_entropy(..., label_smoothing=s)`, which also spreads $s / V$ onto the correct class.

Label smoothing has been standard in image classification and machine translation for years, because it improves how well a model's confidence matches its accuracy.
Modern LLM pretraining mostly does not use it: at web scale the data already contains plenty of genuine ambiguity, and a deliberately blurred target wastes effort on a distribution nobody wants.
It comes back in fine-tuning on small datasets, where overconfidence is the real risk.

**Perplexity.**
A loss of 2.3 nats is hard to picture.
Perplexity converts it into something you can: $\text{perplexity} = \exp(\text{mean loss})$.

It is the number of equally likely options the model is effectively choosing between at each step.
A model that spreads its bets evenly over 8 tokens has loss $\log 8$ and perplexity 8, like rolling an 8-sided die.
A perfect model has loss 0 and perplexity 1: no doubt at all.
A model with loss $\log V$ has perplexity $V$: it is guessing blindly over the whole vocabulary.

This only works because the loss is in nats, matching `exp`.
A loss measured in bits would need $2^{\text{loss}}$ instead.

**KL divergence and distillation.**
Distillation trains a small student model to imitate a big teacher model.
Instead of a one-hot target, the student is graded against the teacher's whole probability distribution, which carries more information: "the next word is probably `cat`, maybe `dog`, certainly not `the`".

The measure of how far the student is from the teacher is the Kullback-Leibler divergence:

```math
\mathrm{KL}(\text{teacher} \parallel \text{student}) = \sum_i p_{\text{teacher},i} \cdot (\log p_{\text{teacher},i} - \log p_{\text{student},i})
```

Read it as "how much extra surprise do I get if I use the student's probabilities, when the data really follows the teacher's?".
It is always at least 0, and exactly 0 when the two distributions are equal.
It is cross-entropy minus the teacher's own entropy, so it measures only the part of the surprise that is the student's fault.

Three things to get right:

- **Stay in log space.** Get $\log p_{\text{teacher}}$ and $\log p_{\text{student}}$ with `log_softmax`, subtract them, and get the teacher's probabilities as `exp` of its log-probabilities. Never form the ratio $p_{\text{teacher}} / p_{\text{student}}$: dividing two tiny probabilities is exactly the rounding problem from 9.1.
- **The order matters.** $\mathrm{KL}(a \parallel b)$ is not $\mathrm{KL}(b \parallel a)$. With `p = [0.7, 0.2, 0.1]` and `q = [0.5, 0.3, 0.2]`, one direction is `0.085` and the other `0.092`. The weights in front always come from the first argument, here the teacher.
- **What "averaged over the batch" means.** Sum over the vocabulary axis, giving one KL per row, then average over the `N` rows. This is what PyTorch calls `reduction="batchmean"`.

`F.kl_div` itself is famously easy to misuse.
It takes the **student's log-probabilities first** and the teacher's distribution second, which is the reverse of how $\mathrm{KL}(\text{teacher} \parallel \text{student})$ is written.
By default it expects the second argument as plain probabilities, unless you pass `log_target=True`.
And its `reduction="mean"` averages over every element, `N * V` of them, which is not the mathematical KL, a documented historical wart.
Computing it by hand, as the exercise asks, sidesteps all three.

### 9.3 The gradient

The derivative of softmax on its own is a `V x V` table, the Jacobian that topic 15 builds.
For cross-entropy, almost all of it cancels, and the result is short enough to remember for life.

Take one row and the loss from 9.2, $\text{loss} = \mathrm{logsumexp}(x) - x_{\text{target}}$.
Ask how it changes when one logit $x_j$ changes, one piece at a time:

- the derivative of $\mathrm{logsumexp}(x)$ with respect to $x_j$ is $e^{x_j} / \sum_k e^{x_k}$, which is just $p_j$, the softmax probability;
- the derivative of $x_{\text{target}}$ with respect to $x_j$ is 1 if $j$ is the target, and 0 otherwise. Over all $j$ that is the one-hot vector of the target.

So for one row, the gradient is $p - \text{onehot}(\text{target})$.
The batch loss is the average over $N$ rows, so each row's share is divided by $N$:

```math
\frac{\partial \text{Loss}}{\partial \text{logits}} = \frac{\mathrm{softmax}(\text{logits}) - \text{onehot}(\text{targets})}{N}
```

Read it entry by entry:

- **On the correct token** the gradient is $p_{\text{target}} - 1$, which is negative. Gradient descent steps against the gradient, so that logit goes **up**.
- **On every wrong token** the gradient is $p_j$, which is positive, so those logits go **down**, each in proportion to how much probability the model wrongly gave it.
- **A perfect prediction**, $p_{\text{target}} = 1$, makes every entry zero. Nothing left to learn.

Each row's gradient also adds up to zero, $1 - 1$, which is shift invariance again: pushing all logits up together would change nothing, so the gradient never asks for it.

Back to the three-word example, target the first word:

```
p                     = [ 0.665,  0.245,  0.090]
onehot                = [ 1,      0,      0    ]
p - onehot (N = 1)    = [-0.335,  0.245,  0.090]
```

PyTorch computes the same gradient automatically: mark the logits with `requires_grad=True`, call `.backward()` on the loss, and read `.grad`.
How that works is topic 15; here it only serves as a check.

```python
>>> logits = torch.tensor([[2.0, 1.0, 0.0]], requires_grad=True)
>>> F.cross_entropy(logits, torch.tensor([0])).backward()
>>> logits.grad
tensor([[-0.3348,  0.2447,  0.0900]])
```

Autograd agrees.

To build the one-hot part in code, `F.one_hot(ids, num_classes)` turns a tensor of ids into rows of 0s and 1s.
It returns `int64`, so cast it to the logits' dtype before mixing it into float arithmetic, and pass `num_classes` explicitly: without it, PyTorch guesses `V` from the largest id it happens to see.

**Why this formula matters beyond the exercise.**
`F.cross_entropy` never builds the `V x V` softmax Jacobian.
It computes the loss, keeps the probabilities, and on the backward pass subtracts the one-hot: one fused operation instead of a softmax layer followed by a separate loss layer.
The same closed form makes chunked cross-entropy possible: the loss splits cleanly by position, so the `(B*T, V)` logit tensor can be processed a block of rows at a time and never held whole.
At a 128k vocabulary that is the difference between fitting in memory and not, and topic 18 builds it.

### 9.4 Properties worth feeling

**Shift invariance, measured.**
On paper, $\mathrm{softmax}(x)$ and $\mathrm{softmax}(x + c)$ are identical.
In float32 they are computed through different intermediate numbers, so they round differently, and the largest difference between them is tiny but usually not exactly zero:

```math
\max \lvert \mathrm{softmax}(x) - \mathrm{softmax}(x + 100) \rvert \approx 6 \times 10^{-7} \qquad \text{for a random } (3, 8)\ x
```

Measuring "how far apart are two tensors" as the largest absolute difference is the same idea `assert_close` uses: subtract, take the absolute value, take the maximum.
Seeing a number at the level of float32 noise, around `1e-7`, rather than something like `0.01`, is the whole point.

**An untrained model's loss is $\log V$.**
A freshly initialized model has small random weights, so its logits are all close to each other.
By shift invariance, close logits mean nearly equal probabilities: about $1/V$ for every token.
The right token therefore gets about $1/V$, and the loss is $-\log(1/V) = \log V$.

| $V$ | $\log V$ | example |
| --- | --- | --- |
| 2 | 0.693 | a coin flip |
| 100 | 4.605 | |
| 32 000 | 10.37 | Llama 2 |
| 50 257 | 10.82 | GPT-2 |
| 128 256 | 11.76 | Llama 3 |

This is the first number to look at on step zero of any training run, and it is a free bug detector:

- **Much higher** than $\log V$: the logits start too spread out, so the initialization or the output layer is scaled too large (topic 08's 8.3).
- **Much lower**: the model is seeing the answers. Usually the causal mask lets a position look at the future, or the inputs and targets are not shifted by one.

Topic 14 checks this on a real model.

`uniform_loss` returns a plain Python `float`, and the test compares it to 9 decimal places.
`math.log` works in double precision and gets there.
A float32 tensor does not: `torch.log(torch.tensor(50257.0))` is off by about `3e-7`.

## API you need

### Softmax family

| Call | Result |
| --- | --- |
| `torch.exp(x)` / `x.exp()`, `torch.log(x)` / `x.log()` | elementwise `exp` and natural log |
| `x.amax(dim, keepdim=True)` | the maximum along `dim`, values only |
| `x.max(dim, keepdim=True)` | a pair `(values, indices)`; use `.values` |
| `x.sum(dim, keepdim=True)` | the total along `dim` |
| `torch.logsumexp(x, dim, keepdim=False)` | $\log \sum e^{x}$ along `dim`, computed stably |
| `F.softmax(x, dim)` / `x.softmax(dim)` | PyTorch's stable softmax, for checking yours |
| `F.log_softmax(x, dim)` | PyTorch's stable log-softmax, not `log(softmax(x))` |

### Losses

| Call | Result |
| --- | --- |
| `F.cross_entropy(logits, targets, ignore_index=-100, label_smoothing=0.0, reduction="mean")` | takes **logits**; the class axis is dim 1 |
| `F.nll_loss(log_probs, targets, ignore_index=-100)` | takes **log-probabilities**, not logits |
| `F.kl_div(student_log_probs, teacher, reduction="batchmean", log_target=False)` | student first, in log space |
| `F.one_hot(ids, num_classes)` | rows of 0s and 1s, dtype `int64` |
| `x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)` | one value per row, the pattern from topic 05 |

`reduction` is `"mean"`, `"sum"` or `"none"`.
`"none"` returns one loss per position, which is what you want when you are doing your own masking.

### Shapes, masks and dtypes

| Call | Result |
| --- | --- |
| `x.reshape(-1, V)` / `t.reshape(-1)` | flatten `(B, T, V)` to `(N, V)` and `(B, T)` to `(N,)` |
| `targets != ignore_index` | boolean mask of the real positions |
| `mask.sum()` | the number of `True` entries, as `int64` |
| `torch.where(cond, a, b)` | `a` where `cond` is true, `b` elsewhere |
| `t.clamp(min=..., max=...)` | limit values to a range |
| `t.to(dtype)` / `t.float()` | cast, for example a one-hot before arithmetic |
| `x.abs()`, `x.max()` | absolute value; the largest element of the whole tensor |
| `math.log(v)` | natural log of a Python number, in double precision |

### A worked example on neutral data

Every piece of this topic on one row of four logits:

```python
>>> x = torch.tensor([3.0, 1.0, 0.0, -2.0])
>>> torch.logsumexp(x, -1)
tensor(3.1755)
>>> x - torch.logsumexp(x, -1)          # log-probabilities
tensor([-0.1755, -2.1755, -3.1755, -5.1755])
>>> F.softmax(x, -1)                    # probabilities, adding up to 1
tensor([0.8390, 0.1135, 0.0418, 0.0057])
>>> F.softmax(x + 50, -1)               # shifted: same probabilities
tensor([0.8390, 0.1135, 0.0418, 0.0057])
>>> F.one_hot(torch.tensor([1, 3]), 4)
tensor([[0, 1, 0, 0],
        [0, 0, 0, 1]])
>>> math.log(4)                          # the loss of a uniform guess over 4 classes
1.3862943611198906
```

## Traps

- `F.cross_entropy` takes logits.
  Applying softmax first trains a worse model without any error.
- `exp` of a large logit overflows to `inf`, and `inf / inf` is `nan`.
  Subtract the maximum first.
- `log(softmax(x))` rounds small probabilities to `0` and returns `-inf`.
  Use `x - logsumexp(x)`.
- The maximum and the total must be taken with `keepdim=True`, or they will not broadcast back along the right axis when `dim` is not the last one.
- `x.max(dim)` returns a `(values, indices)` pair, not a tensor.
- A row that is `-inf` everywhere softmaxes to `nan`.
- `F.cross_entropy` puts the class axis at position 1 for inputs of rank above 2.
  Flatten to `(N, V)` instead of guessing.
- Excluding padding from the sum but not from the divisor makes the loss too small by the padding fraction.
- `gather` with `-100` as an index raises; padded positions need a valid placeholder index before the lookup.
- In label smoothing, $s / V$ goes to every class, the correct one included.
- `F.kl_div` takes the student's log-probabilities first and the teacher second, and its `reduction="mean"` is not the mathematical KL.
- Never divide probabilities to get KL; subtract log-probabilities.
- `F.one_hot` returns `int64`; cast it before arithmetic, and pass `num_classes`.
- The gradient of the mean loss has a $/ N$; the gradient of the summed loss does not.
- Temperature divides the logits, not the probabilities.
- Perplexity is $\exp(\text{loss})$ only when the loss is in nats.
- `uniform_loss` must be a Python `float` from `math.log`, not a float32 tensor.
- An initial loss far from $\log V$ is a bug report, not noise.

## Ready when

Without looking anything up you can:

- [ ] say in one sentence what softmax does to a row of logits, and what `exp` and the division each contribute
- [ ] explain why subtracting the maximum changes nothing mathematically and fixes overflow
- [ ] show on a two-element example why `log(softmax(x))` fails where `x - logsumexp(x)` does not
- [ ] predict what a low and a high temperature do to `softmax([2, 1, 0])`
- [ ] derive $\text{loss} = \mathrm{logsumexp}(\text{logits}) - \text{logits}[\text{target}]$ from $-\log p_{\text{target}}$
- [ ] explain what goes wrong when padding is excluded from the sum but not from the divisor, and why `-100` cannot go into `gather`
- [ ] write down the smoothed target distribution for $V = 4$, $s = 0.1$
- [ ] say what perplexity 8 means in plain words
- [ ] state which argument of $\mathrm{KL}(\text{teacher} \parallel \text{student})$ provides the weights, and why the ratio of probabilities is never formed
- [ ] state the gradient of mean cross-entropy with respect to the logits, derive it in two lines, and say which way each logit moves
- [ ] say what $\log 50257$ is to one decimal place, and what a much higher or a much lower initial loss tells you

## Going deeper

- [`F.cross_entropy`](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html), whose doc page spells out the shape conventions, `ignore_index` and `label_smoothing`.
- [`F.kl_div`](https://pytorch.org/docs/stable/generated/torch.nn.functional.kl_div.html), for the argument order and the `reduction` warning in PyTorch's own words.
- [Rethinking the Inception Architecture](https://arxiv.org/abs/1512.00567), section 7, where label smoothing was introduced.
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531), the paper behind KL-based distillation, which also uses temperature on both teacher and student.
- Andrej Karpathy's [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), whose "verify loss @ init" step is the $\log V$ check from 9.4 and the single highest-value debugging habit in this file.
