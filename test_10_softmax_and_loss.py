"""Topic 10 - softmax, log_softmax, cross-entropy.

Three things to know about softmax:
  1. It is invariant to a shift: softmax(x) == softmax(x - c). Hence the subtraction of the maximum.
  2. log_softmax is NOT computed as log(softmax(x)) - that loses precision. The right way: x - logsumexp(x).
  3. Cross-entropy takes LOGITS, not probabilities. Feed it a softmax and you get a quietly wrong loss.

And the main practical fact: the gradient of cross-entropy with respect to the logits is (p - onehot) / N.
There is probably no prettier formula in deep learning.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 10.1 softmax by hand
# ==========================================================================


def softmax_manual(x: Tensor, dim: int) -> Tensor:
    """A numerically stable softmax along the axis dim, without torch.softmax."""
    e = (x - x.max(dim=dim, keepdim=True).values).exp()
    return e / e.sum(dim=dim, keepdim=True)


def log_softmax_manual(x: Tensor, dim: int) -> Tensor:
    """A numerically stable log_softmax: x - logsumexp(x, dim, keepdim=True). Without torch.log_softmax."""
    lse = x.exp().sum(dim=dim, keepdim=True).log()
    return x - lse


def temperature_softmax(logits: Tensor, temperature: float) -> Tensor:
    """softmax(logits / T). As T -> 0 the distribution tends to the argmax, at large T to the uniform one."""
    assert temperature != 0.0, "Temperature cannot be equal to 0"

    x = logits / temperature
    e = (x - x.max(dim=-1, keepdim=True).values).exp()
    return e / e.sum(dim=-1, keepdim=True)


# ==========================================================================
# 10.2 Cross-entropy
# ==========================================================================


def cross_entropy_from_logits(logits: Tensor, targets: Tensor) -> Tensor:
    """logits: (N, V), targets: (N,) int64 -> a scalar: the batch average of -log p(target).

    Without F.cross_entropy and without F.log_softmax: only logsumexp and gather.
    The formula: loss_i = logsumexp(logits_i) - logits_i[target_i].
    """

    tl = logits.gather(-1, targets[:, None]).squeeze(-1)
    return (torch.logsumexp(logits, dim=-1) - tl).mean()


def sequence_cross_entropy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """logits: (B, T, V), targets: (B, T) -> a scalar.

    The positions where targets == ignore_index enter neither the sum nor the denominator.
    This is the real loss of a language model with padding.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    valid = flat_targets != ignore_index
    safe_targets = flat_targets.clamp(min=0)
    per_token = torch.logsumexp(flat_logits, dim=-1) - flat_logits.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return (per_token * valid).sum() / valid.sum()


def label_smoothing_loss(logits: Tensor, targets: Tensor, smoothing: float) -> Tensor:
    """Cross-entropy with label smoothing: the target distribution is
    (1 - smoothing) on the correct class plus smoothing / V spread uniformly over all the classes.

    Return the batch-averaged loss.
    """
    logp = log_softmax_manual(logits, dim=-1)
    nll = -logp.gather(
        -1,
        targets[:, None],
    ).squeeze(-1)
    smooth = -logp.mean(dim=-1)
    return ((1 - smoothing) * nll + smoothing * smooth).mean()


def perplexity(loss: Tensor) -> Tensor:
    """The perplexity from the mean loss in nats. An understanding check: it is simply exp(loss)."""
    return loss.exp()


def kl_divergence(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    """KL(teacher || student), averaged over the batch. The inputs are logits (N, V).

    This is the distillation loss. It has to be computed through the log_softmax of both,
    without dividing probabilities explicitly.
    """
    tl = log_softmax_manual(teacher_logits, dim=-1)
    sl = log_softmax_manual(student_logits, dim=-1)
    return (tl.exp() * (tl - sl)).sum(dim=-1).mean()


# ==========================================================================
# 10.3 Gradients
# ==========================================================================


def cross_entropy_grad(logits: Tensor, targets: Tensor) -> Tensor:
    """The gradient of the mean CE loss with respect to the logits, derived by hand (without autograd).

    The answer: (softmax(logits) - onehot(targets)) / N.
    """
    one_hot = torch.zeros_like(logits, device=logits.device).scatter_(-1, targets[:, None], 1.0)
    return (softmax_manual(logits, dim=-1) - one_hot) / logits.shape[0]


# ==========================================================================
# 10.4 Properties worth feeling
# ==========================================================================


def shift_invariance_gap(x: Tensor, shift: float) -> Tensor:
    """The maximum absolute difference between softmax(x) and softmax(x + shift).

    ~0 is expected: that is the shift invariance.
    """
    sx = softmax_manual(x + shift, dim=-1)
    x = softmax_manual(x, dim=-1)
    return torch.abs(x - sx).max()


def uniform_loss(vocab_size: int) -> float:
    """What is the loss of an untrained model that outputs a uniform distribution?

    The answer has to be known by heart: it is log(V). For V=50257 that is ~10.8 -
    the first thing checked on the first step of training.
    """
    return math.log(vocab_size)


# ==========================================================================
# TESTS
# ==========================================================================


def test_softmax_manual():
    x = torch.randn(3, 5, 7)
    for dim in (0, 1, 2, -1):
        torch.testing.assert_close(softmax_manual(x, dim), x.softmax(dim=dim))


def test_softmax_manual_stability():
    x = torch.tensor([[1000.0, 1000.0, 1000.0]])
    out = softmax_manual(x, dim=-1)
    assert torch.isfinite(out).all(), "an exp overflow - the maximum was not subtracted"
    torch.testing.assert_close(out, torch.full((1, 3), 1 / 3))

    y = torch.tensor([[0.0, float("-inf")]])
    torch.testing.assert_close(softmax_manual(y, dim=-1), torch.tensor([[1.0, 0.0]]))


def test_log_softmax_manual():
    x = torch.randn(4, 6) * 20
    torch.testing.assert_close(log_softmax_manual(x, -1), F.log_softmax(x, dim=-1))
    torch.testing.assert_close(log_softmax_manual(x, -1).exp().sum(-1), torch.ones(4), atol=1e-5, rtol=0)


def test_temperature_softmax():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    cold = temperature_softmax(logits, 0.1)
    hot = temperature_softmax(logits, 100.0)
    assert cold.argmax(-1).item() == 2
    assert cold.max().item() > 0.99, "a low temperature must almost collapse into the argmax"
    assert (hot - 1 / 3).abs().max().item() < 0.01, "a high temperature must flatten it out"


def test_cross_entropy_from_logits():
    logits = torch.randn(8, 12)
    targets = torch.randint(0, 12, (8,))
    torch.testing.assert_close(cross_entropy_from_logits(logits, targets), F.cross_entropy(logits, targets))


def test_cross_entropy_is_stable():
    logits = torch.tensor([[0.0, 1e4], [1e4, 0.0]])
    targets = torch.tensor([1, 0])
    out = cross_entropy_from_logits(logits, targets)
    assert torch.isfinite(out), "large logits must not produce inf/nan"
    torch.testing.assert_close(out, torch.tensor(0.0), atol=1e-5, rtol=0)


def test_sequence_cross_entropy():
    logits = torch.randn(2, 5, 9)
    targets = torch.randint(0, 9, (2, 5))
    targets[0, 3:] = -100
    expected = F.cross_entropy(logits.reshape(-1, 9), targets.reshape(-1), ignore_index=-100)
    torch.testing.assert_close(sequence_cross_entropy(logits, targets), expected)


def test_label_smoothing_loss():
    logits = torch.randn(6, 10)
    targets = torch.randint(0, 10, (6,))
    torch.testing.assert_close(
        label_smoothing_loss(logits, targets, 0.1),
        F.cross_entropy(logits, targets, label_smoothing=0.1),
    )
    torch.testing.assert_close(label_smoothing_loss(logits, targets, 0.0), F.cross_entropy(logits, targets))


def test_perplexity():
    torch.testing.assert_close(perplexity(torch.tensor(0.0)), torch.tensor(1.0))
    torch.testing.assert_close(perplexity(torch.log(torch.tensor(50257.0))), torch.tensor(50257.0))


def test_kl_divergence():
    s = torch.randn(4, 7)
    t = torch.randn(4, 7)
    expected = F.kl_div(F.log_softmax(s, dim=-1), F.log_softmax(t, dim=-1), log_target=True, reduction="batchmean")
    torch.testing.assert_close(kl_divergence(s, t), expected)
    torch.testing.assert_close(kl_divergence(t, t), torch.tensor(0.0), atol=1e-6, rtol=0)


def test_cross_entropy_grad():
    logits = torch.randn(5, 9, requires_grad=True)
    targets = torch.randint(0, 9, (5,))
    F.cross_entropy(logits, targets).backward() # type: ignore
    torch.testing.assert_close(cross_entropy_grad(logits.detach(), targets), logits.grad, atol=1e-6, rtol=1e-5)


def test_shift_invariance_gap():
    x = torch.randn(3, 8)
    assert shift_invariance_gap(x, 100.0).item() < 1e-6


def test_uniform_loss():
    assert abs(uniform_loss(50257) - math.log(50257)) < 1e-9
    assert abs(uniform_loss(2) - math.log(2)) < 1e-9


@pytest.mark.depends_on("cross_entropy_from_logits", "uniform_loss")
@pytest.mark.parametrize("vocab", [4, 100, 50257])
def test_untrained_model_loss_matches_log_vocab(vocab: int):
    """The real check: a model with zero logits must have a loss of exactly log(V)."""
    logits = torch.zeros(16, vocab)
    targets = torch.randint(0, vocab, (16,))
    torch.testing.assert_close(
        cross_entropy_from_logits(logits, targets),
        torch.tensor(uniform_loss(vocab)),
        atol=1e-5,
        rtol=1e-5,
    )
