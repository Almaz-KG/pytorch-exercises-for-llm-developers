"""Topic 18 (hard) - memory and numerics: online softmax, flash-attention, checkpointing.

Every task in this file is about one thing: how to compute the same result without allocating
a huge intermediate tensor.

  - the T x T attention matrix does not fit in memory -> online softmax and blockwise attention;
  - the logits (N, 50257) do not fit -> cross-entropy in chunks;
  - the activations of all the layers do not fit -> gradient checkpointing;
  - fp16 overflows -> loss scaling.

This is already the level of CS336 assignment 2. If it passes, you are on first-name terms with transformers.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

# ==========================================================================
# 18.1 Online softmax
# ==========================================================================


def online_softmax_update(m_prev: Tensor, l_prev: Tensor, scores_block: Tensor) -> tuple[Tensor, Tensor]:
    """The incremental recomputation of the softmax statistics when a new block of logits is added.

    Input: m_prev (..., 1) - the current maximum, l_prev (..., 1) - the current sum of exp
    relative to THAT maximum, scores_block (..., Sb) - the new logits.

    Return (m_new, l_new), where
        m_new = max(m_prev, max(scores_block))
        l_new = l_prev * exp(m_prev - m_new) + sum(exp(scores_block - m_new))

    This is the core of flash-attention: the sum of exponentials is updated without storing all the logits.
    """
    raise NotImplementedError


def chunked_softmax(x: Tensor, chunk: int) -> Tensor:
    """A softmax along the last axis computed in two passes over blocks of size chunk:
    first the online statistics (m, l), then the probabilities themselves.

    The result is obliged to match an ordinary softmax down to the last bit of float32 precision.
    """
    raise NotImplementedError


# ==========================================================================
# 18.2 Flash-attention (simplified, forward only)
# ==========================================================================


def flash_attention_forward(q: Tensor, k: Tensor, v: Tensor, block_size: int = 32, causal: bool = False) -> Tensor:
    """Blockwise attention without materializing the (T, S) matrix.

    q: (B, H, T, Dh), k, v: (B, H, S, Dh) -> (B, H, T, Dh).

    The scheme (flash-attention v2):
      for every block of queries we keep a running (m, l, accumulated output);
      we walk over the blocks of keys, compute the local logits for each,
      update m and l by the online rule, correct the accumulated output
      by the factor exp(m_old - m_new), and divide by l at the end.

    Fully masked blocks (when causal=True) may be skipped -
      that also saves you from a nan of the form exp(-inf - (-inf)).

    The sizes are NOT obliged to be divisible by block_size.
    """
    raise NotImplementedError


# ==========================================================================
# 18.3 Cross-entropy without materializing the logits
# ==========================================================================


def chunked_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor, chunk: int) -> Tensor:
    """hidden: (N, D), weight: (V, D) - the weight of lm_head, targets: (N,).

    Compute the mean CE WITHOUT creating the logit tensor (N, V) in full:
    walk in chunks over the tokens, and for every chunk compute its own logits and its contribution to the loss.

    The gradients must flow both into hidden and into weight.
    For 8k tokens and a vocabulary of 128k such a logit tensor weighs 4 GB - hence the task.
    """
    raise NotImplementedError


def chunked_logsumexp(x: Tensor, chunk: int) -> Tensor:
    """A logsumexp along the last axis in blocks, through the online update. (N, V) -> (N,)."""
    raise NotImplementedError


# ==========================================================================
# 18.4 Gradient checkpointing
# ==========================================================================


def checkpointed_sequential(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool) -> Tensor:
    """Run x through the list of blocks. With use_checkpoint=True every block is wrapped
    in torch.utils.checkpoint.checkpoint (use_reentrant=False), that is, its activations
    are not saved but recomputed on the backward pass.

    The trade-off: roughly +30% time, but the memory for the activations drops from O(L) to O(sqrt(L))
    or even O(1) under a naive scheme.
    """
    raise NotImplementedError


def count_saved_tensors(blocks: nn.ModuleList, x: Tensor, use_checkpoint: bool) -> int:
    """How many tensors the autograd graph saved for the backward pass. This is a direct measurement
    of what checkpointing buys.

    Hint: the context torch.autograd.graph.saved_tensors_hooks(pack, unpack) calls
    pack on every saved tensor. We count the pack calls during the forward pass.
    """
    raise NotImplementedError


# ==========================================================================
# 18.5 Mixed precision
# ==========================================================================


def scale_and_check(grads: list[Tensor], scale: float) -> tuple[list[Tensor], bool]:
    """Divide the gradients by scale (unscale) and say whether any inf/nan showed up.

    This is the heart of GradScaler: if found_inf, the optimizer step is skipped and the scale is reduced.
    Return (the list of unscaled gradients, found_inf).
    """
    raise NotImplementedError


def next_scale(scale: float, found_inf: bool, growth_interval_reached: bool) -> float:
    """The GradScaler rule: on found_inf halve the scale,
    otherwise, if the growth interval has been reached, double it, otherwise leave it as it is.
    """
    raise NotImplementedError


def safe_softmax_fp16(logits: Tensor) -> Tensor:
    """The logits arrive in float16 and may be large. Compute the softmax in such a way
    that there is no overflow, and return the result in float16.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_online_softmax_update():
    x = torch.randn(4, 20)
    m = torch.full((4, 1), float("-inf"))
    l = torch.zeros(4, 1)
    for start in range(0, 20, 5):
        m, l = online_softmax_update(m, l, x[:, start : start + 5])
    torch.testing.assert_close(m.squeeze(-1), x.max(dim=-1).values)
    torch.testing.assert_close(l.squeeze(-1).log() + m.squeeze(-1), torch.logsumexp(x, dim=-1))


def test_chunked_softmax():
    x = torch.randn(8, 37) * 5
    torch.testing.assert_close(chunked_softmax(x, chunk=8), x.softmax(dim=-1))
    big = torch.randn(4, 16) + 5000.0
    assert torch.isfinite(chunked_softmax(big, chunk=4)).all()


def test_flash_attention_matches_reference():
    torch.manual_seed(0)
    q, k, v = torch.randn(2, 3, 40, 16), torch.randn(2, 3, 40, 16), torch.randn(2, 3, 40, 16)
    torch.testing.assert_close(
        flash_attention_forward(q, k, v, block_size=8),
        F.scaled_dot_product_attention(q, k, v),
        atol=1e-5,
        rtol=1e-5,
    )


def test_flash_attention_causal():
    torch.manual_seed(0)
    q, k, v = torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 8), torch.randn(1, 2, 37, 8)
    out = flash_attention_forward(q, k, v, block_size=8, causal=True)
    assert torch.isfinite(out).all(), "somewhere an exp(-inf - (-inf)) came out"
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q, k, v, is_causal=True), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("block_size", [1, 4, 16, 64])
def test_flash_attention_any_block_size(block_size):
    q, k, v = torch.randn(1, 1, 20, 8), torch.randn(1, 1, 20, 8), torch.randn(1, 1, 20, 8)
    torch.testing.assert_close(
        flash_attention_forward(q, k, v, block_size, causal=True),
        F.scaled_dot_product_attention(q, k, v, is_causal=True),
        atol=1e-5,
        rtol=1e-5,
    )


def test_flash_attention_is_stable_on_large_scores():
    q = torch.full((1, 1, 4, 8), 30.0)
    k = torch.full((1, 1, 4, 8), 30.0)
    v = torch.randn(1, 1, 4, 8)
    out = flash_attention_forward(q, k, v, block_size=2)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out, v.mean(dim=-2, keepdim=True).expand_as(out), atol=1e-4, rtol=1e-4)


def test_chunked_cross_entropy_value_and_grads():
    torch.manual_seed(0)
    N, D, V = 20, 16, 50
    hidden = torch.randn(N, D, requires_grad=True)
    weight = torch.randn(V, D, requires_grad=True)
    targets = torch.randint(0, V, (N,))

    reference = F.cross_entropy(hidden @ weight.T, targets)
    reference.backward()
    assert hidden.grad is not None and weight.grad is not None
    ref_h, ref_w = hidden.grad.clone(), weight.grad.clone()

    hidden.grad = None
    weight.grad = None
    loss = chunked_cross_entropy(hidden, weight, targets, chunk=7)
    torch.testing.assert_close(loss, reference.detach(), atol=1e-5, rtol=1e-5)

    loss.backward()
    torch.testing.assert_close(hidden.grad, ref_h, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(weight.grad, ref_w, atol=1e-5, rtol=1e-5)


def test_chunked_logsumexp():
    x = torch.randn(6, 41) * 3
    torch.testing.assert_close(chunked_logsumexp(x, chunk=7), torch.logsumexp(x, dim=-1))


def test_checkpointing_gives_identical_results():
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(4)])
    x = torch.randn(3, 8, requires_grad=True)

    plain = checkpointed_sequential(blocks, x, use_checkpoint=False)
    plain.sum().backward()
    plain_grads = [p.grad.clone() for p in blocks.parameters() if p.grad is not None]
    assert x.grad is not None
    plain_x_grad = x.grad.clone()

    for p in blocks.parameters():
        p.grad = None
    x.grad = None

    ckpt = checkpointed_sequential(blocks, x, use_checkpoint=True)
    torch.testing.assert_close(ckpt, plain, atol=1e-6, rtol=1e-6)
    ckpt.sum().backward()
    torch.testing.assert_close(x.grad, plain_x_grad, atol=1e-6, rtol=1e-6)
    for p, g in zip(blocks.parameters(), plain_grads, strict=True):
        torch.testing.assert_close(p.grad, g, atol=1e-6, rtol=1e-6)


def test_checkpointing_shrinks_the_graph():
    torch.manual_seed(0)
    blocks = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.Tanh()) for _ in range(6)])
    x = torch.randn(3, 8, requires_grad=True)
    plain = count_saved_tensors(blocks, x, use_checkpoint=False)
    ckpt = count_saved_tensors(blocks, x, use_checkpoint=True)
    assert ckpt < plain, f"checkpointing did not shrink what is saved: {ckpt} against {plain}"
    assert ckpt <= len(blocks), "with checkpointing roughly one input per block is kept"


def test_scale_and_check():
    grads = [torch.tensor([1024.0, -2048.0]), torch.tensor([512.0])]
    unscaled, found = scale_and_check(grads, 1024.0)
    assert found is False
    torch.testing.assert_close(unscaled[0], torch.tensor([1.0, -2.0]))
    torch.testing.assert_close(unscaled[1], torch.tensor([0.5]))

    bad = [torch.tensor([float("inf"), 1.0])]
    _, found = scale_and_check(bad, 1024.0)
    assert found is True

    nan = [torch.tensor([float("nan")])]
    _, found = scale_and_check(nan, 2.0)
    assert found is True


def test_next_scale():
    assert next_scale(1024.0, found_inf=True, growth_interval_reached=False) == 512.0
    assert next_scale(1024.0, found_inf=True, growth_interval_reached=True) == 512.0
    assert next_scale(1024.0, found_inf=False, growth_interval_reached=True) == 2048.0
    assert next_scale(1024.0, found_inf=False, growth_interval_reached=False) == 1024.0


def test_safe_softmax_fp16():
    logits = torch.tensor([[60000.0, 60000.0, 0.0]], dtype=torch.float16)
    out = safe_softmax_fp16(logits)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all(), "fp16 overflowed on the exp"
    torch.testing.assert_close(out.float().sum(), torch.tensor(1.0), atol=1e-3, rtol=1e-3)
    assert abs(out[0, 0].item() - 0.5) < 1e-2
