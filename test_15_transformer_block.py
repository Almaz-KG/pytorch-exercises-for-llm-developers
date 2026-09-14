"""Topic 15 - assembling the transformer block and the whole language model.

Yes, some of these bricks were already written in topics 11-14. Here they are written once more, and that
is deliberate: the goal of the exercises is for a transformer block to come out of the hands without thinking.

The architecture is exactly the one in the CS336 assignment (pre-norm, RMSNorm, RoPE omitted for brevity):

    x = x + attn(ln1(x))
    x = x + ffn(ln2(x))
    ...
    logits = lm_head(ln_final(x))

The submodule names repeat the reference state_dict of the course:
    token_embeddings, layers.{i}.attn, layers.{i}.ffn, layers.{i}.ln1, layers.{i}.ln2,
    ln_final, lm_head
"""

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

# ==========================================================================
# 15.1 The bricks (repetition is the mother of muscle memory)
# ==========================================================================


class RMSNorm(nn.Module):
    """A `weight` parameter of shape (d_model,), initialized with ones."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class SwiGLU(nn.Module):
    """The submodules w1, w2, w3 are nn.Linear without a bias. out = w2(silu(w1(x)) * w3(x))."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class CausalSelfAttention(nn.Module):
    """The submodules q_proj, k_proj, v_proj, output_proj are nn.Linear(d_model, d_model, bias=False).
    Causal multi-head attention.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 15.2 The block
# ==========================================================================


class TransformerBlock(nn.Module):
    """A pre-norm block. The submodules must be named ln1, attn, ln2, ffn.

    x = x + attn(ln1(x))
    x = x + ffn(ln2(x))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 15.3 The whole model
# ==========================================================================


class TransformerLM(nn.Module):
    """The language model.

    The submodules: token_embeddings (nn.Embedding), layers (an nn.ModuleList of TransformerBlock),
    ln_final (RMSNorm), lm_head (nn.Linear(d_model, vocab_size, bias=False)).

    forward(ids: (B, T) int64) -> logits (B, T, vocab_size).

    Initialization: the embeddings and all the linear weights from a normal with std=0.02.
    This matters for the test "the loss of an untrained model equals log(V)".
    """

    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int, num_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, ids: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 15.4 The arithmetic of the model
# ==========================================================================


def count_lm_params(vocab_size: int, d_model: int, d_ff: int, num_layers: int, tied_embeddings: bool) -> int:
    """The number of parameters of the model by formula, without creating the model.

    We count: the embeddings + num_layers * (4 * d^2 of attention + 3 * d * d_ff of SwiGLU) + lm_head.
    Normalizations and biases are ignored. With tied_embeddings=True lm_head is not counted separately.
    """
    raise NotImplementedError


def activation_memory_bytes(batch: int, seq_len: int, d_model: int, num_layers: int, dtype: torch.dtype) -> int:
    """A rough estimate of the memory for the activations of the residual stream: one tensor (B, T, D)
    per layer. Reality is several times worse, but this formula gives the order of magnitude.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_block_structure():
    block = TransformerBlock(32, 4, 64)
    children = dict(block.named_children())
    assert set(children.keys()) == {"ln1", "attn", "ln2", "ffn"}
    x = torch.randn(2, 5, 32)
    assert block(x).shape == x.shape


def test_block_is_pre_norm_with_residuals():
    torch.manual_seed(0)
    block = TransformerBlock(16, 2, 32)
    x = torch.randn(2, 4, 16)
    h = x + block.attn(block.ln1(x))
    expected = h + block.ffn(block.ln2(h))
    torch.testing.assert_close(block(x), expected, atol=1e-5, rtol=1e-5)


def test_block_identity_when_sublayers_are_zero():
    """Zero out the output projections - the residual stream is obliged to carry the input through unchanged."""
    block = TransformerBlock(16, 2, 32)
    with torch.no_grad():
        block.attn.output_proj.weight.zero_()
        block.ffn.w2.weight.zero_()
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(block(x), x, atol=1e-6, rtol=0)


def test_lm_shapes_and_names():
    model = TransformerLM(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    ids = torch.randint(0, 100, (2, 7))
    logits = model(ids)
    assert logits.shape == (2, 7, 100)

    names = {n for n, _ in model.named_parameters()}
    assert "token_embeddings.weight" in names
    assert "ln_final.weight" in names
    assert "lm_head.weight" in names
    assert "layers.0.attn.q_proj.weight" in names
    assert "layers.1.ffn.w1.weight" in names
    assert "layers.0.ln1.weight" in names


def test_lm_is_causal():
    model = TransformerLM(vocab_size=50, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    ids = torch.randint(0, 50, (1, 6))
    emb = model.token_embeddings(ids).detach().requires_grad_(True)

    h = emb
    for layer in model.layers:
        h = layer(h)
    logits = model.lm_head(model.ln_final(h))
    logits[0, 2].sum().backward()

    assert emb.grad[0, 3:].abs().max().item() == 0.0, "the model is peeking into the future"
    assert emb.grad[0, :3].abs().max().item() > 0.0


def test_untrained_loss_is_log_vocab():
    """The first check of any training run: the loss at the start must be ~log(V)."""
    torch.manual_seed(0)
    V = 512
    model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    ids = torch.randint(0, V, (4, 16))
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 0.3, f"loss {loss.item():.3f}, expected ~{math.log(V):.3f}"


def test_lm_can_overfit_one_batch():
    """The model must be able to memorize a single batch - otherwise the graph or the mask is broken somewhere."""
    torch.manual_seed(0)
    V = 32
    model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    ids = torch.randint(0, V, (2, 8))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    first = None
    for step in range(60):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
    assert loss.item() < 0.1 * first, f"the loss did not fall: {first:.3f} -> {loss.item():.3f}"


def test_count_lm_params():
    # a simplified GPT-2 small: 12 layers, d=768, but with a SwiGLU FFN
    n = count_lm_params(vocab_size=50257, d_model=768, d_ff=3072, num_layers=12, tied_embeddings=True)
    per_layer = 4 * 768 * 768 + 3 * 768 * 3072
    assert n == 50257 * 768 + 12 * per_layer

    untied = count_lm_params(50257, 768, 3072, 12, tied_embeddings=False)
    assert untied - n == 50257 * 768


def test_count_matches_real_model():
    model = TransformerLM(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    formula = count_lm_params(100, 32, 64, 2, tied_embeddings=False)
    real = sum(p.numel() for p in model.parameters())
    # the difference is only the normalization weights: 2 per layer plus the final one
    assert real - formula == (2 * 2 + 1) * 32


def test_activation_memory_bytes():
    b = activation_memory_bytes(8, 2048, 4096, 32, torch.bfloat16)
    assert b == 8 * 2048 * 4096 * 32 * 2


@pytest.mark.parametrize("num_layers", [1, 3])
def test_lm_depth(num_layers):
    model = TransformerLM(vocab_size=40, d_model=16, num_heads=2, d_ff=32, num_layers=num_layers)
    assert len(model.layers) == num_layers
    assert model(torch.randint(0, 40, (1, 5))).shape == (1, 5, 40)
