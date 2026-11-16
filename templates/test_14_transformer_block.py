"""Topic 14 - assembling the transformer block and the whole language model.

Yes, some of these bricks were already written in topics 10-13. Here they are written once more, and that
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
# 14.1 The bricks (repetition is the mother of muscle memory)
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
# 14.2 The block
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
# 14.3 The whole model
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
# 14.4 The arithmetic of the model
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
#
# 14.1 first: the three bricks each get their own test. Everything below them is assembled out of
# all three, so those tests declare the bricks as prerequisites and stay skipped until they exist.
#
# Submodules and weights are reached through get_submodule / get_parameter rather than attribute access:
# the stubs above declare no attributes, and these two calls keep the type checker informed.


def _ref_causal_attention(x: Tensor, attn: nn.Module, num_heads: int) -> Tensor:
    """Causal multi-head self-attention built from plain torch and the module's own weights."""
    B, T, D = x.shape
    q, k, v = (
        (x @ attn.get_parameter(f"{name}_proj.weight").T).view(B, T, num_heads, -1).transpose(1, 2) for name in "qkv"
    )
    out = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    return out.transpose(1, 2).reshape(B, T, D) @ attn.get_parameter("output_proj.weight").T


def test_rms_norm():
    norm = RMSNorm(16)
    assert isinstance(norm.weight, nn.Parameter)
    torch.testing.assert_close(norm.weight, torch.ones(16))
    assert set(norm.state_dict().keys()) == {"weight"}, "RMSNorm has no bias"

    x = torch.randn(2, 5, 16)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * norm.weight.detach()
    torch.testing.assert_close(norm(x), expected)


def test_swiglu():
    ffn = SwiGLU(8, 21)
    assert {n for n, _ in ffn.named_parameters()} == {"w1.weight", "w2.weight", "w3.weight"}
    assert sum(p.numel() for p in ffn.parameters()) == 3 * 8 * 21, "three matrices, no biases"

    x = torch.randn(4, 8)
    out = ffn(x)
    assert out.shape == (4, 8)
    w1, w2, w3 = (ffn.get_parameter(f"{name}.weight") for name in ("w1", "w2", "w3"))
    expected = (nn.functional.silu(x @ w1.T) * (x @ w3.T)) @ w2.T
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("num_heads", [1, 2, 4])
def test_causal_self_attention(num_heads):
    attn = CausalSelfAttention(16, num_heads)
    x = torch.randn(2, 5, 16)
    torch.testing.assert_close(attn(x), _ref_causal_attention(x, attn, num_heads), atol=1e-5, rtol=1e-5)


def test_causal_self_attention_names():
    attn = CausalSelfAttention(32, 4)
    names = {n for n, _ in attn.named_parameters()}
    assert names == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight"}
    assert sum(p.numel() for p in attn.parameters()) == 4 * 32 * 32


def test_causal_self_attention_is_causal():
    """The output at position t must read the inputs up to t, and nothing after."""
    x = torch.randn(1, 6, 16, requires_grad=True)
    attn = CausalSelfAttention(16, 2)
    attn(x)[0, 2].sum().backward()
    assert x.grad is not None
    assert x.grad[0, 3:].abs().max().item() == 0.0, "information leaked in from the future"
    assert (x.grad[0, :2].abs().amax(dim=-1) > 0).all(), "the output does not read the earlier tokens"


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention")
def test_transformer_block():
    """A pre-norm block: each sublayer reads a normalized copy of the stream and adds its result back."""
    torch.manual_seed(0)
    block = TransformerBlock(16, 2, 32)
    ln1, attn, ln2, ffn = (block.get_submodule(name) for name in ("ln1", "attn", "ln2", "ffn"))
    x = torch.randn(2, 4, 16)
    h = x + attn(ln1(x))
    expected = h + ffn(ln2(h))
    torch.testing.assert_close(block(x), expected, atol=1e-5, rtol=1e-5)


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention")
def test_transformer_block_names():
    block = TransformerBlock(32, 4, 64)
    assert {name for name, _ in block.named_children()} == {"ln1", "attn", "ln2", "ffn"}
    assert block.get_submodule("ln1") is not block.get_submodule("ln2"), "ln1 and ln2 must be separate instances"


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention")
def test_transformer_block_identity_when_sublayers_are_zero():
    """Zero out the output projections - the residual stream is obliged to carry the input through unchanged."""
    block = TransformerBlock(16, 2, 32)
    with torch.no_grad():
        block.get_parameter("attn.output_proj.weight").zero_()
        block.get_parameter("ffn.w2.weight").zero_()
    x = torch.randn(3, 5, 16)
    torch.testing.assert_close(block(x), x, atol=1e-6, rtol=0)


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock")
@pytest.mark.parametrize("num_layers", [1, 3])
def test_transformer_lm(num_layers):
    """embeddings -> every block in order -> ln_final -> lm_head, assembled from the model's own parts."""
    model = TransformerLM(vocab_size=40, d_model=16, num_heads=2, d_ff=32, num_layers=num_layers)
    ids = torch.randint(0, 40, (2, 5))

    h = model.get_parameter("token_embeddings.weight")[ids]
    for i in range(num_layers):
        h = model.get_submodule(f"layers.{i}")(h)
    expected = model.get_submodule("lm_head")(model.get_submodule("ln_final")(h))

    logits = model(ids)
    assert logits.shape == (2, 5, 40)
    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock")
def test_transformer_lm_names():
    model = TransformerLM(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    assert {name for name, _ in model.named_children()} == {"token_embeddings", "layers", "ln_final", "lm_head"}
    assert isinstance(model.get_submodule("layers"), nn.ModuleList)
    assert isinstance(model.get_submodule("token_embeddings"), nn.Embedding)

    names = {n for n, _ in model.named_parameters()}
    assert "token_embeddings.weight" in names
    assert "ln_final.weight" in names
    assert "lm_head.weight" in names
    assert "lm_head.bias" not in names
    assert "layers.0.attn.q_proj.weight" in names
    assert "layers.1.ffn.w1.weight" in names
    assert "layers.0.ln1.weight" in names
    assert model.get_submodule("layers.0") is not model.get_submodule("layers.1"), "every layer must be its own block"


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock")
def test_transformer_lm_init():
    """The embeddings and every linear weight come from N(0, 0.02); the norm weights stay at one."""
    torch.manual_seed(0)
    model = TransformerLM(vocab_size=1000, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear | nn.Embedding):
            w = module.weight.detach()
            assert abs(w.std().item() - 0.02) < 0.002, f"{name}: std {w.std().item():.4f}, expected 0.02"
            assert abs(w.mean().item()) < 0.002, f"{name}: mean {w.mean().item():.4f}, expected 0"
    torch.testing.assert_close(model.get_parameter("ln_final.weight"), torch.ones(64))


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock")
def test_transformer_lm_is_causal():
    """Changing a token may change the logits at its own position and after it, never before."""
    model = TransformerLM(vocab_size=50, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    ids = torch.randint(0, 50, (1, 6))
    changed = ids.clone()
    changed[0, 3] = (ids[0, 3] + 1) % 50
    with torch.no_grad():
        before, after = model(ids), model(changed)
    torch.testing.assert_close(
        after[:, :3], before[:, :3], atol=1e-6, rtol=0, msg="the model is peeking into the future"
    )
    assert not torch.allclose(after[:, 3:], before[:, 3:]), "the later positions do not read the changed token"


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock")
def test_transformer_lm_untrained_loss_is_log_vocab():
    """The first check of any training run: the loss at the start must be ~log(V)."""
    torch.manual_seed(0)
    V = 512
    model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    ids = torch.randint(0, V, (4, 16))
    logits = model(ids)
    loss = nn.functional.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
    assert abs(loss.item() - math.log(V)) < 0.1, f"loss {loss.item():.3f}, expected ~{math.log(V):.3f}"


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock")
def test_transformer_lm_can_overfit_one_batch():
    """The model must be able to memorize a single batch - otherwise the graph or the mask is broken somewhere."""
    torch.manual_seed(0)
    V = 32
    model = TransformerLM(vocab_size=V, d_model=64, num_heads=4, d_ff=128, num_layers=2)
    ids = torch.randint(0, V, (2, 8))
    inputs, targets = ids[:, :-1], ids[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(60):
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits.reshape(-1, V), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.1 * losses[0], f"the loss did not fall: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_count_lm_params():
    # a simplified GPT-2 small: 12 layers, d=768, but with a SwiGLU FFN
    n = count_lm_params(vocab_size=50257, d_model=768, d_ff=3072, num_layers=12, tied_embeddings=True)
    per_layer = 4 * 768 * 768 + 3 * 768 * 3072
    assert n == 50257 * 768 + 12 * per_layer

    untied = count_lm_params(50257, 768, 3072, 12, tied_embeddings=False)
    assert untied - n == 50257 * 768


@pytest.mark.depends_on("RMSNorm", "SwiGLU", "CausalSelfAttention", "TransformerBlock", "TransformerLM")
def test_count_lm_params_matches_real_model():
    model = TransformerLM(vocab_size=100, d_model=32, num_heads=4, d_ff=64, num_layers=2)
    formula = count_lm_params(100, 32, 64, 2, tied_embeddings=False)
    real = sum(p.numel() for p in model.parameters())
    # the difference is only the normalization weights: 2 per layer plus the final one
    assert real - formula == (2 * 2 + 1) * 32


def test_activation_memory_bytes():
    assert activation_memory_bytes(8, 2048, 4096, 32, torch.bfloat16) == 8 * 2048 * 4096 * 32 * 2
    assert activation_memory_bytes(1, 10, 8, 2, torch.float32) == 10 * 8 * 2 * 4
