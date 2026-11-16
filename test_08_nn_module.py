"""Topic 08 - nn.Module: parameters, buffers, state_dict, initialization.

The key distinctions that have to be felt:
  - Parameter -> is trained, lives in .parameters() and in the state_dict;
  - buffer -> is NOT trained, but lives in the state_dict and travels to the device with the module
    (the typical example: a mask or a RoPE cache);
  - an ordinary tensor attribute -> travels nowhere and is lost on .to(device). The classic bug.
  - a list of modules has to go into an nn.ModuleList, otherwise it is as if the parameters do not exist.
"""

import io
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 8.1 Layers of your own
# ==========================================================================


class MyLinear(nn.Module):
    """A linear layer with no surprises: weight (out_features, in_features), an optional bias.

    The requirements:
      - the parameters are named `weight` and `bias` (this pins the state_dict);
      - with bias=False the bias attribute must be None and must NOT appear in the state_dict;
      - forward works with any number of leading axes.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MyEmbedding(nn.Module):
    """An embedding table: a `weight` parameter of shape (num_embeddings, embedding_dim)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, ids: Tensor) -> Tensor:
        raise NotImplementedError


class CausalMask(nn.Module):
    """Only about buffers: the module stores a causal mask (max_len, max_len) bool as a BUFFER
    named `mask`, and forward returns its top-left (seq_len, seq_len) corner.

    This is what every implementation does: the mask is computed once, not at every step.
    """

    def __init__(self, max_len: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, seq_len: int) -> Tensor:
        raise NotImplementedError


class BlockStack(nn.Module):
    """A stack of n linear layers d->d of identical shape, applied one after another.

    The layers live in an attribute named `layers`, and it must be an nn.ModuleList
    (an ordinary python list does not register the parameters).
    """

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 8.2 Inspecting a model
# ==========================================================================


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """The number of elements (not tensors) in the parameters of the model."""
    raise NotImplementedError


def parameter_names(model: nn.Module) -> list[str]:
    """A sorted list of the full parameter names, as they sit in the state_dict."""
    raise NotImplementedError


def total_bytes(model: nn.Module) -> int:
    """How many bytes all the parameters and buffers of the model take."""
    raise NotImplementedError


# ==========================================================================
# 8.3 Initialization
# ==========================================================================


def init_linear_(layer: nn.Linear, std: float) -> nn.Linear:
    """Initialize the weight with a truncated normal N(0, std) within [-3*std, 3*std],
    and zero out the bias. Return the same layer.

    Hint: nn.init.trunc_normal_ and nn.init.zeros_. Everything in-place, under torch.no_grad
    (the functions from nn.init already do that for you).
    """
    raise NotImplementedError


def init_all_linears_(model: nn.Module, std: float) -> nn.Module:
    """Walk the whole module tree and initialize EVERY nn.Linear through init_linear_.
    Return the same model.

    Hint: model.apply(fn) calls fn on every submodule.
    """
    raise NotImplementedError


def init_residual_projections_(model: nn.Module, base_std: float, n_layers: int) -> nn.Module:
    """The GPT-2 trick: the layers that write into the residual stream get std = base_std / sqrt(2*n_layers),
    otherwise the variance of the residual grows with depth.

    In the CS336 naming those layers are called `output_proj` (attention) and `w2` (feed-forward).
    Find every nn.Linear whose own name, the last piece of its dotted path, is one of those two,
    and initialize it through init_linear_ with the scaled std. Leave every other module untouched.
    Return the same model.

    Hint: model.named_modules() yields the dotted path of every submodule.
    """
    raise NotImplementedError


# ==========================================================================
# 8.4 state_dict
# ==========================================================================


def save_load_roundtrip(model: nn.Module, fresh: nn.Module) -> nn.Module:
    """Save the state_dict of model into a byte buffer (io.BytesIO) through torch.save,
    load it back from that buffer into fresh, a new model of the same architecture, and return fresh.
    """
    raise NotImplementedError


def strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Remove the prefix from the keys (for example `_orig_mod.`, which torch.compile prepends,
    or `module.`, which DDP prepends). Leave the keys without the prefix as they are.

    This exact function lives in the conftest.py of the first CS336 assignment.
    """
    raise NotImplementedError


def tie_weights(embedding: nn.Embedding, head: nn.Linear) -> None:
    """Tie the weights of the embedding and the output projection (weight tying): one and the same tensor.

    After the call a change in one must be visible in the other, and the number of unique
    parameters of the model must go down.
    """
    raise NotImplementedError


# ==========================================================================
# 8.5 Modes and hooks
# ==========================================================================


def dropout_behaviour(p: float, x: Tensor) -> tuple[Tensor, Tensor]:
    """Create an nn.Dropout(p) and return (the output in train mode, the output in eval mode).

    The point: to convince yourself by hand that model.eval() switches dropout off rather than "reducing" it.
    """
    raise NotImplementedError


def capture_activations(model: nn.Sequential, x: Tensor) -> dict[str, Tensor]:
    """Hang forward hooks on all the submodules and return a dict {module_name: output}.

    This is the main way to debug "where the activations blew up" in a large model.
    The hooks are better removed once they have been used.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


@pytest.mark.parametrize("bias", [True, False])
def test_my_linear(bias):
    layer = MyLinear(4, 6, bias=bias)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (6, 4)
    if bias:
        assert isinstance(layer.bias, nn.Parameter)
        assert layer.bias.shape == (6,)
        assert set(layer.state_dict().keys()) == {"weight", "bias"}
    else:
        assert layer.bias is None
        assert set(layer.state_dict().keys()) == {"weight"}, "an absent bias must stay out of the state_dict"

    ref = nn.Linear(4, 6, bias=bias)
    layer.load_state_dict(ref.state_dict())
    x = torch.randn(2, 3, 4)
    out = layer(x)
    assert out.shape == (2, 3, 6)
    torch.testing.assert_close(out, ref(x))


def test_my_embedding():
    emb = MyEmbedding(10, 4)
    assert isinstance(emb.weight, nn.Parameter)
    assert emb.weight.shape == (10, 4)
    assert set(emb.state_dict().keys()) == {"weight"}

    ids = torch.tensor([[1, 2], [3, 3]])
    out = emb(ids)
    assert out.shape == (2, 2, 4)
    torch.testing.assert_close(out, F.embedding(ids, emb.weight))


def test_causal_mask():
    layer = CausalMask(8)
    assert "mask" in dict(layer.named_buffers()), "the mask must be a buffer"
    assert len(list(layer.parameters())) == 0, "a buffer is not trained"

    m = layer(3)
    assert m.dtype == torch.bool
    assert torch.equal(m, torch.ones(3, 3, dtype=torch.bool).tril())
    assert layer(8).shape == (8, 8)

    # a plain tensor attribute would stay behind on the cpu
    assert layer.to("meta")(3).device.type == "meta", "a buffer moves with the module"


@pytest.mark.parametrize("n_layers", [1, 3])
def test_block_stack(n_layers):
    stack = BlockStack(4, n_layers)
    layers = stack.get_submodule("layers")
    assert isinstance(layers, nn.ModuleList)
    # a plain python list of layers would leave both of these at zero
    assert sum(p.numel() for p in stack.parameters()) == n_layers * (4 * 4 + 4)
    assert len(list(stack.named_parameters())) == 2 * n_layers

    x = torch.randn(2, 4)
    expected = x
    for layer in layers:
        expected = layer(expected)
    out = stack(x)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, expected, msg="forward must apply every layer in order")


def test_count_parameters():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    model[0].weight.requires_grad_(False)
    total = count_parameters(model, trainable_only=False)
    assert total == (2 * 3 + 3) + (3 * 4 + 4)
    assert count_parameters(model, trainable_only=True) == total - 6
    assert count_parameters(model) == total - 6, "trainable_only is on by default"


def test_parameter_names():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    assert parameter_names(model) == ["0.bias", "0.weight", "1.bias", "1.weight"]


def test_total_bytes():
    model = nn.Linear(4, 6)  # 24 weights + 6 bias, float32
    assert total_bytes(model) == (24 + 6) * 4

    model.register_buffer("cache", torch.zeros(10, dtype=torch.float64))
    assert total_bytes(model) == (24 + 6) * 4 + 10 * 8, "buffers take memory too"


def test_init_linear():
    layer = nn.Linear(64, 64)
    assert init_linear_(layer, std=0.02) is layer
    assert layer.weight.abs().max().item() <= 3 * 0.02 + 1e-6
    assert abs(layer.weight.std().item() - 0.02) < 0.005
    torch.testing.assert_close(layer.bias, torch.zeros(64))


@pytest.mark.depends_on("init_linear_")
def test_init_all_linears():
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Sequential(nn.Linear(32, 32)))
    assert init_all_linears_(model, std=0.01) is model
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    assert len(linears) == 2
    for m in linears:
        assert m.weight.abs().max().item() <= 3 * 0.01 + 1e-6
        assert abs(m.weight.std().item() - 0.01) < 0.003
        torch.testing.assert_close(m.bias, torch.zeros(32))


@pytest.mark.depends_on("init_linear_")
def test_init_residual_projections():
    def block() -> nn.Module:
        attn = nn.ModuleDict({"q_proj": nn.Linear(64, 64, bias=False), "output_proj": nn.Linear(64, 64, bias=False)})
        ffn = nn.ModuleDict({"w1": nn.Linear(64, 256, bias=False), "w2": nn.Linear(256, 64, bias=False)})
        return nn.ModuleDict({"attn": attn, "ffn": ffn})

    model = nn.ModuleDict({"layers": nn.ModuleList([block(), block()]), "lm_head": nn.Linear(64, 100, bias=False)})
    before = {k: v.clone() for k, v in model.state_dict().items()}
    assert init_residual_projections_(model, base_std=0.02, n_layers=8) is model

    std = 0.02 / math.sqrt(2 * 8)
    for name, w in model.state_dict().items():
        if name.endswith(("output_proj.weight", "w2.weight")):
            assert w.abs().max().item() <= 3 * std + 1e-6, name
            assert abs(w.std().item() - std) < 0.001, f"{name}: std {w.std().item():.4f}, expected {std:.4f}"
        else:
            assert torch.equal(w, before[name]), f"{name} does not write into the residual stream"


def test_save_load_roundtrip():
    def make() -> nn.Module:
        return nn.Sequential(nn.Linear(3, 5), nn.ReLU(), nn.Linear(5, 2))

    model, fresh = make(), make()
    loaded = save_load_roundtrip(model, fresh)
    assert loaded is fresh
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)
        assert v1.data_ptr() != v2.data_ptr(), "the fresh model must hold its own copy of the weights"
    x = torch.randn(4, 3)
    torch.testing.assert_close(model(x), loaded(x))


def test_strip_prefix():
    w, ln = torch.zeros(1), torch.ones(1)
    sd = {"_orig_mod.layers.0.weight": w, "ln_final.weight": ln}
    out = strip_prefix(sd, "_orig_mod.")
    assert set(out.keys()) == {"layers.0.weight", "ln_final.weight"}
    assert out["layers.0.weight"] is w and out["ln_final.weight"] is ln, "the values must not change"


def test_tie_weights():
    emb = nn.Embedding(10, 4)
    head = nn.Linear(4, 10, bias=False)
    tie_weights(emb, head)
    assert head.weight is emb.weight, "both modules must hold the same Parameter object"
    assert "weight" in dict(head.named_parameters()), "the head must still own its weight as a parameter"

    model = nn.Sequential(emb, head)
    assert len(list(model.parameters())) == 1, "after tying there must be exactly one unique parameter"


def test_dropout_behaviour():
    x = torch.ones(1000)
    train_out, eval_out = dropout_behaviour(0.5, x)
    zeros = train_out == 0
    assert 0.4 < zeros.float().mean().item() < 0.6, "in train mode about half of the elements are dropped"
    torch.testing.assert_close(train_out[~zeros], torch.full_like(train_out[~zeros], 2.0), msg="survivors x 1/(1-p)")
    torch.testing.assert_close(eval_out, x, msg="in eval mode dropout is the identity")


def test_capture_activations():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(5, 3)
    acts = capture_activations(model, x)
    assert set(acts.keys()) == {"0", "1", "2"}
    torch.testing.assert_close(acts["0"], model[0](x))
    torch.testing.assert_close(acts["1"], model[1](model[0](x)))
    torch.testing.assert_close(acts["2"], model(x))
    assert all(len(m._forward_hooks) == 0 for m in model.modules()), "the hooks must be removed afterwards"
