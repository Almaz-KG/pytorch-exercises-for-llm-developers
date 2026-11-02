"""Topic 09 - nn.Module: parameters, buffers, state_dict, initialization.

The key distinctions that have to be felt:
  - Parameter -> is trained, lives in .parameters() and in the state_dict;
  - buffer -> is NOT trained, but lives in the state_dict and travels to the device with the module
    (the typical example: a mask or a RoPE cache);
  - an ordinary tensor attribute -> travels nowhere and is lost on .to(device). The classic bug.
  - a list of modules has to go into an nn.ModuleList, otherwise it is as if the parameters do not exist.
"""

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 9.1 Layers of your own
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


class CausalSelfAttentionSkeleton(nn.Module):
    """Only about buffers: the layer stores a causal mask (max_len, max_len) bool as a BUFFER
    named `mask`, and forward returns a slice of the mask for the current length.

    This is what every implementation does: the mask is computed once, not at every step.
    """

    def __init__(self, max_len: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, seq_len: int) -> Tensor:
        raise NotImplementedError


class BlockStack(nn.Module):
    """A stack of n linear layers d->d of identical shape, applied one after another.

    The layers must live in an nn.ModuleList (an ordinary python list does not register the parameters).
    """

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 9.2 Inspecting a model
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
# 9.3 Initialization
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

    Hint: model.apply(fn) calls fn on every submodule.
    """
    raise NotImplementedError


def scaled_residual_std(base_std: float, n_layers: int) -> float:
    """The GPT-2 trick: the weights that write into the residual stream are initialized with
    std / sqrt(2*n_layers), otherwise the variance of the residual grows with depth. Return that value.
    """
    raise NotImplementedError


# ==========================================================================
# 9.4 state_dict
# ==========================================================================


def save_load_roundtrip(model: nn.Module) -> nn.Module:
    """Save the state_dict into a byte buffer (io.BytesIO) through torch.save,
    create a NEW model of the same class and shape, load the weights into it, and return it.

    For simplicity assume the model is a MyLinear(in, out).
    """
    raise NotImplementedError


def strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Remove the prefix from the keys (for example `_orig_mod.`, which torch.compile prepends,
    or `module.`, which DDP prepends). Leave the keys without the prefix as they are.

    This exact function lives in the conftest.py of the first CS336 assignment.
    """
    raise NotImplementedError


def tie_weights(embedding: MyEmbedding, head: MyLinear) -> None:
    """Tie the weights of the embedding and the output projection (weight tying): one and the same tensor.

    After the call a change in one must be visible in the other, and the number of unique
    parameters of the model must go down.
    """
    raise NotImplementedError


# ==========================================================================
# 9.5 Modes and hooks
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


def test_my_linear_matches_functional():
    layer = MyLinear(4, 6)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight, layer.bias))  # type: ignore
    assert layer(x).shape == (2, 3, 6)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (6, 4)
    assert set(layer.state_dict().keys()) == {"weight", "bias"}


def test_my_linear_no_bias():
    layer = MyLinear(4, 6, bias=False)
    assert layer.bias is None
    assert set(layer.state_dict().keys()) == {"weight"}
    x = torch.randn(5, 4)
    torch.testing.assert_close(layer(x), x @ layer.weight.T)  # type: ignore


def test_my_embedding():
    emb = MyEmbedding(10, 4)
    ids = torch.tensor([[1, 2], [3, 3]])
    out = emb(ids)
    assert out.shape == (2, 2, 4)
    torch.testing.assert_close(out, emb.weight[ids])  # type: ignore
    assert sum(p.numel() for p in emb.parameters()) == 40


def test_buffer_is_not_a_parameter():
    layer = CausalSelfAttentionSkeleton(8)
    assert "mask" in dict(layer.named_buffers()), "the mask must be a buffer"
    assert "mask" not in dict(layer.named_parameters())
    assert "mask" in layer.state_dict()
    assert sum(p.numel() for p in layer.parameters()) == 0, "a buffer is not trained"

    m = layer(3)
    assert m.shape == (3, 3)
    assert m.dtype == torch.bool
    assert bool(m[0, 0]) and not bool(m[0, 1]) and bool(m[2, 1])


def test_module_list_registers_parameters():
    stack = BlockStack(4, 3)
    # a plain python list of layers would leave both of these at zero
    assert sum(p.numel() for p in stack.parameters()) == 3 * (4 * 4 + 4)
    assert len(list(stack.named_parameters())) == 6
    x = torch.randn(2, 4)
    expected = x
    for layer in stack.layers:  # type: ignore
        expected = F.linear(expected, layer.weight, layer.bias)
    out = stack(x)
    assert out.shape == (2, 4)
    torch.testing.assert_close(out, expected, msg="forward must apply every layer in order")


def test_parameter_names():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    assert parameter_names(model) == ["0.bias", "0.weight", "1.bias", "1.weight"]


def test_total_bytes():
    model = nn.Linear(4, 6)  # 24 weights + 6 bias, float32
    assert total_bytes(model) == (24 + 6) * 4

    model.register_buffer("cache", torch.zeros(10, dtype=torch.float64))
    assert total_bytes(model) == (24 + 6) * 4 + 10 * 8, "buffers take memory too"


def test_count_parameters_trainable_flag():
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 4))
    model[0].weight.requires_grad_(False)
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    assert total == (2 * 3 + 3) + (3 * 4 + 4)
    assert trainable == total - 6


def test_init_linear():
    layer = nn.Linear(64, 64)
    init_linear_(layer, std=0.02)
    assert layer.weight.abs().max().item() <= 3 * 0.02 + 1e-6
    assert abs(layer.weight.std().item() - 0.02) < 0.005
    torch.testing.assert_close(layer.bias, torch.zeros(64))


def test_init_all_linears():
    model = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Sequential(nn.Linear(32, 32)))
    init_all_linears_(model, std=0.01)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert m.weight.abs().max().item() <= 3 * 0.01 + 1e-6
            torch.testing.assert_close(m.bias, torch.zeros(32))


def test_scaled_residual_std():
    assert abs(scaled_residual_std(0.02, 12) - 0.02 / (24**0.5)) < 1e-12


@pytest.mark.depends_on("MyLinear")
def test_save_load_roundtrip():
    model = MyLinear(3, 5)
    loaded = save_load_roundtrip(model)
    assert loaded is not model
    for (k1, v1), (k2, v2) in zip(model.state_dict().items(), loaded.state_dict().items(), strict=True):
        assert k1 == k2
        torch.testing.assert_close(v1, v2)
    x = torch.randn(4, 3)
    torch.testing.assert_close(model(x), loaded(x))


def test_strip_prefix():
    sd = {"_orig_mod.layers.0.weight": torch.zeros(1), "ln_final.weight": torch.zeros(1)}
    out = strip_prefix(sd, "_orig_mod.")
    assert set(out.keys()) == {"layers.0.weight", "ln_final.weight"}


@pytest.mark.depends_on("MyEmbedding", "MyLinear")
def test_tie_weights():
    emb = MyEmbedding(10, 4)
    head = MyLinear(4, 10, bias=False)
    tie_weights(emb, head)
    assert emb.weight.data_ptr() == head.weight.data_ptr()  # type: ignore

    model = nn.Sequential(emb, head)
    unique = {p.data_ptr() for p in model.parameters()}
    assert len(unique) == 1, "after tying there must be exactly one unique parameter"


def test_dropout_behaviour():
    x = torch.ones(100)
    train_out, eval_out = dropout_behaviour(1.0, x)
    torch.testing.assert_close(train_out, torch.zeros(100))
    torch.testing.assert_close(eval_out, x)


def test_capture_activations():
    model = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
    x = torch.randn(5, 3)
    acts = capture_activations(model, x)
    assert set(acts.keys()) == {"0", "1", "2"}
    assert acts["0"].shape == (5, 4)
    assert acts["2"].shape == (5, 2)
    assert (acts["1"] >= 0).all(), "after a ReLU there must be no negative values"
    assert model._forward_hooks == {} or len(model._forward_hooks) == 0


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_block_stack_depth(n_layers):
    stack = BlockStack(8, n_layers)
    assert len(list(stack.parameters())) == 2 * n_layers
