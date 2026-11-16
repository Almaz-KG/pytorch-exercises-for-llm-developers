"""Reference solutions for topic 08."""

import io
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, std=in_features**-0.5)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.weight, self.bias)


class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim) * 0.02)

    def forward(self, ids: Tensor) -> Tensor:
        return self.weight[ids]


class CausalMask(nn.Module):
    mask: Tensor

    def __init__(self, max_len: int):
        super().__init__()
        idx = torch.arange(max_len)
        self.register_buffer("mask", idx[:, None] >= idx[None, :], persistent=True)

    def forward(self, seq_len: int) -> Tensor:
        return self.mask[:seq_len, :seq_len]


class BlockStack(nn.Module):
    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def parameter_names(model: nn.Module) -> list[str]:
    return sorted(name for name, _ in model.named_parameters())


def total_bytes(model: nn.Module) -> int:
    params = sum(p.numel() * p.element_size() for p in model.parameters())
    buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    return params + buffers


def init_linear_(layer: nn.Linear, std: float) -> nn.Linear:
    nn.init.trunc_normal_(layer.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def init_all_linears_(model: nn.Module, std: float) -> nn.Module:
    def fn(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            init_linear_(module, std)

    model.apply(fn)
    return model


def init_residual_projections_(model: nn.Module, base_std: float, n_layers: int) -> nn.Module:
    std = base_std / math.sqrt(2 * n_layers)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in ("output_proj", "w2"):
            init_linear_(module, std)
    return model


def save_load_roundtrip(model: nn.Module, fresh: nn.Module) -> nn.Module:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    buf.seek(0)
    fresh.load_state_dict(torch.load(buf, weights_only=True))
    return fresh


def strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    return {k.removeprefix(prefix): v for k, v in state_dict.items()}


def tie_weights(embedding: nn.Embedding, head: nn.Linear) -> None:
    head.weight = embedding.weight


def dropout_behaviour(p: float, x: Tensor) -> tuple[Tensor, Tensor]:
    drop = nn.Dropout(p)
    drop.train()
    train_out = drop(x)
    drop.eval()
    return train_out, drop(x)


def capture_activations(model: nn.Sequential, x: Tensor) -> dict[str, Tensor]:
    acts: dict[str, Tensor] = {}
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            acts[name] = output.detach()

        return hook

    for name, module in model.named_children():
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        model(x)
    finally:
        for h in handles:
            h.remove()
    return acts
