"""Reference solutions for topic 20."""

import io

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


def get_batch(
    data: np.ndarray, batch_size: int, context_length: int, generator: np.random.Generator | None = None
) -> tuple[Tensor, Tensor]:
    rng = generator if generator is not None else np.random.default_rng()
    high = len(data) - context_length
    starts = rng.integers(0, high, size=batch_size)
    x = np.stack([data[s : s + context_length] for s in starts])
    y = np.stack([data[s + 1 : s + context_length + 1] for s in starts])
    return torch.from_numpy(x).to(torch.int64), torch.from_numpy(y).to(torch.int64)


def max_start_index(num_tokens: int, context_length: int) -> int:
    return num_tokens - context_length - 1


def steps_per_epoch(num_tokens: int, batch_size: int, context_length: int) -> int:
    return num_tokens // (batch_size * context_length)


def sequential_windows(data: np.ndarray, context_length: int) -> Tensor:
    n = len(data) // context_length
    trimmed = np.asarray(data[: n * context_length])
    return torch.from_numpy(trimmed).to(torch.int64).reshape(n, context_length)


def collate_with_padding(sequences: list[list[int]], pad_id: int) -> tuple[Tensor, Tensor]:
    max_len = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.int64)
    mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.int64)
        mask[i, : len(seq)] = True
    return ids, mask


def make_labels(ids: Tensor, mask: Tensor, ignore_index: int = -100) -> Tensor:
    labels = torch.full_like(ids, ignore_index)
    labels[:, :-1] = ids[:, 1:]
    valid = mask.clone()
    valid[:, :-1] &= mask[:, 1:]
    valid[:, -1] = False
    return torch.where(valid, labels, torch.full_like(labels, ignore_index))


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: io.BytesIO) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src: io.BytesIO, model: nn.Module, optimizer: torch.optim.Optimizer) -> int:
    payload = torch.load(src, weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    return payload["iteration"]
