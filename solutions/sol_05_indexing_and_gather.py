"""Reference solutions for topic 05."""

import torch
from torch import Tensor


def embedding_lookup(table: Tensor, ids: Tensor) -> Tensor:
    return table[ids]


def select_rows(x: Tensor, rows: list[int]) -> Tensor:
    return x[torch.tensor(rows, device=x.device)]


def fancy_index_pairs(x: Tensor, batch_idx: Tensor, pos_idx: Tensor) -> Tensor:
    return x[batch_idx, pos_idx]


def gather_target_logits(logits: Tensor, targets: Tensor) -> Tensor:
    return logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)


def last_valid_hidden(h: Tensor, lengths: Tensor) -> Tensor:
    idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, h.shape[-1])
    return h.gather(dim=1, index=idx).squeeze(1)


def gather_beams(x: Tensor, beam_idx: Tensor) -> Tensor:
    return x.index_select(0, beam_idx)


def scatter_one_hot(ids: Tensor, vocab_size: int) -> Tensor:
    out = torch.zeros(*ids.shape, vocab_size, device=ids.device)
    return out.scatter_(-1, ids.unsqueeze(-1), 1.0)


def bag_of_words(ids: Tensor, vocab_size: int) -> Tensor:
    out = torch.zeros(ids.shape[0], vocab_size, device=ids.device)
    return out.scatter_add_(1, ids, torch.ones_like(ids, dtype=out.dtype))


def keep_top_k(logits: Tensor, k: int) -> Tensor:
    values, idx = torch.topk(logits, k, dim=-1)
    out = torch.full_like(logits, float("-inf"))
    return out.scatter_(-1, idx, values)


def select_valid(x: Tensor, mask: Tensor) -> Tensor:
    return x[mask]


def replace_where(x: Tensor, cond: Tensor, value: float) -> Tensor:
    return torch.where(cond, torch.full_like(x, value), x)


def find_positions(ids: Tensor, token_id: int) -> Tensor:
    return (ids == token_id).nonzero(as_tuple=True)[0]


def mask_after_eos(ids: Tensor, eos_id: int) -> Tensor:
    is_eos = ids == eos_id
    seen_before = is_eos.cumsum(dim=-1) - is_eos.to(torch.int64)
    return seen_before == 0


def make_lm_pairs(ids: Tensor) -> tuple[Tensor, Tensor]:
    return ids[:, :-1], ids[:, 1:]


def expand_for_beams(x: Tensor, num_beams: int) -> Tensor:
    return x.repeat_interleave(num_beams, dim=0)
