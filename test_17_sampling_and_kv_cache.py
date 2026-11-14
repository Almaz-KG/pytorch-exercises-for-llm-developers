"""Topic 17 - generation: sampling and the KV cache.

Two independent topics that always go together.

Sampling: a token has to be picked out of the logits. The temperature stretches or squeezes
the distribution, top-k and top-p cut off the tail out of which the model talks nonsense.
The generation loop runs the model, picks a token, appends it, and stops at a length or at EOS.

The KV cache: when generating one token at a time, recomputing the attention over the whole prefix
from scratch is O(T^2) per step. Instead the K and V of the past positions are stored and
appended to. The main thing to check is that the cache gives EXACTLY the same result.
"""

from collections.abc import Callable

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 17.1 Filters for the logits
# ==========================================================================


def apply_temperature(logits: Tensor, temperature: float) -> Tensor:
    """Divide the logits by the temperature. At temperature == 0 return logits
    from which the argmax is picked deterministically (for example a one-hot of -inf and 0):
    that is greedy mode.
    """
    raise NotImplementedError


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """(B, V) -> (B, V): replace everything except the top-k with -inf. At k <= 0 return it as is."""
    raise NotImplementedError


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """(B, V) -> (B, V): nucleus sampling.

    Walk the tokens from the most probable down, adding up their probabilities, and stop at
    the first token that brings the total to >= p. That token and every one before it are kept,
    everything after it -> -inf. So at least one token always survives.

    In tensors: sort descending, softmax, cumsum. A token is dropped when the sum of the tokens
    strictly BEFORE it is already >= p. That mask is in sorted order, so move it back to the
    original vocabulary order (a scatter, topic 05) before applying it to the logits.
    """
    raise NotImplementedError


def repetition_penalty(logits: Tensor, generated: Tensor, penalty: float) -> Tensor:
    """The penalty for repetition: the logits of the already generated tokens are divided by penalty
    if they are positive, and multiplied by penalty if they are negative.

    logits: (B, V), generated: (B, T) the tokens already produced.
    """
    raise NotImplementedError


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """(B, V) -> (B,) int64. The order of the operations is the standard one: temperature, top-k, top-p, softmax,
    multinomial. At temperature == 0 - a greedy choice without any randomness.
    """
    raise NotImplementedError


# ==========================================================================
# 17.2 Generation
# ==========================================================================


# The model as generation sees it: token ids (B, T) int64 -> logits (B, T, V), one row of scores
# for the token after each position. A whole transformer is a StepFn, and so is any function with
# that contract; the tests pass small stand-ins.
type StepFn = Callable[[Tensor], Tensor]


def greedy_generate(step_fn: StepFn, prompt: Tensor, max_new_tokens: int) -> Tensor:
    """(B, T_prompt) -> (B, T_prompt + max_new_tokens) int64, the prompt unchanged at the front.

    The naive loop without a cache: at every step the whole sequence so far goes through step_fn,
    and the argmax of the logits at the LAST position is appended to every row.
    Generation never calls backward, so step_fn must run with autograd switched off.
    """
    raise NotImplementedError


def sample_generate(
    step_fn: StepFn,
    prompt: Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> Tensor:
    """The same loop as greedy_generate, but every new token is drawn by sample_from_logits from 17.1
    with these settings. At temperature == 0 the result is exactly greedy_generate's.

    One generator serves the whole generation: the same seed gives the same text, another seed
    another text.
    """
    raise NotImplementedError


def generate_until_eos(step_fn: StepFn, prompt: Tensor, max_new_tokens: int, eos_id: int) -> Tensor:
    """Greedy generation that stops early: (B, T_prompt) -> (B, T_prompt + n), n <= max_new_tokens.

    A row is finished once it has GENERATED eos_id; an eos_id already in the prompt does not count.
    A finished row gets eos_id appended from then on, whatever the model predicts, so all rows keep
    the same length. The loop stops as soon as every row is finished, or after max_new_tokens steps.
    """
    raise NotImplementedError


# ==========================================================================
# 17.3 The KV cache
# ==========================================================================


class KVCache:
    """The store of K and V for one layer.

    append(k, v) appends the new positions along the time axis and returns the full (k, v).
    len(cache) is how many positions have accumulated. The shape of the tensors: (B, H, T, Dh).
    """

    def __init__(self):
        raise NotImplementedError

    def append(self, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class CachedSelfAttention(nn.Module):
    """Causal self-attention that can work step by step.

    forward(x, cache=None):
      - x: (B, T, D) - the whole prefix on the first call, one token at a time afterwards;
      - if a cache is passed, the new k and v are appended to it and the attention is computed
        over the whole accumulated history;
      - the causal mask is only needed when T > 1: a single new token sees the whole past anyway.

    The projections: q_proj, k_proj, v_proj, output_proj without a bias.
    """

    q_proj: nn.Linear
    k_proj: nn.Linear
    v_proj: nn.Linear
    output_proj: nn.Linear

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor, cache: KVCache | None = None) -> Tensor:
        raise NotImplementedError


def cache_memory_bytes(
    batch: int, num_layers: int, num_kv_heads: int, seq_len: int, head_dim: int, dtype: torch.dtype
) -> int:
    """The memory for the KV cache: two tensors (K and V) per layer.

    This exact quantity is the reason GQA and MQA exist.
    """
    raise NotImplementedError


# ==========================================================================
# 17.4 Generation with the cache
# ==========================================================================


def generate_with_cache(step_fn: StepFn, prompt: Tensor, max_new_tokens: int) -> Tensor:
    """Greedy generation driven the way a cached model is driven. The result is the same as
    greedy_generate's, (B, T_prompt + max_new_tokens).

    Here step_fn keeps a KV cache of its own, so it must only be given what it has not seen yet:
    the whole prompt once (the prefill), then just the token picked on the previous step, (B, 1).
    It returns logits for exactly the positions it was given.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_apply_temperature():
    logits = torch.tensor([[1.0, 2.0, 3.0]])
    torch.testing.assert_close(apply_temperature(logits, 2.0), logits / 2.0)
    hot = apply_temperature(logits, 0.5).softmax(-1)
    warm = apply_temperature(logits, 1.0).softmax(-1)
    assert hot.max() > warm.max(), "a low temperature must sharpen the distribution"

    greedy = apply_temperature(logits, 0.0)
    assert greedy.argmax(-1).item() == 2
    assert greedy.softmax(-1).max().item() > 0.999


def test_top_k_filter():
    logits = torch.tensor([[1.0, 5.0, 3.0, 2.0]])
    out = top_k_filter(logits, 2)
    assert torch.isfinite(out).sum().item() == 2
    assert torch.isfinite(out[0, 1]) and torch.isfinite(out[0, 2])
    torch.testing.assert_close(top_k_filter(logits, 0), logits)


def test_top_p_filter():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    out = top_p_filter(logits, 0.8)
    assert torch.isfinite(out[0, 0]) and torch.isfinite(out[0, 1])
    assert torch.isinf(out[0, 2]) and torch.isinf(out[0, 3])

    # p = 1.0 must not cut anything off
    torch.testing.assert_close(top_p_filter(logits, 1.0).softmax(-1), probs, atol=1e-6, rtol=1e-5)
    # even at a tiny p one token is obliged to remain, otherwise there is nothing to sample from
    assert torch.isfinite(top_p_filter(logits, 0.01)).sum().item() >= 1


def test_top_p_keeps_order_of_original_vocab():
    logits = torch.tensor([[0.1, 5.0, 0.2, 4.0]])
    out = top_p_filter(logits, 0.9)
    kept = torch.isfinite(out)[0]
    assert bool(kept[1]) and bool(kept[3]), "the most probable ones are kept, but in their original places"
    assert not bool(kept[0])


@pytest.mark.parametrize("p", [0.1, 0.5, 0.9, 1.0])
def test_top_p_never_empties_the_row(p):
    logits = torch.randn(4, 20)
    out = top_p_filter(logits, p)
    assert (torch.isfinite(out).sum(dim=-1) >= 1).all()
    probs = out.softmax(dim=-1)
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(4), atol=1e-5, rtol=0)


def test_repetition_penalty():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    generated = torch.tensor([[0, 1]])
    out = repetition_penalty(logits, generated, 2.0)
    torch.testing.assert_close(out, torch.tensor([[1.0, -4.0, 1.0]]))


def test_sample_from_logits_is_greedy_at_zero_temperature():
    logits = torch.tensor([[0.1, 0.2, 5.0], [7.0, 0.0, 0.0]])
    out = sample_from_logits(logits, temperature=0.0)
    assert out.shape == (2,)
    assert out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([2, 0]))


def test_sample_from_logits_respects_top_k():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[3.0, 2.9, 0.0, -5.0]])
    samples = torch.stack([sample_from_logits(logits, top_k=2, generator=g) for _ in range(200)])
    assert set(samples.flatten().tolist()) <= {0, 1}, "top-k let the cut-off tokens through"


def test_sample_from_logits_distribution():
    g = torch.Generator().manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    samples = torch.stack([sample_from_logits(logits, generator=g) for _ in range(4000)])
    counts = torch.bincount(samples.flatten(), minlength=4).float() / 4000
    assert (counts - 0.25).abs().max().item() < 0.03


def _ref_plus_one(ids: Tensor, vocab: int = 10) -> Tensor:
    """A stand-in model that always predicts "the next token = this token + 1, modulo vocab"."""
    logits = torch.full((*ids.shape, vocab), -10.0)
    return logits.scatter(-1, ((ids + 1) % vocab).unsqueeze(-1), 10.0)


def test_greedy_generate():
    calls = []

    def step_fn(ids: Tensor) -> Tensor:
        assert not torch.is_grad_enabled(), "generation must run with autograd switched off"
        calls.append(ids.shape[1])
        return _ref_plus_one(ids)

    prompt = torch.tensor([[0, 1], [7, 8]])
    out = greedy_generate(step_fn, prompt, max_new_tokens=3)
    assert out.shape == (2, 5) and out.dtype == torch.int64
    torch.testing.assert_close(out, torch.tensor([[0, 1, 2, 3, 4], [7, 8, 9, 0, 1]]))
    assert calls == [2, 3, 4], "without a cache every step runs the whole sequence"


@pytest.mark.depends_on("sample_from_logits")
def test_sample_generate_draws_only_what_the_filters_allow():
    row = torch.tensor([3.0, 2.9, 0.0, -5.0, -5.0, -5.0])

    def step_fn(ids: Tensor) -> Tensor:
        assert not torch.is_grad_enabled(), "generation must run with autograd switched off"
        return row.expand(*ids.shape, -1)

    prompt = torch.tensor([[4], [5], [4], [5]])
    g = torch.Generator().manual_seed(0)
    out = sample_generate(step_fn, prompt, max_new_tokens=20, top_k=2, generator=g)
    assert out.shape == (4, 21) and out.dtype == torch.int64
    torch.testing.assert_close(out[:, :1], prompt)
    assert set(out[:, 1:].flatten().tolist()) == {0, 1}, "top-k=2 must allow tokens 0 and 1, and only them"


@pytest.mark.depends_on("sample_from_logits")
def test_sample_generate_seed_and_greedy_limit():
    def step_fn(ids: Tensor) -> Tensor:
        return torch.zeros(*ids.shape, 50)

    prompt = torch.zeros(2, 3, dtype=torch.long)

    def run(seed: int) -> Tensor:
        return sample_generate(step_fn, prompt, 16, generator=torch.Generator().manual_seed(seed))

    torch.testing.assert_close(run(0), run(0))
    assert not torch.equal(run(0), run(1)), "another seed must give another text"
    assert len(set(run(0)[0, 3:].tolist())) > 1, "every step must continue the generator's stream"

    out = sample_generate(_ref_plus_one, torch.tensor([[0, 1], [7, 8]]), 3, temperature=0.0)
    torch.testing.assert_close(out, torch.tensor([[0, 1, 2, 3, 4], [7, 8, 9, 0, 1]]))


def test_generate_until_eos_stops_and_pads():
    calls = []

    def step_fn(ids: Tensor) -> Tensor:
        assert not torch.is_grad_enabled(), "generation must run with autograd switched off"
        calls.append(ids.shape[1])
        return _ref_plus_one(ids)

    prompt = torch.tensor([[0, 1], [3, 4]])
    out = generate_until_eos(step_fn, prompt, max_new_tokens=10, eos_id=5)
    # the second row says 5 on the first step and is then padded, although the model goes on to 6, 7, ...
    torch.testing.assert_close(out, torch.tensor([[0, 1, 2, 3, 4, 5], [3, 4, 5, 5, 5, 5]]))
    assert len(calls) == 4, "the loop must stop once every row has produced EOS"


def test_generate_until_eos_ignores_the_prompt_and_respects_the_limit():
    out = generate_until_eos(_ref_plus_one, torch.tensor([[5, 6]]), max_new_tokens=3, eos_id=5)
    torch.testing.assert_close(out, torch.tensor([[5, 6, 7, 8, 9]]))


def test_kv_cache_append():
    cache = KVCache()
    assert len(cache) == 0
    k1, v1 = torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)
    k, v = cache.append(k1, v1)
    assert k.shape == (1, 2, 3, 4) and len(cache) == 3

    k2, v2 = torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4)
    k, v = cache.append(k2, v2)
    assert k.shape == (1, 2, 4, 4) and len(cache) == 4
    torch.testing.assert_close(k[:, :, :3], k1)
    torch.testing.assert_close(v[:, :, 3:], v2)


def test_cached_self_attention_without_cache():
    """With no cache passed in it is plain causal self-attention - that part stands on its own."""
    torch.manual_seed(0)
    attn = CachedSelfAttention(32, 4)
    names = {n for n, _ in attn.named_parameters()}
    assert names == {"q_proj.weight", "k_proj.weight", "v_proj.weight", "output_proj.weight"}

    x = torch.randn(2, 6, 32)
    out = attn(x)
    assert out.shape == (2, 6, 32)

    q = attn.q_proj(x).unflatten(-1, (4, 8)).transpose(1, 2)
    k = attn.k_proj(x).unflatten(-1, (4, 8)).transpose(1, 2)
    v = attn.v_proj(x).unflatten(-1, (4, 8)).transpose(1, 2)
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = attn.output_proj(ref.transpose(1, 2).reshape(2, 6, 32))
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.depends_on("KVCache", "CachedSelfAttention")
def test_incremental_decoding_matches_full_forward():
    """The main test of the topic: step-by-step generation with a cache is obliged to give the same numbers
    as a single pass over the whole sequence."""
    torch.manual_seed(0)
    attn = CachedSelfAttention(32, 4)
    x = torch.randn(2, 6, 32)

    full = attn(x)

    cache = KVCache()
    steps = [attn(x[:, t : t + 1], cache=cache) for t in range(6)]
    incremental = torch.cat(steps, dim=1)

    torch.testing.assert_close(incremental, full, atol=1e-5, rtol=1e-5)
    assert len(cache) == 6


@pytest.mark.depends_on("KVCache", "CachedSelfAttention")
def test_prefill_then_decode():
    """The real mode: first a prefill of the whole prompt, then one token at a time."""
    torch.manual_seed(0)
    attn = CachedSelfAttention(16, 2)
    x = torch.randn(1, 5, 16)
    full = attn(x)

    cache = KVCache()
    prefill = attn(x[:, :3], cache=cache)
    assert len(cache) == 3
    torch.testing.assert_close(prefill, full[:, :3], atol=1e-5, rtol=1e-5)

    rest = torch.cat([attn(x[:, t : t + 1], cache=cache) for t in (3, 4)], dim=1)
    torch.testing.assert_close(rest, full[:, 3:], atol=1e-5, rtol=1e-5)


def test_cache_memory_bytes():
    # LLaMA-2-7B: 32 layers, 32 kv heads, head_dim 128, bf16, batch 1, a context of 4096
    mha = cache_memory_bytes(1, 32, 32, 4096, 128, torch.bfloat16)
    assert mha == 2 * 1 * 32 * 32 * 4096 * 128 * 2
    # GQA with 8 kv heads shrinks the cache exactly fourfold
    gqa = cache_memory_bytes(1, 32, 8, 4096, 128, torch.bfloat16)
    assert mha // gqa == 4


def test_generate_with_cache_feeds_only_new_tokens():
    seen = []

    def step_fn(new_ids: Tensor) -> Tensor:
        assert not torch.is_grad_enabled(), "generation must run with autograd switched off"
        seen.append(new_ids.clone())
        return _ref_plus_one(new_ids)

    prompt = torch.tensor([[0, 1], [7, 8]])
    out = generate_with_cache(step_fn, prompt, max_new_tokens=3)
    torch.testing.assert_close(out, torch.tensor([[0, 1, 2, 3, 4], [7, 8, 9, 0, 1]]))

    assert len(seen) == 3, "the prefill, then one call per token still to pick"
    torch.testing.assert_close(seen[0], prompt)
    torch.testing.assert_close(seen[1], torch.tensor([[2], [9]]))
    torch.testing.assert_close(seen[2], torch.tensor([[3], [0]]))


@pytest.mark.depends_on("KVCache", "CachedSelfAttention", "greedy_generate", "generate_with_cache")
def test_cached_generation_matches_uncached():
    """A tiny one-layer model, generated from twice: rerunning the whole prefix, and through the cache."""
    torch.manual_seed(0)
    V, D = 13, 16
    emb = nn.Embedding(V, D)
    attn = CachedSelfAttention(D, 2)
    head = nn.Linear(D, V, bias=False)

    def full(ids: Tensor) -> Tensor:
        return head(attn(emb(ids)))

    cache = KVCache()

    def cached(new_ids: Tensor) -> Tensor:
        return head(attn(emb(new_ids), cache=cache))

    prompt = torch.randint(0, V, (2, 4))
    torch.testing.assert_close(generate_with_cache(cached, prompt, 8), greedy_generate(full, prompt, 8))
    assert len(cache) == 4 + 8 - 1, "the last token picked is never fed back"
