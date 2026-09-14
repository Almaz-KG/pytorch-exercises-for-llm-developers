"""Topic 19 (the hardest) - the advanced mechanisms of modern LLMs.

Five storylines, each one a separate paper that became a standard:
  1. a backward of your own for RMSNorm, derived on paper (the basis of custom kernels);
  2. masks: the sliding window (Mistral) and packing documents into one example;
  3. Mixture-of-Experts: top-2 routing, expert capacity, the balancing loss;
  4. speculative decoding: how to get exactly the same distribution, but faster;
  5. beam search: a beam step through a flat topk.

If this file passes in full, you hold the whole conceptual apparatus
of modern LLM inference and training.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 19.1 A backward of your own for RMSNorm
# ==========================================================================


class RMSNormFunction(torch.autograd.Function):
    """y_i = g_i * x_i * r, where r = 1 / sqrt(mean(x^2) + eps), the statistics over the last axis.

    The derivation of backward (do it yourself, the answer below is only for cross-checking):
        r = (mean(x^2) + eps)^(-1/2)
        dr/dx_j = -x_j * r^3 / D
        let a_i = grad_y_i * g_i, then
        dL/dx_j = a_j * r - x_j * r^3 * sum_i(a_i * x_i) / D
        dL/dg_i = the sum over all the leading axes of grad_y_i * x_i * r

    Implement forward and backward. backward is obliged to return the gradients for (x, weight, eps),
    and for eps that is None.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        raise NotImplementedError


# ==========================================================================
# 19.2 Harder masks
# ==========================================================================


def sliding_window_mask(seq_len: int, window: int) -> Tensor:
    """(T, T) bool: position i sees the positions j for which i - window < j <= i.

    That is, causality plus the restriction "no further back than window" (Mistral, Longformer).
    At window >= T it must coincide with an ordinary causal mask.
    """
    raise NotImplementedError


def document_mask(doc_ids: Tensor) -> Tensor:
    """doc_ids: (T,) int64 - the document number for every position of the packed sequence.
    Return (T, T) bool: causality AND a ban on looking into somebody else's document.

    This is how computation is saved during training: several short documents are packed into one
    example of length T, but the attention between them has to be blocked.
    """
    raise NotImplementedError


def block_diagonal_lengths_to_doc_ids(lengths: list[int]) -> Tensor:
    """[3, 2, 4] -> tensor([0,0,0,1,1,2,2,2,2]). A helper for document_mask."""
    raise NotImplementedError


# ==========================================================================
# 19.3 Mixture of Experts
# ==========================================================================


def top2_routing(router_logits: Tensor) -> tuple[Tensor, Tensor]:
    """router_logits: (N, E) -> (the expert indices (N, 2) int64, the weights (N, 2)).

    The weights are a softmax over the two chosen logits ONLY (as in Mixtral), that is, they
    sum to 1 for every token. The indices are sorted by descending logit.
    """
    raise NotImplementedError


def capacity_mask(expert_idx: Tensor, num_experts: int, capacity: int) -> Tensor:
    """expert_idx: (N, K) - which experts every token was sent to.
    Return (N, K) bool: True if the token fits into the capacity of the expert.

    The rule: every expert has no more than `capacity` slots, and tokens occupy them
    in order of increasing token index; slot number 0 first (the most confident choice),
    then slot 1. The surplus tokens are "dropped" - their contribution is simply lost.

    Without loops over the tokens: a cumsum over the one-hot gives the position of a token in the expert's queue.
    """
    raise NotImplementedError


def load_balancing_loss(router_probs: Tensor, top1_idx: Tensor, num_experts: int) -> Tensor:
    """The auxiliary loss from the Switch Transformer:
        loss = E * sum_e f_e * P_e,
    where f_e is the fraction of tokens that went to expert e (by top-1),
        P_e is the mean probability of expert e over all the tokens.

    Under perfectly uniform routing this is exactly 1.0, under a degenerate one it is E.
    """
    raise NotImplementedError


def moe_forward(x: Tensor, router: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    """A sparse MoE layer.

    x: (N, D), router: (E, D), w1: (E, D_ff, D), w2: (E, D, D_ff).
    Every token goes to two experts (top-2 by the router logits), and the outputs are added
    with the routing weights. An expert: out = w2_e @ relu(w1_e @ x).

    The idea of the implementation: do not push everyone through everyone. For every expert pick the tokens
    that landed on it (a mask and indexing), compute only those, and add them back
    through index_add_. This is exactly what real implementations do.
    """
    raise NotImplementedError


# ==========================================================================
# 19.4 Speculative decoding
# ==========================================================================


def speculative_accept(
    draft_probs: Tensor, target_probs: Tensor, draft_tokens: Tensor, u: Tensor
) -> tuple[int, Tensor]:
    """The acceptance rule from the speculative decoding paper.

    draft_probs: (K, V) - the distributions of the draft model at the K proposed positions;
    target_probs: (K+1, V) - the distributions of the big model at the same positions plus the bonus one;
    draft_tokens: (K,) - what the draft model proposed;
    u: (K,) - uniform random numbers from [0, 1), passed in from outside for the sake of determinism.

    At position i we accept the token if u[i] <= p_target(token) / p_draft(token).
    On the first rejection, at position j, we return (j, the distribution norm(relu(p_target - p_draft))
    at that position). If all K are accepted, we return (K, target_probs[K]) - the bonus token
    comes for free.

    The key property of the scheme: the resulting distribution of tokens matches the distribution
    of the big model EXACTLY. The speedup comes from the big model
    checking K tokens in a single pass.
    """
    raise NotImplementedError


def expected_speedup(acceptance_rate: float, k: int) -> float:
    """The expected number of tokens per verification cycle at an acceptance probability alpha:
        (1 - alpha^(k+1)) / (1 - alpha),
    and at alpha == 1 it is simply k + 1.
    """
    raise NotImplementedError


# ==========================================================================
# 19.5 Beam search
# ==========================================================================


def beam_search_step(beam_scores: Tensor, next_log_probs: Tensor, beam_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """beam_scores: (num_beams,) - the accumulated log probabilities of the beams;
    next_log_probs: (num_beams, V) - the log probabilities of the next token.

    Return (the new scores (beam_size,), the indices of the parent beams (beam_size,),
    the token indices (beam_size,)), having picked the beam_size best variants out of all num_beams * V.

    The trick worth remembering: add, flatten into (num_beams * V,), take a topk,
    and recover the pairs through a divmod by the vocabulary size.
    """
    raise NotImplementedError


# ==========================================================================
# TESTS
# ==========================================================================


def test_rms_norm_function_forward():
    x = torch.randn(2, 3, 8)
    w = torch.randn(8)
    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    torch.testing.assert_close(RMSNormFunction.apply(x, w, 1e-5), expected, atol=1e-6, rtol=1e-5)


def test_rms_norm_function_backward_matches_autograd():
    x = torch.randn(4, 6, 16, requires_grad=True)
    w = torch.randn(16, requires_grad=True)
    g = torch.randn(4, 6, 16)

    # the reference is ordinary autograd through elementary operations
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * w
    ref.backward(g)
    ref_gx, ref_gw = x.grad.clone(), w.grad.clone()

    x.grad = None
    w.grad = None
    out = RMSNormFunction.apply(x, w, 1e-5)
    out.backward(g)
    torch.testing.assert_close(x.grad, ref_gx, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(w.grad, ref_gw, atol=1e-5, rtol=1e-4)


def test_rms_norm_function_gradcheck():
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: RMSNormFunction.apply(a, b, 1e-5), (x, w))


def test_sliding_window_mask():
    m = sliding_window_mask(5, window=3)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)

    idx = torch.arange(6)
    torch.testing.assert_close(sliding_window_mask(6, 99), (idx[:, None] >= idx[None, :]))


def test_document_mask():
    doc_ids = block_diagonal_lengths_to_doc_ids([2, 3])
    torch.testing.assert_close(doc_ids, torch.tensor([0, 0, 1, 1, 1]))

    m = document_mask(doc_ids)
    expected = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(m, expected)


def test_top2_routing():
    logits = torch.tensor([[1.0, 3.0, 2.0, 0.0], [5.0, 0.0, 0.0, 4.0]])
    idx, weights = top2_routing(logits)
    torch.testing.assert_close(idx, torch.tensor([[1, 2], [0, 3]]))
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(weights[0], torch.tensor([3.0, 2.0]).softmax(0))
    assert (weights[:, 0] >= weights[:, 1]).all()


def test_capacity_mask():
    # four tokens, all of them going to expert 0 as their first choice, capacity 2
    expert_idx = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    m = capacity_mask(expert_idx, num_experts=2, capacity=2)
    torch.testing.assert_close(m[:, 0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(m[:, 1], torch.tensor([True, True, False, False]))

    # with enough capacity nothing gets dropped
    big = capacity_mask(expert_idx, num_experts=2, capacity=10)
    assert big.all()


def test_capacity_mask_slot_order():
    """Slot 0 (the token's main expert) takes up capacity before slot 1."""
    expert_idx = torch.tensor([[1, 0], [0, 1]])
    m = capacity_mask(expert_idx, num_experts=2, capacity=1)
    # expert 1: token 0 arrives as slot 0, token 1 as slot 1 -> token 0 wins
    assert bool(m[0, 0]) and not bool(m[1, 1])
    # expert 0: token 1 arrives as slot 0, token 0 as slot 1 -> token 1 wins
    assert bool(m[1, 0]) and not bool(m[0, 1])


def test_load_balancing_loss():
    E = 4
    uniform_probs = torch.full((8, E), 1 / E)
    uniform_idx = torch.arange(8) % E
    torch.testing.assert_close(load_balancing_loss(uniform_probs, uniform_idx, E), torch.tensor(1.0), atol=1e-6, rtol=0)

    collapsed = torch.zeros(8, E)
    collapsed[:, 0] = 1.0
    collapsed_idx = torch.zeros(8, dtype=torch.int64)
    torch.testing.assert_close(
        load_balancing_loss(collapsed, collapsed_idx, E), torch.tensor(float(E)), atol=1e-6, rtol=0
    )


def test_moe_forward_matches_dense_reference():
    torch.manual_seed(0)
    N, D, E, D_ff = 12, 8, 4, 16
    x = torch.randn(N, D)
    router = torch.randn(E, D)
    w1 = torch.randn(E, D_ff, D) * 0.1
    w2 = torch.randn(E, D, D_ff) * 0.1

    idx, weights = top2_routing(x @ router.T)
    expected = torch.zeros(N, D)
    for n in range(N):
        for slot in range(2):
            e = int(idx[n, slot])
            h = F.relu(w1[e] @ x[n])
            expected[n] += weights[n, slot] * (w2[e] @ h)

    torch.testing.assert_close(moe_forward(x, router, w1, w2), expected, atol=1e-5, rtol=1e-4)


def test_moe_uses_only_two_experts_per_token():
    """If every expert except the ones chosen for a token is zeroed out, the output must not change."""
    torch.manual_seed(0)
    N, D, E, D_ff = 6, 8, 4, 16
    x = torch.randn(N, D)
    router = torch.randn(E, D)
    w1 = torch.randn(E, D_ff, D) * 0.1
    w2 = torch.randn(E, D, D_ff) * 0.1

    idx, _ = top2_routing(x @ router.T)
    out = moe_forward(x, router, w1, w2)

    unused = set(range(E)) - set(idx.flatten().tolist())
    if unused:
        w1_mod, w2_mod = w1.clone(), w2.clone()
        for e in unused:
            w1_mod[e].zero_()
            w2_mod[e].zero_()
        torch.testing.assert_close(moe_forward(x, router, w1_mod, w2_mod), out, atol=1e-5, rtol=1e-4)


def test_speculative_accept_all_accepted():
    draft = torch.full((3, 4), 0.25)
    target = torch.full((4, 4), 0.25)
    tokens = torch.tensor([0, 1, 2])
    n, dist = speculative_accept(draft, target, tokens, u=torch.zeros(3))
    assert n == 3
    torch.testing.assert_close(dist, target[3])


def test_speculative_accept_rejection():
    draft = torch.tensor([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]])
    target = torch.tensor([[0.9, 0.05, 0.05], [0.1, 0.6, 0.3], [0.2, 0.3, 0.5]])
    tokens = torch.tensor([0, 0])
    # the first position: p/q = 1, we accept at any u; the second: p/q = 0.111, u = 0.5 -> a rejection
    n, dist = speculative_accept(draft, target, tokens, u=torch.tensor([0.5, 0.5]))
    assert n == 1
    residual = torch.clamp(target[1] - draft[1], min=0)
    torch.testing.assert_close(dist, residual / residual.sum(), atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(dist.sum(), torch.tensor(1.0))


def test_speculative_accept_is_a_distribution():
    torch.manual_seed(0)
    V, K = 8, 4
    draft = torch.rand(K, V).softmax(-1)
    target = torch.rand(K + 1, V).softmax(-1)
    tokens = torch.randint(0, V, (K,))
    for u_val in (0.0, 0.3, 0.7, 0.99):
        n, dist = speculative_accept(draft, target, tokens, u=torch.full((K,), u_val))
        assert 0 <= n <= K
        assert torch.isfinite(dist).all()
        torch.testing.assert_close(dist.sum(), torch.tensor(1.0), atol=1e-5, rtol=0)


def test_expected_speedup():
    assert abs(expected_speedup(1.0, 4) - 5.0) < 1e-9
    assert abs(expected_speedup(0.0, 4) - 1.0) < 1e-9
    assert abs(expected_speedup(0.8, 4) - (1 - 0.8**5) / 0.2) < 1e-9
    assert expected_speedup(0.9, 8) > expected_speedup(0.5, 8)


def test_beam_search_step():
    beam_scores = torch.tensor([-1.0, -2.0])
    next_log_probs = torch.tensor([[-0.5, -3.0, -4.0], [-0.1, -0.2, -5.0]])
    scores, beam_idx, token_idx = beam_search_step(beam_scores, next_log_probs, beam_size=2)
    torch.testing.assert_close(scores, torch.tensor([-1.5, -2.1]))
    torch.testing.assert_close(beam_idx, torch.tensor([0, 1]))
    torch.testing.assert_close(token_idx, torch.tensor([0, 0]))


def test_beam_search_step_shapes():
    torch.manual_seed(0)
    beam_scores = torch.randn(4)
    log_probs = torch.randn(4, 50).log_softmax(-1)
    scores, beam_idx, token_idx = beam_search_step(beam_scores, log_probs, beam_size=3)
    assert scores.shape == beam_idx.shape == token_idx.shape == (3,)
    assert (beam_idx < 4).all() and (token_idx < 50).all()
    # the scores are obliged to be sorted in descending order
    assert (scores[:-1] >= scores[1:]).all()
    for i in range(3):
        torch.testing.assert_close(scores[i], beam_scores[beam_idx[i]] + log_probs[beam_idx[i], token_idx[i]])


@pytest.mark.parametrize("window", [1, 2, 5, 100])
def test_sliding_window_counts(window):
    T = 10
    m = sliding_window_mask(T, window)
    for i in range(T):
        assert int(m[i].sum()) == min(i + 1, window)
