"""Топик 19 (самый сложный) - продвинутые механизмы современных LLM.

Пять сюжетов, каждый - отдельная статья, которая стала стандартом:
  1. свой backward для RMSNorm, выведенный на бумаге (основа кастомных ядер);
  2. маски: скользящее окно (Mistral) и упаковка документов в один пример;
  3. Mixture-of-Experts: маршрутизация top-2, ёмкость экспертов, лосс балансировки;
  4. спекулятивное декодирование: как получить точно то же распределение, но быстрее;
  5. beam search: шаг луча через плоский topk.

Если этот файл проходит целиком - у тебя в руках весь понятийный аппарат
современного инференса и обучения LLM.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 19.1 Свой backward для RMSNorm
# ==========================================================================


class RMSNormFunction(torch.autograd.Function):
    """y_i = g_i * x_i * r, где r = 1 / sqrt(mean(x^2) + eps), статистики по последней оси.

    Вывод backward (сделай его сам, ответ ниже нужен только для сверки):
        r = (mean(x^2) + eps)^(-1/2)
        dr/dx_j = -x_j * r^3 / D
        пусть a_i = grad_y_i * g_i, тогда
        dL/dx_j = a_j * r - x_j * r^3 * sum_i(a_i * x_i) / D
        dL/dg_i = sum по всем ведущим осям от grad_y_i * x_i * r

    Реализуй forward и backward. backward обязан вернуть градиенты по (x, weight, eps),
    причём для eps это None.
    """

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        raise NotImplementedError


# ==========================================================================
# 19.2 Маски посложнее
# ==========================================================================


def sliding_window_mask(seq_len: int, window: int) -> Tensor:
    """(T, T) bool: позиция i видит позиции j, для которых i - window < j <= i.

    То есть причинность плюс ограничение "не дальше window назад" (Mistral, Longformer).
    При window >= T должно совпасть с обычной причинной маской.
    """
    raise NotImplementedError


def document_mask(doc_ids: Tensor) -> Tensor:
    """doc_ids: (T,) int64 - номер документа для каждой позиции упакованной последовательности.
    Вернуть (T, T) bool: причинность И запрет смотреть в чужой документ.

    Так экономят вычисления при обучении: несколько коротких документов пакуют в один
    пример длины T, но внимание между ними должно быть заблокировано.
    """
    raise NotImplementedError


def block_diagonal_lengths_to_doc_ids(lengths: list[int]) -> Tensor:
    """[3, 2, 4] -> tensor([0,0,0,1,1,2,2,2,2]). Вспомогательное к document_mask."""
    raise NotImplementedError


# ==========================================================================
# 19.3 Mixture of Experts
# ==========================================================================


def top2_routing(router_logits: Tensor) -> tuple[Tensor, Tensor]:
    """router_logits: (N, E) -> (индексы экспертов (N, 2) int64, веса (N, 2)).

    Веса - softmax ТОЛЬКО по двум выбранным логитам (как в Mixtral), то есть они
    суммируются в 1 для каждого токена. Индексы отсортированы по убыванию логита.
    """
    raise NotImplementedError


def capacity_mask(expert_idx: Tensor, num_experts: int, capacity: int) -> Tensor:
    """expert_idx: (N, K) - к каким экспертам отправлен каждый токен.
    Вернуть (N, K) bool: True, если токен помещается в ёмкость эксперта.

    Правило: у каждого эксперта не больше `capacity` слотов, занимают их токены
    в порядке возрастания индекса токена; сначала слот номер 0 (самый уверенный выбор),
    затем слот 1. Лишние токены "дропаются" - их вклад просто теряется.

    Без циклов по токенам: cumsum по one-hot даёт позицию токена в очереди эксперта.
    """
    raise NotImplementedError


def load_balancing_loss(router_probs: Tensor, top1_idx: Tensor, num_experts: int) -> Tensor:
    """Вспомогательный лосс из Switch Transformer:
        loss = E * sum_e f_e * P_e,
    где f_e - доля токенов, ушедших к эксперту e (по top-1),
        P_e - средняя вероятность эксперта e по всем токенам.

    При идеально равномерной маршрутизации это ровно 1.0, при вырожденной - E.
    """
    raise NotImplementedError


def moe_forward(x: Tensor, router: Tensor, w1: Tensor, w2: Tensor) -> Tensor:
    """Разреженный MoE-слой.

    x: (N, D), router: (E, D), w1: (E, D_ff, D), w2: (E, D, D_ff).
    Каждый токен идёт к двум экспертам (top-2 по router-логитам), выходы складываются
    с весами маршрутизации. Эксперт: out = w2_e @ relu(w1_e @ x).

    Идея реализации: не гонять всех через всех. Для каждого эксперта выбери токены,
    которые к нему попали (маска и индексация), посчитай только их, и сложи обратно
    через index_add_. Именно так это делают настоящие реализации.
    """
    raise NotImplementedError


# ==========================================================================
# 19.4 Спекулятивное декодирование
# ==========================================================================


def speculative_accept(
    draft_probs: Tensor, target_probs: Tensor, draft_tokens: Tensor, u: Tensor
) -> tuple[int, Tensor]:
    """Правило приёма из статьи про speculative decoding.

    draft_probs: (K, V) - распределения черновой модели на K предложенных позициях;
    target_probs: (K+1, V) - распределения большой модели на тех же позициях плюс бонусная;
    draft_tokens: (K,) - что предложила черновая модель;
    u: (K,) - равномерные случайные числа из [0, 1), поданы снаружи ради детерминизма.

    Для позиции i принимаем токен, если u[i] <= p_target(token) / p_draft(token).
    При первом отказе на позиции j возвращаем (j, распределение norm(relu(p_target - p_draft))
    в этой позиции). Если приняты все K, возвращаем (K, target_probs[K]) - бонусный токен
    достаётся бесплатно.

    Ключевое свойство схемы: итоговое распределение токенов В ТОЧНОСТИ совпадает
    с распределением большой модели. Ускорение берётся из того, что большая модель
    проверяет K токенов за один проход.
    """
    raise NotImplementedError


def expected_speedup(acceptance_rate: float, k: int) -> float:
    """Ожидаемое число токенов за один цикл проверки при вероятности приёма alpha:
        (1 - alpha^(k+1)) / (1 - alpha),
    при alpha == 1 это просто k + 1.
    """
    raise NotImplementedError


# ==========================================================================
# 19.5 Beam search
# ==========================================================================


def beam_search_step(beam_scores: Tensor, next_log_probs: Tensor, beam_size: int) -> tuple[Tensor, Tensor, Tensor]:
    """beam_scores: (num_beams,) - накопленные логарифмические вероятности лучей;
    next_log_probs: (num_beams, V) - логарифмы вероятностей следующего токена.

    Вернуть (новые счёта (beam_size,), индексы лучей-родителей (beam_size,),
    индексы токенов (beam_size,)), выбрав beam_size лучших вариантов из всех num_beams * V.

    Приём, который надо запомнить: сложить, сплющить в (num_beams * V,), взять topk,
    восстановить пары через divmod по размеру словаря.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
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

    # эталон - обычный autograd через элементарные операции
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
    # четыре токена, все первым выбором идут к эксперту 0, ёмкость 2
    expert_idx = torch.tensor([[0, 1], [0, 1], [0, 1], [0, 1]])
    m = capacity_mask(expert_idx, num_experts=2, capacity=2)
    torch.testing.assert_close(m[:, 0], torch.tensor([True, True, False, False]))
    torch.testing.assert_close(m[:, 1], torch.tensor([True, True, False, False]))

    # при достаточной ёмкости не дропается ничего
    big = capacity_mask(expert_idx, num_experts=2, capacity=10)
    assert big.all()


def test_capacity_mask_slot_order():
    """Слот 0 (главный эксперт токена) занимает ёмкость раньше слота 1."""
    expert_idx = torch.tensor([[1, 0], [0, 1]])
    m = capacity_mask(expert_idx, num_experts=2, capacity=1)
    # эксперт 1: токен 0 приходит слотом 0, токен 1 - слотом 1 -> побеждает токен 0
    assert bool(m[0, 0]) and not bool(m[1, 1])
    # эксперт 0: токен 1 приходит слотом 0, токен 0 - слотом 1 -> побеждает токен 1
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
    """Если занулить всех экспертов, кроме выбранных для токена, выход не изменится."""
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
    # первая позиция: p/q = 1, принимаем при любом u; вторая: p/q = 0.111, u = 0.5 -> отказ
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
    # счета обязаны быть отсортированы по убыванию
    assert (scores[:-1] >= scores[1:]).all()
    for i in range(3):
        torch.testing.assert_close(scores[i], beam_scores[beam_idx[i]] + log_probs[beam_idx[i], token_idx[i]])


@pytest.mark.parametrize("window", [1, 2, 5, 100])
def test_sliding_window_counts(window):
    T = 10
    m = sliding_window_mask(T, window)
    for i in range(T):
        assert int(m[i].sum()) == min(i + 1, window)
