"""Топик 09 - nn.Module: параметры, буферы, state_dict, инициализация.

Ключевые различия, которые надо чувствовать:
  - Parameter -> обучается, лежит в .parameters() и в state_dict;
  - buffer -> НЕ обучается, но лежит в state_dict и едет на device вместе с модулем
    (типичный пример: маска или RoPE-кеш);
  - обычный атрибут-тензор -> не едет никуда и теряется при .to(device). Классический баг.
  - список модулей надо класть в nn.ModuleList, иначе параметров как будто нет.
"""

import io

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ==========================================================================
# 9.1 Свои слои
# ==========================================================================


class MyLinear(nn.Module):
    """Линейный слой без сюрпризов: вес (out_features, in_features), опциональный bias.

    Требования:
      - параметры называются `weight` и `bias` (это фиксирует state_dict);
      - при bias=False атрибут bias должен быть None и НЕ появляться в state_dict;
      - forward работает с любым числом ведущих осей.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MyEmbedding(nn.Module):
    """Таблица эмбеддингов: параметр `weight` формы (num_embeddings, embedding_dim)."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, ids: Tensor) -> Tensor:
        raise NotImplementedError


class CausalSelfAttentionSkeleton(nn.Module):
    """Только про буферы: слой хранит причинную маску (max_len, max_len) bool как БУФЕР
    с именем `mask`, а forward возвращает срез маски под текущую длину.

    Так делают все реализации: маску считают один раз, а не на каждом шаге.
    """

    def __init__(self, max_len: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, seq_len: int) -> Tensor:
        raise NotImplementedError


class BlockStack(nn.Module):
    """Стек из n одинаковых по форме линейных слоёв d->d, применяемых последовательно.

    Слои обязаны лежать в nn.ModuleList (обычный python-список не регистрирует параметры).
    """

    def __init__(self, d_model: int, n_layers: int):
        super().__init__()
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


# ==========================================================================
# 9.2 Инспекция модели
# ==========================================================================


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Число элементов (не тензоров) в параметрах модели."""
    raise NotImplementedError


def parameter_names(model: nn.Module) -> list[str]:
    """Отсортированный список полных имён параметров, как они лежат в state_dict."""
    raise NotImplementedError


def total_bytes(model: nn.Module) -> int:
    """Сколько байт занимают все параметры и буферы модели."""
    raise NotImplementedError


# ==========================================================================
# 9.3 Инициализация
# ==========================================================================


def init_linear_(layer: nn.Linear, std: float) -> nn.Linear:
    """Инициализировать вес усечённым нормальным N(0, std) в пределах [-3*std, 3*std],
    bias занулить. Вернуть тот же слой.

    Подсказка: nn.init.trunc_normal_ и nn.init.zeros_. Всё in-place, под torch.no_grad
    (функции из nn.init уже это делают за тебя).
    """
    raise NotImplementedError


def init_all_linears_(model: nn.Module, std: float) -> nn.Module:
    """Пройти по всему дереву модулей и инициализировать КАЖДЫЙ nn.Linear через init_linear_.

    Подсказка: model.apply(fn) вызывает fn на каждом подмодуле.
    """
    raise NotImplementedError


def scaled_residual_std(base_std: float, n_layers: int) -> float:
    """GPT-2-трюк: веса, пишущие в остаточный поток, инициализируют с std / sqrt(2*n_layers),
    иначе дисперсия резидуала растёт с глубиной. Верни это значение.
    """
    raise NotImplementedError


# ==========================================================================
# 9.4 state_dict
# ==========================================================================


def save_load_roundtrip(model: nn.Module) -> nn.Module:
    """Сохранить state_dict в байтовый буфер (io.BytesIO) через torch.save,
    создать НОВУЮ модель того же класса и формы, загрузить в неё веса, вернуть её.

    Для простоты считаем, что модель - это MyLinear(in, out).
    """
    raise NotImplementedError


def strip_prefix(state_dict: dict[str, Tensor], prefix: str) -> dict[str, Tensor]:
    """Убрать префикс из ключей (например `_orig_mod.`, который дописывает torch.compile,
    или `module.`, который дописывает DDP). Ключи без префикса оставить как есть.

    Ровно эта функция живёт в conftest.py первого ассайнмента CS336.
    """
    raise NotImplementedError


def tie_weights(embedding: MyEmbedding, head: MyLinear) -> None:
    """Связать веса эмбеддинга и выходной проекции (weight tying): один и тот же тензор.

    После вызова изменение одного должно быть видно в другом, а число уникальных
    параметров модели - уменьшиться.
    """
    raise NotImplementedError


# ==========================================================================
# 9.5 Режимы и хуки
# ==========================================================================


def dropout_behaviour(p: float, x: Tensor) -> tuple[Tensor, Tensor]:
    """Создать nn.Dropout(p) и вернуть (выход в train-режиме, выход в eval-режиме).

    Смысл: убедиться руками, что model.eval() выключает dropout, а не "уменьшает" его.
    """
    raise NotImplementedError


def capture_activations(model: nn.Sequential, x: Tensor) -> dict[str, Tensor]:
    """Повесить forward-хуки на все подмодули и вернуть словарь {имя_модуля: выход}.

    Это основной способ дебажить "где взорвались активации" в большой модели.
    Хуки после использования лучше снимать.
    """
    raise NotImplementedError


# ==========================================================================
# ТЕСТЫ
# ==========================================================================


def test_my_linear_matches_functional():
    layer = MyLinear(4, 6)
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight, layer.bias))
    assert layer(x).shape == (2, 3, 6)
    assert isinstance(layer.weight, nn.Parameter)
    assert layer.weight.shape == (6, 4)
    assert set(layer.state_dict().keys()) == {"weight", "bias"}


def test_my_linear_no_bias():
    layer = MyLinear(4, 6, bias=False)
    assert layer.bias is None
    assert set(layer.state_dict().keys()) == {"weight"}
    x = torch.randn(5, 4)
    torch.testing.assert_close(layer(x), x @ layer.weight.T)


def test_my_embedding():
    emb = MyEmbedding(10, 4)
    ids = torch.tensor([[1, 2], [3, 3]])
    out = emb(ids)
    assert out.shape == (2, 2, 4)
    torch.testing.assert_close(out, emb.weight[ids])
    assert count_parameters(emb) == 40


def test_buffer_is_not_a_parameter():
    layer = CausalSelfAttentionSkeleton(8)
    assert "mask" in dict(layer.named_buffers()), "маска должна быть буфером"
    assert "mask" not in dict(layer.named_parameters())
    assert "mask" in layer.state_dict()
    assert count_parameters(layer) == 0

    m = layer(3)
    assert m.shape == (3, 3)
    assert m.dtype == torch.bool
    assert bool(m[0, 0]) and not bool(m[0, 1]) and bool(m[2, 1])


def test_module_list_registers_parameters():
    stack = BlockStack(4, 3)
    assert count_parameters(stack) == 3 * (4 * 4 + 4)
    x = torch.randn(2, 4)
    assert stack(x).shape == (2, 4)
    assert len(parameter_names(stack)) == 6


def test_parameter_names():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    assert parameter_names(model) == ["0.bias", "0.weight", "1.bias", "1.weight"]


def test_total_bytes():
    model = MyLinear(4, 6)  # 24 весов + 6 bias, float32
    assert total_bytes(model) == (24 + 6) * 4


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


def test_tie_weights():
    emb = MyEmbedding(10, 4)
    head = MyLinear(4, 10, bias=False)
    tie_weights(emb, head)
    assert emb.weight.data_ptr() == head.weight.data_ptr()

    model = nn.Sequential(emb, head)
    unique = {p.data_ptr() for p in model.parameters()}
    assert len(unique) == 1, "после tying уникальный параметр должен быть один"


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
    assert (acts["1"] >= 0).all(), "после ReLU не должно быть отрицательных"
    assert model._forward_hooks == {} or len(model._forward_hooks) == 0


@pytest.mark.parametrize("n_layers", [1, 2, 5])
def test_block_stack_depth(n_layers):
    stack = BlockStack(8, n_layers)
    assert len(list(stack.parameters())) == 2 * n_layers
