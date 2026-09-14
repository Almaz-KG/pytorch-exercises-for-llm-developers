"""Общая настройка для всех упражнений.

Два режима запуска:

1. Обычный: `uv run pytest` - тесты гоняют то, что ты реализовал в файлах `test_*.py`.
2. Проверочный: `SOLUTIONS=1 uv run pytest` - тесты гоняют эталонные реализации из `solutions/`.
   Второй режим нужен, чтобы убедиться, что сами тесты корректны, и чтобы подсмотреть ответ.

Механика второго режима: перед запуском в модуль теста подставляются одноимённые
функции и классы из `solutions/sol_<то же имя>.py`.
"""

import importlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _solutions_enabled() -> bool:
    return os.environ.get("SOLUTIONS", "").lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config, items):
    if not _solutions_enabled():
        return
    patched: set[str] = set()
    for item in items:
        module = getattr(item, "module", None)
        if module is None or module.__name__ in patched:
            continue
        patched.add(module.__name__)
        sol_name = "solutions.sol_" + module.__name__.removeprefix("test_")
        try:
            sol = importlib.import_module(sol_name)
        except ModuleNotFoundError:
            continue
        for name in dir(sol):
            if name.startswith("_"):
                continue
            obj = getattr(sol, name)
            if getattr(obj, "__module__", None) == sol.__name__:
                setattr(module, name, obj)


@pytest.fixture(autouse=True)
def _deterministic():
    """Каждый тест стартует с одного и того же состояния генераторов."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_printoptions(precision=4, sci_mode=False)
    yield
