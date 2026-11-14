"""Shared setup for every exercise.

Two ways to run:

1. The normal one: `uv run pytest` - the tests exercise what you implemented in the `test_*.py` files.
2. The checking one: `SOLUTIONS=1 uv run pytest` - the tests exercise the reference implementations
   from `solutions/`. The second mode is there to confirm that the tests themselves are correct,
   and to peek at an answer.

The mechanics of the second mode: before the run, the functions and classes of the same name from
`solutions/sol_<the same name>.py` are substituted into the test module.

This file also provides the `depends_on` marker. Almost every test exercises exactly one stub, so a
failure names the single function to go and write. The exceptions are the tests that check that
several finished pieces agree with each other - a split/merge roundtrip, a KV cache against a full
forward pass - and those carry `@pytest.mark.depends_on("a", "b")`. Such a test is skipped until
every stub it names is implemented, so it stays quiet while you work on the parts and turns itself
on once the last one is done.

The working copies `test_NN_*.py` are git-ignored, so a fresh clone has none. Before collection this
file creates every missing one from `templates/`, which makes the first `uv run pytest` work as is.
"""

import importlib
import inspect
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


STUB = "raise NotImplementedError"


def _solutions_enabled() -> bool:
    return os.environ.get("SOLUTIONS", "").lower() in {"1", "true", "yes", "on"}


def _is_stub(obj) -> bool:
    """True while `obj` is still an untouched exercise stub.

    Read off the source rather than by calling it: a call would need arguments, and for a class
    the answer has to cover every method at once. A name that does not resolve, or has no source
    (a builtin, or a solution swapped in by SOLUTIONS=1), counts as implemented.
    """
    try:
        return STUB in inspect.getsource(obj)
    except (OSError, TypeError):
        return False


def pytest_configure(config):
    from reset import ensure_working_copies

    ensure_working_copies()
    config.addinivalue_line(
        "markers",
        "depends_on(*names): a test that checks several pieces against each other; "
        "skipped until every named stub in the module is implemented",
    )


def pytest_runtest_setup(item):
    marker = item.get_closest_marker("depends_on")
    if marker is None:
        return
    module = getattr(item, "module", None)
    if module is None:
        return
    pending = [name for name in marker.args if _is_stub(getattr(module, name, None))]
    if pending:
        pytest.skip("not implemented yet: " + ", ".join(pending))


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
    """Every test starts from the same state of the random generators."""
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_printoptions(precision=4, sci_mode=False)
    yield
