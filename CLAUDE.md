# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone `uv` project: rustlings-style PyTorch drills, one file per topic, 20 topics ordered by difficulty.
It is not part of any course assignment; it is deliberate practice built by the user for muscle memory on tensor mechanics and transformer internals.
The exercises began as `pytorch-exercises/` inside another working copy and were later extracted into this repo, so there is no parent repo whose conventions (the `sessions.yaml` work log) apply here.
Docs at the repo root (`README.md`, `AGENTS.md`, this file) are in English; docstrings and comments inside the exercises are in Russian.
Keep new prose in the language of the file being edited.

## Commands

```sh
uv sync                                          # once
uv run pytest test_01_tensor_basics.py -x        # current topic, stop at first failure
uv run pytest -x                                 # everything, in order
uv run pytest -k rope                            # one test by name
SOLUTIONS=1 uv run pytest test_14_attention.py   # run the same tests against solutions/
uv run ruff check . && uv run ruff format .
uv run python reset.py                           # status: which topics are touched, stubs left
uv run python reset.py 14                        # roll topic 14 back to its template
uv run python reset.py --diff 14                 # my version vs the original template
uv run python reset.py --all                     # roll everything back
uv run python reset.py --restore attempts/<ts>/test_14_attention.py
```

A fresh topic file fails every test with `NotImplementedError`.
That is the intended starting state, not a broken environment.

## Structure of a topic file

Each `test_NN_<topic>.py` is both the exercise and its test suite, split by a `# ТЕСТЫ` banner:

- Above the banner: function and `nn.Module` stubs with docstrings that state the contract (shapes, dtypes, hints, and any "do it without `torch.softmax`" constraints). The body is `raise NotImplementedError`, and solving means replacing it.
- Below the banner: the tests, mostly `torch.testing.assert_close` against a `torch`/`F` reference, plus `@pytest.mark.parametrize` sweeps.

So the file is edited and the tests are not.
Changing an assertion to make a stub pass defeats the whole point; fix the implementation instead.

## The three parallel copies

The same 20 filenames exist in three places, and confusing them breaks the workflow:

- `./test_NN_*.py` - the live working copy, the only files meant to be edited while solving.
- `templates/test_NN_*.py` - pristine starting versions. `reset.py` compares against these byte-for-byte to decide whether a topic is untouched, so never edit them.
- `solutions/sol_NN_*.py` - reference implementations, imported as the `solutions` package.

`attempts/<timestamp>/` holds working copies archived by `reset.py` before it overwrites them.
`pyproject.toml` keeps `templates/`, `solutions/`, and `attempts/` out of pytest collection via `norecursedirs`; a new sibling copy of the exercises would need the same treatment.

## How SOLUTIONS=1 works

`conftest.py` hooks `pytest_collection_modifyitems`: with `SOLUTIONS=1` set, for every collected test module it imports `solutions.sol_<same suffix>` and rebinds every top-level name defined there onto the test module.
The tests are unchanged; only what they call is swapped.
This is the check that the tests themselves are correct, so a test that fails under `SOLUTIONS=1` is a bug in the test or in the solution, not in the user's work.
It follows that a solution file must export names matching the stubs exactly, and that helpers private to a solution should be prefixed with `_` to stay out of the swap.

## Naming contract with CS336

Module and parameter names deliberately mirror `assignment1-basics`: `q_proj`, `k_proj`, `v_proj`, `output_proj`, `w1`, `w2`, `w3`, `ln1`, `ln2`, `ln_final`, `token_embeddings`, `lm_head`, no biases anywhere.
Several tests assert on `state_dict` keys, so these names are load-bearing, not cosmetic.

## Working with the user here

`AGENTS.md` sits next to this file and states the conduct policy for every agent working here, including which paths are editable and which are not.
It agrees with this file: running tests, ruff and `reset.py`, and editing non-exercise files (`conftest.py`, `reset.py`, `pyproject.toml`, docs) is normal work.
What matters most in both is the next paragraph.

This project exists so the user writes the code.
Do not fill in `NotImplementedError` stubs or hand over a solution for an unsolved topic unless asked directly; explain, review, and debug through dialog instead, and point at `solutions/` when the user wants the answer.
If asked for direct implementation work anyway, say once why that undercuts the exercise, then follow the user's decision.

## Conventions

- Python >= 3.12, < 3.14, ruff with line length 120 and `UP` enabled.
- `F401` is ignored in `test_*.py`: imports are staged in advance for the solution, so unused ones are expected and should stay.
- The exercises ban element-wise Python loops unless a docstring says otherwise; `einops` is available and is the intended tool for head splitting and attention algebra.
- `conftest.py` reseeds `torch`, `numpy`, and `random` before every test, so tests using randomness are still expected to be exactly reproducible.
