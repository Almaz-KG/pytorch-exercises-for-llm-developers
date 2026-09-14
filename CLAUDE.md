# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A standalone `uv` project: rustlings-style PyTorch drills, one file per topic, 20 topics ordered by difficulty.
It is not part of any course assignment; it is deliberate practice built by the user for muscle memory on tensor mechanics and transformer internals.
`README.md` carries the topic table and the arc through it: 01-07 tensor mechanics, 08-09 autograd and `nn.Module`, 10-15 the transformer brick by brick, 16-17 training and inference, 18-19 memory and modern tricks, 20 data plumbing.
Everything in this repository is in English: the docs at the root (`README.md`, `AGENTS.md`, this file),
and the docstrings and comments inside the exercises and the solutions.
The exercises were originally written in Russian and translated wholesale; a Russian docstring anywhere
under `templates/` or the working copies is a leftover, not a convention to follow.
`attempts/` is the exception and stays as it was: it is an archive of past work, not living text.

The exercises began as `pytorch-exercises/` inside a CS336 working copy.
They now live here as their own repository, checked out as a git submodule of `ai-learning-path`.
Nothing from the parent applies here: no `sessions.yaml` work log, no shared toolchain.
What the submodule relationship does mean is that a commit here is invisible to the parent until a second commit there moves the pinned SHA, and that the parent's `git status` never shows an edit made in this tree.
Pushing to this repo's own remote comes first; bumping the pin in `ai-learning-path` is a separate, explicit act.

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
uv run python reset.py 14 15 16                  # several at once
uv run python reset.py --diff 14                 # my version vs the original template
uv run python reset.py --all                     # roll everything back
uv run python reset.py --restore attempts/<ts>/test_14_attention.py
```

A topic is named to `reset.py` by number (`14`), by file stem (`test_14_attention`), by the stem without its prefix (`14_attention`), or by any unambiguous substring (`attention`).
`--no-archive` skips the copy into `attempts/` and is the one flag that can lose work.

A fresh topic file fails every test with `NotImplementedError`.
That is the intended starting state, not a broken environment.

## Structure of a topic file

Each `test_NN_<topic>.py` is both the exercise and its test suite, split by a `# TESTS` banner:

- Above the banner: function and `nn.Module` stubs grouped into numbered sections (`14.0`, `14.1`, ...), with docstrings that state the contract (shapes, dtypes, hints, and any "do it without `torch.softmax`" constraints). The body is `raise NotImplementedError`, and solving means replacing it.
- Below the banner: the tests, mostly `torch.testing.assert_close` against a `torch`/`F` reference, plus `@pytest.mark.parametrize` sweeps.

So the file is edited and the tests are not.
Changing an assertion to make a stub pass defeats the whole point; fix the implementation instead.

No topic file imports another one, and none of them import `solutions/`.
Every file stands alone, and a later topic that needs an earlier helper re-declares its stub rather than importing it: `split_heads` is drilled again at the top of topic 14, the transformer bricks again at the top of topic 15.
That duplication is the exercise.
Do not factor it out into a shared module.

## The three parallel copies

The same 20 filenames exist in three places, and confusing them breaks the workflow:

- `./test_NN_*.py` - the live working copy, the only files meant to be edited while solving.
- `templates/test_NN_*.py` - pristine starting versions. `reset.py` compares against these byte-for-byte to decide whether a topic is untouched, so never edit them.
- `solutions/sol_NN_*.py` - reference implementations, imported as the `solutions` package.

`attempts/<timestamp>/` holds working copies archived by `reset.py` before it overwrites them.
`pyproject.toml` keeps `templates/`, `solutions/`, and `attempts/` out of pytest collection via `norecursedirs`; a new sibling copy of the exercises would need the same treatment.

## "Untouched" does not mean "never solved"

`attempts/` is tracked in git, not ignored, and it is the only record of past passes through a topic.
Redoing a topic after a break is the intended use of `reset.py`, so a working file can be byte-identical to its template while the user has solved that topic twice already.
`uv run python reset.py` reports the state of the working copy and nothing more.
Before concluding that a topic is new to the user, look at whether `attempts/*/` holds a version of it.

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

- Python >= 3.12, < 3.14, ruff with line length 120 and `UP` enabled. `F722`, `E741` and `PLR0402` are ignored repo-wide, and `F401` additionally in `test_*.py`: imports are staged in advance for the solution, so unused ones are expected and should stay.
- The exercises ban element-wise Python loops unless a docstring says otherwise; `einops` is available and is the intended tool for head splitting and attention algebra.
- `conftest.py` reseeds `torch`, `numpy`, and `random` before every test, so tests using randomness are still expected to be exactly reproducible.
