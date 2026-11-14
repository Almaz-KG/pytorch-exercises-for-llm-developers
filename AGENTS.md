# AI agent guidelines

This file tells AI coding assistants (Claude Code, Codex, Cursor, Copilot and the rest) how to behave in this repository.
It is about conduct.
The layout, the commands and the invariants of the repo are described in `CLAUDE.md`, and agents that can read it should read it too.

## What this repo is

Rustlings-style PyTorch drills: twenty `test_NN_<topic>.py` files, ordered by difficulty, each one both the exercise and its own test suite.
Above the `# TESTS` banner sit function and `nn.Module` stubs whose docstrings state the contract; every body is `raise NotImplementedError`.
Below the banner sit the tests that check those stubs, one test per stub.
Solving a topic means replacing the `NotImplementedError` bodies until the file goes green.
A few tests check that several finished pieces agree with each other; those carry
`@pytest.mark.depends_on(...)` and skip until every stub they name is implemented.

This is not a course assignment and there is no grader.
It is deliberate practice, and the point is muscle memory: after twenty files a transformer block, cross-entropy, AdamW and a KV cache should come out of the fingers without lookup.
An agent that writes the drills defeats the only purpose the repo has.

## The one rule that matters

The learner writes the exercise code.

Do not fill in `NotImplementedError` stubs, do not hand over a working implementation for an unsolved topic, and do not paste one into chat as "an example".
Explain, review and debug through dialog instead.
When the learner wants the answer outright, point at the matching `solutions/sol_NN_*.py` rather than retyping it.

If asked directly for implementation work anyway, say once why that undercuts the exercise, then follow the decision without further argument.

## What agents should do

- Explain the concept behind a topic: shapes, dtypes, broadcasting rules, the algebra of attention, what a gradient of some expression actually is.
- Review code the learner has already written, and point at bugs, edge cases, missing invariants or clumsy formulations.
- Debug through questions: what was expected, what came out, what the intermediate tensors look like on a toy input.
- Decode error messages and `assert_close` diffs, including shape mismatches and dtype surprises.
- Suggest sanity checks: a length-3 toy sequence, a shape assertion, a comparison against a `torch`/`F` reference, a printed intermediate.
- Run the tooling and report what it says: `uv run pytest`, `SOLUTIONS=1 uv run pytest`, `uv run ruff check .`, `uv run python reset.py`, `uv run mkdocs build`.
- Edit the non-exercise files when asked: `conftest.py`, `reset.py`, `pyproject.toml`, `README.md`, `CLAUDE.md`, this file.
- Write and maintain `theory/`: the written material for each topic, which is what makes this repository solvable without a search engine or a second assistant. It states formulas, shapes and API semantics, and its examples run on neutral data. It must never contain a solution to a stub, in any form, including a "sketch" that only needs renaming.
- Author new topics, new stubs and new tests when asked for them; that is writing the exercise, not solving it.
- Fix a test that is genuinely wrong, which is what a failure under `SOLUTIONS=1` means.
- Keep the one-test-per-stub shape when adding tests: a new test gets at most one subject, and anything
  else it needs goes into `@pytest.mark.depends_on(...)` so it skips until those are written.

## What agents should not do

- Replace a `raise NotImplementedError` in a `test_NN_*.py` file that the learner is working through.
- Produce a pasteable solution for an unsolved stub in chat, in a comment, in pseudocode or as a "sketch" that only needs renaming.
- Touch anything below the `# TESTS` banner to make a stub pass; a test bent to fit an implementation teaches nothing.
- Edit `templates/` for anything but authoring a topic or fixing a broken test: it is the tracked source of every exercise, and `reset.py` compares the working copies against it byte for byte.
- Edit `attempts/`, which is the learner's archive of past work.
- Volunteer the answer mid-debugging because the bug happens to be obvious.
- Reach for a third-party implementation of a topic that the repo already covers.
- Put a stub's implementation into `theory/` because it is documentation rather than code. The same rule applies there: a reader should finish a theory file knowing every piece and still have to assemble them.

## Which files are editable

| Path | Editable by an agent |
| --- | --- |
| `templates/` | Only when authoring a new topic or fixing a broken test, never to solve one |
| `test_NN_*.py` | Only below the `# TESTS` banner, to carry over a test fixed in `templates/`; these git-ignored working copies are the learner's own work |
| `solutions/` | Yes, when a solution is wrong or a new topic needs one |
| `attempts/` | No |
| `conftest.py`, `reset.py`, `pyproject.toml`, `mkdocs.yml`, docs | Yes |
| `theory/` | Yes, but never as a place to leak a solution |

## House rules that apply to any code an agent does write

- Python >= 3.12, < 3.14; ruff with line length 120 and `UP` enabled.
- A pre-commit hook (`.pre-commit-config.yaml`) runs `ruff check --fix` and `ruff format` on every commit; do not skip it with `--no-verify`.
- No element-wise Python loops over tensors unless a docstring explicitly allows them; `einops` is the intended tool for head splitting and attention algebra.
- Names mirror Stanford CS336's `assignment1-basics`: `q_proj`, `k_proj`, `v_proj`, `output_proj`, `w1`, `w2`, `w3`, `ln1`, `ln2`, `ln_final`, `token_embeddings`, `lm_head`, no biases anywhere.
  Several tests assert on `state_dict` keys, so these names are load-bearing.
- Everything is in English: the root docs (`README.md`, `CLAUDE.md`, this file) and the docstrings and comments
  inside the exercises and the solutions. Files under `attempts/` are the learner's archive and stay as they are.
- `F401` is ignored in `test_*.py` on purpose: imports are staged ahead of the solution, so unused ones stay.

## Example interactions

**Good:**
> Learner: "My causal mask is wrong in topic 14, what is my mistake?"
>
> Agent: "Three things to check, in order: whether the mask is applied before the softmax, whether it broadcasts to the shape of the score tensor you expect, and whether masked positions become a large negative value rather than zero. Take a length-3 toy sequence and print the scores before and after masking. What do those look like?"

**Good:**
> Learner: "Just show me the RoPE implementation."
>
> Agent: "That is in `solutions/sol_13_embeddings_and_rope.py` if you want the whole thing. Before you open it: you already have the angle table right, and the part that is off is the pairing of dimensions in the rotation. Want to try that first?"

**Bad:**
> Learner: "Topic 12 does not pass."
>
> Agent: "Here is the full SwiGLU implementation: ..."
