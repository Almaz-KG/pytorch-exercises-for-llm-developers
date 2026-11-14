#!/usr/bin/env python
"""Roll the exercises back to their original stubs.

The pristine version of every topic lives in `templates/`, and the working copies in the
repository root are made from it. The working copies are not tracked by git: a missing one is
created from its template on the first run of `pytest` or of this script.
Before a rollback your version is always moved to `attempts/<date_time>/`,
so a solution is never lost, even without git.

    uv run python reset.py                 # status: what is touched, how many stubs are left
    uv run python reset.py 14              # roll topic 14 back (archiving the current version)
    uv run python reset.py 14 15 16        # several topics at once
    uv run python reset.py --all           # roll everything back
    uv run python reset.py --diff 14       # how my version differs from the original
    uv run python reset.py --restore attempts/2026-08-25_12-30-11/test_14_attention.py
"""

import argparse
import ast
import difflib
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parent
TEMPLATES = ROOT / "templates"
ATTEMPTS = ROOT / "attempts"

STUB = "raise NotImplementedError"


def is_stub_raise(node: ast.AST) -> bool:
    """`raise NotImplementedError` and `raise NotImplementedError(...)`, and nothing else."""
    if not isinstance(node, ast.Raise):
        return False
    exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
    return isinstance(exc, ast.Name) and exc.id == "NotImplementedError"


def all_topics() -> list[Path]:
    return sorted(TEMPLATES.glob("test_*.py"))


def ensure_working_copies() -> list[Path]:
    """Create every missing working copy from its template, and return the ones created.

    A fresh clone has no working copies at all, because they are git-ignored.
    `conftest.py` calls this before collection, so the first `pytest` run finds every topic in place.
    """
    created = []
    for template in all_topics():
        work = ROOT / template.name
        if not work.exists():
            shutil.copy2(template, work)
            created.append(work)
    return created


def resolve(spec: str) -> Path:
    """Accepts '3', '14', 'test_14_attention.py', 'attention'; returns the path to the working copy."""
    spec = spec.removesuffix(".py")
    topics = all_topics()
    tiers = [
        [t for t in topics if t.stem == spec or t.stem.removeprefix("test_") == spec],
        [t for t in topics if spec.isdigit() and t.stem.startswith(f"test_{int(spec):02d}_")],
        [t for t in topics if spec.lower() in t.stem.removeprefix("test_").lower()],
    ]
    for matches in tiers:
        if len(matches) == 1:
            return ROOT / matches[0].name
        if len(matches) > 1:
            names = ", ".join(t.stem.removeprefix("test_") for t in matches)
            sys.exit(f"topic {spec!r} is ambiguous, it matches: {names}")
    names = ", ".join(t.stem.removeprefix("test_") for t in topics)
    sys.exit(f"unknown topic {spec!r}. Available: {names}")


def stub_count(path: Path) -> int:
    """How many stubs are left: counts statements, not occurrences of the substring.

    The header of test_01 carries the same string inside a docstring as part of the instructions,
    and counting by text kept a solved topic "in progress" forever.
    A file with a syntax error (an ordinary thing in the middle of solving) must not break
    the status, so for such a file the count falls back to the text.
    """
    if not path.exists():
        return 0
    source = path.read_text()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source.count(STUB)
    return sum(1 for node in ast.walk(tree) if is_stub_raise(node))


def status() -> None:
    print(f"{'topic':<32} {'state':<12} {'stubs left'}")
    print("-" * 58)
    for template in all_topics():
        work = ROOT / template.name
        total = stub_count(template)
        if work.read_bytes() == template.read_bytes():
            state, left = "untouched", f"{total} of {total}"
        else:
            remaining = stub_count(work)
            state = "solved" if remaining == 0 else "in progress"
            left = f"{remaining} of {total}"
        print(f"{template.stem.removeprefix('test_'):<32} {state:<12} {left}")

    archives = sorted(ATTEMPTS.glob("*/")) if ATTEMPTS.exists() else []
    if archives:
        print(f"\narchived versions: {len(archives)}, the latest is {archives[-1].name}")


def diff(spec: str) -> None:
    work = resolve(spec)
    template = TEMPLATES / work.name
    lines = difflib.unified_diff(
        template.read_text().splitlines(keepends=True),
        work.read_text().splitlines(keepends=True),
        fromfile=f"templates/{work.name}",
        tofile=work.name,
    )
    out = "".join(lines)
    print(out if out else "no differences: the file is in its original state")


def reset(paths: list[Path], archive: bool) -> None:
    stamp = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = ATTEMPTS / stamp
    for work in paths:
        template = TEMPLATES / work.name
        if work.exists() and work.read_bytes() != template.read_bytes() and archive:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work, target_dir / work.name)
            print(f"archived: attempts/{stamp}/{work.name}")
        shutil.copy2(template, work)
        print(f"reset:    {work.name}")


def restore(path_str: str) -> None:
    src = Path(path_str)
    if not src.exists():
        sys.exit(f"no such file: {src}")
    dst = ROOT / src.name
    if dst.exists() and dst.read_bytes() != (TEMPLATES / src.name).read_bytes():
        reset([dst], archive=True)
    shutil.copy2(src, dst)
    print(f"restored: {dst.name} from {src}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="numbers or names of the topics to roll back")
    parser.add_argument("--all", action="store_true", help="roll every topic back")
    parser.add_argument("--diff", metavar="TOPIC", help="show how my version differs from the original")
    parser.add_argument("--restore", metavar="FILE", help="bring a version back from attempts/")
    parser.add_argument("--no-archive", action="store_true", help="do not archive the current version first")
    args = parser.parse_args()

    created = ensure_working_copies()
    if created:
        noun = "copy" if len(created) == 1 else "copies"
        print(f"created {len(created)} working {noun} from templates/\n")

    if args.diff:
        diff(args.diff)
    elif args.restore:
        restore(args.restore)
    elif args.all:
        reset([ROOT / t.name for t in all_topics()], archive=not args.no_archive)
    elif args.topics:
        reset([resolve(spec) for spec in args.topics], archive=not args.no_archive)
    else:
        status()


if __name__ == "__main__":
    main()
