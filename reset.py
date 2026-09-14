#!/usr/bin/env python
"""Откат упражнений к исходным заготовкам.

Исходные (нетронутые) версии всех файлов лежат в `templates/`.
Твоя рабочая версия перед откатом всегда уезжает в `attempts/<дата_время>/`,
поэтому решение не теряется никогда - даже если история git недоступна.

    uv run python reset.py                 # статус: что тронуто, сколько заготовок осталось
    uv run python reset.py 14              # откатить тему 14 (с архивацией текущей версии)
    uv run python reset.py 14 15 16        # несколько тем сразу
    uv run python reset.py --all           # откатить всё
    uv run python reset.py --diff 14       # чем моя версия отличается от исходной
    uv run python reset.py --restore attempts/2026-08-25_12-30-11/test_14_attention.py
"""

import argparse
import difflib
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).parent
TEMPLATES = ROOT / "templates"
ATTEMPTS = ROOT / "attempts"

STUB = "raise NotImplementedError"


def all_topics() -> list[Path]:
    return sorted(TEMPLATES.glob("test_*.py"))


def resolve(spec: str) -> Path:
    """Принимает '3', '14', 'test_14_attention.py', 'attention' - возвращает путь к рабочему файлу."""
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
            sys.exit(f"тема {spec!r} неоднозначна, подходят: {names}")
    names = ", ".join(t.stem.removeprefix("test_") for t in topics)
    sys.exit(f"не понял тему {spec!r}. Доступные: {names}")


def stub_count(path: Path) -> int:
    return path.read_text().count(STUB) if path.exists() else 0


def status() -> None:
    print(f"{'тема':<32} {'состояние':<12} {'заготовок осталось'}")
    print("-" * 66)
    for template in all_topics():
        work = ROOT / template.name
        total = stub_count(template)
        if not work.exists():
            state, left = "нет файла", "-"
        elif work.read_bytes() == template.read_bytes():
            state, left = "не тронут", f"{total} из {total}"
        else:
            remaining = stub_count(work)
            state = "решён" if remaining == 0 else "в работе"
            left = f"{remaining} из {total}"
        print(f"{template.stem.removeprefix('test_'):<32} {state:<12} {left}")

    archives = sorted(ATTEMPTS.glob("*/")) if ATTEMPTS.exists() else []
    if archives:
        print(f"\nархивы прошлых версий: {len(archives)}, последний - {archives[-1].name}")


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
    print(out if out else "отличий нет - файл в исходном состоянии")


def reset(paths: list[Path], archive: bool) -> None:
    stamp = datetime.now(UTC).astimezone().strftime("%Y-%m-%d_%H-%M-%S")
    target_dir = ATTEMPTS / stamp
    for work in paths:
        template = TEMPLATES / work.name
        if work.exists() and work.read_bytes() != template.read_bytes() and archive:
            target_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(work, target_dir / work.name)
            print(f"сохранено: attempts/{stamp}/{work.name}")
        shutil.copy2(template, work)
        print(f"откачено:  {work.name}")


def restore(path_str: str) -> None:
    src = Path(path_str)
    if not src.exists():
        sys.exit(f"файла нет: {src}")
    dst = ROOT / src.name
    if dst.exists() and dst.read_bytes() != (TEMPLATES / src.name).read_bytes():
        reset([dst], archive=True)
    shutil.copy2(src, dst)
    print(f"восстановлено: {dst.name} из {src}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("topics", nargs="*", help="номера или имена тем для отката")
    parser.add_argument("--all", action="store_true", help="откатить все темы")
    parser.add_argument("--diff", metavar="ТЕМА", help="показать отличия моей версии от исходной")
    parser.add_argument("--restore", metavar="ФАЙЛ", help="вернуть версию из attempts/")
    parser.add_argument("--no-archive", action="store_true", help="не сохранять текущую версию перед откатом")
    args = parser.parse_args()

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
