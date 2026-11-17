"""Shared command-line safeguards for generated publication artifacts."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Iterable
from pathlib import Path


def add_force_argument(parser: argparse.ArgumentParser) -> None:
    """Add the repository-wide explicit rebuild switch."""

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite generated outputs that already exist.",
    )


def refuse_existing_outputs(
    paths: Iterable[Path],
    *,
    force: bool,
    project_root: Path | None = None,
) -> None:
    """Exit cleanly before computation if an output would be overwritten."""

    if force:
        return
    existing: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved.exists() and resolved not in seen:
            existing.append(resolved)
            seen.add(resolved)
    if not existing:
        return

    print(
        "Existing generated outputs would be overwritten; no files were changed:",
        file=sys.stderr,
    )
    root = None if project_root is None else Path(project_root).expanduser().resolve()
    for path in existing:
        try:
            displayed = path.relative_to(root) if root is not None else path
        except ValueError:
            displayed = path
        print(f"  {displayed}", file=sys.stderr)

    command = (
        "uv",
        "run",
        "python",
        sys.argv[0],
        *sys.argv[1:],
        "--force",
    )
    print("\nRerun with --force to rebuild them:", file=sys.stderr)
    print(f"  {shlex.join(command)}", file=sys.stderr)
    raise SystemExit(2)
