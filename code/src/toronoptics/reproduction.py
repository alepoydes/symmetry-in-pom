"""Small helpers used by the high-level reproduction drivers."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class ReproductionStep:
    """One documented command in a reproducible data-processing pipeline."""

    title: str
    explanation: str
    command: tuple[str, ...]
    accepts_force: bool = True
    force_on_partial: bool = True
    outputs: tuple[Path, ...] = ()
    output_is_current: Callable[[Path], bool] | None = None


def run_steps(
    steps: tuple[ReproductionStep, ...],
    *,
    project_root: Path,
    force: bool,
    dry_run: bool = False,
) -> None:
    """Run incomplete steps, forwarding force only to the affected step."""

    environment = os.environ.copy()
    matplotlib_config = project_root / "build" / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    environment.setdefault("MPLCONFIGDIR", str(matplotlib_config))
    for index, step in enumerate(steps, start=1):
        print(f"\n[{index}/{len(steps)}] {step.title}", flush=True)
        print(step.explanation, flush=True)
        existing = tuple(path for path in step.outputs if path.exists())
        current = tuple(
            path
            for path in existing
            if step.output_is_current is None or step.output_is_current(path)
        )
        complete = bool(step.outputs) and len(current) == len(step.outputs)
        if not force and complete:
            print(
                f"[skip] all {len(step.outputs)} declared output(s) exist",
                flush=True,
            )
            continue
        stale = tuple(path for path in existing if path not in current)
        if not force and stale:
            print(
                f"[stale] {len(stale)} existing output(s) do not match the "
                "requested calculation; rebuilding this step",
                flush=True,
            )
        if not force and existing:
            missing_count = len(step.outputs) - len(existing)
            if missing_count:
                print(
                    f"[resume] {len(existing)} output(s) exist; "
                    f"{missing_count} missing; rebuilding this step",
                    flush=True,
                )
        local_force = force or bool(stale) or (
            bool(existing) and step.force_on_partial
        )
        command = (
            (*step.command, "--force")
            if local_force and step.accepts_force
            else step.command
        )
        if dry_run:
            print(f"[would run] {shlex.join(command)}", flush=True)
            continue
        completed = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            if not force:
                rerun = (
                    "uv",
                    "run",
                    "python",
                    sys.argv[0],
                    *sys.argv[1:],
                    "--force",
                )
                print(
                    "\nTo rebuild the complete high-level stage, run:\n"
                    f"  {shlex.join(rerun)}",
                    file=sys.stderr,
                )
            raise SystemExit(completed.returncode)
        missing = tuple(path for path in step.outputs if not path.exists())
        if missing:
            displayed = []
            for path in missing:
                try:
                    displayed.append(str(path.relative_to(project_root)))
                except ValueError:
                    displayed.append(str(path))
            raise RuntimeError(
                f"{step.title!r} completed without creating declared outputs: "
                + ", ".join(displayed)
            )
        stale = tuple(
            path
            for path in step.outputs
            if step.output_is_current is not None
            and not step.output_is_current(path)
        )
        if stale:
            displayed = []
            for path in stale:
                try:
                    displayed.append(str(path.relative_to(project_root)))
                except ValueError:
                    displayed.append(str(path))
            raise RuntimeError(
                f"{step.title!r} completed with incompatible outputs: "
                + ", ".join(displayed)
            )
