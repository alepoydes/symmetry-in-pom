#!/usr/bin/env python3
"""Validate and list figures in the standalone reproduction manifest."""

import argparse
from pathlib import Path

from toronoptics.figure_output import figure_output_dir


ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = figure_output_dir(ROOT)
GENERATED_SUFFIXES = frozenset((".pdf", ".png"))


def publication_figures() -> tuple[str, ...]:
    """Read PDF basenames without depending on manuscript sources."""

    lines = (ROOT / "figures.txt").read_text(encoding="utf-8").splitlines()
    names = [line.strip() for line in lines if line.strip()]
    if not names or len(set(names)) != len(names) or any(
        Path(name).name != name or Path(name).suffix != ".pdf" for name in names
    ):
        raise ValueError("figures.txt must contain unique PDF basenames")
    return tuple(sorted(names))


def figures_read_from_fls(paths: tuple[Path, ...]) -> tuple[str, ...]:
    """Return publication-figure basenames actually opened by TeX."""

    names: set[str] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("INPUT "):
                continue
            value = line.removeprefix("INPUT ").strip().replace("//", "/")
            if value.startswith("figs/") or "/figs/" in value:
                names.add(Path(value).name)
    return tuple(sorted(names))


def unreferenced_generated_files(directory: Path) -> tuple[Path, ...]:
    """Return generated image files absent from figures.txt."""

    expected = set(publication_figures())
    if not directory.is_dir():
        return ()
    return tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in GENERATED_SUFFIXES
            and path.name not in expected
        )
    )


def prune_unreferenced_generated_files(directory: Path) -> tuple[Path, ...]:
    """Remove stale generated images from a publication-figure directory."""

    stale = unreferenced_generated_files(directory)
    for path in stale:
        path.unlink()
    return stale


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="print the PDF basenames listed in figures.txt",
    )
    parser.add_argument(
        "--prune-unreferenced",
        action="store_true",
        help="remove generated PDF/PNG files not listed in figures.txt",
    )
    parser.add_argument(
        "--fls",
        action="append",
        type=Path,
        default=[],
        help="verify the figure inputs recorded by a completed LaTeX run",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    figures = publication_figures()
    if args.list:
        print("\n".join(figures))
        return

    if args.prune_unreferenced:
        removed = prune_unreferenced_generated_files(FIGURE_DIR)
        for path in removed:
            print(f"removed unreferenced publication image: {path}")

    missing = [name for name in figures if not (FIGURE_DIR / name).is_file()]
    empty = [
        name
        for name in figures
        if (FIGURE_DIR / name).is_file() and (FIGURE_DIR / name).stat().st_size == 0
    ]
    if missing or empty:
        raise SystemExit(f"missing={missing}; empty={empty}")

    if args.fls:
        logged = figures_read_from_fls(tuple(path.resolve() for path in args.fls))
        expected = tuple(sorted(figures))
        if logged != expected:
            missing_from_log = sorted(set(expected).difference(logged))
            extra_in_log = sorted(set(logged).difference(expected))
            raise SystemExit(
                "LaTeX recorder inputs disagree with the TeX sources: "
                f"missing_from_log={missing_from_log}; extra_in_log={extra_in_log}"
            )
        print(f"verified {len(logged)} publication PDFs in the LaTeX recorder logs")

    print(f"verified {len(figures)} publication PDFs in {FIGURE_DIR}")


if __name__ == "__main__":
    main()
