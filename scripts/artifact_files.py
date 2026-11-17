"""Package and verify exactly the reviewed data files, without directory recursion."""

import argparse
from pathlib import Path, PurePosixPath
import stat
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
LISTS = {
    "fourfold-source-data.zip": "source-data-files.txt",
    "fourfold-derived-data.zip": "derived-data-files.txt",
}


def payload_names(archive_name, root=ROOT):
    names = (root / LISTS[archive_name]).read_text().splitlines()
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate artifact file list")
    for name in names:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or ".." in path.parts
                or any(part.startswith(".") for part in path.parts)
                or path.as_posix() != name or "\\" in name):
            raise ValueError(f"invalid artifact path: {name!r}")
    return names


def check_archive_members(archive, archive_name, root=ROOT):
    expected = set(payload_names(archive_name, root))
    entries = archive.infolist()
    actual = [entry.filename for entry in entries]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        extra = sorted(set(actual) - expected)
        missing = sorted(expected - set(actual))
        raise ValueError(f"{archive_name}: payload differs from file list; extra={extra[:3]}, missing={missing[:3]}")
    for entry in entries:
        mode = entry.external_attr >> 16
        if entry.is_dir() or (stat.S_IFMT(mode) and not stat.S_ISREG(mode)):
            raise ValueError(f"{archive_name}: non-regular payload {entry.filename}")


def pack(archive_name, output, *, force=False, root=ROOT):
    output = output.absolute()
    if output.suffix != ".zip":
        raise ValueError("artifact must have a .zip suffix")
    if output.exists() and not force:
        raise ValueError(f"output already exists: {output}; rerun with --force")
    names = payload_names(archive_name, root)
    # Check all inputs before creating or replacing any archive. Symlinks in
    # parent directories are also forbidden, even when their target is local.
    for name in names:
        path = root / name
        if not path.is_file() or path.resolve() != root.resolve() / name:
            raise ValueError(f"missing or symlinked artifact input: {name}")
        if path.resolve() == output.resolve():
            raise ValueError("output archive cannot replace a payload file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".artifact-", suffix=".zip", delete=False) as stream:
            temporary = Path(stream.name)
        # NPZ, PNG and publication PDF files are already compressed. ZIP_STORED
        # streams them directly, with bounded memory and no second compression.
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in names:
                archive.write(root / name, name)
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"Prepared {output}: {len(names)} listed files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("source", "derived"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        pack(f"fourfold-{args.kind}-data.zip", args.output, force=args.force)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"pack_artifact: {exc}\n")
