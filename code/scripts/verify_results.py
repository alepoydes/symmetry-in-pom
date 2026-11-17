#!/usr/bin/env python3
"""Check saved numerical reports against the rounded publication values."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify(root: Path = ROOT) -> int:
    checks = json.loads((root / "expected-results.json").read_text())
    failures = []
    for check in checks:
        value = json.loads((root / check["file"]).read_text())
        for key in check["keys"]:
            value = value[key]
        value = float(value)
        if not math.isfinite(value) or abs(value - check["value"]) > check["atol"]:
            failures.append(f'{check["name"]}: {value} != {check["value"]} +/- {check["atol"]}')
    if failures:
        raise SystemExit("Publication-result mismatch:\n" + "\n".join(failures))
    print(f"Verified {len(checks)} numerical values against the publication precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(verify())
