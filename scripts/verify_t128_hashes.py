#!/usr/bin/env python3
"""Verify every extracted T=128 result file against the packaged SHA-256 list."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "t128" / "results_128"
DEFAULT_HASHES = REPO_ROOT / "data" / "t128" / "raw_results.sha256"
MARKER = "/results_128/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--hashes", type=Path, default=DEFAULT_HASHES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    entries = []
    for line_number, line in enumerate(args.hashes.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            expected, recorded_path = line.split(maxsplit=1)
        except ValueError as exc:
            raise ValueError(f"{args.hashes}:{line_number}: malformed line") from exc
        if MARKER not in recorded_path:
            raise ValueError(
                f"{args.hashes}:{line_number}: missing {MARKER!r} path marker"
            )
        relative = recorded_path.split(MARKER, 1)[1]
        entries.append((expected, results_root / relative))

    failures = []
    for index, (expected, path) in enumerate(entries, 1):
        if not path.is_file():
            failures.append(f"missing {path}")
        else:
            actual = sha256_file(path)
            if actual != expected:
                failures.append(f"hash mismatch {path}: {actual} != {expected}")
        if index % 20 == 0 or index == len(entries):
            print(f"verified hashes {index}/{len(entries)}", flush=True)
    if failures:
        raise ValueError("\n".join(failures))
    print(f"all {len(entries)} extracted result-file hashes match", flush=True)


if __name__ == "__main__":
    main()
