#!/usr/bin/env python3
"""Patch MLX CUDA headers for an H100/SM90 NVRTC header mismatch.

Some MLX CUDA wheel combinations expose CCCL's NVFP8_E8M0 definitions to
NVRTC on H100, but the active compiler headers do not provide the complete
type. A simple BF16 gather can then fail while compiling generated CUDA. This
script applies the narrow workaround used for the replication H100 smoke tests:
disable the CCCL E8M0 feature probe inside the installed MLX package.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


ORIGINAL = (
    "#define _CCCL_HAS_NVFP8_E8M0() "
    "(_CCCL_HAS_NVFP8() && _CCCL_CTK_AT_LEAST(12, 8))"
)
PATCHED = (
    "#define _CCCL_HAS_NVFP8_E8M0() 0  "
    "// patched for SM90/H100 NVRTC e8m0 header mismatch"
)


def mlx_package_dir() -> Path:
    spec = importlib.util.find_spec("mlx")
    if spec is None:
        raise RuntimeError("Could not import-locate the installed mlx package.")
    if spec.origin is not None:
        return Path(spec.origin).resolve().parent
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    raise RuntimeError("Located mlx, but could not determine its package directory.")


def header_path() -> Path:
    return (
        mlx_package_dir()
        / "include"
        / "cccl"
        / "cuda"
        / "std"
        / "__cccl"
        / "extended_data_types.h"
    )


def patch_header(path: Path, *, dry_run: bool = False) -> str:
    if not path.exists():
        raise FileNotFoundError(
            f"MLX CCCL header not found at {path}. The installed wheel layout may differ."
        )

    text = path.read_text(encoding="utf-8")
    if PATCHED in text:
        return f"already patched: {path}"
    if ORIGINAL not in text:
        raise RuntimeError(
            "Expected NVFP8_E8M0 feature-probe line was not found. "
            f"Refusing to edit {path}."
        )

    if dry_run:
        return f"would patch: {path}"

    backup = path.with_suffix(path.suffix + ".bak_e8m0")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(ORIGINAL, PATCHED, 1), encoding="utf-8")
    return f"patched: {path}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Report the edit without writing.")
    args = parser.parse_args()
    print(patch_header(header_path(), dry_run=args.dry_run))


if __name__ == "__main__":
    main()
