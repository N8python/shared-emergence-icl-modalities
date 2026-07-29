#!/usr/bin/env python3
"""Download, verify, and safely extract the complete T=128 raw result archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "t128" / "artifact_manifest.json"
CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, manifest: dict) -> None:
    expected_size = int(manifest["size_bytes"])
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"{path}: size {actual_size:,} does not match {expected_size:,}"
        )
    print(f"hashing {path} ({actual_size:,} bytes)", flush=True)
    actual_hash = sha256_file(path)
    if actual_hash != manifest["sha256"]:
        raise ValueError(
            f"{path}: SHA-256 {actual_hash} does not match {manifest['sha256']}"
        )
    print(f"verified SHA-256 {actual_hash}", flush=True)


def download(url: str, destination: Path) -> None:
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise FileExistsError(
            f"Partial download already exists: {partial}. Remove it or pass "
            "--archive pointing at a complete local copy."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "shared-emergence-icl-modalities-artifact-fetcher/1"},
    )
    print(f"downloading {url}", flush=True)
    started = time.monotonic()
    last_report = started
    downloaded = 0
    try:
        with urllib.request.urlopen(request) as response, partial.open("wb") as out:
            total_header = response.headers.get("Content-Length")
            total = int(total_header) if total_header else None
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_report >= 5:
                    if total:
                        print(
                            f"downloaded {downloaded:,}/{total:,} bytes "
                            f"({100 * downloaded / total:.1f}%)",
                            flush=True,
                        )
                    else:
                        print(f"downloaded {downloaded:,} bytes", flush=True)
                    last_report = now
        partial.replace(destination)
    except BaseException:
        print(f"incomplete download retained at {partial}", file=sys.stderr)
        raise
    elapsed = max(time.monotonic() - started, 1e-9)
    print(
        f"downloaded {downloaded:,} bytes in {elapsed:.1f}s "
        f"({downloaded / elapsed / 1024**2:.1f} MiB/s)",
        flush=True,
    )


def validate_member(member: tarfile.TarInfo, destination: Path) -> None:
    if not (member.isfile() or member.isdir()):
        raise ValueError(f"Archive contains unsupported member type: {member.name}")
    resolved = (destination / member.name).resolve()
    try:
        resolved.relative_to(destination.resolve())
    except ValueError as exc:
        raise ValueError(f"Archive path escapes destination: {member.name}") from exc


def extract_archive(path: Path, destination: Path, archive_root: str) -> None:
    expected_root = destination / archive_root
    if expected_root.exists():
        print(
            f"{expected_root} already exists; leaving it untouched. "
            "Run the validator to check it.",
            flush=True,
        )
        return
    destination.mkdir(parents=True, exist_ok=True)
    print(f"validating archive members before extraction to {destination}", flush=True)
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            validate_member(member, destination)
        roots = {Path(member.name).parts[0] for member in members if member.name}
        if roots != {archive_root}:
            raise ValueError(
                f"Archive roots {sorted(roots)} do not match {archive_root!r}"
            )
        print(f"extracting {len(members):,} files/directories", flush=True)
        archive.extractall(destination, members=members)
    print(f"extracted {expected_root}", flush=True)


def parse_args() -> argparse.Namespace:
    manifest = json.loads(MANIFEST_PATH.read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=REPO_ROOT / "data" / "t128" / manifest["artifact"],
        help="Local archive destination or an already-downloaded archive.",
    )
    parser.add_argument(
        "--extract-parent",
        type=Path,
        default=REPO_ROOT / manifest["default_extract_parent"],
        help="Parent directory under which results_128 will be extracted.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download and verify the archive without extracting it.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require an existing local archive and verify it without downloading.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text())
    archive = args.archive.expanduser().resolve()
    if not archive.exists():
        if args.verify_only:
            raise FileNotFoundError(archive)
        download(manifest["download_url"], archive)
    verify_archive(archive, manifest)
    if not args.no_extract and not args.verify_only:
        extract_archive(
            archive,
            args.extract_parent.expanduser().resolve(),
            manifest["archive_root"],
        )


if __name__ == "__main__":
    main()
