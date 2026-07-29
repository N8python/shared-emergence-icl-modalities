#!/usr/bin/env python3
"""Download pinned source revisions and prepare the converted ImageGPT/ProGen2 rows."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_REVISIONS = REPO_ROOT / "data" / "t128" / "model_revisions.json"
MODELS_ROOT = REPO_ROOT / "models"

CONVERSIONS = {
    "imagegpt_small": {
        "source": "openai/imagegpt-small",
        "dest": "imagegpt-small-bf16",
        "converter": "imagegpt",
    },
    "imagegpt_medium": {
        "source": "openai/imagegpt-medium",
        "dest": "imagegpt-medium-bf16",
        "converter": "imagegpt",
    },
    "progen2_small": {
        "source": "hugohrban/progen2-small",
        "dest": "ProGen2-small-bf16",
        "converter": "progen2",
    },
    "progen2_medium": {
        "source": "hugohrban/progen2-medium",
        "dest": "ProGen2-medium-bf16",
        "converter": "progen2",
    },
    "progen2_large": {
        "source": "hugohrban/progen2-large",
        "dest": "ProGen2-large-bf16",
        "converter": "progen2",
    },
    "progen2_xlarge": {
        "source": "hugohrban/progen2-xlarge",
        "dest": "ProGen2-xlarge-bf16",
        "converter": "progen2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-key",
        nargs="+",
        choices=tuple(CONVERSIONS),
        help="Converted run key(s) to prepare. Defaults to all six.",
    )
    parser.add_argument("--hf-exe", default="hf")
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert an existing destination (large and potentially slow).",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    revisions_data = json.loads(MODEL_REVISIONS.read_text())
    revisions = {row["id"]: row["sha"] for row in revisions_data["models"]}
    selected = args.run_key or list(CONVERSIONS)
    source_root = MODELS_ROOT / "source"
    source_root.mkdir(parents=True, exist_ok=True)

    for run_key in selected:
        spec = CONVERSIONS[run_key]
        source_id = spec["source"]
        source_dir = source_root / source_id.replace("/", "--")
        destination = MODELS_ROOT / spec["dest"]
        complete = (destination / "config.json").is_file() and any(
            destination.glob("model*.safetensors")
        )
        if complete and not args.force:
            print(f"[{run_key}] prepared model already exists: {destination}", flush=True)
            continue
        revision = revisions[source_id]
        print(f"[{run_key}] downloading {source_id}@{revision}", flush=True)
        run(
            [
                args.hf_exe,
                "download",
                source_id,
                "--revision",
                revision,
                "--local-dir",
                str(source_dir),
            ]
        )
        if spec["converter"] == "imagegpt":
            command = [
                args.python_exe,
                str(REPO_ROOT / "scripts" / "convert_hf_imagegpt_bf16.py"),
                str(source_dir),
                str(destination),
            ]
            if args.force:
                command.append("--overwrite")
        else:
            command = [
                args.python_exe,
                str(REPO_ROOT / "scripts" / "convert_progen2_to_mlx_bf16.py"),
                "--source",
                str(source_dir),
                "--dest",
                str(destination),
            ]
            if args.force:
                command.append("--force")
        print(f"[{run_key}] converting to {destination}", flush=True)
        run(command)


if __name__ == "__main__":
    main()
