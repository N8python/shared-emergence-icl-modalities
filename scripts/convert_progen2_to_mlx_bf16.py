#!/usr/bin/env python3
"""Package ProGen2 as an MLX-LM custom model with bf16 safetensors."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MLX_MODEL_FILE = (
    REPO_ROOT / "src" / "model_implementations" / "progen2_mlx.py"
)


def copy_metadata(source: Path, dest: Path) -> None:
    for name in (
        "README.md",
        "tokenizer.json",
        "configuration_progen.py",
        "modeling_progen.py",
        "generation_config.json",
        "model.safetensors.index.json",
    ):
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)

    model_impl = DEFAULT_MLX_MODEL_FILE
    if model_impl.exists():
        shutil.copy2(model_impl, dest / "progen2_mlx.py")

    config = json.loads((source / "config.json").read_text())
    config["model_file"] = "progen2_mlx.py"
    config["dtype"] = "bfloat16"
    config["torch_dtype"] = "bfloat16"
    config.setdefault("pad_token_id", 0)
    config["bos_token_id"] = 1
    config["eos_token_id"] = 2
    (dest / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    tokenizer_config = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|eos|>",
        "model_max_length": int(config.get("n_positions", 1024)),
        "add_prefix_space": False,
    }
    special_tokens_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|eos|>",
    }
    (dest / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, indent=2) + "\n"
    )
    (dest / "special_tokens_map.json").write_text(
        json.dumps(special_tokens_map, indent=2) + "\n"
    )


def convert_shard(src: Path, dst: Path) -> None:
    tensors = {}
    with safe_open(src, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).to(torch.bfloat16)
    save_file(tensors, dst, metadata={"format": "pt"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    copy_metadata(args.source, args.dest)

    shards = sorted(args.source.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No model*.safetensors files found in {args.source}")

    for src in shards:
        dst = args.dest / src.name
        if dst.exists() and not args.force:
            print(f"exists: {dst}")
            continue
        print(f"converting {src.name} -> {dst}")
        convert_shard(src, dst)

    print(f"wrote {args.dest}")


if __name__ == "__main__":
    main()
