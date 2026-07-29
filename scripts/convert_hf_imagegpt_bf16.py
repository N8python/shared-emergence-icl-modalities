#!/usr/bin/env python3
"""Convert a local Hugging Face ImageGPT checkpoint from fp32 to bf16."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file


COPY_FILES = (
    "README.md",
    "preprocessor_config.json",
    "generation_config.json",
)

DEFAULT_MLX_MODEL_FILE = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "model_implementations"
    / "imagegpt_mlx_lm.py"
)


def should_skip_tensor(name: str) -> bool:
    """Drop legacy non-parameter attention mask buffers from old HF checkpoints."""
    return name.endswith(".attn.bias") or name.endswith(".attn.masked_bias")


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local HF ImageGPT directory.")
    parser.add_argument("dest", type=Path, help="Output HF directory with bf16 safetensors.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output directory.",
    )
    parser.add_argument(
        "--mlx-model-file",
        type=Path,
        default=DEFAULT_MLX_MODEL_FILE if DEFAULT_MLX_MODEL_FILE.exists() else None,
        help=(
            "Optional local mlx-lm model implementation to copy into the output "
            "directory and register as config['model_file']."
        ),
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest = args.dest.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    if dest.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dest} exists; pass --overwrite to replace it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    config_path = source / "config.json"
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    config["torch_dtype"] = "bfloat16"
    if args.mlx_model_file is not None:
        config["model_file"] = "imagegpt_mlx_lm.py"
    with (dest / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    for name in COPY_FILES:
        src = source / name
        if src.exists():
            shutil.copy2(src, dest / name)
    if args.mlx_model_file is not None:
        mlx_model_file = args.mlx_model_file.expanduser().resolve()
        if not mlx_model_file.exists():
            raise FileNotFoundError(mlx_model_file)
        shutil.copy2(mlx_model_file, dest / "imagegpt_mlx_lm.py")

    bin_path = source / "pytorch_model.bin"
    if not bin_path.exists():
        raise FileNotFoundError(f"Expected {bin_path}")

    print(f"Loading {bin_path}")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict, got {type(state)!r}")

    before_bytes = 0
    after_bytes = 0
    converted = {}
    skipped = []
    for name, tensor in state.items():
        if not torch.is_tensor(tensor):
            continue
        if should_skip_tensor(name):
            skipped.append(name)
            continue
        before_bytes += tensor_nbytes(tensor)
        if tensor.is_floating_point():
            tensor = tensor.to(torch.bfloat16)
        tensor = tensor.contiguous()
        after_bytes += tensor_nbytes(tensor)
        converted[name] = tensor

    out_path = dest / "model.safetensors"
    print(f"Saving {out_path}")
    save_file(converted, str(out_path), metadata={"format": "pt"})

    metadata = {
        "source": str(source),
        "source_file": str(bin_path),
        "output_file": str(out_path),
        "param_tensors": len(converted),
        "source_bytes": before_bytes,
        "bf16_bytes": after_bytes,
        "dtype": "bfloat16",
        "skipped_tensors": skipped,
    }
    with (dest / "conversion_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    print(
        "Converted "
        f"{metadata['param_tensors']} tensors: "
        f"{before_bytes / 2**30:.2f} GiB -> {after_bytes / 2**30:.2f} GiB"
    )


if __name__ == "__main__":
    main()
