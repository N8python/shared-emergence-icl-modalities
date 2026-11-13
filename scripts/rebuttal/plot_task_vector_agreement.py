#!/usr/bin/env python3
"""6x6 cross-modality agreement of task-vector similarity matrices.

Each cell is the Spearman correlation of the off-diagonal entries of two models'
task-vector similarity matrices. Rows, columns, labels and colour scale follow the
paper's Figure 2 so the two can be read side by side. Reads task_vectors_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ORDER = [
    ("evo2_7b", "Genome\n(Evo2-7B)"),
    ("progen2_base", "Protein\n(ProGen2-base)"),
    ("qwen3_14b", "Language\n(Qwen3-14B)"),
    ("nextterm_440m", "Integer seq.\n(NextTerm-440M)"),
    ("timesfm_2_5", "Time-series\n(TimesFM-2.5)"),
    ("imagegpt_large", "Image\n(ImageGPT-large)"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/task_vectors_summary.json")
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    n = len(ORDER)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axis = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    grid = np.ones((n, n))
    for i, (a, _) in enumerate(ORDER):
        for j, (b, _) in enumerate(ORDER):
            if i != j:
                pair = summary["pairs"].get(f"{a}|{b}") or summary["pairs"][f"{b}|{a}"]
                grid[i, j] = pair["task_vectors"]["rho"]
    image = axis.imshow(grid, cmap="viridis", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            axis.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9,
                      color="#111827" if grid[i, j] > 0.5 else "white")
    labels = [label for _, label in ORDER]
    axis.set_xticks(range(n), labels=labels, fontsize=7.5, rotation=35, ha="right", rotation_mode="anchor")
    axis.set_yticks(range(n), labels=labels, fontsize=7.5)
    axis.set_xticks(np.arange(-0.5, n), minor=True)
    axis.set_yticks(np.arange(-0.5, n), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="both", length=0)
    fig.colorbar(image, ax=axis, shrink=0.8, label="Spearman ρ")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.05)
    print("wrote", args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
