#!/usr/bin/env python3
"""Rebuttal figure: encoding-by-encoding Spearman matrices for language and image.

Reads results_rebuttal/analysis/summary.json (written by
analyze_encoding_variants.py) and draws one heatmap per modality in the style of
the paper's cross-modality correlation figure. The paper-seed rerun is omitted
(it reproduces the published reference, rho = 0.99).
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

PANELS = {
    "language": ("Language (Qwen3-14B, 64 shots)", [
        ("reference_published", "Reference"),
        ("reference_reseed", "Reference\n(new seed)"),
        ("two_sep", "Two\nseparators"),
        ("no_sep", "No\nseparator"),
        ("nondigit", "Non-digit\ntokens"),
        ("nondigit_two_sep", "Non-digit,\ntwo separators"),
    ]),
    "image": ("Image (ImageGPT-large, 31 shots)", [
        ("reference_published", "Reference\n(left separator)"),
        ("reference_reseed", "Reference\n(new seed)"),
        ("right_sep", "Right\nseparator"),
        ("middle_sep", "Middle\nseparator"),
        ("left_two_sep", "Left, two\nseparators"),
        ("right_two_sep", "Right, two\nseparators"),
    ]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/summary.json")
    parser.add_argument("--output-stem", type=Path, default=REPO_ROOT / "results_rebuttal/figures/encoding_self_correlation")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.9), constrained_layout=True)
    image = None
    for axis, (modality, (title, rows)) in zip(axes, PANELS.items()):
        block = summary[modality]
        full = np.asarray(block["variant_spearman"])
        index = [block["ids"].index(run_id) for run_id, _ in rows]
        matrix = full[np.ix_(index, index)]
        labels = [label for _, label in rows]
        image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
        positions = np.arange(len(labels))
        axis.set_xticks(positions, labels=labels, rotation=42, ha="right", rotation_mode="anchor")
        axis.set_yticks(positions, labels=labels)
        axis.tick_params(axis="both", which="major", length=0, pad=4)
        axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for row in range(len(labels)):
            for column in range(len(labels)):
                value = matrix[row, column]
                text = "1" if row == column else f"{value:.2f}"
                axis.text(column, row, text, ha="center", va="center", fontsize=9,
                          color="black" if value >= 0.6 else "white")
        axis.set_title(title, fontsize=10, pad=8)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(r"Spearman $\rho$ of per-task clean $-$ deranged effect")
    colorbar.outline.set_visible(False)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"))
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220)
    print("wrote", args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
