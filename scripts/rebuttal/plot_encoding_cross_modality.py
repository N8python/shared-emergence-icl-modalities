#!/usr/bin/env python3
"""Rebuttal figure: each language/image encoding against the other modalities.

Rows are the encodings of one modality; columns are the other five modalities'
published profiles. Cells show the Spearman correlation of per-task
clean-minus-deranged effects; hatched cells are not significant (one-sided
permutation p >= 0.05). Reads results_rebuttal/analysis/summary.json.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("selfcorr", REPO_ROOT / "scripts/rebuttal/plot_encoding_self_correlation.py")
selfcorr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selfcorr)

COLUMN_LABELS = {
    "Genome": "Genome\n(Evo2-40B)", "Protein": "Protein\n(ProGen2-xl)", "Language": "Language\n(Qwen3-14B)",
    "Integer seq.": "Integer seq.\n(NextTerm)", "Time-series": "Time series\n(TimesFM)", "Image": "Image\n(ImageGPT-L)",
}
SIGNIFICANCE_LEVEL = 0.05
HATCH_COLOR = (0.10, 0.10, 0.10, 0.82)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/summary.json")
    parser.add_argument("--output-stem", type=Path, default=REPO_ROOT / "results_rebuttal/figures/encoding_cross_modality")
    parser.add_argument("--paper-labels", action="store_true", help="Use the paper script's own modality labels.")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    paper_spec = importlib.util.spec_from_file_location("paper", REPO_ROOT / "scripts/analysis/plot_task_effect_correlation_t128.py")
    paper = importlib.util.module_from_spec(paper_spec)
    paper_spec.loader.exec_module(paper)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.7})
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.9), constrained_layout=True)
    image = None
    for axis, (modality, (title, rows)) in zip(axes, selfcorr.PANELS.items()):
        block = summary[modality]
        others = block["other_modalities"]
        by_id = {row["id"]: row for row in block["cross_modality"]}
        rho = np.array([[by_id[run_id][o]["spearman"] for o in others] for run_id, _ in rows])
        pval = np.array([[by_id[run_id][o]["one_sided_permutation_p"] for o in others] for run_id, _ in rows])
        image = axis.imshow(rho, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        axis.set_xticks(np.arange(len(others)), labels=[paper.SPECS[o]["label"] for o in others],
                        rotation=42, ha="right", rotation_mode="anchor")
        axis.set_yticks(np.arange(len(rows)), labels=[label for _, label in rows])
        axis.tick_params(axis="both", which="major", length=0, pad=4)
        axis.set_xticks(np.arange(-0.5, len(others), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
        axis.grid(which="minor", color="white", linewidth=1.0)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for r in range(rho.shape[0]):
            for c in range(rho.shape[1]):
                significant = pval[r, c] < SIGNIFICANCE_LEVEL
                if not significant:
                    axis.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor="none",
                                             edgecolor=HATCH_COLOR, linewidth=0, hatch="////"))
                text = axis.text(c, r, f"{rho[r, c]:.2f}".replace("-", "−"), ha="center", va="center", fontsize=9,
                                 color="black" if rho[r, c] >= 0.6 else "white")
                if not significant:
                    text.set_bbox({"facecolor": plt.cm.viridis(max(rho[r, c], 0.0)), "edgecolor": "none", "pad": 1.2})
        axis.set_title(f"{title}\nvs. other modalities (reference)", fontsize=10, pad=8)
    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label(r"Spearman $\rho$ of per-task clean $-$ deranged effect")
    colorbar.outline.set_visible(False)
    fig.legend(handles=[Patch(facecolor=plt.cm.viridis(0.1), edgecolor=HATCH_COLOR, linewidth=0, hatch="////",
                              label=r"not significant (one-sided permutation $p \geq 0.05$)")],
               loc="outside lower center", frameon=False, fontsize=9)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"))
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220)
    print("wrote", args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
