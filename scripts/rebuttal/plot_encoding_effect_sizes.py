#!/usr/bin/env python3
"""Rebuttal figure: clean vs deranged accuracy and the ICL effect per encoding.

One row per encoding; the deranged (red) and clean (blue) mean accuracies are
joined by a bar whose length is the paired ICL effect, annotated with its
task-bootstrap 95% interval. Reads results_rebuttal/analysis/summary.json.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("selfcorr", REPO_ROOT / "scripts/rebuttal/plot_encoding_self_correlation.py")
selfcorr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(selfcorr)

BLUE, RED, GRID, INK = "#2563EB", "#DC2626", "#D1D5DB", "#111827"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/summary.json")
    parser.add_argument("--output-stem", type=Path, default=REPO_ROOT / "results_rebuttal/figures/encoding_effect_sizes")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.3), constrained_layout=True, sharex=True)
    for axis, (modality, (title, rows)) in zip(axes, selfcorr.PANELS.items()):
        effects = {e["id"]: e for e in summary[modality]["effects"]}
        y = np.arange(len(rows))[::-1]
        for position, (run_id, _) in zip(y, rows):
            e = effects[run_id]
            clean, deranged = 100 * e["clean"], 100 * e["deranged"]
            low, high = (100 * v for v in e["gap_ci95"])
            axis.plot([deranged, clean], [position, position], color="#9CA3AF", linewidth=3.2, solid_capstyle="butt", zorder=1)
            axis.scatter([deranged], [position], s=46, color=RED, zorder=3, edgecolor="white", linewidth=0.8)
            axis.scatter([clean], [position], s=46, color=BLUE, zorder=3, edgecolor="white", linewidth=0.8)
            axis.text(clean + 1.4, position, f"+{100 * e['gap']:.1f} pp  [{low:.1f}, {high:.1f}]",
                      va="center", ha="left", fontsize=8.5, color=INK)
        axis.set_yticks(y, labels=[label.replace("\n", " ") for _, label in rows])
        axis.set_ylim(-0.6, len(rows) - 0.4)
        axis.set_xlim(0, 76)
        axis.set_xticks(range(0, 61, 10))
        axis.set_xlabel("Mean task accuracy (%)")
        axis.grid(True, axis="x", color=GRID, alpha=0.55, linewidth=0.7)
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(INK)
        axis.set_title(title, fontsize=10, pad=8)
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=RED, markersize=6.5, label="Deranged (label-shuffled) accuracy"),
        plt.Line2D([], [], marker="o", linestyle="", color=BLUE, markersize=6.5, label="Clean accuracy"),
        plt.Line2D([], [], color="#9CA3AF", linewidth=3.2, label="ICL effect, with 95% bootstrap interval in brackets"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=9)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_stem.with_suffix(".pdf"))
    fig.savefig(args.output_stem.with_suffix(".png"), dpi=220)
    print("wrote", args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
