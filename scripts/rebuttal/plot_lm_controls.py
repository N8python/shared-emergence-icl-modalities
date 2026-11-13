#!/usr/bin/env python3
"""Rebuttal figures: shot scaling for the two language-model controls.

Writes one figure per control, each with three panels over 1-64 shots:
mean clean (solid) and deranged (dashed) accuracy, the paired ICL effect with
its task-bootstrap 95% interval, and the Spearman correlation between the
100-task effect profiles of the runs being compared.

* literal: Qwen3-14B on literal bitstrings vs the published digit encoding.
* mistral: Mistral 7B v0.3 vs published Qwen3-8B and Qwen3-14B, digit encoding.

Reads results_rebuttal/analysis/lm_controls_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID, INK = "#D1D5DB", "#111827"
SHOTS = [1, 2, 4, 8, 16, 32, 64]

STYLE = {
    "qwen3_14b_digits": ("Qwen3-14B, digits", "#2563EB"),
    "qwen3_8b_digits": ("Qwen3-8B, digits", "#7C3AED"),
    "qwen3_14b_literal": ("Qwen3-14B, literal bitstrings", "#D97706"),
    "mistral7b_digits": ("Mistral 7B v0.3, digits", "#059669"),
}

FIGURES = {
    "literal": {
        "runs": ["qwen3_14b_literal", "qwen3_14b_digits"],
        "pairs": [
            ("qwen3_14b_literal", "qwen3_14b_digits", "Literal vs digits (Qwen3-14B)", "#D97706", "-"),
            ("qwen3_14b_digits", "qwen3_14b_digits_reseed", "Noise ceiling: digits vs digits, new seed", "#6B7280", ":"),
        ],
    },
    "mistral": {
        "runs": ["mistral7b_digits", "qwen3_8b_digits", "qwen3_14b_digits"],
        "pairs": [
            ("mistral7b_digits", "qwen3_8b_digits", "Mistral vs Qwen3-8B", "#7C3AED", "-"),
            ("mistral7b_digits", "qwen3_14b_digits", "Mistral vs Qwen3-14B", "#2563EB", "-"),
            ("qwen3_14b_digits", "qwen3_14b_digits_reseed", "Noise ceiling: Qwen3-14B vs itself, new seed", "#6B7280", ":"),
        ],
    },
}


def style_axis(axis, ylabel, title):
    axis.set_xscale("log", base=2)
    axis.set_xticks(SHOTS, labels=[str(s) for s in SHOTS])
    axis.minorticks_off()
    axis.set_xlabel("In-context examples (shots)")
    axis.set_ylabel(ylabel)
    axis.set_title(title, fontsize=10, loc="left", color=INK)
    axis.grid(True, color=GRID, alpha=0.55, linewidth=0.7)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/lm_controls_summary.json")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results_rebuttal/figures")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})

    for name, spec in FIGURES.items():
        fig, (ax_acc, ax_gap, ax_rho) = plt.subplots(1, 3, figsize=(11.4, 3.5), constrained_layout=True)
        for run in spec["runs"]:
            label, color = STYLE[run]
            block = [summary[str(s)]["runs"][run] for s in SHOTS]
            ax_acc.plot(SHOTS, [100 * b["clean"] for b in block], color=color, marker="o", markersize=4, linewidth=1.8, label=label)
            ax_acc.plot(SHOTS, [100 * b["deranged"] for b in block], color=color, marker="o", markersize=3,
                        linewidth=1.2, linestyle="--", alpha=0.75)
            ax_gap.fill_between(SHOTS, [100 * b["gap_ci95"][0] for b in block], [100 * b["gap_ci95"][1] for b in block],
                                color=color, alpha=0.16, linewidth=0)
            ax_gap.plot(SHOTS, [100 * b["gap"] for b in block], color=color, marker="o", markersize=4, linewidth=1.8, label=label)
        ax_acc.plot([], [], color="#6B7280", linewidth=1.8, label="clean (solid)")
        ax_acc.plot([], [], color="#6B7280", linewidth=1.2, linestyle="--", label="deranged (dashed)")
        ax_acc.legend(frameon=False, fontsize=7.8, loc="upper left")
        ax_gap.axhline(0, color=INK, linewidth=0.7)
        style_axis(ax_acc, "Mean task accuracy (%)", "Accuracy")
        style_axis(ax_gap, "Clean minus deranged (pp)", "ICL gain (95% CI)")

        # Profiles are identically zero at one shot (a single label cannot be deranged).
        rho_shots = SHOTS[1:]
        for left, right, label, color, linestyle in spec["pairs"]:
            have = [s for s in rho_shots if right in summary[str(s)]["pairwise_spearman"].get(left, {})]
            values = [summary[str(s)]["pairwise_spearman"][left][right] for s in have]
            ax_rho.plot(have, values, color=color, marker="o", markersize=4, linewidth=1.8, linestyle=linestyle, label=label)
        ax_rho.axhline(0, color=INK, linewidth=0.7)
        ax_rho.legend(frameon=False, fontsize=7.8, loc="upper left")
        ax_rho.set_ylim(-0.3, 1.25)
        ax_rho.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        style_axis(ax_rho, "Spearman, 100-task effect profiles", "Task-profile agreement")
        ax_rho.set_xticks(rho_shots, labels=[str(s) for s in rho_shots])
        ax_rho.set_xlim(1.7, 76)

        for suffix in ("pdf", "png"):
            fig.savefig(args.output_dir / f"lm_control_{name}.{suffix}", dpi=300)
        plt.close(fig)
        print("wrote", args.output_dir / f"lm_control_{name}.pdf")


if __name__ == "__main__":
    main()
