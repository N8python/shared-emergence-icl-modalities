#!/usr/bin/env python3
"""Plot clean-minus-deranged accuracy across model sizes using the T=128 data.

The shot count for each model is selected exactly as in the original Figure 6:
choose the shot with the highest clean accuracy, then subtract deranged
accuracy at that same shot. Error bars are standard errors of the 100 paired
per-task accuracy differences.
"""

import argparse
import csv
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_ANALYSIS = (
    Path.home()
    / "Documents"
    / "ICLManyReplication_128_2026-07-28"
    / "expanded"
    / "analysis_128_full"
)

# Display order is smallest to largest within each family.
MODEL_GROUPS = [
    (
        "ImageGPT",
        "#2B6FB0",
        [
            ("imagegpt_small", "small"),
            ("imagegpt_medium", "medium"),
            ("imagegpt", "large"),
        ],
    ),
    (
        "NextTerm",
        "#E8622A",
        [
            ("nextterm_47m", "47M"),
            ("nextterm", "440M"),
        ],
    ),
    (
        "Qwen3",
        "#2B5FD6",
        [
            ("qwen3_0_6b", "0.6B"),
            ("qwen3_1_7b", "1.7B"),
            ("qwen3_4b", "4B"),
            ("qwen3_8b", "8B"),
            ("qwen3", "14B"),
        ],
    ),
    (
        "Evo2",
        "#2E8B57",
        [
            ("evo2_1b_base", "1B"),
            ("evo2_7b", "7B"),
            ("evo2", "40B"),
        ],
    ),
    (
        "ProGen2",
        "#7B3FBF",
        [
            ("progen2_small", "small"),
            ("progen2_medium", "medium"),
            ("progen2", "base"),
            ("progen2_large", "large"),
            ("progen2_xlarge", "xlarge"),
        ],
    ),
]


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def shade(base_hex, fraction):
    base = np.array(mcolors.to_rgb(base_hex))
    white = np.array([1.0, 1.0, 1.0])
    amount = 0.40 + 0.60 * fraction
    return tuple(white * (1 - amount) + base * amount)


def select_rows(best_shots_path, paired_gaps_path, cell_summary_path):
    best = {row["run_key"]: row for row in read_csv(best_shots_path)}
    paired = {
        (row["run_key"], int(row["shot"])): row
        for row in read_csv(paired_gaps_path)
    }
    cells = {
        (row["run_key"], row["condition"], int(row["shot"])): row
        for row in read_csv(cell_summary_path)
    }

    selected = []
    for family, _, models in MODEL_GROUPS:
        for run_key, size_label in models:
            shot = int(best[run_key]["best_clean_shot"])
            gap = paired[run_key, shot]
            clean_cell = cells[run_key, "clean", shot]
            deranged_cell = cells[run_key, "deranged", shot]
            for cell in (clean_cell, deranged_cell):
                if int(cell["tasks"]) != 100 or int(cell["trials_per_task"]) != 128:
                    raise ValueError(
                        f"{run_key} at {shot} shots is not a 100-task T=128 cell"
                    )
            selected.append(
                {
                    "family": family,
                    "run_key": run_key,
                    "model_size": size_label,
                    "shot": shot,
                    "clean_accuracy": float(gap["clean_accuracy"]),
                    "deranged_accuracy": float(gap["deranged_accuracy"]),
                    "clean_minus_deranged": float(gap["clean_minus_deranged"]),
                    "paired_task_se": float(gap["paired_task_se"]),
                    "task_bootstrap_ci95_low": float(
                        gap["task_bootstrap_ci95_low"]
                    ),
                    "task_bootstrap_ci95_high": float(
                        gap["task_bootstrap_ci95_high"]
                    ),
                    "tasks": 100,
                    "trials_per_task": 128,
                }
            )
    return selected


def write_selected_csv(path, rows):
    fields = [
        "family",
        "run_key",
        "model_size",
        "shot",
        "clean_accuracy",
        "deranged_accuracy",
        "clean_minus_deranged",
        "paired_task_se",
        "task_bootstrap_ci95_low",
        "task_bootstrap_ci95_high",
        "tasks",
        "trials_per_task",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows, output_path):
    by_key = {row["run_key"]: row for row in rows}
    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    x = 0.0
    group_gap = 1.0
    bar_width = 0.82
    xticks = []
    xticklabels = []
    group_centers = []
    group_names = []
    ymax = 0.0

    for family, base_color, models in MODEL_GROUPS:
        start = x
        for index, (run_key, size_label) in enumerate(models):
            row = by_key[run_key]
            fraction = 0.5 if len(models) == 1 else index / (len(models) - 1)
            effect = row["clean_minus_deranged"]
            se = row["paired_task_se"]
            ax.bar(
                x,
                effect,
                width=bar_width,
                color=shade(base_color, fraction),
                edgecolor="#222222",
                linewidth=0.7,
                zorder=2,
            )
            ax.errorbar(
                x,
                effect,
                yerr=se,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.9,
                capsize=2.5,
                zorder=3,
            )
            xticks.append(x)
            xticklabels.append(size_label)
            ymax = max(ymax, effect + se)
            x += 1.0
        group_centers.append((start + x - 1.0) / 2.0)
        group_names.append(family)
        x += group_gap

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Clean - deranged accuracy", fontsize=12)
    ax.set_ylim(0, ymax * 1.20)
    ax.set_xlim(-0.8, x - group_gap)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for center, family in zip(group_centers, group_names):
        ax.text(
            center,
            ymax * 1.15,
            family,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--best-shots",
        type=Path,
        default=DEFAULT_ANALYSIS / "best_shot_zgaps.csv",
    )
    parser.add_argument(
        "--paired-gaps",
        type=Path,
        default=DEFAULT_ANALYSIS / "paired_gaps.csv",
    )
    parser.add_argument(
        "--cell-summary",
        type=Path,
        default=DEFAULT_ANALYSIS / "cell_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "modalities_scale_bars.pdf",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=HERE / "modalities_scale_effects_128.csv",
    )
    args = parser.parse_args()

    rows = select_rows(args.best_shots, args.paired_gaps, args.cell_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    write_selected_csv(args.csv_output, rows)
    plot(rows, args.output)
    print(args.output)
    print(args.csv_output)


if __name__ == "__main__":
    main()
