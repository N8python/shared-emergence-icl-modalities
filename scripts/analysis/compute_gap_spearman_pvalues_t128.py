#!/usr/bin/env python3
"""One-sided permutation tests for clean-minus-deranged Spearman correlations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import rankdata, spearmanr


SPECS = [
    (
        "Language",
        "qwen3/clean/Qwen3-14B-Base_ic64.json",
        "qwen3/deranged/Qwen3-14B-Base_deranged_ic64.json",
    ),
    (
        "Genome",
        "evo2/clean/evo2_40b_ic64.json",
        "evo2/deranged/evo2_40b_deranged_ic64.json",
    ),
    (
        "Integer",
        "nextterm/clean/NextTerm-440M_ic64.json",
        "nextterm/deranged/NextTerm-440M_ic64.json",
    ),
    (
        "Image",
        "imagegpt/clean/imagegpt-large-bf16_31shot_leftpad_clean.json",
        "imagegpt/deranged/imagegpt-large-bf16_31shot_leftpad_deranged.json",
    ),
    (
        "Time",
        "timesfm/clean/timesfm2_5_fp32_sine_lobe_2bit_iozero_ic48.json",
        "timesfm/deranged/timesfm2_5_fp32_sine_lobe_2bit_iozero_deranged_ic48.json",
    ),
    (
        "Protein",
        "progen2/clean/ProGen2-base-bf16_ic48.json",
        "progen2/deranged/ProGen2-base-bf16_ic48.json",
    ),
]

CLUSTERED_ORDER = ("Integer", "Genome", "Protein", "Time", "Language", "Image")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=1_000_000)
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def program_key(task: dict) -> str:
    return " -> ".join(task["program"])


def load_accuracy(path: Path) -> tuple[list[str], np.ndarray]:
    data = json.loads(path.read_text())
    tasks = data["tasks"]
    keys = [program_key(task) for task in tasks]
    if len(tasks) != 100 or len(set(keys)) != 100:
        raise ValueError(f"{path}: expected 100 unique tasks")
    trials = np.array([int(task["total_trials"]) for task in tasks])
    if not np.all(trials == 128):
        raise ValueError(f"{path}: expected 128 trials per task")
    counts = np.array(
        [
            sum(bool(record["correct"]) for record in task.get("trials", []))
            for task in tasks
        ],
        dtype=float,
    )
    if any(len(task.get("trials", [])) != 128 for task in tasks):
        raise ValueError(f"{path}: incomplete trial records")
    return keys, counts / 128.0


def load_gap_profiles(data_root: Path) -> tuple[list[str], np.ndarray]:
    names = []
    profiles = []
    canonical_keys = None
    for name, clean_relative, deranged_relative in SPECS:
        clean_keys, clean = load_accuracy(data_root / clean_relative)
        deranged_keys, deranged = load_accuracy(data_root / deranged_relative)
        if clean_keys != deranged_keys:
            raise ValueError(f"{name}: clean and deranged task order differs")
        if canonical_keys is None:
            canonical_keys = clean_keys
        elif clean_keys != canonical_keys:
            raise ValueError(f"{name}: task order differs across modalities")
        names.append(name)
        profiles.append(clean - deranged)
    return names, np.asarray(profiles)


def adjust_holm(values: np.ndarray) -> np.ndarray:
    count = len(values)
    order = np.argsort(values)
    adjusted_sorted = np.maximum.accumulate(
        (count - np.arange(count)) * values[order]
    )
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def adjust_bh(values: np.ndarray) -> np.ndarray:
    count = len(values)
    order = np.argsort(values)
    ranked = values[order] * count / np.arange(1, count + 1)
    adjusted_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.empty(count)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def format_p(value: float, floor: float) -> str:
    if value <= floor:
        return f"≤{floor:.1e}"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names, profiles = load_gap_profiles(args.data_root)
    count, task_count = profiles.shape

    ranks = rankdata(profiles, axis=1, method="average")
    ranks -= ranks.mean(axis=1, keepdims=True)
    ranks /= np.sqrt(np.sum(ranks**2, axis=1, keepdims=True))
    observed = ranks @ ranks.T

    pairs = [(left, right) for left in range(count) for right in range(left + 1, count)]
    exceedances = np.zeros(len(pairs), dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    completed = 0
    while completed < args.permutations:
        size = min(args.chunk_size, args.permutations - completed)
        permutations = np.argsort(
            rng.random((size, task_count), dtype=np.float32), axis=1
        )
        for right in range(1, count):
            permuted_right = ranks[right][permutations]
            correlations = permuted_right @ ranks[:right].T
            for left in range(right):
                pair_index = pairs.index((left, right))
                exceedances[pair_index] += np.count_nonzero(
                    correlations[:, left] >= observed[left, right] - 1e-15
                )
        completed += size
        print(f"permutations: {completed:,}/{args.permutations:,}", flush=True)

    permutation_p = (exceedances + 1) / (args.permutations + 1)
    asymptotic_p = np.array(
        [
            spearmanr(
                profiles[left], profiles[right], alternative="greater"
            ).pvalue
            for left, right in pairs
        ]
    )
    holm_p = adjust_holm(permutation_p)
    bh_p = adjust_bh(permutation_p)
    monte_carlo_se = np.sqrt(
        permutation_p * (1.0 - permutation_p) / (args.permutations + 1)
    )

    rows = []
    for pair_index, (left, right) in enumerate(pairs):
        rows.append(
            {
                "left": names[left],
                "right": names[right],
                "observed_spearman": float(observed[left, right]),
                "permutation_exceedances": int(exceedances[pair_index]),
                "permutations": args.permutations,
                "one_sided_permutation_p": float(permutation_p[pair_index]),
                "permutation_monte_carlo_se": float(monte_carlo_se[pair_index]),
                "holm_p_15_tests": float(holm_p[pair_index]),
                "bh_fdr_p_15_tests": float(bh_p[pair_index]),
                "one_sided_asymptotic_p": float(asymptotic_p[pair_index]),
            }
        )

    csv_path = args.output_dir / "gap_spearman_positive_pvalues.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    floor = 1.0 / (args.permutations + 1)
    report_path = args.output_dir / "gap_spearman_positive_pvalues.md"
    with report_path.open("w") as handle:
        handle.write("# One-sided permutation tests for positive gap correlation\n\n")
        handle.write(
            f"- Null hypothesis: task labels are exchangeable between modalities "
            f"(`H0: rho <= 0`).\n"
        )
        handle.write(f"- Aligned tasks: {task_count}\n")
        handle.write(f"- Monte Carlo permutations: {args.permutations:,}\n")
        handle.write(f"- Seed: {args.seed}\n")
        handle.write(
            "- Holm and Benjamini-Hochberg corrections cover all 15 modality pairs.\n"
        )
        handle.write(
            f"- Minimum reportable add-one permutation p-value: {floor:.2e}.\n\n"
        )
        handle.write(
            "| Pair | Spearman rho | One-sided permutation p | Holm p | BH-FDR p | "
            "Asymptotic p |\n"
        )
        handle.write("|---|---:|---:|---:|---:|---:|\n")
        for row in sorted(rows, key=lambda item: item["one_sided_permutation_p"]):
            handle.write(
                f"| {row['left']}–{row['right']} | "
                f"{row['observed_spearman']:.3f} | "
                f"{format_p(row['one_sided_permutation_p'], floor)} | "
                f"{format_p(row['holm_p_15_tests'], floor)} | "
                f"{format_p(row['bh_fdr_p_15_tests'], floor)} | "
                f"{row['one_sided_asymptotic_p']:.2e} |\n"
            )

    name_to_index = {name: index for index, name in enumerate(names)}
    order = np.array([name_to_index[name] for name in CLUSTERED_ORDER])
    ordered = observed[np.ix_(order, order)]
    p_matrix = np.full((count, count), np.nan)
    for pair_index, (left, right) in enumerate(pairs):
        p_matrix[left, right] = p_matrix[right, left] = permutation_p[pair_index]
    ordered_p = p_matrix[np.ix_(order, order)]

    fig, ax = plt.subplots(figsize=(10.2, 8.0), constrained_layout=True)
    image = ax.imshow(ordered, cmap="Reds", vmin=0, vmax=1)
    ordered_names = [names[index] for index in order]
    ax.set_xticks(
        range(count), labels=ordered_names, rotation=35, ha="right"
    )
    ax.set_yticks(range(count), labels=ordered_names)
    ax.set_title(
        "Clean − deranged per-task effects: Spearman ρ and one-sided p\n"
        "128 trials/task; 1,000,000 task-label permutations",
        pad=16,
        fontsize=15,
    )
    ax.set_xticks(np.arange(-0.5, count, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, count, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(count):
        for column in range(count):
            if row == column:
                text = "1.00"
            else:
                text = (
                    f"{ordered[row, column]:.2f}\n"
                    f"p{format_p(ordered_p[row, column], floor)}"
                )
            ax.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color="white" if ordered[row, column] >= 0.58 else "black",
                fontsize=9.5,
            )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86, pad=0.04)
    colorbar.set_label("Spearman ρ", rotation=270, labelpad=18)
    plot_path = args.output_dir / "gap_spearman_positive_pvalues_clustered.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")

    print(csv_path)
    print(report_path)
    print(plot_path)


if __name__ == "__main__":
    main()
