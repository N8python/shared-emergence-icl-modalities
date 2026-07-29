#!/usr/bin/env python3
"""Analyze clean, deranged, gap, and residualized task correlations.

The observed matrices use the per-task exact-match accuracies in each JSON.
Two uncertainty analyses are reported: (1) Jeffreys-posterior propagation for
each task's Bernoulli success probability while holding the fixed 100-task
suite constant, and (2) a paired bootstrap of the 100 aligned tasks.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

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

DISCRETE = ("Language", "Genome", "Integer", "Protein")
REPO_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--posterior-draws", type=int, default=10_000)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def program_key(task: dict) -> str:
    return " -> ".join(task["program"])


def load_condition(path: Path) -> dict:
    data = json.loads(path.read_text())
    tasks = data["tasks"]
    keys = [program_key(task) for task in tasks]
    if len(tasks) != 100 or len(set(keys)) != 100:
        raise ValueError(f"{path}: expected 100 unique tasks, found {len(set(keys))}")

    trials = np.array([int(task["total_trials"]) for task in tasks], dtype=int)
    if len(set(trials.tolist())) != 1:
        raise ValueError(f"{path}: inconsistent trial counts")
    n = int(trials[0])

    counts = []
    for task in tasks:
        records = task.get("trials", [])
        if len(records) != n:
            raise ValueError(
                f"{path}: {program_key(task)} has {len(records)} records, expected {n}"
            )
        counts.append(sum(bool(record["correct"]) for record in records))
    counts_array = np.array(counts, dtype=int)
    reported = np.array([float(task["accuracy"]) for task in tasks])
    calculated = counts_array / n
    if not np.allclose(reported, calculated, atol=1e-12):
        raise ValueError(f"{path}: stored task accuracies disagree with trial records")

    return {
        "path": str(path),
        "data": data,
        "keys": keys,
        "counts": counts_array,
        "n": n,
        "accuracy": calculated,
    }


def load_root(root: Path) -> dict:
    loaded = {}
    canonical_keys = None
    for name, clean_relative, deranged_relative in SPECS:
        clean = load_condition(root / clean_relative)
        deranged = load_condition(root / deranged_relative)
        if clean["keys"] != deranged["keys"]:
            raise ValueError(f"{name}: clean and deranged task order differs")
        if canonical_keys is None:
            canonical_keys = clean["keys"]
        elif clean["keys"] != canonical_keys:
            raise ValueError(f"{name}: task order differs from the other modalities")
        loaded[name] = {"clean": clean, "deranged": deranged}
    return loaded


def matrix(profiles: np.ndarray) -> np.ndarray:
    count = profiles.shape[0]
    result = np.eye(count)
    for left in range(count):
        for right in range(left + 1, count):
            rho = float(spearmanr(profiles[left], profiles[right]).statistic)
            result[left, right] = result[right, left] = rho
    return result


def residual_profiles(clean: np.ndarray, deranged: np.ndarray) -> np.ndarray:
    """Residualize each modality's clean profile against its deranged profile."""
    clean_centered = clean - clean.mean(axis=-1, keepdims=True)
    deranged_centered = deranged - deranged.mean(axis=-1, keepdims=True)
    denominator = np.sum(deranged_centered**2, axis=-1, keepdims=True)
    slope = np.divide(
        np.sum(clean_centered * deranged_centered, axis=-1, keepdims=True),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return clean_centered - slope * deranged_centered


def observed_matrices(loaded: dict) -> dict[str, np.ndarray]:
    clean = np.array([loaded[name]["clean"]["accuracy"] for name, *_ in SPECS])
    deranged = np.array([loaded[name]["deranged"]["accuracy"] for name, *_ in SPECS])
    return {
        "clean": matrix(clean),
        "deranged": matrix(deranged),
        "gap": matrix(clean - deranged),
        "residual": matrix(residual_profiles(clean, deranged)),
    }


def row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = left - left.mean(axis=1, keepdims=True)
    right = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left * right, axis=1)
    denominator = np.sqrt(np.sum(left * left, axis=1) * np.sum(right * right, axis=1))
    return numerator / denominator


def posterior_intervals(
    loaded: dict, *, draws: int, seed: int
) -> dict[tuple[str, int, int], dict[str, float]]:
    rng = np.random.default_rng(seed)
    names = [name for name, *_ in SPECS]
    counts_clean = np.array([loaded[name]["clean"]["counts"] for name in names])
    counts_deranged = np.array([loaded[name]["deranged"]["counts"] for name in names])
    trials_clean = np.array([loaded[name]["clean"]["n"] for name in names])[:, None]
    trials_deranged = np.array([loaded[name]["deranged"]["n"] for name in names])[:, None]

    samples = {
        (metric, left, right): []
        for metric in ("clean", "deranged", "gap", "residual")
        for left in range(len(names))
        for right in range(left + 1, len(names))
    }

    chunk_size = min(500, draws)
    completed = 0
    while completed < draws:
        size = min(chunk_size, draws - completed)
        clean = rng.beta(
            counts_clean[None, :, :] + 0.5,
            trials_clean[None, :, :] - counts_clean[None, :, :] + 0.5,
            size=(size, len(names), counts_clean.shape[1]),
        )
        deranged = rng.beta(
            counts_deranged[None, :, :] + 0.5,
            trials_deranged[None, :, :] - counts_deranged[None, :, :] + 0.5,
            size=(size, len(names), counts_deranged.shape[1]),
        )
        profiles = {
            "clean": clean,
            "deranged": deranged,
            "gap": clean - deranged,
            "residual": residual_profiles(clean, deranged),
        }
        for metric, values in profiles.items():
            ranks = rankdata(values, axis=2, method="average")
            for left in range(len(names)):
                for right in range(left + 1, len(names)):
                    samples[(metric, left, right)].append(
                        row_correlations(ranks[:, left, :], ranks[:, right, :])
                    )
        completed += size
        print(f"posterior draws: {completed:,}/{draws:,}", flush=True)

    summaries = {}
    for key, chunks in samples.items():
        values = np.concatenate(chunks)
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        summaries[key] = {
            "posterior_mean": float(values.mean()),
            "posterior_median": float(median),
            "ci95_low": float(low),
            "ci95_high": float(high),
        }
    return summaries


def task_bootstrap_intervals(
    loaded: dict, *, draws: int, seed: int
) -> dict[tuple[str, int, int], dict[str, float]]:
    """Resample the 100 aligned tasks, preserving all cross-modality pairing."""
    rng = np.random.default_rng(seed)
    names = [name for name, *_ in SPECS]
    clean_base = np.array(
        [loaded[name]["clean"]["accuracy"] for name in names]
    )
    deranged_base = np.array(
        [loaded[name]["deranged"]["accuracy"] for name in names]
    )
    task_count = clean_base.shape[1]
    samples = {
        (metric, left, right): []
        for metric in ("clean", "deranged", "gap", "residual")
        for left in range(len(names))
        for right in range(left + 1, len(names))
    }

    chunk_size = min(500, draws)
    completed = 0
    while completed < draws:
        size = min(chunk_size, draws - completed)
        indices = rng.integers(0, task_count, size=(size, task_count))
        clean = np.take_along_axis(
            np.broadcast_to(clean_base, (size, *clean_base.shape)),
            indices[:, None, :],
            axis=2,
        )
        deranged = np.take_along_axis(
            np.broadcast_to(deranged_base, (size, *deranged_base.shape)),
            indices[:, None, :],
            axis=2,
        )
        profiles = {
            "clean": clean,
            "deranged": deranged,
            "gap": clean - deranged,
            "residual": residual_profiles(clean, deranged),
        }
        for metric, values in profiles.items():
            ranks = rankdata(values, axis=2, method="average")
            for left in range(len(names)):
                for right in range(left + 1, len(names)):
                    samples[(metric, left, right)].append(
                        row_correlations(ranks[:, left, :], ranks[:, right, :])
                    )
        completed += size
        print(f"task bootstrap draws: {completed:,}/{draws:,}", flush=True)

    summaries = {}
    for key, chunks in samples.items():
        values = np.concatenate(chunks)
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        summaries[key] = {
            "task_bootstrap_mean": float(values.mean()),
            "task_bootstrap_median": float(median),
            "task_bootstrap_ci95_low": float(low),
            "task_bootstrap_ci95_high": float(high),
        }
    return summaries


def serialize_matrix(value: np.ndarray) -> list[list[float]]:
    return [[float(cell) for cell in row] for row in value]


def range_for(matrix_value: np.ndarray, names: list[str]) -> tuple[float, float]:
    indices = [names.index(name) for name in DISCRETE]
    values = [
        float(matrix_value[left, right])
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    ]
    return min(values), max(values)


def plot_matrices(
    observed: dict[str, np.ndarray],
    reference: dict[str, np.ndarray] | None,
    names: list[str],
    output_dir: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = ("clean", "deranged", "gap", "residual")
    rows = [("128 trials/task", observed)]
    if reference is not None:
        rows.insert(0, ("8 trials/task", reference))
    fig, axes = plt.subplots(
        len(rows),
        len(metrics),
        figsize=(14.5, 3.5 * len(rows)),
        squeeze=False,
        constrained_layout=True,
    )
    short_names = ["Lang.", "Genome", "Integer", "Image", "Time", "Protein"]
    image = None
    for row_index, (row_label, matrices) in enumerate(rows):
        for column_index, metric in enumerate(metrics):
            axis = axes[row_index, column_index]
            values = matrices[metric]
            image = axis.imshow(values, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            axis.set_xticks(range(len(names)), short_names, rotation=45, ha="right")
            axis.set_yticks(range(len(names)), short_names)
            axis.set_title(f"{row_label}: {metric}")
            for left in range(len(names)):
                for right in range(len(names)):
                    value = values[left, right]
                    axis.text(
                        right,
                        left,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(value) > 0.55 else "black",
                    )
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, label="Spearman rho")
    fig.savefig(output_dir / "correlation_matrices_8_vs_128.png", dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    loaded = load_root(args.data_root)
    observed = observed_matrices(loaded)
    reference = None
    if args.reference_root is not None:
        reference = observed_matrices(load_root(args.reference_root))

    intervals = posterior_intervals(
        loaded, draws=args.posterior_draws, seed=args.seed
    )
    bootstrap_intervals = task_bootstrap_intervals(
        loaded, draws=args.bootstrap_draws, seed=args.seed + 1
    )
    names = [name for name, *_ in SPECS]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            for metric in ("clean", "deranged", "gap", "residual"):
                row = {
                    "left": names[left],
                    "right": names[right],
                    "metric": metric,
                    "observed_spearman": float(observed[metric][left, right]),
                    **intervals[(metric, left, right)],
                    **bootstrap_intervals[(metric, left, right)],
                }
                if reference is not None:
                    row["reference_spearman"] = float(
                        reference[metric][left, right]
                    )
                rows.append(row)

    with (args.output_dir / "pairwise_correlations.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data_root": portable_path(args.data_root),
        "reference_root": portable_path(args.reference_root),
        "posterior_draws": args.posterior_draws,
        "task_bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "modalities": names,
        "trial_counts": {
            name: {
                condition: loaded[name][condition]["n"]
                for condition in ("clean", "deranged")
            }
            for name in names
        },
        "overall_accuracy": {
            name: {
                condition: float(loaded[name][condition]["accuracy"].mean())
                for condition in ("clean", "deranged")
            }
            for name in names
        },
        "observed_spearman": {
            metric: serialize_matrix(value) for metric, value in observed.items()
        },
        "reference_spearman": (
            {
                metric: serialize_matrix(value)
                for metric, value in reference.items()
            }
            if reference is not None
            else None
        ),
        "discrete_ranges": {
            metric: list(range_for(value, names))
            for metric, value in observed.items()
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    lines = [
        "# 128-trial task-correlation analysis",
        "",
        f"- Data root: `{portable_path(args.data_root)}`",
        f"- Posterior draws: {args.posterior_draws:,}",
        f"- Paired task-bootstrap draws: {args.bootstrap_draws:,}",
        "- Posterior-propagation intervals: Jeffreys uncertainty for each "
        "task's Bernoulli success probability; the 100-task suite is held fixed.",
        "- Task-bootstrap intervals: uncertainty from resampling the aligned "
        "100 tasks; every modality and condition uses the same sampled indices.",
        "",
        "## Overall exact-match accuracy",
        "",
        "| Modality | Clean | Deranged | Gap | Trials/task/condition |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in names:
        clean = summary["overall_accuracy"][name]["clean"]
        deranged = summary["overall_accuracy"][name]["deranged"]
        n = loaded[name]["clean"]["n"]
        lines.append(
            f"| {name} | {clean:.4f} | {deranged:.4f} | "
            f"{clean - deranged:.4f} | {n} |"
        )

    lines.extend(["", "## Discrete-modality observed Spearman ranges", ""])
    for metric in ("clean", "deranged", "gap", "residual"):
        low, high = summary["discrete_ranges"][metric]
        lines.append(f"- {metric}: {low:.3f} to {high:.3f}")

    lines.extend(
        [
            "",
            "## Discrete-modality clean-minus-deranged pairs",
            "",
            "| Pair | Observed rho | Posterior 95% interval | "
            "Task-bootstrap 95% interval |"
            + (" Reference rho |" if reference is not None else ""),
            "|---|---:|---:|---:|"
            + ("---:|" if reference is not None else ""),
        ]
    )
    discrete_indices = [names.index(name) for name in DISCRETE]
    for offset, left in enumerate(discrete_indices):
        for right in discrete_indices[offset + 1 :]:
            result = intervals[("gap", left, right)]
            bootstrap = bootstrap_intervals[("gap", left, right)]
            line = (
                f"| {names[left]}–{names[right]} | "
                f"{observed['gap'][left, right]:.3f} | "
                f"[{result['ci95_low']:.3f}, {result['ci95_high']:.3f}] |"
                f" [{bootstrap['task_bootstrap_ci95_low']:.3f}, "
                f"{bootstrap['task_bootstrap_ci95_high']:.3f}] |"
            )
            if reference is not None:
                line += f" {reference['gap'][left, right]:.3f} |"
            lines.append(line)

    lines.extend(
        [
            "",
            "## Discrete-modality residualized-clean pairs",
            "",
            "| Pair | Observed rho | Posterior 95% interval | "
            "Task-bootstrap 95% interval |"
            + (" Reference rho |" if reference is not None else ""),
            "|---|---:|---:|---:|"
            + ("---:|" if reference is not None else ""),
        ]
    )
    for offset, left in enumerate(discrete_indices):
        for right in discrete_indices[offset + 1 :]:
            result = intervals[("residual", left, right)]
            bootstrap = bootstrap_intervals[("residual", left, right)]
            line = (
                f"| {names[left]}–{names[right]} | "
                f"{observed['residual'][left, right]:.3f} | "
                f"[{result['ci95_low']:.3f}, {result['ci95_high']:.3f}] |"
                f" [{bootstrap['task_bootstrap_ci95_low']:.3f}, "
                f"{bootstrap['task_bootstrap_ci95_high']:.3f}] |"
            )
            if reference is not None:
                line += f" {reference['residual'][left, right]:.3f} |"
            lines.append(line)

    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n")
    plot_matrices(observed, reference, names, args.output_dir)
    print(f"Wrote analysis to {args.output_dir}")


if __name__ == "__main__":
    main()
