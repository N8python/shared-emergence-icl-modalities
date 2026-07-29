#!/usr/bin/env python3
"""Quantify correlation distortion from finite Bernoulli trials per task.

For each canonical endpoint, fit a beta distribution to the across-task latent
accuracy distribution after subtracting estimated binomial measurement noise.
Then simulate two modalities whose latent task accuracies have population
correlation zero (independent beta draws) or one (the same beta draw), with
independent Bernoulli evaluation trials in the two modalities.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata


SPECS = [
    ("Language", "qwen3/clean/Qwen3-14B-Base_ic64.json"),
    ("Genome", "evo2/clean/evo2_40b_ic64.json"),
    ("Integer", "nextterm/clean/NextTerm-440M_ic64.json"),
    (
        "Image",
        "imagegpt/clean/imagegpt-large-bf16_31shot_leftpad_clean.json",
    ),
    (
        "Time",
        "timesfm/clean/timesfm2_5_fp32_sine_lobe_2bit_iozero_ic48.json",
    ),
    ("Protein", "progen2/clean/ProGen2-base-bf16_ic48.json"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=20_000)
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser.parse_args()


def load_task_accuracies(path: Path) -> tuple[np.ndarray, int]:
    data = json.loads(path.read_text())
    n = int(data["config"]["trials_per_program"])
    counts = np.asarray(
        [
            sum(bool(record["correct"]) for record in task["trials"])
            for task in data["tasks"]
        ],
        dtype=float,
    )
    if len(counts) != 100:
        raise ValueError(f"{path}: expected 100 tasks")
    return counts / n, n


def fit_beta(accuracy: np.ndarray, measurement_trials: int) -> dict[str, float]:
    mean = float(accuracy.mean())
    observed_variance = float(accuracy.var(ddof=1))
    # p_hat * (1-p_hat) / (n-1) is unbiased for p*(1-p)/n.
    estimated_noise_variance = float(
        np.mean(accuracy * (1.0 - accuracy)) / (measurement_trials - 1)
    )
    latent_variance = max(observed_variance - estimated_noise_variance, 1e-8)
    max_beta_variance = mean * (1.0 - mean)
    concentration = max(max_beta_variance / latent_variance - 1.0, 1e-4)
    return {
        "mean": mean,
        "observed_variance": observed_variance,
        "estimated_measurement_noise_variance": estimated_noise_variance,
        "estimated_latent_variance": latent_variance,
        "alpha": mean * concentration,
        "beta": (1.0 - mean) * concentration,
        "concentration": concentration,
    }


def row_pearson(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    return numerator / denominator


def simulate(
    *,
    alpha: float,
    beta: float,
    true_correlation: int,
    trials: int,
    tasks: int,
    draws: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    pearson_chunks = []
    spearman_chunks = []
    completed = 0
    while completed < draws:
        size = min(500, draws - completed)
        latent_left = rng.beta(alpha, beta, size=(size, tasks))
        if true_correlation == 1:
            latent_right = latent_left
        else:
            latent_right = rng.beta(alpha, beta, size=(size, tasks))
        observed_left = rng.binomial(trials, latent_left) / trials
        observed_right = rng.binomial(trials, latent_right) / trials
        pearson_chunks.append(row_pearson(observed_left, observed_right))
        left_ranks = rankdata(observed_left, method="average", axis=1)
        right_ranks = rankdata(observed_right, method="average", axis=1)
        spearman_chunks.append(row_pearson(left_ranks, right_ranks))
        completed += size
    return np.concatenate(pearson_chunks), np.concatenate(spearman_chunks)


def summarize(values: np.ndarray) -> dict[str, float]:
    low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
    return {
        "mean": float(values.mean()),
        "median": float(median),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def main() -> None:
    args = parse_args()
    profiles = {}
    source_trials = None
    for name, relative in SPECS:
        profile, n = load_task_accuracies(args.data_root / relative)
        profiles[name] = profile
        if source_trials is None:
            source_trials = n
        elif source_trials != n:
            raise ValueError("Canonical endpoint files have different trial counts")
    profiles["Pooled"] = np.concatenate(list(profiles.values()))

    rng = np.random.default_rng(args.seed)
    fits = {
        name: fit_beta(profile, int(source_trials))
        for name, profile in profiles.items()
    }
    rows = []
    for name, fit in fits.items():
        for true_correlation in (0, 1):
            for trials in (8, 128):
                pearson, spearman = simulate(
                    alpha=fit["alpha"],
                    beta=fit["beta"],
                    true_correlation=true_correlation,
                    trials=trials,
                    tasks=args.tasks,
                    draws=args.draws,
                    rng=rng,
                )
                theoretical_pearson = (
                    0.0
                    if true_correlation == 0
                    else trials / (trials + fit["concentration"])
                )
                for metric, values in (
                    ("pearson", pearson),
                    ("spearman", spearman),
                ):
                    rows.append(
                        {
                            "calibration": name,
                            "true_latent_correlation": true_correlation,
                            "trials_per_task": trials,
                            "tasks": args.tasks,
                            "metric": metric,
                            "theoretical_population_pearson": (
                                theoretical_pearson
                            ),
                            **summarize(values),
                        }
                    )
        print(f"simulated {name}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (args.output_dir / "sampling_correlation_simulation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "beta_calibrations.json").write_text(
        json.dumps(
            {
                "source_trials_per_task": source_trials,
                "fits": fits,
                "simulation_draws": args.draws,
                "tasks_per_draw": args.tasks,
                "seed": args.seed,
            },
            indent=2,
        )
        + "\n"
    )

    pooled = [row for row in rows if row["calibration"] == "Pooled"]
    lines = [
        "# Bernoulli sampling-noise correlation simulation",
        "",
        f"- Calibration data: `{portable_path(args.data_root)}`",
        f"- Tasks per simulated experiment: {args.tasks}",
        f"- Monte Carlo draws: {args.draws:,}",
        (
            "- True correlation 1 uses the same latent task accuracy in both "
            "modalities but independent Bernoulli trials."
        ),
        (
            "- True correlation 0 uses independent latent task accuracies with "
            "the same fitted marginal distribution."
        ),
        "",
        "## Pooled canonical calibration",
        "",
        "| True latent corr. | Trials/task | Metric | Mean observed | 95% simulation interval |",
        "|---:|---:|---|---:|---:|",
    ]
    for row in pooled:
        lines.append(
            f"| {row['true_latent_correlation']} | {row['trials_per_task']} | "
            f"{row['metric']} | {row['mean']:.3f} | "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}] |"
        )
    (args.output_dir / "sampling_correlation_report.md").write_text(
        "\n".join(lines) + "\n"
    )
    print(f"Wrote simulation to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
