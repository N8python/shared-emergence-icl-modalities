#!/usr/bin/env python3
"""Exact paired task-level tests of clean > deranged at each canonical maximum shot."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def task_key(task: dict) -> str:
    return " -> ".join(task["program"])


def load_successes(path: Path, expected_shot: int) -> tuple[list[str], np.ndarray]:
    data = json.loads(path.read_text())
    config = data["config"]
    if int(config["in_context_examples"]) != expected_shot:
        raise ValueError(f"{path}: wrong shot count")
    if int(config["trials_per_program"]) != 128:
        raise ValueError(f"{path}: expected 128 trials/task")
    tasks = data["tasks"]
    if len(tasks) != 100:
        raise ValueError(f"{path}: expected 100 tasks")

    keys = [task_key(task) for task in tasks]
    successes = []
    for task in tasks:
        trials = task["trials"]
        if len(trials) != 128:
            raise ValueError(f"{path}: {task_key(task)} does not have 128 trials")
        success = sum(bool(trial["correct"]) for trial in trials)
        if success != int(task["correct"]):
            raise ValueError(f"{path}: stored task success count disagrees")
        successes.append(success)
    return keys, np.asarray(successes, dtype=int)


def validate_condition_pair(clean_path: Path, deranged_path: Path) -> None:
    clean = json.loads(clean_path.read_text())
    deranged = json.loads(deranged_path.read_text())
    clean_config = clean["config"]
    deranged_config = deranged["config"]
    for field in (
        "model",
        "in_context_examples",
        "trials_per_program",
        "bit_length",
        "seed",
    ):
        if clean_config[field] != deranged_config[field]:
            raise ValueError(
                f"{clean_path} vs {deranged_path}: config {field} differs"
            )
    if clean_config["ablate_labels"] is not False:
        raise ValueError(f"{clean_path}: expected clean ablate_labels=False")
    if deranged_config["ablate_labels"] is not True:
        raise ValueError(f"{deranged_path}: expected deranged ablate_labels=True")


def exact_task_swap_p(differences: np.ndarray) -> tuple[int, int, int]:
    """Return numerator, denominator, and nonzero task count for P(T >= T_obs).

    Under the paired sharp null, clean/deranged labels are exchangeable within
    each task. Swapping labels negates that task's clean-minus-deranged count.
    Dynamic programming enumerates the exact distribution without Monte Carlo.
    """

    integer_differences = np.asarray(differences, dtype=int)
    observed = int(integer_differences.sum())
    weights = [abs(int(value)) for value in integer_differences if value != 0]
    counts: dict[int, int] = {0: 1}
    for weight in weights:
        updated: defaultdict[int, int] = defaultdict(int)
        for total, count in counts.items():
            updated[total + weight] += count
            updated[total - weight] += count
        counts = dict(updated)
    numerator = sum(count for total, count in counts.items() if total >= observed)
    denominator = 1 << len(weights)
    if sum(counts.values()) != denominator:
        raise AssertionError("Exact randomization state count is inconsistent")
    if any(counts[total] != counts.get(-total, 0) for total in counts):
        raise AssertionError("Exact randomization distribution is not symmetric")
    return numerator, denominator, len(weights)


def holm_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * p_values[int(index)]
        running = max(running, candidate)
        adjusted[int(index)] = min(1.0, running)
    return adjusted.tolist()


def bh_adjust(p_values: list[float]) -> list[float]:
    count = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running = 1.0
    for reversed_rank in range(count - 1, -1, -1):
        index = int(order[reversed_rank])
        rank = reversed_rank + 1
        candidate = p_values[index] * count / rank
        running = min(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    rows: list[dict[str, object]] = []
    canonical_keys = config["experiment"]["canonical_run_keys"]

    for run_key in canonical_keys:
        run = config["runs"][run_key]
        clean_config = run["conditions"]["clean"]
        deranged_config = run["conditions"]["deranged"]
        common_shots = sorted(
            set(map(int, clean_config["shots"]))
            & set(map(int, deranged_config["shots"]))
        )
        if not common_shots:
            raise ValueError(f"{run_key}: no common clean/deranged shots")
        shot = common_shots[-1]
        clean_path = args.results_root / clean_config["output_template"].format(
            run=run_key,
            condition="clean",
            shot=shot,
        )
        deranged_path = args.results_root / deranged_config[
            "output_template"
        ].format(
            run=run_key,
            condition="deranged",
            shot=shot,
        )

        clean_keys, clean_successes = load_successes(clean_path, shot)
        deranged_keys, deranged_successes = load_successes(deranged_path, shot)
        if clean_keys != deranged_keys:
            raise ValueError(f"{run_key}: clean/deranged task order differs")
        validate_condition_pair(clean_path, deranged_path)

        differences = clean_successes - deranged_successes
        task_effects = differences.astype(float) / 128.0
        numerator, denominator, nonzero_tasks = exact_task_swap_p(differences)
        exact_p = numerator / denominator
        paired_t = stats.ttest_rel(
            clean_successes / 128.0,
            deranged_successes / 128.0,
            alternative="greater",
        )
        wilcoxon = stats.wilcoxon(
            task_effects,
            alternative="greater",
            zero_method="wilcox",
            method="approx",
        )
        mean_effect = float(task_effects.mean())
        paired_se = float(task_effects.std(ddof=1) / math.sqrt(len(task_effects)))

        rows.append(
            {
                "run_key": run_key,
                "label": run["label"],
                "highest_shot": shot,
                "tasks": 100,
                "trials_per_task": 128,
                "clean_accuracy": float(clean_successes.mean() / 128.0),
                "deranged_accuracy": float(
                    deranged_successes.mean() / 128.0
                ),
                "clean_minus_deranged": mean_effect,
                "paired_task_se": paired_se,
                "normal_ci95_low": mean_effect - 1.96 * paired_se,
                "normal_ci95_high": mean_effect + 1.96 * paired_se,
                "tasks_clean_gt_deranged": int(np.sum(differences > 0)),
                "tasks_clean_eq_deranged": int(np.sum(differences == 0)),
                "tasks_clean_lt_deranged": int(np.sum(differences < 0)),
                "exact_task_swap_nonzero_tasks": nonzero_tasks,
                "exact_task_swap_extreme_states": numerator,
                "exact_task_swap_total_states": denominator,
                "one_sided_exact_task_swap_p": exact_p,
                "one_sided_paired_t_p": float(paired_t.pvalue),
                "one_sided_wilcoxon_approx_p": float(wilcoxon.pvalue),
                "clean_path": str(clean_path.relative_to(args.results_root)),
                "deranged_path": str(
                    deranged_path.relative_to(args.results_root)
                ),
            }
        )

    raw_p = [float(row["one_sided_exact_task_swap_p"]) for row in rows]
    for row, holm_p, bh_p in zip(
        rows,
        holm_adjust(raw_p),
        bh_adjust(raw_p),
        strict=True,
    ):
        row["holm_p_6_tests"] = holm_p
        row["bh_fdr_p_6_tests"] = bh_p

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        "Exact one-sided task-pair randomization tests "
        "(clean > deranged at maximum shot):"
    )
    for row in rows:
        print(
            f"{row['run_key']:10s} n={row['highest_shot']:2d} "
            f"gap={row['clean_minus_deranged']:.6f} "
            f"p={row['one_sided_exact_task_swap_p']:.10g} "
            f"Holm={row['holm_p_6_tests']:.10g}"
        )
    print(args.output)


if __name__ == "__main__":
    main()
