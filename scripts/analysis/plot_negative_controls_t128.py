#!/usr/bin/env python3
"""Rebuild Figure 7 from the T=128 chess and MusicRoll evaluations.

The script writes the two paper panels plus a CSV containing the plotted
accuracies, paired task-level standard errors, and exact task-swap p-values.
"""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_EXPANDED = (
    Path.home()
    / "Documents"
    / "ICLManyReplication_128_2026-07-28"
    / "expanded"
)
RUNS = {
    "chessgpt": {
        "display": "ChessGPT-50M",
        "shots": [1, 2, 4, 8],
    },
    "musicroll": {
        "display": "Musicroll-50M",
        "shots": [1, 2, 4, 8, 16, 32],
    },
}


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def local_result_path(remote_path, results_root):
    marker = "/results_128/"
    if marker in remote_path:
        return results_root / remote_path.split(marker, 1)[1]
    path = Path(remote_path)
    if path.is_absolute():
        raise ValueError(f"Cannot map packaged result path: {remote_path}")
    return results_root / path


def task_key(task):
    return " -> ".join(task["program"])


def load_successes(path, expected_shot):
    data = json.loads(path.read_text())
    config = data["config"]
    if int(config["in_context_examples"]) != expected_shot:
        raise ValueError(f"{path}: wrong shot count")
    if int(config["trials_per_program"]) != 128:
        raise ValueError(f"{path}: expected 128 trials per task")
    if len(data["tasks"]) != 100:
        raise ValueError(f"{path}: expected 100 tasks")

    keys = []
    successes = []
    for task in data["tasks"]:
        if len(task["trials"]) != 128:
            raise ValueError(f"{path}: task does not contain 128 trials")
        success = sum(bool(trial["correct"]) for trial in task["trials"])
        if success != int(task["correct"]):
            raise ValueError(f"{path}: stored success count disagrees")
        keys.append(task_key(task))
        successes.append(success)
    return data, keys, np.asarray(successes, dtype=int)


def exact_task_swap_p(differences):
    """Return exact one- and two-sided paired task-swap p-values."""

    differences = np.asarray(differences, dtype=int)
    observed = int(differences.sum())
    weights = [abs(int(value)) for value in differences if value != 0]
    counts = {0: 1}
    for weight in weights:
        updated = defaultdict(int)
        for total, count in counts.items():
            updated[total + weight] += count
            updated[total - weight] += count
        counts = dict(updated)
    denominator = 1 << len(weights)
    one_sided = (
        sum(count for total, count in counts.items() if total >= observed)
        / denominator
    )
    two_sided = (
        sum(
            count
            for total, count in counts.items()
            if abs(total) >= abs(observed)
        )
        / denominator
    )
    return one_sided, min(1.0, two_sided), len(weights)


def holm_adjust(p_values):
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def condition_lookup(cell_summary):
    return {
        (row["run_key"], row["condition"], int(row["shot"])): row
        for row in cell_summary
    }


def build_rows(cell_summary_path, results_root):
    cells = condition_lookup(read_csv(cell_summary_path))
    rows = []

    for run_key, spec in RUNS.items():
        run_rows = []
        for shot in spec["shots"]:
            clean_cell = cells[run_key, "clean", shot]
            clean_path = local_result_path(clean_cell["path"], results_root)
            clean_data, clean_keys, clean_successes = load_successes(
                clean_path, shot
            )

            deranged_cell = cells.get((run_key, "deranged", shot))
            if deranged_cell is None:
                if run_key != "musicroll" or shot != 1:
                    raise ValueError(
                        f"Missing deranged cell for {run_key} at {shot} shots"
                    )
                # With one demonstration there is no nontrivial label
                # permutation, so the deranged condition is exactly clean.
                deranged_path = clean_path
                deranged_data = clean_data
                deranged_keys = clean_keys
                deranged_successes = clean_successes.copy()
                deranged_source = "clean identity at one shot"
                test_performed = False
            else:
                deranged_path = local_result_path(
                    deranged_cell["path"], results_root
                )
                (
                    deranged_data,
                    deranged_keys,
                    deranged_successes,
                ) = load_successes(deranged_path, shot)
                deranged_source = "evaluated deranged cell"
                test_performed = True

            if clean_keys != deranged_keys:
                raise ValueError(
                    f"{run_key} at {shot} shots: task order differs"
                )
            if test_performed:
                for field in (
                    "model",
                    "in_context_examples",
                    "trials_per_program",
                    "bit_length",
                    "seed",
                ):
                    if (
                        clean_data["config"][field]
                        != deranged_data["config"][field]
                    ):
                        raise ValueError(
                            f"{run_key} at {shot} shots: {field} differs"
                        )
                if clean_data["config"]["ablate_labels"] is not False:
                    raise ValueError(f"{clean_path}: expected clean condition")
                if deranged_data["config"]["ablate_labels"] is not True:
                    raise ValueError(
                        f"{deranged_path}: expected deranged condition"
                    )

            clean_task_accuracy = clean_successes / 128.0
            deranged_task_accuracy = deranged_successes / 128.0
            differences = clean_successes - deranged_successes
            task_effects = differences / 128.0
            one_sided_p, two_sided_p, nonzero_tasks = exact_task_swap_p(
                differences
            )

            run_rows.append(
                {
                    "run_key": run_key,
                    "model": spec["display"],
                    "shot": shot,
                    "tasks": 100,
                    "trials_per_task": 128,
                    "clean_accuracy": float(clean_task_accuracy.mean()),
                    "clean_task_se": float(
                        clean_task_accuracy.std(ddof=1) / math.sqrt(100)
                    ),
                    "deranged_accuracy": float(
                        deranged_task_accuracy.mean()
                    ),
                    "deranged_task_se": float(
                        deranged_task_accuracy.std(ddof=1) / math.sqrt(100)
                    ),
                    "clean_minus_deranged": float(task_effects.mean()),
                    "paired_task_se": float(
                        task_effects.std(ddof=1) / math.sqrt(100)
                    ),
                    "one_sided_exact_task_swap_p": one_sided_p,
                    "two_sided_exact_task_swap_p": two_sided_p,
                    "exact_task_swap_nonzero_tasks": nonzero_tasks,
                    "test_performed": test_performed,
                    "deranged_source": deranged_source,
                    "clean_path": str(clean_path.relative_to(results_root)),
                    "deranged_path": str(
                        deranged_path.relative_to(results_root)
                    ),
                }
            )

        tested = [row for row in run_rows if row["test_performed"]]
        adjusted = holm_adjust(
            [row["one_sided_exact_task_swap_p"] for row in tested]
        )
        for row, adjusted_p in zip(tested, adjusted):
            row["holm_one_sided_p_within_model_shots"] = adjusted_p
        for row in run_rows:
            if "holm_one_sided_p_within_model_shots" not in row:
                row["holm_one_sided_p_within_model_shots"] = 1.0
        rows.extend(run_rows)

    return rows


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_panel(rows, run_key, output_path):
    selected = [row for row in rows if row["run_key"] == run_key]
    shots = np.asarray([row["shot"] for row in selected], dtype=float)
    clean = np.asarray([row["clean_accuracy"] for row in selected])
    clean_se = np.asarray([row["clean_task_se"] for row in selected])
    deranged = np.asarray([row["deranged_accuracy"] for row in selected])
    deranged_se = np.asarray(
        [row["deranged_task_se"] for row in selected]
    )

    fig, ax = plt.subplots(figsize=(5.0, 3.09))
    ax.errorbar(
        shots,
        clean,
        yerr=clean_se,
        color="#2563eb",
        marker="o",
        linewidth=2.2,
        markersize=6,
        capsize=3,
        label="clean",
    )
    ax.errorbar(
        shots,
        deranged,
        yerr=deranged_se,
        color="#dc2626",
        marker="o",
        linewidth=2.2,
        markersize=6,
        capsize=3,
        label="deranged",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(shots)
    ax.set_xticklabels([str(int(shot)) for shot in shots])
    ax.set_xlabel("in-context examples")
    ax.set_ylabel("Exact Accuracy")
    ax.set_ylim(0.0, 0.25)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", axis="x", alpha=0.08)
    ax.legend(frameon=True, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cell-summary",
        type=Path,
        default=DEFAULT_EXPANDED / "analysis_128_full" / "cell_summary.csv",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_EXPANDED / "results_128",
    )
    parser.add_argument(
        "--chess-output",
        type=Path,
        default=HERE / "chess_clean_vs_deranged.pdf",
    )
    parser.add_argument(
        "--music-output",
        type=Path,
        default=HERE / "musicroll_clean_vs_deranged.pdf",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=HERE / "negative_controls_128.csv",
    )
    args = parser.parse_args()

    rows = build_rows(args.cell_summary, args.results_root)
    for path in (args.chess_output, args.music_output, args.csv_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.csv_output, rows)
    plot_panel(rows, "chessgpt", args.chess_output)
    plot_panel(rows, "musicroll", args.music_output)

    print(args.chess_output)
    print(args.music_output)
    print(args.csv_output)
    for run_key in RUNS:
        last = [row for row in rows if row["run_key"] == run_key][-1]
        print(
            f"{run_key}: n={last['shot']}, "
            f"gap={last['clean_minus_deranged']:.8f}, "
            f"p={last['one_sided_exact_task_swap_p']:.10g}, "
            "Holm="
            f"{last['holm_one_sided_p_within_model_shots']:.10g}"
        )


if __name__ == "__main__":
    main()
