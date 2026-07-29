#!/usr/bin/env python3
"""Validate and summarize every cell in the complete 128-trial paper grid."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper_runs_t128.json"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "t128" / "results_128"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "t128" / "analysis"

FAMILY_ORDER = [
    "qwen3",
    "evo2",
    "nextterm",
    "imagegpt",
    "timesfm",
    "progen2",
    "chessgpt",
    "musicroll",
]


def portable_path(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def family_for_key(key: str) -> str:
    if key.startswith("qwen3"):
        return "qwen3"
    if key.startswith("evo2"):
        return "evo2"
    if key.startswith("nextterm"):
        return "nextterm"
    if key.startswith("imagegpt"):
        return "imagegpt"
    if key.startswith("timesfm"):
        return "timesfm"
    if key.startswith("progen2"):
        return "progen2"
    if key == "chessgpt":
        return "chessgpt"
    if key == "musicroll":
        return "musicroll"
    raise ValueError(f"Unrecognized paper-family run key: {key}")


def expected_cells(config: dict, root: Path) -> list[dict]:
    rows = []
    canonical = set(config["experiment"]["canonical_run_keys"])
    for run_key, run in config["runs"].items():
        for condition, condition_config in run["conditions"].items():
            for shot in condition_config["shots"]:
                relative = condition_config["output_template"].format(
                    run=run_key,
                    condition=condition,
                    shot=shot,
                )
                rows.append(
                    {
                        "run_key": run_key,
                        "label": run["label"],
                        "family": family_for_key(run_key),
                        "canonical": run_key in canonical,
                        "default_model": run["default_model"],
                        "condition": condition,
                        "shot": int(shot),
                        "path": root / relative,
                    }
                )
    return rows


def task_key(task: dict) -> str:
    return " -> ".join(task["program"])


def load_and_validate(cell: dict) -> dict:
    path = cell["path"]
    data = json.loads(path.read_text())
    config = data["config"]
    if int(config["trials_per_program"]) != 128:
        raise ValueError(f"{path}: trials_per_program != 128")
    if int(config["in_context_examples"]) != cell["shot"]:
        raise ValueError(
            f"{path}: shot {config['in_context_examples']} != {cell['shot']}"
        )
    expected_ablation = cell["condition"] == "deranged"
    if bool(config["ablate_labels"]) != expected_ablation:
        raise ValueError(f"{path}: wrong ablate_labels value")

    tasks = data["tasks"]
    keys = [task_key(task) for task in tasks]
    indices = [int(task["index"]) for task in tasks]
    if len(tasks) != 100 or len(set(keys)) != 100:
        raise ValueError(f"{path}: expected 100 unique tasks")
    if indices != list(range(100)):
        raise ValueError(f"{path}: task indices are not exactly 0..99 in order")

    successes = []
    edits = []
    for task in tasks:
        records = task["trials"]
        if len(records) != 128 or int(task["total_trials"]) != 128:
            raise ValueError(f"{path}: {task_key(task)} does not have 128 trials")
        success = sum(bool(record["correct"]) for record in records)
        task_edits = [float(record["edit_distance"]) for record in records]
        successes.append(success)
        edits.append(float(np.mean(task_edits)))
        if int(task["correct"]) != success:
            raise ValueError(f"{path}: stored correct count disagrees with records")
        if not math.isclose(float(task["accuracy"]), success / 128, abs_tol=1e-12):
            raise ValueError(f"{path}: stored task accuracy disagrees with records")
        if not math.isclose(
            float(task["average_edit_distance"]),
            float(np.mean(task_edits)),
            abs_tol=1e-12,
        ):
            raise ValueError(f"{path}: stored task edit distance disagrees with records")

    accuracy = np.asarray(successes, dtype=float) / 128
    edit = np.asarray(edits, dtype=float)
    mean_accuracy = float(accuracy.mean())
    mean_edit = float(edit.mean())
    accuracy_se = float(accuracy.std(ddof=1) / math.sqrt(len(accuracy)))
    edit_se = float(edit.std(ddof=1) / math.sqrt(len(edit)))
    overall = data["overall"]
    comparisons = {
        "mean_accuracy": mean_accuracy,
        "stderr": accuracy_se,
        "mean_edit_distance": mean_edit,
        "edit_distance_stderr": edit_se,
    }
    for name, calculated in comparisons.items():
        if not math.isclose(float(overall[name]), calculated, abs_tol=1e-12):
            raise ValueError(f"{path}: overall {name} disagrees with records")

    return {
        **cell,
        "config": config,
        "keys": keys,
        "successes": np.asarray(successes, dtype=int),
        "task_accuracy": accuracy,
        "task_edit": edit,
        "mean_accuracy": mean_accuracy,
        "accuracy_task_cluster_se": accuracy_se,
        "mean_edit_distance": mean_edit,
        "edit_distance_task_cluster_se": edit_se,
    }


def bootstrap_interval(
    values: np.ndarray, *, draws: int, rng: np.random.Generator
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def make_plots(loaded: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lookup = {
        (row["run_key"], row["condition"], row["shot"]): row for row in loaded
    }
    by_family: dict[str, list[str]] = defaultdict(list)
    labels = {}
    for row in loaded:
        labels[row["run_key"]] = row["label"]
        if row["run_key"] not in by_family[row["family"]]:
            by_family[row["family"]].append(row["run_key"])

    for family in FAMILY_ORDER:
        run_keys = by_family.get(family, [])
        if not run_keys:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
        family_shots = set()
        for run_key in run_keys:
            clean_shots = {
                shot
                for key, condition, shot in lookup
                if key == run_key and condition == "clean"
            }
            deranged_shots = {
                shot
                for key, condition, shot in lookup
                if key == run_key and condition == "deranged"
            }
            clean_shots = sorted(clean_shots)
            deranged_shots = sorted(deranged_shots)
            paired_shots = sorted(set(clean_shots) & set(deranged_shots))
            if not clean_shots and not deranged_shots:
                continue
            family_shots.update(clean_shots)
            family_shots.update(deranged_shots)
            clean = np.array(
                [
                    lookup[(run_key, "clean", shot)]["mean_accuracy"]
                    for shot in clean_shots
                ]
            )
            deranged = np.array(
                [
                    lookup[(run_key, "deranged", shot)]["mean_accuracy"]
                    for shot in deranged_shots
                ]
            )
            clean_se = np.array(
                [
                    lookup[(run_key, "clean", shot)]["accuracy_task_cluster_se"]
                    for shot in clean_shots
                ]
            )
            deranged_se = np.array(
                [
                    lookup[(run_key, "deranged", shot)][
                        "accuracy_task_cluster_se"
                    ]
                    for shot in deranged_shots
                ]
            )
            line = axes[0].errorbar(
                clean_shots,
                clean,
                yerr=1.96 * clean_se,
                marker="o",
                capsize=2,
                label=f"{labels[run_key]} clean",
            )
            color = line[0].get_color()
            axes[0].errorbar(
                deranged_shots,
                deranged,
                yerr=1.96 * deranged_se,
                marker="s",
                linestyle="--",
                color=color,
                alpha=0.75,
                capsize=2,
                label=f"{labels[run_key]} deranged",
            )
            if paired_shots:
                axes[1].plot(
                    paired_shots,
                    [
                        lookup[(run_key, "clean", shot)]["mean_accuracy"]
                        - lookup[(run_key, "deranged", shot)]["mean_accuracy"]
                        for shot in paired_shots
                    ],
                    marker="o",
                    color=color,
                    label=labels[run_key],
                )

        for axis in axes:
            axis.set_xscale("log", base=2)
            ordered_family_shots = sorted(family_shots)
            axis.set_xticks(ordered_family_shots)
            axis.set_xticklabels([str(shot) for shot in ordered_family_shots])
            axis.set_xlabel("In-context examples")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Exact-match accuracy")
        axes[0].set_title("Clean and deranged (95% task-cluster normal CI)")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("Clean minus deranged accuracy")
        axes[1].set_title("Paired-mapping effect")
        axes[0].legend(fontsize=7, ncol=2)
        axes[1].legend(fontsize=8)
        fig.suptitle(f"{family}: 128 trials per task")
        fig.savefig(output_dir / f"{family}_full_grid_128.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.config = args.config.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    config = json.loads(args.config.read_text())
    expected = expected_cells(config, args.results_root)
    missing = [
        cell["path"].relative_to(args.results_root).as_posix()
        for cell in expected
        if not cell["path"].is_file()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "missing_files.json").write_text(
        json.dumps(missing, indent=2) + "\n"
    )
    print(
        f"expected={len(expected)} present={len(expected) - len(missing)} "
        f"missing={len(missing)}",
        flush=True,
    )
    if missing and not args.allow_incomplete:
        raise SystemExit(2)

    loaded = []
    for cell in expected:
        if not cell["path"].is_file():
            continue
        loaded.append(load_and_validate(cell))
        if len(loaded) % 10 == 0 or len(loaded) == len(expected) - len(missing):
            print(f"validated {len(loaded)}/{len(expected) - len(missing)}", flush=True)

    canonical_keys = None
    for row in loaded:
        if canonical_keys is None:
            canonical_keys = row["keys"]
        elif row["keys"] != canonical_keys:
            raise ValueError(f"{row['path']}: task order differs from other cells")

    cell_rows = []
    for row in loaded:
        cell_rows.append(
            {
                "run_key": row["run_key"],
                "label": row["label"],
                "family": row["family"],
                "canonical": row["canonical"],
                "condition": row["condition"],
                "shot": row["shot"],
                "model": row["default_model"],
                "seed": row["config"]["seed"],
                "tasks": 100,
                "trials_per_task": 128,
                "total_predictions": 12_800,
                "mean_accuracy": row["mean_accuracy"],
                "accuracy_task_cluster_se": row["accuracy_task_cluster_se"],
                "mean_edit_distance": row["mean_edit_distance"],
                "edit_distance_task_cluster_se": (
                    row["edit_distance_task_cluster_se"]
                ),
                "path": row["path"].relative_to(args.results_root).as_posix(),
            }
        )
    cell_fields = [
        "run_key",
        "label",
        "family",
        "canonical",
        "condition",
        "shot",
        "model",
        "seed",
        "tasks",
        "trials_per_task",
        "total_predictions",
        "mean_accuracy",
        "accuracy_task_cluster_se",
        "mean_edit_distance",
        "edit_distance_task_cluster_se",
        "path",
    ]
    write_csv(args.output_dir / "cell_summary.csv", cell_rows, cell_fields)

    rng = np.random.default_rng(args.seed)
    lookup = {
        (row["run_key"], row["condition"], row["shot"]): row for row in loaded
    }
    gap_rows = []
    for run_key, run in config["runs"].items():
        for shot in run["conditions"]["clean"]["shots"]:
            clean = lookup.get((run_key, "clean", int(shot)))
            deranged = lookup.get((run_key, "deranged", int(shot)))
            if clean is None or deranged is None:
                continue
            difference = clean["task_accuracy"] - deranged["task_accuracy"]
            low, high = bootstrap_interval(
                difference, draws=args.bootstrap_draws, rng=rng
            )
            gap_rows.append(
                {
                    "run_key": run_key,
                    "label": run["label"],
                    "family": family_for_key(run_key),
                    "canonical": clean["canonical"],
                    "shot": int(shot),
                    "clean_accuracy": clean["mean_accuracy"],
                    "deranged_accuracy": deranged["mean_accuracy"],
                    "clean_minus_deranged": float(difference.mean()),
                    "paired_task_se": float(
                        difference.std(ddof=1) / math.sqrt(len(difference))
                    ),
                    "task_bootstrap_ci95_low": low,
                    "task_bootstrap_ci95_high": high,
                }
            )
    gap_fields = [
        "run_key",
        "label",
        "family",
        "canonical",
        "shot",
        "clean_accuracy",
        "deranged_accuracy",
        "clean_minus_deranged",
        "paired_task_se",
        "task_bootstrap_ci95_low",
        "task_bootstrap_ci95_high",
    ]
    write_csv(args.output_dir / "paired_gaps.csv", gap_rows, gap_fields)

    best_shot_rows = []
    for run_key, run in config["runs"].items():
        clean_candidates = [
            lookup[(run_key, "clean", int(shot))]
            for shot in run["conditions"]["clean"]["shots"]
            if (run_key, "clean", int(shot)) in lookup
        ]
        deranged_candidates = [
            lookup[(run_key, "deranged", int(shot))]
            for shot in run["conditions"]["deranged"]["shots"]
            if (run_key, "deranged", int(shot)) in lookup
        ]
        if not clean_candidates or not deranged_candidates:
            continue
        clean_best = max(clean_candidates, key=lambda row: row["mean_accuracy"])
        deranged_best = max(
            deranged_candidates, key=lambda row: row["mean_accuracy"]
        )
        gap = clean_best["mean_accuracy"] - deranged_best["mean_accuracy"]
        denominator = math.sqrt(
            clean_best["accuracy_task_cluster_se"] ** 2
            + deranged_best["accuracy_task_cluster_se"] ** 2
        )
        best_shot_rows.append(
            {
                "run_key": run_key,
                "label": run["label"],
                "family": family_for_key(run_key),
                "canonical": clean_best["canonical"],
                "best_clean_shot": clean_best["shot"],
                "best_clean_accuracy": clean_best["mean_accuracy"],
                "best_clean_task_cluster_se": clean_best[
                    "accuracy_task_cluster_se"
                ],
                "best_deranged_shot": deranged_best["shot"],
                "best_deranged_accuracy": deranged_best["mean_accuracy"],
                "best_deranged_task_cluster_se": deranged_best[
                    "accuracy_task_cluster_se"
                ],
                "best_accuracy_gap": gap,
                "z_gap_unpaired": gap / denominator if denominator else math.nan,
            }
        )
    best_shot_fields = [
        "run_key",
        "label",
        "family",
        "canonical",
        "best_clean_shot",
        "best_clean_accuracy",
        "best_clean_task_cluster_se",
        "best_deranged_shot",
        "best_deranged_accuracy",
        "best_deranged_task_cluster_se",
        "best_accuracy_gap",
        "z_gap_unpaired",
    ]
    write_csv(
        args.output_dir / "best_shot_zgaps.csv",
        best_shot_rows,
        best_shot_fields,
    )

    summary = {
        "config": portable_path(args.config),
        "results_root": portable_path(args.results_root),
        "expected_cells": len(expected),
        "validated_cells": len(loaded),
        "missing_cells": len(missing),
        "predictions_per_complete_cell": 12_800,
        "validated_predictions": len(loaded) * 12_800,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
    }
    (args.output_dir / "validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    if not missing:
        make_plots(loaded, args.output_dir)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
