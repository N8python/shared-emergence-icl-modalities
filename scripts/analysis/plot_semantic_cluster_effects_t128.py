#!/usr/bin/env python3
"""Plot cluster-level clean-minus-deranged effects for representative models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPECS = (
    {
        "model": "ImageGPT-large",
        "clean": (
            "imagegpt/clean/"
            "imagegpt-large-bf16_31shot_leftpad_clean.json"
        ),
        "deranged": (
            "imagegpt/deranged/"
            "imagegpt-large-bf16_31shot_leftpad_deranged.json"
        ),
        "shot": 31,
        "color": "#2B6FB0",
    },
    {
        "model": "NextTerm-440M",
        "clean": "nextterm/clean/NextTerm-440M_ic64.json",
        "deranged": "nextterm/deranged/NextTerm-440M_ic64.json",
        "shot": 64,
        "color": "#E8622A",
    },
    {
        "model": "Evo2-40B",
        "clean": "evo2/clean/evo2_40b_ic64.json",
        "deranged": "evo2/deranged/evo2_40b_deranged_ic64.json",
        "shot": 64,
        "color": "#2E8B57",
    },
    {
        "model": "Qwen3-14B",
        "clean": "qwen3/clean/Qwen3-14B-Base_ic64.json",
        "deranged": "qwen3/deranged/Qwen3-14B-Base_deranged_ic64.json",
        "shot": 64,
        "color": "#2B5FD6",
    },
    {
        "model": "TimesFM-2.5",
        "clean": (
            "timesfm/clean/"
            "timesfm2_5_fp32_sine_lobe_2bit_iozero_ic48.json"
        ),
        "deranged": (
            "timesfm/deranged/"
            "timesfm2_5_fp32_sine_lobe_2bit_iozero_deranged_ic48.json"
        ),
        "shot": 48,
        "color": "#159A9A",
    },
    {
        "model": "ProGen2-base",
        "clean": "progen2/clean/ProGen2-base-bf16_ic48.json",
        "deranged": "progen2/deranged/ProGen2-base-bf16_ic48.json",
        "shot": 48,
        "color": "#7B3FBF",
    },
)

SIMPLE_CLUSTER_NAMES = {
    1: "Global Broadcast",
    2: "Half Segments",
    3: "Parity Masking",
    4: "Half Swaps",
    5: "Bit Inversion",
    6: "Bit Reversal",
    7: "Local Shifts",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument("--log-floor", type=float, default=0.01)
    return parser.parse_args()


def task_key(task: dict) -> tuple[str, ...]:
    return tuple(task["program"])


def load_clusters(path: Path) -> list[tuple[str, tuple[tuple[str, ...], ...]]]:
    data = json.loads(path.read_text())
    clusters = []
    all_programs: list[tuple[str, ...]] = []
    for cluster in data["clusters"]:
        if "programs" in cluster:
            programs = tuple(tuple(program) for program in cluster["programs"])
        else:
            programs = tuple(
                tuple(function["program"]) for function in cluster["functions"]
            )
        cluster_id = int(cluster["cluster_id"])
        simple_name = cluster.get(
            "simple_name",
            SIMPLE_CLUSTER_NAMES.get(cluster_id),
        )
        if simple_name is None:
            raise ValueError(f"{path}: no display name for cluster {cluster_id}")
        clusters.append((simple_name, programs))
        all_programs.extend(programs)
    if len(clusters) != 7:
        raise ValueError(f"{path}: expected 7 clusters")
    if len(all_programs) != 100 or len(set(all_programs)) != 100:
        raise ValueError(f"{path}: expected 100 uniquely assigned tasks")
    return clusters


def load_accuracy(path: Path, *, expected_shot: int) -> dict[tuple[str, ...], float]:
    data = json.loads(path.read_text())
    config = data["config"]
    if int(config["in_context_examples"]) != expected_shot:
        raise ValueError(f"{path}: expected {expected_shot} shots")
    if int(config["trials_per_program"]) != 128:
        raise ValueError(f"{path}: expected 128 trials per task")

    tasks = data["tasks"]
    if len(tasks) != 100:
        raise ValueError(f"{path}: expected 100 tasks")
    result = {}
    for task in tasks:
        key = task_key(task)
        if key in result:
            raise ValueError(f"{path}: duplicate task {key}")
        trials = task["trials"]
        if len(trials) != 128:
            raise ValueError(f"{path}: incomplete trials for {key}")
        correct = sum(bool(trial["correct"]) for trial in trials)
        if correct != int(task["correct"]):
            raise ValueError(f"{path}: stored correct count disagrees for {key}")
        result[key] = correct / 128.0
    return result


def compute_table(
    data_root: Path,
    clusters: list[tuple[str, tuple[tuple[str, ...], ...]]],
) -> list[dict[str, object]]:
    expected_programs = {
        program for _, programs in clusters for program in programs
    }
    rows = []
    for spec in SPECS:
        clean = load_accuracy(
            data_root / spec["clean"], expected_shot=spec["shot"]
        )
        deranged = load_accuracy(
            data_root / spec["deranged"], expected_shot=spec["shot"]
        )
        if clean.keys() != deranged.keys():
            raise ValueError(
                f"{spec['model']}: clean and deranged task sets differ"
            )
        if set(clean) != expected_programs:
            missing = expected_programs - set(clean)
            extra = set(clean) - expected_programs
            raise ValueError(
                f"{spec['model']}: cluster mismatch; "
                f"missing={len(missing)}, extra={len(extra)}"
            )

        for category, programs in clusters:
            clean_values = np.asarray([clean[program] for program in programs])
            deranged_values = np.asarray(
                [deranged[program] for program in programs]
            )
            rows.append(
                {
                    "model": spec["model"],
                    "category": category,
                    "mean_clean_accuracy": float(clean_values.mean()),
                    "mean_deranged_accuracy": float(deranged_values.mean()),
                    "clean_minus_deranged": float(
                        (clean_values - deranged_values).mean()
                    ),
                    "task_count": len(programs),
                    "in_context_examples": spec["shot"],
                    "trials_per_task_per_condition": 128,
                }
            )
        print(
            f"computed {spec['model']}: {spec['shot']} shots, "
            f"100 tasks x 128 trials x 2 conditions",
            flush=True,
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def plot(
    rows: list[dict[str, object]],
    categories: list[str],
    pdf_output: Path,
    png_output: Path,
    *,
    log_floor: float,
) -> None:
    if not 0.0 < log_floor < 1.0:
        raise ValueError("log_floor must lie between 0 and 1")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 11,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    count = len(categories)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
    closed_angles = np.concatenate([angles, angles[:1]])
    by_model = {
        spec["model"]: {
            row["category"]: float(row["clean_minus_deranged"])
            for row in rows
            if row["model"] == spec["model"]
        }
        for spec in SPECS
    }

    figure, axis = plt.subplots(
        figsize=(7.5, 7.0), subplot_kw={"projection": "polar"}
    )
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)

    for spec in SPECS:
        effects = np.asarray(
            [by_model[spec["model"]][category] for category in categories]
        )
        displayed = np.maximum(effects, log_floor)
        closed = np.concatenate([displayed, displayed[:1]])
        axis.plot(
            closed_angles,
            closed,
            color=spec["color"],
            linewidth=3,
            label=spec["model"],
            zorder=2,
        )
        axis.fill(
            closed_angles,
            closed,
            color=spec["color"],
            alpha=0.12,
            zorder=1,
        )

    axis.set_rscale("log")
    axis.set_ylim(log_floor, 1.0)
    radial_ticks = [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    radial_ticks = [tick for tick in radial_ticks if tick >= log_floor]
    if not np.isclose(radial_ticks[0], log_floor):
        radial_ticks.insert(0, log_floor)
    radial_labels = [rf"$\leq {log_floor:g}$"] + [
        f"{tick:g}" for tick in radial_ticks[1:]
    ]
    axis.set_xticks(angles)
    axis.set_yticks(radial_ticks)
    axis.set_xticklabels([])
    axis.set_yticklabels([])

    category_transform = axis.get_xaxis_transform()
    for angle, name in zip(angles, categories, strict=True):
        screen_angle = np.pi / 2 - angle
        horizontal = np.cos(screen_angle)
        vertical = np.sin(screen_angle)
        horizontal_alignment = (
            "center"
            if abs(horizontal) < 0.30
            else ("left" if horizontal > 0 else "right")
        )
        vertical_alignment = (
            "center"
            if abs(vertical) < 0.30
            else ("bottom" if vertical > 0 else "top")
        )
        axis.text(
            angle,
            1.06,
            name,
            transform=category_transform,
            fontsize=20,
            ha=horizontal_alignment,
            va=vertical_alignment,
            zorder=20,
            clip_on=False,
        )

    label_angle = (angles[4] + angles[5]) / 2
    for tick, label in zip(radial_ticks, radial_labels, strict=True):
        axis.text(
            label_angle,
            tick,
            label,
            fontsize=13,
            ha="center",
            va="center",
            zorder=20,
            clip_on=False,
            bbox={
                "boxstyle": "round,pad=0.05",
                "fc": "white",
                "ec": "none",
                "alpha": 1.0,
            },
        )

    axis.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.65)
    axis.xaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    legend = axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        fontsize=16,
        frameon=True,
    )
    legend.set_zorder(20)
    figure.tight_layout()

    for output in (pdf_output, png_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(pdf_output, bbox_inches="tight")
    figure.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    clusters = load_clusters(args.clusters)
    rows = compute_table(args.data_root, clusters)
    write_csv(args.csv_output, rows)
    plot(
        rows,
        [name for name, _ in clusters],
        args.pdf_output,
        args.png_output,
        log_floor=args.log_floor,
    )
    print(f"wrote {args.pdf_output}", flush=True)
    print(f"wrote {args.png_output}", flush=True)
    print(f"wrote {args.csv_output}", flush=True)


if __name__ == "__main__":
    main()
