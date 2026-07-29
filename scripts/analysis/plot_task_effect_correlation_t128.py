#!/usr/bin/env python3
"""Plot cross-modality correlations of per-task clean-minus-deranged effects."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


SPECS = {
    "Language": {
        "label": "Language\n(Qwen3-14B)",
        "clean": "qwen3/clean/Qwen3-14B-Base_ic64.json",
        "deranged": "qwen3/deranged/Qwen3-14B-Base_deranged_ic64.json",
        "shot": 64,
    },
    "Genome": {
        "label": "Genome\n(Evo2-40B)",
        "clean": "evo2/clean/evo2_40b_ic64.json",
        "deranged": "evo2/deranged/evo2_40b_deranged_ic64.json",
        "shot": 64,
    },
    "Integer seq.": {
        "label": "Integer seq.\n(NextTerm-440M)",
        "clean": "nextterm/clean/NextTerm-440M_ic64.json",
        "deranged": "nextterm/deranged/NextTerm-440M_ic64.json",
        "shot": 64,
    },
    "Image": {
        "label": "Image\n(ImageGPT-large)",
        "clean": (
            "imagegpt/clean/"
            "imagegpt-large-bf16_31shot_leftpad_clean.json"
        ),
        "deranged": (
            "imagegpt/deranged/"
            "imagegpt-large-bf16_31shot_leftpad_deranged.json"
        ),
        "shot": 31,
    },
    "Time-series": {
        "label": "Time-series\n(TimesFM-2.5)",
        "clean": (
            "timesfm/clean/"
            "timesfm2_5_fp32_sine_lobe_2bit_iozero_ic48.json"
        ),
        "deranged": (
            "timesfm/deranged/"
            "timesfm2_5_fp32_sine_lobe_2bit_iozero_deranged_ic48.json"
        ),
        "shot": 48,
    },
    "Protein": {
        "label": "Protein\n(ProGen2-base)",
        "clean": "progen2/clean/ProGen2-base-bf16_ic48.json",
        "deranged": "progen2/deranged/ProGen2-base-bf16_ic48.json",
        "shot": 48,
    },
}

# Put the shared non-image block in the upper left and the image outlier last.
DISPLAY_ORDER = (
    "Genome",
    "Protein",
    "Language",
    "Integer seq.",
    "Time-series",
    "Image",
)

PVALUE_NAMES = {
    "Language": "Language",
    "Genome": "Genome",
    "Integer": "Integer seq.",
    "Image": "Image",
    "Time": "Time-series",
    "Protein": "Protein",
}

SIGNIFICANCE_LEVEL = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--png-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument(
        "--pvalues-csv",
        type=Path,
        required=True,
        help="Pairwise one-sided permutation-test results.",
    )
    return parser.parse_args()


def task_key(task: dict) -> str:
    return " -> ".join(task["program"])


def load_accuracy(
    path: Path, *, expected_shot: int
) -> tuple[list[str], np.ndarray]:
    data = json.loads(path.read_text())
    config = data["config"]
    if int(config["in_context_examples"]) != expected_shot:
        raise ValueError(f"{path}: expected {expected_shot} shots")
    if int(config["trials_per_program"]) != 128:
        raise ValueError(f"{path}: expected 128 trials per task")

    tasks = data["tasks"]
    keys = [task_key(task) for task in tasks]
    if len(tasks) != 100 or len(set(keys)) != 100:
        raise ValueError(f"{path}: expected 100 unique tasks")

    successes = []
    for task in tasks:
        trials = task["trials"]
        if len(trials) != 128:
            raise ValueError(f"{path}: incomplete trials for {task_key(task)}")
        correct = sum(bool(trial["correct"]) for trial in trials)
        if correct != int(task["correct"]):
            raise ValueError(f"{path}: stored correct count disagrees")
        successes.append(correct)
    return keys, np.asarray(successes, dtype=float) / 128.0


def load_profiles(data_root: Path) -> np.ndarray:
    profiles = []
    shared_keys = None
    for name in DISPLAY_ORDER:
        spec = SPECS[name]
        clean_keys, clean = load_accuracy(
            data_root / spec["clean"], expected_shot=spec["shot"]
        )
        deranged_keys, deranged = load_accuracy(
            data_root / spec["deranged"], expected_shot=spec["shot"]
        )
        if clean_keys != deranged_keys:
            raise ValueError(f"{name}: clean and deranged task order differs")
        if shared_keys is None:
            shared_keys = clean_keys
        elif clean_keys != shared_keys:
            raise ValueError(f"{name}: task order differs across modalities")
        profiles.append(clean - deranged)
        print(
            f"loaded {name}: {spec['shot']} shots, "
            f"mean effect={(clean - deranged).mean():.4f}",
            flush=True,
        )
    return np.asarray(profiles)


def correlation_matrix(profiles: np.ndarray) -> np.ndarray:
    count = len(profiles)
    result = np.eye(count)
    for left in range(count):
        for right in range(left + 1, count):
            rho = float(spearmanr(profiles[left], profiles[right]).statistic)
            result[left, right] = result[right, left] = rho
    return result


def write_csv(path: Path, matrix: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["modality", *DISPLAY_ORDER])
        for name, row in zip(DISPLAY_ORDER, matrix, strict=True):
            writer.writerow([name, *[f"{value:.12g}" for value in row]])


def load_pvalue_matrix(path: Path, correlations: np.ndarray) -> np.ndarray:
    index = {name: position for position, name in enumerate(DISPLAY_ORDER)}
    pvalues = np.full_like(correlations, np.nan)
    seen_pairs: set[tuple[int, int]] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            left_name = PVALUE_NAMES[row["left"]]
            right_name = PVALUE_NAMES[row["right"]]
            left = index[left_name]
            right = index[right_name]
            pair = tuple(sorted((left, right)))
            if pair in seen_pairs:
                raise ValueError(f"{path}: duplicate pair {pair}")
            seen_pairs.add(pair)

            stored_rho = float(row["observed_spearman"])
            if not np.isclose(
                stored_rho, correlations[left, right], atol=1e-12, rtol=0
            ):
                raise ValueError(
                    f"{path}: correlation mismatch for "
                    f"{left_name}/{right_name}"
                )
            value = float(row["one_sided_permutation_p"])
            pvalues[left, right] = pvalues[right, left] = value

    expected_pairs = len(DISPLAY_ORDER) * (len(DISPLAY_ORDER) - 1) // 2
    if len(seen_pairs) != expected_pairs:
        raise ValueError(
            f"{path}: expected {expected_pairs} unique pairs, "
            f"found {len(seen_pairs)}"
        )
    return pvalues


def plot(
    matrix: np.ndarray,
    pvalues: np.ndarray,
    pdf_output: Path,
    png_output: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axis = plt.subplots(figsize=(6.2, 5.05), constrained_layout=True)
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)

    labels = [SPECS[name]["label"] for name in DISPLAY_ORDER]
    positions = np.arange(len(labels))
    axis.set_xticks(positions, labels=labels, rotation=42, ha="right")
    axis.set_yticks(positions, labels=labels)
    axis.tick_params(axis="both", which="major", length=0, pad=4)
    axis.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.0)
    axis.tick_params(which="minor", bottom=False, left=False)

    for row in range(len(labels)):
        for column in range(len(labels)):
            if (
                row != column
                and pvalues[row, column] >= SIGNIFICANCE_LEVEL
            ):
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="none",
                        edgecolor=(0.10, 0.10, 0.10, 0.82),
                        linewidth=0,
                        hatch="////",
                    )
                )

    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.43 else "black",
                fontsize=9,
            )

    colorbar = fig.colorbar(image, ax=axis, shrink=0.83, pad=0.04)
    colorbar.set_label(r"Spearman $\rho$", rotation=270, labelpad=14)
    colorbar.ax.tick_params(length=3)

    for output in (pdf_output, png_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_output, bbox_inches="tight")
    fig.savefig(png_output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    profiles = load_profiles(args.data_root)
    matrix = correlation_matrix(profiles)
    pvalues = load_pvalue_matrix(args.pvalues_csv, matrix)
    write_csv(args.csv_output, matrix)
    plot(matrix, pvalues, args.pdf_output, args.png_output)
    print(f"wrote {args.pdf_output}", flush=True)
    print(f"wrote {args.png_output}", flush=True)
    print(f"wrote {args.csv_output}", flush=True)


if __name__ == "__main__":
    main()
