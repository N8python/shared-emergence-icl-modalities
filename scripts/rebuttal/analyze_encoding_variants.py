#!/usr/bin/env python3
"""Rebuttal appendix analysis: encoding resampling for language and image.

For each of the two modalities this builds the 100-task clean-minus-deranged
profile of every encoding variant, then reports

* the variant-by-variant Spearman matrix (published reference, a same-seed
  rerun, a reseeded reference that serves as the sampling-noise ceiling, and
  each new encoding);
* each variant's Spearman correlation with the published profiles of the other
  five modalities, with the paper's one-sided permutation p-value;
* each variant's mean clean accuracy, deranged accuracy and paired effect.

Run from the repository root once `results_rebuttal/` has been collected and
the published T=128 archive has been fetched into `data/t128/results_128/`.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]

_spec = importlib.util.spec_from_file_location(
    "plot_task_effect",
    REPO_ROOT / "scripts/analysis/plot_task_effect_correlation_t128.py",
)
paper = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(paper)

MODALITY_TO_PAPER_NAME = {"language": "Language", "image": "Image"}

VARIANTS = {
    "language": [
        ("reference_sameseed", "Reference, paper seed (rerun)"),
        ("reference_reseed", "Reference, new seed"),
        ("no_sep", "No separator"),
        ("two_sep", "Two separators"),
        ("nondigit", "Non-digit tokens"),
        ("nondigit_two_sep", "Non-digit tokens, two separators"),
    ],
    "image": [
        ("reference_sameseed", "Reference, paper seed (rerun)"),
        ("reference_reseed", "Reference, new seed"),
        ("right_sep", "Right separator"),
        ("middle_sep", "Middle separator"),
        ("left_two_sep", "Left, two separators"),
        ("right_two_sep", "Right, two separators"),
    ],
}


def load_gap(clean_path: Path, deranged_path: Path, shot: int, order):
    clean_keys, clean = paper.load_accuracy(clean_path, expected_shot=shot)
    deranged_keys, deranged = paper.load_accuracy(deranged_path, expected_shot=shot)
    if clean_keys != deranged_keys:
        raise ValueError(f"{clean_path}: clean and deranged task order differs")
    if order is not None and clean_keys != order:
        raise ValueError(f"{clean_path}: task order differs from the published runs")
    return clean_keys, clean, deranged


def permutation_p(left: np.ndarray, right: np.ndarray, *, draws: int, rng) -> float:
    """One-sided (positive) permutation p-value for Spearman, add-one estimate."""
    left_rank = rankdata(left)
    right_rank = rankdata(right)
    left_rank = (left_rank - left_rank.mean()) / left_rank.std()
    right_rank = (right_rank - right_rank.mean()) / right_rank.std()
    observed = float(np.mean(left_rank * right_rank))
    count = 0
    done = 0
    while done < draws:
        n = min(20000, draws - done)
        perms = np.argsort(rng.random((n, left_rank.size)), axis=1)
        stats = (right_rank[perms] * left_rank).mean(axis=1)
        count += int(np.sum(stats >= observed - 1e-12))
        done += n
    return (count + 1) / (draws + 1)


def bootstrap_ci(diff: np.ndarray, rng, draws: int = 10000):
    idx = rng.integers(0, diff.size, size=(draws, diff.size))
    means = diff[idx].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results_rebuttal")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/t128/results_128")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results_rebuttal/analysis")
    parser.add_argument("--permutations", type=int, default=1_000_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261111)

    # Published profiles of all six modalities, in the paper's task order.
    published = {}
    order = None
    for name, spec in paper.SPECS.items():
        keys, clean, deranged = load_gap(
            args.data_root / spec["clean"], args.data_root / spec["deranged"], spec["shot"], order
        )
        order = order or keys
        published[name] = {"clean": clean, "deranged": deranged, "gap": clean - deranged}

    summary: dict = {"task_order": order, "permutations": args.permutations}
    for modality, variants in VARIANTS.items():
        paper_name = MODALITY_TO_PAPER_NAME[modality]
        shot = paper.SPECS[paper_name]["shot"]
        ids = ["reference_published"]
        labels = ["Reference (published)"]
        clean = [published[paper_name]["clean"]]
        deranged = [published[paper_name]["deranged"]]
        for variant_id, label in variants:
            base = args.results_root / modality / variant_id
            if not ((base / "clean.json").exists() and (base / "deranged.json").exists()):
                print(f"[missing] {modality}/{variant_id}")
                continue
            _, c, d = load_gap(base / "clean.json", base / "deranged.json", shot, order)
            ids.append(variant_id)
            labels.append(label)
            clean.append(c)
            deranged.append(d)
        clean_arr, deranged_arr = np.array(clean), np.array(deranged)
        gap = clean_arr - deranged_arr
        n = len(ids)

        rho = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                rho[i, j] = rho[j, i] = float(spearmanr(gap[i], gap[j]).statistic)

        effects = []
        for i in range(n):
            effects.append(
                {
                    "id": ids[i],
                    "label": labels[i],
                    "clean": float(clean_arr[i].mean()),
                    "deranged": float(deranged_arr[i].mean()),
                    "gap": float(gap[i].mean()),
                    "gap_ci95": bootstrap_ci(gap[i], rng),
                    "gap_task_se": float(gap[i].std(ddof=1) / np.sqrt(gap[i].size)),
                }
            )

        others = [name for name in paper.DISPLAY_ORDER if name != paper_name]
        cross = []
        for i in range(n):
            row = {"id": ids[i], "label": labels[i]}
            for other in others:
                value = float(spearmanr(gap[i], published[other]["gap"]).statistic)
                p = permutation_p(gap[i], published[other]["gap"], draws=args.permutations, rng=rng)
                row[other] = {"spearman": value, "one_sided_permutation_p": p}
            cross.append(row)

        summary[modality] = {
            "ids": ids,
            "labels": labels,
            "variant_spearman": rho.tolist(),
            "effects": effects,
            "other_modalities": others,
            "cross_modality": cross,
            "profiles": {ids[i]: gap[i].tolist() for i in range(n)},
        }

        with (args.output_dir / f"{modality}_variant_spearman.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([""] + ids)
            for i in range(n):
                writer.writerow([ids[i]] + [f"{rho[i, j]:.4f}" for j in range(n)])
        with (args.output_dir / f"{modality}_effects.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "label", "clean", "deranged", "gap", "gap_ci95_low", "gap_ci95_high", "gap_task_se"])
            for e in effects:
                writer.writerow([e["id"], e["label"], f"{e['clean']:.6f}", f"{e['deranged']:.6f}", f"{e['gap']:.6f}",
                                 f"{e['gap_ci95'][0]:.6f}", f"{e['gap_ci95'][1]:.6f}", f"{e['gap_task_se']:.6f}"])
        with (args.output_dir / f"{modality}_cross_modality.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            header = ["id"]
            for other in others:
                header += [f"{other} rho", f"{other} p"]
            writer.writerow(header)
            for row in cross:
                line = [row["id"]]
                for other in others:
                    line += [f"{row[other]['spearman']:.4f}", f"{row[other]['one_sided_permutation_p']:.6g}"]
                writer.writerow(line)

        print(f"\n=== {modality} ===")
        for e in effects:
            print(f"  {e['id']:<22} clean {e['clean']:.4f}  deranged {e['deranged']:.4f}  gap {e['gap']:+.4f} "
                  f"[{e['gap_ci95'][0]:+.4f}, {e['gap_ci95'][1]:+.4f}]")
        print("  variant-by-variant Spearman:")
        for i in range(n):
            print("   ", f"{ids[i]:<22}", " ".join(f"{rho[i, j]:+.2f}" for j in range(n)))
        print("  vs other modalities (published):", others)
        for row in cross:
            print("   ", f"{row['id']:<22}",
                  " ".join(f"{row[o]['spearman']:+.2f}{'*' if row[o]['one_sided_permutation_p'] < 0.05 else ' '}" for o in others))

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\nwrote", args.output_dir)


if __name__ == "__main__":
    main()
