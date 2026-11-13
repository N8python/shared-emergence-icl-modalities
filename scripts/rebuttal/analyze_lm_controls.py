#!/usr/bin/env python3
"""Rebuttal language controls, per shot count.

Compares, at every shot count with both conditions available:
  * Qwen3-14B on literal 0/1 bitstrings ("00001111->00001111\\n...") against the
    published Qwen3-14B digit-encoding runs;
  * Mistral 7B v0.3 (base) on the paper digit encoding against the published
    Qwen3-14B and size-matched Qwen3-8B runs.
Reports clean/deranged accuracy, the paired effect, profile Spearman
correlations between language runs, and each run's correlation with the other
five modalities' published profiles (one-sided permutation p).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("enc", REPO_ROOT / "scripts/rebuttal/analyze_encoding_variants.py")
enc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enc)
paper = enc.paper

SHOTS = [1, 2, 4, 8, 16, 32, 64]
RUNS = {
    "qwen3_14b_digits": ("Qwen3-14B, digits (published)", "data", "qwen3/clean/Qwen3-14B-Base_ic{s}.json", "qwen3/deranged/Qwen3-14B-Base_deranged_ic{s}.json"),
    "qwen3_8b_digits": ("Qwen3-8B, digits (published)", "data", "qwen3_family/qwen3_8b/clean/Qwen3-8B-Base_ic{s}.json", "qwen3_family/qwen3_8b/deranged/Qwen3-8B-Base_deranged_ic{s}.json"),
    "qwen3_14b_literal": ("Qwen3-14B, literal bitstrings", "ctl", "qwen3_literal_bits/clean/ic{s}.json", "qwen3_literal_bits/deranged/ic{s}.json"),
    # Noise ceiling: the paper's Qwen3-14B digit run with a new seed (64 shots lives with the encoding variants).
    "qwen3_14b_digits_reseed": ("Qwen3-14B, digits (new seed)", "ctl", "qwen3_digits_reseed/clean/ic{s}.json", "qwen3_digits_reseed/deranged/ic{s}.json"),
    "mistral7b_digits": ("Mistral 7B v0.3, digits", "ctl", "mistral7b_v03_digits/clean/ic{s}.json", "mistral7b_v03_digits/deranged/ic{s}.json"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-root", type=Path, default=REPO_ROOT / "results_rebuttal/lm_controls")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/t128/results_128")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results_rebuttal/analysis")
    parser.add_argument("--permutations", type=int, default=1_000_000)
    args = parser.parse_args()
    rng = np.random.default_rng(20261112)
    roots = {"data": args.data_root, "ctl": args.controls_root}

    published, order = {}, None
    for name, spec in paper.SPECS.items():
        keys, clean, deranged = enc.load_gap(args.data_root / spec["clean"], args.data_root / spec["deranged"], spec["shot"], order)
        order = order or keys
        published[name] = clean - deranged
    others = [n for n in paper.DISPLAY_ORDER if n != "Language"]

    summary, rows = {}, []
    for shot in SHOTS:
        gaps, block = {}, {}
        for run_id, (label, root, clean_t, deranged_t) in RUNS.items():
            cp, dp = roots[root] / clean_t.format(s=shot), roots[root] / deranged_t.format(s=shot)
            if run_id == "qwen3_14b_digits_reseed" and shot == 64:
                base = args.controls_root.parent / "language/reference_reseed"
                cp, dp = base / "clean.json", base / "deranged.json"
            if not (cp.exists() and dp.exists()):
                continue
            _, clean, deranged = enc.load_gap(cp, dp, shot, order)
            gap = clean - deranged
            gaps[run_id] = gap
            cross = {}
            for other in others:
                cross[other] = {
                    "spearman": float(spearmanr(gap, published[other]).statistic),
                    "p": enc.permutation_p(gap, published[other], draws=args.permutations, rng=rng),
                }
            block[run_id] = {
                "label": label, "clean": float(clean.mean()), "deranged": float(deranged.mean()),
                "gap": float(gap.mean()), "gap_ci95": enc.bootstrap_ci(gap, rng), "cross": cross,
                "profile": gap.tolist(),
            }
        ids = list(gaps)
        pair = {a: {b: float(spearmanr(gaps[a], gaps[b]).statistic) for b in ids} for a in ids}
        summary[str(shot)] = {"runs": block, "pairwise_spearman": pair}
        print(f"\n=== {shot} shots ===")
        for run_id in ids:
            b = block[run_id]
            print(f"  {run_id:<20} clean {b['clean']:.4f} deranged {b['deranged']:.4f} gap {b['gap']:+.4f} "
                  f"[{b['gap_ci95'][0]:+.4f}, {b['gap_ci95'][1]:+.4f}]  vs " +
                  " ".join(f"{o[:4]} {b['cross'][o]['spearman']:+.2f}{'*' if b['cross'][o]['p'] < 0.05 else ' '}" for o in others))
            rows.append([shot, run_id, b["clean"], b["deranged"], b["gap"], *b["gap_ci95"]] +
                        [x for o in others for x in (b["cross"][o]["spearman"], b["cross"][o]["p"])])
        print("  pairwise profile Spearman:", ids)
        for a in ids:
            print("   ", f"{a:<20}", " ".join(f"{pair[a][b]:+.2f}" for b in ids))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "lm_controls_summary.json").write_text(json.dumps(summary, indent=1))
    with (args.output_dir / "lm_controls.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot", "run", "clean", "deranged", "gap", "gap_ci95_low", "gap_ci95_high"] +
                        [f"{o} {k}" for o in others for k in ("rho", "p")])
        writer.writerows(rows)


if __name__ == "__main__":
    main()
