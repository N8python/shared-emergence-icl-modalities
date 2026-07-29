#!/usr/bin/env python3
"""Rebuild all paper-facing T=128 summaries, tests, and figures."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_SCRIPTS = REPO_ROOT / "scripts" / "analysis"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "data" / "t128" / "results_128"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "t128" / "analysis"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper_runs_t128.json"
DEFAULT_CLUSTERS = REPO_ROOT / "data" / "t128" / "canonical_semantic_clusters_k7.json"
LEGACY_T8_ROOT = REPO_ROOT / "data" / "legacy_t8" / "principal_results"


def run_step(label: str, command: list[str]) -> None:
    print(f"\n[{label}]", flush=True)
    print("+ " + " ".join(command), flush=True)
    environment = dict(os.environ)
    environment.setdefault("MPLBACKEND", "Agg")
    subprocess.run(command, cwd=REPO_ROOT, env=environment, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--clusters", type=Path, default=DEFAULT_CLUSTERS)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use reduced Monte Carlo counts for a fast smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    config = args.config.expanduser().resolve()
    clusters = args.clusters.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(
            f"{results_root} does not exist. Run scripts/fetch_t128_results.py first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    correlations = output_dir / "correlations"
    sampling = output_dir / "sampling_noise"
    paper = output_dir / "paper"
    for directory in (correlations, sampling, paper):
        directory.mkdir(parents=True, exist_ok=True)

    posterior_draws = 1_000 if args.quick else 10_000
    bootstrap_draws = 1_000 if args.quick else 10_000
    permutations = 10_000 if args.quick else 1_000_000
    sampling_draws = 2_000 if args.quick else 20_000
    py = args.python_exe

    run_step(
        "raw result hashes",
        [
            py,
            str(REPO_ROOT / "scripts" / "verify_t128_hashes.py"),
            "--results-root",
            str(results_root),
        ],
    )
    run_step(
        "strict validation and full-grid summaries",
        [
            py,
            str(ANALYSIS_SCRIPTS / "validate_analyze_t128.py"),
            "--config",
            str(config),
            "--results-root",
            str(results_root),
            "--output-dir",
            str(output_dir),
            "--bootstrap-draws",
            str(bootstrap_draws),
        ],
    )
    highest_pvalues = output_dir / "highest_shot_pvalues_t128.csv"
    run_step(
        "maximum-shot paired tests",
        [
            py,
            str(ANALYSIS_SCRIPTS / "compute_highest_shot_pvalues_t128.py"),
            "--config",
            str(config),
            "--results-root",
            str(results_root),
            "--output",
            str(highest_pvalues),
        ],
    )
    correlation_command = [
        py,
        str(ANALYSIS_SCRIPTS / "analyze_correlations_t128.py"),
        "--data-root",
        str(results_root),
        "--output-dir",
        str(correlations),
        "--posterior-draws",
        str(posterior_draws),
        "--bootstrap-draws",
        str(bootstrap_draws),
    ]
    if LEGACY_T8_ROOT.is_dir():
        correlation_command.extend(["--reference-root", str(LEGACY_T8_ROOT)])
    run_step("paired-effect correlation analysis", correlation_command)
    correlation_pvalues = correlations / "gap_spearman_positive_pvalues.csv"
    run_step(
        "one-sided correlation permutation tests",
        [
            py,
            str(ANALYSIS_SCRIPTS / "compute_gap_spearman_pvalues_t128.py"),
            "--data-root",
            str(results_root),
            "--output-dir",
            str(correlations),
            "--permutations",
            str(permutations),
        ],
    )
    run_step(
        "Bernoulli sampling-noise calibration",
        [
            py,
            str(ANALYSIS_SCRIPTS / "simulate_sampling_correlations.py"),
            "--data-root",
            str(results_root),
            "--output-dir",
            str(sampling),
            "--draws",
            str(sampling_draws),
        ],
    )
    run_step(
        "Figure 3 and Table 3",
        [
            py,
            str(ANALYSIS_SCRIPTS / "make_modalities_clean_vs_deranged_t128.py"),
            "--cell-summary",
            str(output_dir / "cell_summary.csv"),
            "--best-shot-summary",
            str(output_dir / "best_shot_zgaps.csv"),
            "--pvalues",
            str(highest_pvalues),
            "--output-dir",
            str(paper),
        ],
    )
    run_step(
        "Figure 4 paired-effect correlations",
        [
            py,
            str(ANALYSIS_SCRIPTS / "plot_task_effect_correlation_t128.py"),
            "--data-root",
            str(results_root),
            "--pvalues-csv",
            str(correlation_pvalues),
            "--pdf-output",
            str(paper / "task_effect_spearman.pdf"),
            "--png-output",
            str(paper / "task_effect_spearman.png"),
            "--csv-output",
            str(paper / "task_effect_spearman.csv"),
        ],
    )
    run_step(
        "Figure 5 semantic-cluster effects",
        [
            py,
            str(ANALYSIS_SCRIPTS / "plot_semantic_cluster_effects_t128.py"),
            "--data-root",
            str(results_root),
            "--clusters",
            str(clusters),
            "--pdf-output",
            str(paper / "model_semantic_cluster_radar.pdf"),
            "--png-output",
            str(paper / "model_semantic_cluster_radar.png"),
            "--csv-output",
            str(paper / "model_semantic_cluster_radar.csv"),
        ],
    )
    run_step(
        "Figure 6 model-scale effects",
        [
            py,
            str(ANALYSIS_SCRIPTS / "plot_model_scale_effects_t128.py"),
            "--best-shots",
            str(output_dir / "best_shot_zgaps.csv"),
            "--paired-gaps",
            str(output_dir / "paired_gaps.csv"),
            "--cell-summary",
            str(output_dir / "cell_summary.csv"),
            "--output",
            str(paper / "modalities_scale_bars.pdf"),
            "--csv-output",
            str(paper / "modalities_scale_effects_t128.csv"),
        ],
    )
    run_step(
        "Figure 7 controls",
        [
            py,
            str(ANALYSIS_SCRIPTS / "plot_negative_controls_t128.py"),
            "--cell-summary",
            str(output_dir / "cell_summary.csv"),
            "--results-root",
            str(results_root),
            "--chess-output",
            str(paper / "chess_clean_vs_deranged.pdf"),
            "--music-output",
            str(paper / "musicroll_clean_vs_deranged.pdf"),
            "--csv-output",
            str(paper / "negative_controls_t128.csv"),
        ],
    )
    print(f"\nAll T=128 analyses completed: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
