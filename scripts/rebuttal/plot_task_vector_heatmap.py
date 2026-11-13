#!/usr/bin/env python3
"""Heatmap of one model's 100x100 task-vector similarity matrix.

Tasks are ordered by the paper's seven semantic clusters (cluster boundaries
drawn, labels on the left); within a cluster the order is the task index.
Reads results_rebuttal/analysis/task_vectors_summary.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SHORT = {1: "Global broadcast", 2: "Half segments", 3: "Parity masking", 4: "Half swaps",
         5: "Bit inversion", 6: "Bit reversal", 7: "Local shifts"}


def cluster_of_tasks(path: Path, programs_path: Path):
    clusters = json.loads(path.read_text())["clusters"]
    programs = [json.dumps(json.loads(line)["program"]) for line in programs_path.read_text().splitlines() if line.strip()]
    assignment, labels = {}, {}

    def walk(node, cluster_id):
        if isinstance(node, dict):
            if "program" in node and isinstance(node["program"], list):
                assignment[json.dumps(node["program"])] = cluster_id
            for value in node.values():
                walk(value, cluster_id)
        elif isinstance(node, list):
            for value in node:
                walk(value, cluster_id)

    for cluster in clusters:
        labels[cluster["cluster_id"]] = cluster["label"]
        walk(cluster, cluster["cluster_id"])
    missing = [p for p in programs if p not in assignment]
    if missing:
        raise ValueError(f"{len(missing)} tasks have no cluster, e.g. {missing[:3]}")
    return np.array([assignment[p] for p in programs]), labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/task_vectors_summary.json")
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--title", nargs="+", required=True, help="One panel title per model")
    parser.add_argument("--matrix", default="similarity",
                        choices=["similarity", "similarity_uncentered", "similarity_clean_only", "similarity_deranged_only"])
    parser.add_argument("--clusters", type=Path, default=REPO_ROOT / "data/t128/canonical_semantic_clusters_k7.json")
    parser.add_argument("--programs", type=Path, default=REPO_ROOT / "src/curated_transformations.jsonl")
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text())
    cluster, labels = cluster_of_tasks(args.clusters, args.programs)
    cluster = cluster[np.array(summary["task_indices"])]  # tasks with a zero task vector are excluded
    order = np.argsort(cluster, kind="stable")
    sorted_cluster = cluster[order]
    bounds = [0] + [int(i) for i in np.flatnonzero(np.diff(sorted_cluster)) + 1] + [len(order)]
    centers = [(a + b) / 2 - 0.5 for a, b in zip(bounds, bounds[1:])]
    ids = [int(sorted_cluster[a]) for a in bounds[:-1]]

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    n = len(args.model)
    fig, axes = plt.subplots(1, n, figsize=(1.9 + 4.6 * n, 4.9), constrained_layout=True, squeeze=False)
    for k, (axis, model, panel_title) in enumerate(zip(axes[0], args.model, args.title)):
        matrix = np.array(summary["models"][model][args.matrix])[np.ix_(order, order)]
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
        for b in bounds[1:-1]:
            axis.axhline(b - 0.5, color="black", linewidth=0.7)
            axis.axvline(b - 0.5, color="black", linewidth=0.7)
        if k == 0:
            axis.set_yticks(centers, labels=[f"{SHORT.get(i, labels[i])} ({b - a})" for i, a, b in zip(ids, bounds, bounds[1:])])
        else:
            axis.set_yticks([])
        axis.set_xticks([])
        axis.tick_params(axis="y", length=0)
        axis.set_title(panel_title, fontsize=10, loc="left")
    bar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    bar.set_label("Cosine similarity of task vectors")
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{suffix}"), dpi=300)
    print("wrote", args.output_stem.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
