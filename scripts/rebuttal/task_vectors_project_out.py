#!/usr/bin/env python3
"""Exploratory: task-vector similarity after projecting out each model's top direction.

For every model, at the layer already chosen by analyze_task_vectors.py, the centered
clean-minus-deranged task vectors have their first k principal directions removed
before cosine similarity is taken. Writes a summary JSON in the same layout as
task_vectors_summary.json (so plot_task_vector_heatmap.py can read it) and prints the
pairwise Spearman agreement before and after.
"""
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("tv", REPO_ROOT / "scripts/rebuttal/analyze_task_vectors.py")
tv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "results_rebuttal/task_vectors")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/task_vectors_summary.json")
    parser.add_argument("--components", type=int, default=1)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results_rebuttal/analysis/task_vectors_projected_summary.json")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    valid = np.array(summary["task_indices"])
    out = {"task_indices": summary["task_indices"], "excluded_tasks": summary["excluded_tasks"],
           "components_removed": args.components, "models": {}, "pairs": {}}
    halves = {}
    for name, block in summary["models"].items():
        layer = block["layer"]
        clean = np.load(args.root / name / "clean.npz")["half_means"].astype(np.float64)[valid][:, :, layer]
        deranged = np.load(args.root / name / "deranged.npz")["half_means"].astype(np.float64)[valid][:, :, layer]
        diff = clean - deranged  # [tasks, halves, hidden]
        full = diff.mean(axis=1)
        centered = full - full.mean(axis=0)
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[: args.components]
        explained = float(np.sum(s[: args.components] ** 2) / np.sum(s**2))

        def project(x):
            x = x - x.mean(axis=0)
            return x - (x @ basis.T) @ basis

        matrix = tv.cosine_matrix(project(full), center=False)
        # Split-half reliability with the direction estimated on the full data (slightly optimistic).
        rel = tv.spearman(tv.upper(tv.cosine_matrix(project(diff[:, 0]), center=False)),
                          tv.upper(tv.cosine_matrix(project(diff[:, 1]), center=False)))
        out["models"][name] = {"layer": layer, "similarity": matrix.tolist(), "variance_removed": explained, "split_half": rel}
        halves[name] = matrix
        print(f"{name}: layer {layer}, removed {100 * explained:.0f}% of variance, split-half after projection {rel:.3f}")
    for a, b in itertools.combinations(halves, 2):
        before = tv.spearman(tv.upper(np.array(summary["models"][a]["similarity"])), tv.upper(np.array(summary["models"][b]["similarity"])))
        after = tv.spearman(tv.upper(halves[a]), tv.upper(halves[b]))
        out["pairs"][f"{a}|{b}"] = {"before": before, "after": after}
        print(f"{a} vs {b}: before {before:.3f}  after {after:.3f}")
    args.output.write_text(json.dumps(out))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
