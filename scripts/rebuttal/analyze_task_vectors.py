#!/usr/bin/env python3
"""Task-vector similarity matrices and their agreement across models.

Input: results_rebuttal/task_vectors/<model>/{clean,deranged}.npz as written by
`program_synth.py --dump-hidden-states` (per-task mean last-position hidden
states at every layer, split into even/odd trial halves; the deranged run uses
--paired-ablation so each deranged prompt is the twin of a clean prompt).

Per model and layer the task vector of task t is

    v(t) = mean_trials h_clean - mean_trials h_deranged,

centered across the 100 tasks; the similarity matrix is the cosine between
task vectors. The layer is chosen per model by split-half reliability of that
matrix (never by agreement with another model). Models are then compared by
the Spearman correlation of the 4,950 off-diagonal entries, with a Mantel
permutation p-value (task labels permuted) and a partial correlation that
controls for (a) the similarity of the tasks' truth tables and (b) the two
models' deranged-only matrices.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def cosine_matrix(vectors: np.ndarray, center: bool = True) -> np.ndarray:
    centered = vectors - vectors.mean(axis=0, keepdims=True) if center else vectors
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    unit = centered / np.maximum(norms, 1e-12)
    return unit @ unit.T


def upper(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(matrix.shape[0], k=1)]


def zrank(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values)
    return (ranks - ranks.mean()) / ranks.std()


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(zrank(a) * zrank(b)))


def residual(target: np.ndarray, covariates: list) -> np.ndarray:
    """Rank-transform, then remove the least-squares fit on the ranked covariates."""
    y = zrank(target)
    if not covariates:
        return y
    x = np.column_stack([np.ones_like(y)] + [zrank(c) for c in covariates])
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return y - x @ beta


def mantel(a: np.ndarray, b: np.ndarray, covariates_a: list, covariates_b: list, *, draws: int, rng) -> dict:
    """Spearman of off-diagonal entries; p from permuting the task labels of `b`.

    With covariates this is a partial Mantel test: each matrix is residualized on
    its covariate matrices (after ranking) and b's residual matrix is permuted.
    """
    n = a.shape[0]
    iu = np.triu_indices(n, k=1)
    ra = residual(a[iu], [c[iu] for c in covariates_a])
    rb_vec = residual(b[iu], [c[iu] for c in covariates_b])
    rb = np.zeros((n, n))
    rb[iu] = rb_vec
    rb = rb + rb.T
    scale = np.sqrt(np.sum(ra**2) * np.sum(rb_vec**2))
    observed = float(np.sum(ra * rb_vec) / scale)
    count = 0
    for _ in range(draws):
        perm = rng.permutation(n)
        count += float(np.sum(ra * rb[np.ix_(perm, perm)][iu]) / scale) >= observed - 1e-12
    return {"rho": observed, "p": (count + 1) / (draws + 1)}


def truth_table_similarity(programs_path: Path, bit_length: int = 8) -> np.ndarray:
    """Mean per-bit agreement of two tasks' outputs over all inputs."""
    import dsl

    programs = [json.loads(line)["program"] for line in programs_path.read_text().splitlines() if line.strip()]
    inputs = [format(i, f"0{bit_length}b") for i in range(2**bit_length)]
    constant = "10110010"  # fixed stand-in for the per-trial random constant of meta_constant tasks
    tables = np.array([
        [[int(bit) for bit in dsl.run_program(program, s, constant)] for s in inputs] for program in programs
    ], dtype=np.float64)
    flat = tables.reshape(len(programs), -1)
    agree = (flat @ flat.T + (1 - flat) @ (1 - flat).T) / flat.shape[1]
    return agree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT / "results_rebuttal/task_vectors")
    parser.add_argument("--programs", type=Path, default=REPO_ROOT / "src/curated_transformations.jsonl")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results_rebuttal/analysis")
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args()
    rng = np.random.default_rng(20261112)

    # Pass 1: load, and find tasks whose clean and deranged prompts coincide (constant
    # outputs within a prompt cannot be deranged), which leaves a zero task vector.
    raw = {}
    for directory in sorted(p for p in args.root.iterdir() if p.is_dir()):
        clean_path, deranged_path = directory / "clean.npz", directory / "deranged.npz"
        if not (clean_path.exists() and deranged_path.exists()):
            print(f"[missing] {directory.name}")
            continue
        clean, deranged = np.load(clean_path), np.load(deranged_path)
        ct, dt = clean["prompt_tokens"], deranged["prompt_tokens"]
        paired = ct.shape == dt.shape and all(sorted(x) == sorted(y) for x, y in zip(ct[::97], dt[::97]))
        raw[directory.name] = (clean["half_means"].astype(np.float64), deranged["half_means"].astype(np.float64), paired)
    n_tasks = next(iter(raw.values()))[0].shape[0]
    valid = np.ones(n_tasks, dtype=bool)
    for c, d, _ in raw.values():
        valid &= np.linalg.norm((c - d).mean(axis=1)[:, -1], axis=1) > 1e-6
    excluded = [int(i) for i in np.flatnonzero(~valid)]
    print(f"excluded tasks with zero task vector: {excluded}")

    models = {}
    for name, (c, d, paired) in raw.items():
        c, d = c[valid], d[valid]
        diff = c - d  # [tasks, halves, layers, hidden]
        layers = diff.shape[2]
        reliability = [0.0] + [spearman(upper(cosine_matrix(diff[:, 0, l])), upper(cosine_matrix(diff[:, 1, l]))) for l in range(1, layers)]
        # A layer whose receptive field at the last position excludes the labels has a zero
        # task vector for every task (undefined reliability); it can never be chosen.
        reliability = [0.0 if not np.isfinite(r) else r for r in reliability]
        best = int(np.argmax(reliability))
        models[name] = {
            "paired_prompts": bool(paired),
            "reliability_by_layer": reliability,
            "layer": best,
            "diff_by_layer": [cosine_matrix(diff[:, :, l].mean(axis=1)) if l else np.eye(diff.shape[0]) for l in range(layers)],
            "uncentered": cosine_matrix(diff[:, :, best].mean(axis=1), center=False),
            "clean": cosine_matrix(c[:, :, best].mean(axis=1)),
            "deranged": cosine_matrix(d[:, :, best].mean(axis=1)),
            "diff_norm": np.linalg.norm(diff[:, :, best].mean(axis=1), axis=1),
        }
        print(f"{name}: layers={layers - 1} chosen layer={best} split-half rho={reliability[best]:.3f} paired={paired}")

    truth = truth_table_similarity(args.programs)[np.ix_(valid, valid)]
    n = truth.shape[0]
    summary = {"models": {}, "pairs": {}, "permutations": args.permutations,
               "excluded_tasks": excluded, "task_indices": [int(i) for i in np.flatnonzero(valid)]}
    for name, m in models.items():
        if m["diff_by_layer"][0].shape[0] != n:
            raise ValueError(f"{name}: {m['diff_by_layer'][0].shape[0]} tasks, expected {n}")
        s = m["diff_by_layer"][m["layer"]]
        summary["models"][name] = {
            "layer": m["layer"],
            "paired_prompts": m["paired_prompts"],
            "reliability_by_layer": m["reliability_by_layer"],
            "vs_truth_table": mantel(s, truth, [], [], draws=args.permutations, rng=rng),
            "deranged_vs_truth_table": mantel(m["deranged"], truth, [], [], draws=args.permutations, rng=rng),
            "similarity": s.tolist(),
            "similarity_uncentered": m["uncentered"].tolist(),
            "similarity_clean_only": m["clean"].tolist(),
            "similarity_deranged_only": m["deranged"].tolist(),
            "diff_norm": m["diff_norm"].tolist(),
        }
        print(f"  {name}: vs truth tables rho={summary['models'][name]['vs_truth_table']['rho']:.3f}")

    for a, b in itertools.combinations(models, 2):
        ma, mb = models[a], models[b]
        sa, sb = ma["diff_by_layer"][ma["layer"]], mb["diff_by_layer"][mb["layer"]]
        block = {
            "task_vectors": mantel(sa, sb, [], [], draws=args.permutations, rng=rng),
            "task_vectors_partial_truth": mantel(sa, sb, [truth], [truth], draws=args.permutations, rng=rng),
            "task_vectors_partial_truth_and_deranged": mantel(
                sa, sb, [truth, ma["deranged"]], [truth, mb["deranged"]], draws=args.permutations, rng=rng),
            "task_vectors_uncentered": mantel(ma["uncentered"], mb["uncentered"], [], [], draws=args.permutations, rng=rng),
            "clean_only": mantel(ma["clean"], mb["clean"], [], [], draws=args.permutations, rng=rng),
            "deranged_only": mantel(ma["deranged"], mb["deranged"], [], [], draws=args.permutations, rng=rng),
            # Every layer pair, so the chosen layers can be seen in context.
            "layer_grid": [[spearman(upper(x), upper(y)) if i and j else 0.0 for j, y in enumerate(mb["diff_by_layer"])]
                           for i, x in enumerate(ma["diff_by_layer"])],
        }
        summary["pairs"][f"{a}|{b}"] = block
        print(f"{a} vs {b}: task vectors rho={block['task_vectors']['rho']:.3f} (p={block['task_vectors']['p']:.3g}); "
              f"partial truth rho={block['task_vectors_partial_truth']['rho']:.3f}; "
              f"partial truth+deranged rho={block['task_vectors_partial_truth_and_deranged']['rho']:.3f} "
              f"(p={block['task_vectors_partial_truth_and_deranged']['p']:.3g}); "
              f"uncentered rho={block['task_vectors_uncentered']['rho']:.3f}; clean-only rho={block['clean_only']['rho']:.3f}; "
              f"deranged-only rho={block['deranged_only']['rho']:.3f}")

    summary["truth_table_similarity"] = truth.tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "task_vectors_summary.json").write_text(json.dumps(summary))
    print("wrote", args.output_dir / "task_vectors_summary.json")


if __name__ == "__main__":
    main()
