#!/usr/bin/env python3
"""Recompute the representative clean-vs-deranged result summaries.

It consumes the downloaded per-run JSON outputs and recomputes the headline
numeric table without importing the experiment harness or plotting machinery.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "t128" / "results_128"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "t128" / "principal_results"


@dataclass(frozen=True)
class RunSpec:
    key: str
    label: str
    modality: str
    expected_signal: str
    run_dir: str | None = None
    clean_dir: str | None = None
    deranged_dir: str | None = None
    note: str = ""


@dataclass(frozen=True)
class RunPoint:
    shot: int
    path: Path
    sha256: str
    accuracy: float
    accuracy_stderr: float
    edit_distance: float | None
    edit_distance_stderr: float | None
    task_count: int | None
    trial_count: int | None


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        key="qwen3",
        label="Qwen3 14B",
        modality="language tokens",
        expected_signal="positive",
        run_dir="qwen3",
        note="Representative Qwen3 language-token run.",
    ),
    RunSpec(
        key="evo2",
        label="Evo2 40B",
        modality="genomics",
        expected_signal="positive",
        run_dir="evo2",
        note="Representative Evo2 nucleotide-token run.",
    ),
    RunSpec(
        key="nextterm",
        label="NextTerm 440M",
        modality="integer sequence tokens",
        expected_signal="positive",
        run_dir="nextterm",
        note="Representative NextTerm digit-token run.",
    ),
    RunSpec(
        key="imagegpt",
        label="ImageGPT large",
        modality="raster image tokens",
        expected_signal="positive",
        run_dir="imagegpt",
        note="Representative ImageGPT left-pad raster-token run.",
    ),
    RunSpec(
        key="timesfm",
        label="TimesFM 2.5 200M",
        modality="time-series patches",
        expected_signal="positive",
        run_dir="timesfm",
        note="Representative TimesFM 2-bit sine-lobe run.",
    ),
    RunSpec(
        key="progen2",
        label="ProGen2 base",
        modality="protein amino-acid tokens",
        expected_signal="positive",
        run_dir="progen2",
        note="Representative ProGen2 protein-token run.",
    ),
    RunSpec(
        key="chessgpt",
        label="ChessGPT 50M",
        modality="chess PGN tokens",
        expected_signal="negative",
        run_dir="chessgpt",
        note="Signal2-wipe ChessGPT negative/control run.",
    ),
    RunSpec(
        key="musicroll",
        label="MusicRoll 50M",
        modality="music time-slice tokens",
        expected_signal="negative",
        run_dir="musicroll",
        note="ROLL time-slice music run; aggregate paired-mapping result is negative despite retrieval-family behavior.",
    ),
)


def log(message: str, *, quiet: bool) -> None:
    if not quiet:
        print(message, file=sys.stderr, flush=True)


def portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def z_score(numerator: float, denominator: float) -> float:
    if denominator == 0:
        if numerator > 0:
            return math.inf
        if numerator < 0:
            return -math.inf
        return math.nan
    return numerator / denominator


def z_gap_denominator(stderr_a: float, stderr_b: float) -> float:
    return math.hypot(stderr_a, stderr_b)


def infer_shot(path: Path, data: dict[str, Any]) -> int:
    config = data.get("config", {})
    if isinstance(config, dict):
        for key in ("in_context_examples", "num_shots", "shots", "n_shots"):
            value = config.get(key)
            if isinstance(value, int):
                return value

    for pattern in (
        r"_ic(\d+)\.json$",
        r"_(\d+)shot(?:_[^.]+)?\.json$",
        r"_(\d+)shot_",
    ):
        match = re.search(pattern, path.name)
        if match:
            return int(match.group(1))

    raise ValueError(f"Could not infer shot count from {path}")


def count_trials(tasks: Any) -> int | None:
    if not isinstance(tasks, list):
        return None
    total = 0
    for task in tasks:
        if not isinstance(task, dict):
            return None
        trials = task.get("trials")
        if not isinstance(trials, list):
            return None
        total += len(trials)
    return total


def load_point(path: Path) -> RunPoint | None:
    data = json.loads(path.read_text())
    if "overall" not in data or "tasks" not in data:
        return None

    overall = data["overall"]
    if not isinstance(overall, dict):
        raise ValueError(f"{path} has a non-object overall field")

    try:
        accuracy = float(overall["mean_accuracy"])
        accuracy_stderr = float(overall["stderr"])
    except KeyError as exc:
        raise KeyError(f"{path} is missing required overall metric {exc}") from exc

    edit_distance = overall.get("mean_edit_distance")
    edit_distance_stderr = overall.get("edit_distance_stderr")
    tasks = data.get("tasks")

    return RunPoint(
        shot=infer_shot(path, data),
        path=path,
        sha256=sha256_file(path),
        accuracy=accuracy,
        accuracy_stderr=accuracy_stderr,
        edit_distance=None if edit_distance is None else float(edit_distance),
        edit_distance_stderr=(
            None if edit_distance_stderr is None else float(edit_distance_stderr)
        ),
        task_count=len(tasks) if isinstance(tasks, list) else None,
        trial_count=count_trials(tasks),
    )


def load_condition(condition_dir: Path) -> dict[int, RunPoint]:
    if not condition_dir.is_dir():
        raise FileNotFoundError(f"Missing condition directory: {condition_dir}")

    points: dict[int, RunPoint] = {}
    for path in sorted(condition_dir.glob("*.json")):
        if path.name.startswith("."):
            continue
        point = load_point(path)
        if point is None:
            continue
        if point.shot in points:
            raise ValueError(
                f"Duplicate shot {point.shot} in {condition_dir}: "
                f"{points[point.shot].path.name} and {path.name}"
            )
        points[point.shot] = point

    if not points:
        raise ValueError(f"No result JSONs found in {condition_dir}")
    return points


def resolve_spec_dirs(data_root: Path, spec: RunSpec) -> tuple[Path, Path]:
    if spec.run_dir is not None:
        run_dir = data_root / spec.run_dir
        return run_dir / "clean", run_dir / "deranged"
    if spec.clean_dir is None or spec.deranged_dir is None:
        raise ValueError(f"Run spec {spec.key!r} is missing result directories")
    return data_root / spec.clean_dir, data_root / spec.deranged_dir


def best_accuracy_result(clean: dict[int, RunPoint], deranged: dict[int, RunPoint]) -> dict[str, Any]:
    common_shots = sorted(set(clean) & set(deranged))
    if not common_shots:
        raise ValueError("Clean and deranged conditions have no shared shot counts")

    best_clean = max((clean[shot] for shot in common_shots), key=lambda p: p.accuracy)
    best_deranged = max(
        (deranged[shot] for shot in common_shots), key=lambda p: p.accuracy
    )
    gap = best_clean.accuracy - best_deranged.accuracy
    denominator = z_gap_denominator(
        best_clean.accuracy_stderr,
        best_deranged.accuracy_stderr,
    )
    return {
        "metric": "best_accuracy_z_gap",
        "z_gap": z_score(gap, denominator),
        "gap": gap,
        "denominator": denominator,
        "best_clean": point_summary(best_clean, metric="accuracy"),
        "best_deranged": point_summary(best_deranged, metric="accuracy"),
        "common_shots": common_shots,
    }


def best_edit_result(clean: dict[int, RunPoint], deranged: dict[int, RunPoint]) -> dict[str, Any] | None:
    common_shots = sorted(set(clean) & set(deranged))
    clean_points = [
        clean[shot]
        for shot in common_shots
        if clean[shot].edit_distance is not None
        and clean[shot].edit_distance_stderr is not None
    ]
    deranged_points = [
        deranged[shot]
        for shot in common_shots
        if deranged[shot].edit_distance is not None
        and deranged[shot].edit_distance_stderr is not None
    ]
    if not clean_points or not deranged_points:
        return None

    best_clean = min(clean_points, key=lambda p: p.edit_distance if p.edit_distance is not None else math.inf)
    best_deranged = min(
        deranged_points,
        key=lambda p: p.edit_distance if p.edit_distance is not None else math.inf,
    )
    assert best_clean.edit_distance is not None
    assert best_clean.edit_distance_stderr is not None
    assert best_deranged.edit_distance is not None
    assert best_deranged.edit_distance_stderr is not None

    gap = best_deranged.edit_distance - best_clean.edit_distance
    denominator = z_gap_denominator(
        best_clean.edit_distance_stderr,
        best_deranged.edit_distance_stderr,
    )
    return {
        "metric": "best_edit_z_gap",
        "z_gap": z_score(gap, denominator),
        "gap": gap,
        "denominator": denominator,
        "best_clean": point_summary(best_clean, metric="edit_distance"),
        "best_deranged": point_summary(best_deranged, metric="edit_distance"),
        "common_shots": common_shots,
    }


def point_summary(point: RunPoint, *, metric: str) -> dict[str, Any]:
    summary = {
        "shot": point.shot,
        "file": portable_path(point.path),
        "sha256": point.sha256,
        "task_count": point.task_count,
        "trial_count": point.trial_count,
    }
    if metric == "accuracy":
        summary["value"] = point.accuracy
        summary["stderr"] = point.accuracy_stderr
    elif metric == "edit_distance":
        summary["value"] = point.edit_distance
        summary["stderr"] = point.edit_distance_stderr
    else:
        raise ValueError(f"Unsupported metric {metric!r}")
    return summary


def summarize_run(data_root: Path, spec: RunSpec, *, quiet: bool) -> dict[str, Any]:
    clean_dir, deranged_dir = resolve_spec_dirs(data_root, spec)
    log(f"[{spec.key}] loading {clean_dir} and {deranged_dir}", quiet=quiet)
    clean = load_condition(clean_dir)
    deranged = load_condition(deranged_dir)

    accuracy = best_accuracy_result(clean, deranged)
    edit = best_edit_result(clean, deranged)
    common_shots = accuracy["common_shots"]

    by_shot = []
    for shot in common_shots:
        clean_point = clean[shot]
        deranged_point = deranged[shot]
        by_shot.append(
            {
                "key": spec.key,
                "label": spec.label,
                "shot": shot,
                "clean_accuracy": clean_point.accuracy,
                "clean_accuracy_stderr": clean_point.accuracy_stderr,
                "deranged_accuracy": deranged_point.accuracy,
                "deranged_accuracy_stderr": deranged_point.accuracy_stderr,
                "accuracy_gap": clean_point.accuracy - deranged_point.accuracy,
                "clean_edit_distance": clean_point.edit_distance,
                "clean_edit_distance_stderr": clean_point.edit_distance_stderr,
                "deranged_edit_distance": deranged_point.edit_distance,
                "deranged_edit_distance_stderr": deranged_point.edit_distance_stderr,
                "edit_distance_gap": (
                    None
                    if clean_point.edit_distance is None
                    or deranged_point.edit_distance is None
                    else deranged_point.edit_distance - clean_point.edit_distance
                ),
                "clean_file": portable_path(clean_point.path),
                "deranged_file": portable_path(deranged_point.path),
                "clean_sha256": clean_point.sha256,
                "deranged_sha256": deranged_point.sha256,
            }
        )

    return {
        "spec": asdict(spec),
        "source": {
            "clean_dir": portable_path(clean_dir),
            "deranged_dir": portable_path(deranged_dir),
            "clean_shots": sorted(clean),
            "deranged_shots": sorted(deranged),
            "common_shots": common_shots,
            "clean_file_count": len(clean),
            "deranged_file_count": len(deranged),
        },
        "best_accuracy_z_gap": accuracy,
        "best_edit_z_gap": edit,
        "by_shot": by_shot,
    }


def flatten_summary(result: dict[str, Any]) -> dict[str, Any]:
    spec = result["spec"]
    accuracy = result["best_accuracy_z_gap"]
    edit = result["best_edit_z_gap"]

    row = {
        "key": spec["key"],
        "label": spec["label"],
        "modality": spec["modality"],
        "expected_signal": spec["expected_signal"],
        "common_shots": " ".join(str(shot) for shot in accuracy["common_shots"]),
        "best_clean_accuracy_shot": accuracy["best_clean"]["shot"],
        "best_clean_accuracy": accuracy["best_clean"]["value"],
        "best_clean_accuracy_stderr": accuracy["best_clean"]["stderr"],
        "best_deranged_accuracy_shot": accuracy["best_deranged"]["shot"],
        "best_deranged_accuracy": accuracy["best_deranged"]["value"],
        "best_deranged_accuracy_stderr": accuracy["best_deranged"]["stderr"],
        "best_accuracy_gap": accuracy["gap"],
        "best_accuracy_z_gap": accuracy["z_gap"],
        "best_accuracy_z_denominator": accuracy["denominator"],
        "best_clean_accuracy_file": accuracy["best_clean"]["file"],
        "best_deranged_accuracy_file": accuracy["best_deranged"]["file"],
        "note": spec["note"],
    }
    if edit is not None:
        row.update(
            {
                "best_clean_edit_shot": edit["best_clean"]["shot"],
                "best_clean_edit_distance": edit["best_clean"]["value"],
                "best_clean_edit_stderr": edit["best_clean"]["stderr"],
                "best_deranged_edit_shot": edit["best_deranged"]["shot"],
                "best_deranged_edit_distance": edit["best_deranged"]["value"],
                "best_deranged_edit_stderr": edit["best_deranged"]["stderr"],
                "best_edit_gap": edit["gap"],
                "best_edit_z_gap": edit["z_gap"],
                "best_edit_z_denominator": edit["denominator"],
            }
        )
    else:
        row.update(
            {
                "best_clean_edit_shot": None,
                "best_clean_edit_distance": None,
                "best_clean_edit_stderr": None,
                "best_deranged_edit_shot": None,
                "best_deranged_edit_distance": None,
                "best_deranged_edit_stderr": None,
                "best_edit_gap": None,
                "best_edit_z_gap": None,
                "best_edit_z_denominator": None,
            }
        )
    return row


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def selected_specs(keys: list[str] | None) -> list[RunSpec]:
    if keys is None:
        return list(RUN_SPECS)
    by_key = {spec.key: spec for spec in RUN_SPECS}
    missing = [key for key in keys if key not in by_key]
    if missing:
        raise ValueError(
            f"Unknown run key(s): {', '.join(missing)}. "
            f"Known keys: {', '.join(sorted(by_key))}"
        )
    return [by_key[key] for key in keys]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        "--source-root",
        dest="data_root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=(
            "Directory containing per-run clean/deranged JSON folders. "
            f"Defaults to bundled data: {DEFAULT_DATA_ROOT}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for summary outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        help="Optional subset of run keys to recompute.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print available run keys and exit.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_runs:
        for spec in RUN_SPECS:
            print(f"{spec.key}\t{spec.label}\t{spec.expected_signal}")
        return

    data_root = args.data_root.expanduser().resolve()
    output_dir = args.output_dir.resolve()
    specs = selected_specs(args.runs)

    log(f"Data root:  {data_root}", quiet=args.quiet)
    log(f"Output dir: {output_dir}", quiet=args.quiet)
    log(f"Runs:       {', '.join(spec.key for spec in specs)}", quiet=args.quiet)

    results = []
    for spec in specs:
        results.append(summarize_run(data_root, spec, quiet=args.quiet))

    summary_rows = [flatten_summary(result) for result in results]
    by_shot_rows = [
        shot_row
        for result in results
        for shot_row in result["by_shot"]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summary_rows)
    write_csv(output_dir / "by_shot.csv", by_shot_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "data_root": portable_path(data_root),
                "runs": results,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n"
    )

    log(f"Wrote {output_dir / 'summary.csv'}", quiet=args.quiet)
    log(f"Wrote {output_dir / 'by_shot.csv'}", quiet=args.quiet)
    log(f"Wrote {output_dir / 'summary.json'}", quiet=args.quiet)

    if not args.quiet:
        print("\nPrincipal clean-vs-deranged summary:")
        for row in summary_rows:
            print(
                f"{row['key']:10s} "
                f"clean={row['best_clean_accuracy']:.4f}@{row['best_clean_accuracy_shot']:>2} "
                f"deranged={row['best_deranged_accuracy']:.4f}@{row['best_deranged_accuracy_shot']:>2} "
                f"gap={row['best_accuracy_gap']:+.4f} "
                f"z={row['best_accuracy_z_gap']:.3f} "
                f"({row['expected_signal']})"
            )


if __name__ == "__main__":
    main()
