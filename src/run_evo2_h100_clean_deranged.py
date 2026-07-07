from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

from tqdm import tqdm

from program_synth_evo_smartbatch import (
    Evo2,
    TaskResult,
    evaluate_task,
    load_programs,
    patch_torch_load_for_evo2,
    resolve_evo2_model_name,
)


DEFAULT_SHOTS: Sequence[int] = (1, 2, 4, 8, 16, 32, 64)
DEFAULT_CONDITIONS: Sequence[str] = ("clean", "deranged")
REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clean/deranged Evo2 program-synthesis sweeps while loading each model once."
    )
    parser.add_argument("--model", required=True, help="Evo2 model id, e.g. evo2_1b_base.")
    parser.add_argument(
        "--programs",
        type=Path,
        default=Path("curated_transformations.jsonl"),
        help="Program JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sweep_results_evo2_h100_ablate"),
        help="Directory for per-shot result JSONs.",
    )
    parser.add_argument("--shots", type=int, nargs="*", default=list(DEFAULT_SHOTS))
    parser.add_argument("--conditions", nargs="*", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--trials-per-program", type=int, default=8)
    parser.add_argument("--bit-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def slugify_model(identifier: str) -> str:
    return identifier.replace("/", "_")


def summarize_tasks(tasks: Sequence[TaskResult]) -> Dict[str, float]:
    accuracies = [task.accuracy() for task in tasks]
    mean_accuracy = statistics.mean(accuracies) if accuracies else 0.0
    accuracy_stderr = (
        statistics.stdev(accuracies) / math.sqrt(len(accuracies))
        if len(accuracies) > 1
        else 0.0
    )

    edit_distances = [task.average_edit_distance() for task in tasks]
    mean_edit_distance = statistics.mean(edit_distances) if edit_distances else 0.0
    edit_distance_stderr = (
        statistics.stdev(edit_distances) / math.sqrt(len(edit_distances))
        if len(edit_distances) > 1
        else 0.0
    )

    return {
        "mean_accuracy": mean_accuracy,
        "stderr": accuracy_stderr,
        "mean_edit_distance": mean_edit_distance,
        "edit_distance_stderr": edit_distance_stderr,
    }


def write_result(
    output_path: Path,
    *,
    model_name: str,
    programs_path: Path,
    in_context_examples: int,
    trials_per_program: int,
    bit_length: int,
    max_new_tokens: int,
    seed: int,
    ablate_labels: bool,
    task_results: Sequence[TaskResult],
) -> Dict[str, Any]:
    overall = summarize_tasks(task_results)
    output_data = {
        "config": {
            "model": model_name,
            "programs_file": portable_path(programs_path),
            "in_context_examples": in_context_examples,
            "trials_per_program": trials_per_program,
            "bit_length": bit_length,
            "max_new_tokens": max_new_tokens,
            "seed": seed,
            "per_trial_random": True,
            "ablate_labels": ablate_labels,
        },
        "overall": overall,
        "tasks": [task.to_json() for task in task_results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)
    return overall


def main() -> None:
    args = parse_args()
    if Evo2 is None:
        raise ImportError("The evo2 package is required.")
    if args.trials_per_program <= 0:
        raise ValueError("--trials-per-program must be positive")
    if args.bit_length <= 0:
        raise ValueError("--bit-length must be positive")

    shots = list(args.shots)
    conditions = list(args.conditions)
    invalid_conditions = [c for c in conditions if c not in {"clean", "deranged"}]
    if invalid_conditions:
        raise ValueError(f"Invalid conditions: {invalid_conditions}")

    max_new_tokens = args.max_new_tokens or args.bit_length
    programs = load_programs(args.programs)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runtime_model = resolve_evo2_model_name(args.model)
    if runtime_model == args.model:
        print(f"Loading {args.model} once for {len(shots) * len(conditions)} runs...")
    else:
        print(
            f"Loading {args.model} as Evo2 alias {runtime_model} once for "
            f"{len(shots) * len(conditions)} runs..."
        )
    patch_torch_load_for_evo2()
    model = Evo2(runtime_model)
    tokenizer = None

    summary_runs: List[Dict[str, Any]] = []
    model_slug = slugify_model(runtime_model)

    for condition in conditions:
        ablate_labels = condition == "deranged"
        condition_part = "_deranged" if ablate_labels else ""
        for shot in shots:
            output_path = args.output_dir / f"{model_slug}{condition_part}_ic{shot}.json"
            if output_path.exists() and not args.overwrite:
                with output_path.open("r", encoding="utf-8") as handle:
                    existing = json.load(handle)
                overall = existing.get("overall", {})
                print(
                    f"Skipping existing {output_path}: "
                    f"acc={overall.get('mean_accuracy')} edit={overall.get('mean_edit_distance')}"
                )
                summary_runs.append(
                    {
                        "model": args.model,
                        "condition": condition,
                        "in_context_examples": shot,
                        "output_file": portable_path(output_path),
                        **overall,
                    }
                )
                continue

            random.seed(args.seed)
            task_results: List[TaskResult] = []
            total_predictions = len(programs) * args.trials_per_program
            desc = f"{args.model} {condition} ic{shot}"
            with tqdm(total=total_predictions, desc=desc) as progress:
                for idx, entry in enumerate(programs):
                    task = evaluate_task(
                        model,
                        tokenizer,
                        entry["program"],
                        description=entry.get("description"),
                        trials_per_program=args.trials_per_program,
                        in_context_examples=shot,
                        bit_length=args.bit_length,
                        max_new_tokens=max_new_tokens,
                        backend="evo",
                        per_trial_random=True,
                        ablate_labels=ablate_labels,
                        progress=progress,
                    )
                    task.index = idx
                    task_results.append(task)

            overall = write_result(
                output_path,
                model_name=args.model,
                programs_path=args.programs,
                in_context_examples=shot,
                trials_per_program=args.trials_per_program,
                bit_length=args.bit_length,
                max_new_tokens=max_new_tokens,
                seed=args.seed,
                ablate_labels=ablate_labels,
                task_results=task_results,
            )
            print(
                f"Wrote {output_path}: "
                f"acc={overall['mean_accuracy']:.4f} edit={overall['mean_edit_distance']:.4f}"
            )
            summary_runs.append(
                {
                    "model": args.model,
                    "condition": condition,
                    "in_context_examples": shot,
                    "output_file": portable_path(output_path),
                    **overall,
                }
            )

    condition_slug = "_".join(conditions)
    summary_dir = args.output_dir
    if len(conditions) == 1 and args.output_dir.name == conditions[0]:
        summary_dir = args.output_dir.parent
    summary_path = summary_dir / f"{model_slug}_{condition_slug}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model": args.model,
                "programs_file": portable_path(args.programs),
                "shots": shots,
                "conditions": conditions,
                "trials_per_program": args.trials_per_program,
                "bit_length": args.bit_length,
                "max_new_tokens": max_new_tokens,
                "seed": args.seed,
                "runs": summary_runs,
            },
            handle,
            indent=2,
        )
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
