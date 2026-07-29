#!/usr/bin/env python3
"""Validate and summarize the paper experiment manifest without loading models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper_runs_t128.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--list-cells", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    expected_trials = int(config["experiment"]["trials_per_program"])
    cells = []
    for run_key, run in config["runs"].items():
        base_args = list(map(str, run["base_args"]))
        try:
            trial_index = base_args.index("--trials-per-program") + 1
            configured_trials = int(base_args[trial_index])
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{run_key}: missing trial-count argument") from exc
        if configured_trials != expected_trials:
            raise ValueError(
                f"{run_key}: trial count {configured_trials} != {expected_trials}"
            )
        for condition_name, condition in run["conditions"].items():
            if condition_name not in {"clean", "deranged"}:
                raise ValueError(f"{run_key}: unsupported condition {condition_name}")
            for shot in condition["shots"]:
                relative = condition["output_template"].format(
                    run=run_key,
                    condition=condition_name,
                    shot=shot,
                )
                cells.append((run_key, condition_name, int(shot), relative))

    predictions = len(cells) * 100 * expected_trials
    print(f"run keys:             {len(config['runs'])}")
    print(f"experiment cells:     {len(cells)}")
    print("tasks per cell:        100")
    print(f"trials per task:       {expected_trials}")
    print(f"expected predictions:  {predictions}")
    if args.list_cells:
        for run_key, condition, shot, relative in cells:
            print(f"{run_key}\t{condition}\t{shot}\t{relative}")


if __name__ == "__main__":
    main()
