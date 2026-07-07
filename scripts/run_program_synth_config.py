#!/usr/bin/env python3
"""Build or execute principal-run commands from a run config.

Default behavior is a dry run that prints the commands. Pass --execute to run
them. Model locations are taken from each config's environment variable when it
is set, otherwise from the config's default_model field.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "principal_runs.json"


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def selected_runs(config: dict[str, Any], names: list[str] | None) -> list[str]:
    runs = config.get("runs", {})
    if not isinstance(runs, dict) or not runs:
        raise ValueError("Config must contain a nonempty 'runs' object")
    if names is None:
        return list(runs)
    missing = [name for name in names if name not in runs]
    if missing:
        known = ", ".join(sorted(runs))
        raise ValueError(f"Unknown run key(s): {', '.join(missing)}. Known: {known}")
    return names


def resolve_model(run_config: dict[str, Any]) -> str | None:
    env_name = run_config.get("model_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name]
    return run_config.get("default_model")


def condition_names(raw: str) -> list[str]:
    if raw == "both":
        return ["clean", "deranged"]
    return [raw]


def shot_seed(condition: dict[str, Any], shot: int) -> int | None:
    seeds = condition.get("seeds")
    if isinstance(seeds, dict):
        value = seeds.get(str(shot))
        if value is not None:
            return int(value)
    return None


def replace_arg(args: list[str], flag: str, value: str) -> list[str]:
    if flag in args:
        idx = args.index(flag)
        if idx + 1 >= len(args):
            raise ValueError(f"Flag {flag} has no value in base_args")
        return args[: idx + 1] + [value] + args[idx + 2 :]
    return args + [flag, value]


def build_command(
    *,
    python_exe: str,
    default_script: Path,
    programs_file: Path,
    output_root: Path,
    run_key: str,
    run_config: dict[str, Any],
    condition_name: str,
    condition: dict[str, Any],
    shot: int,
) -> tuple[list[str], Path]:
    runner = run_config.get("runner", "program_synth")
    script = (REPO_ROOT / run_config.get("script", str(default_script))).resolve()
    output_template = condition["output_template"]
    output_path = output_root / output_template.format(
        run=run_key,
        condition=condition_name,
        shot=shot,
    )

    command = [
        python_exe,
        str(script),
        "--programs",
        str(programs_file),
    ]

    model = resolve_model(run_config)
    if model:
        command.extend(["--model", model])

    command.extend(str(part) for part in run_config.get("base_args", []))
    command.extend(str(part) for part in condition.get("extra_args", []))

    seed = shot_seed(condition, shot)
    if seed is not None:
        command = replace_arg(command, "--seed", str(seed))

    if runner == "program_synth":
        command.extend(
            [
                "--in-context-examples",
                str(shot),
                "--output",
                str(output_path),
            ]
        )
    elif runner == "evo2_h100":
        command.extend(
            [
                "--shots",
                str(shot),
                "--conditions",
                condition_name,
                "--output-dir",
                str(output_path.parent),
            ]
        )
    else:
        raise ValueError(f"Unsupported runner for {run_key}: {runner!r}")

    return command, output_path


def build_evo2_h100_command(
    *,
    python_exe: str,
    default_script: Path,
    programs_file: Path,
    output_root: Path,
    run_key: str,
    run_config: dict[str, Any],
    condition_name: str,
    condition: dict[str, Any],
    shots: list[int],
) -> tuple[list[str], Path]:
    if not shots:
        raise ValueError("Evo2 H100 command requires at least one shot")

    script = (REPO_ROOT / run_config.get("script", str(default_script))).resolve()
    first_output_path = output_root / condition["output_template"].format(
        run=run_key,
        condition=condition_name,
        shot=shots[0],
    )
    output_dir = first_output_path.parent

    command = [
        python_exe,
        str(script),
        "--programs",
        str(programs_file),
    ]

    model = resolve_model(run_config)
    if model:
        command.extend(["--model", model])

    command.extend(str(part) for part in run_config.get("base_args", []))
    command.extend(str(part) for part in condition.get("extra_args", []))
    command.append("--shots")
    command.extend(str(shot) for shot in shots)
    command.append("--conditions")
    command.append(condition_name)
    command.extend(["--output-dir", str(output_dir)])
    return command, first_output_path


def build_program_synth_multi_command(
    *,
    python_exe: str,
    default_script: Path,
    programs_file: Path,
    output_root: Path,
    run_key: str,
    run_config: dict[str, Any],
    condition_name: str,
    condition: dict[str, Any],
    shots: list[int],
) -> tuple[list[str], Path]:
    if not shots:
        raise ValueError("program_synth multi-shot command requires at least one shot")

    script = (REPO_ROOT / run_config.get("script", str(default_script))).resolve()
    output_template = condition["output_template"]
    if "{shot}" not in output_template:
        raise ValueError(
            f"{run_key}/{condition_name} output_template must contain '{{shot}}' "
            "to use the multi-shot runner"
        )

    first_output_path = output_root / output_template.format(
        run=run_key,
        condition=condition_name,
        shot=shots[0],
    )
    output_pattern = output_root / output_template

    command = [
        python_exe,
        str(script),
        "--programs",
        str(programs_file),
    ]

    model = resolve_model(run_config)
    if model:
        command.extend(["--model", model])

    command.extend(str(part) for part in run_config.get("base_args", []))
    command.extend(str(part) for part in condition.get("extra_args", []))
    command.append("--shots")
    command.extend(str(shot) for shot in shots)
    command.extend(["--output", str(output_pattern)])
    return command, first_output_path


def validate_required_env(run_key: str, run_config: dict[str, Any], *, execute: bool) -> None:
    required = run_config.get("requires_env", [])
    missing = [name for name in required if not os.environ.get(name)]
    if missing and execute:
        raise RuntimeError(
            f"{run_key} requires environment variable(s): {', '.join(missing)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run", nargs="+", help="Run key(s) to include. Defaults to all.")
    parser.add_argument(
        "--condition",
        choices=("clean", "deranged", "both"),
        default="both",
        help="Condition to build/run. Defaults to both.",
    )
    parser.add_argument("--shots", type=int, nargs="+", help="Optional shot subset.")
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Output root for generated result JSONs. Defaults to config default_output_root.",
    )
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run commands instead of printing them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    default_script = (REPO_ROOT / config["program_synth"]).resolve()
    programs_file = (REPO_ROOT / config["programs"]).resolve()
    output_root = (
        args.output_root
        if args.output_root is not None
        else REPO_ROOT / config.get("default_output_root", "results/generated_runs")
    ).resolve()

    commands: list[tuple[list[str], Path]] = []
    for run_key in selected_runs(config, args.run):
        run_config = config["runs"][run_key]
        validate_required_env(run_key, run_config, execute=args.execute)
        conditions = run_config.get("conditions", {})
        runner = run_config.get("runner", "program_synth")
        for cond_name in condition_names(args.condition):
            if cond_name not in conditions:
                raise ValueError(f"{run_key} has no condition {cond_name!r}")
            condition = conditions[cond_name]
            shots = list(condition.get("shots", []))
            if args.shots is not None:
                wanted = set(args.shots)
                shots = [shot for shot in shots if shot in wanted]
            if not shots:
                continue
            if runner == "evo2_h100":
                commands.append(
                    build_evo2_h100_command(
                        python_exe=args.python_exe,
                        default_script=default_script,
                        programs_file=programs_file,
                        output_root=output_root,
                        run_key=run_key,
                        run_config=run_config,
                        condition_name=cond_name,
                        condition=condition,
                        shots=shots,
                    )
                )
                continue
            if runner == "program_synth" and not condition.get("seeds"):
                commands.append(
                    build_program_synth_multi_command(
                        python_exe=args.python_exe,
                        default_script=default_script,
                        programs_file=programs_file,
                        output_root=output_root,
                        run_key=run_key,
                        run_config=run_config,
                        condition_name=cond_name,
                        condition=condition,
                        shots=shots,
                    )
                )
                continue
            for shot in shots:
                commands.append(
                    build_command(
                        python_exe=args.python_exe,
                        default_script=default_script,
                        programs_file=programs_file,
                        output_root=output_root,
                        run_key=run_key,
                        run_config=run_config,
                        condition_name=cond_name,
                        condition=condition,
                        shot=shot,
                    )
                )

    if not commands:
        raise ValueError("No commands selected")

    for command, output_path in commands:
        if args.execute:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            print("+ " + shlex.join(command), flush=True)
            subprocess.run(command, check=True, cwd=REPO_ROOT)
        else:
            print(shlex.join(command))


if __name__ == "__main__":
    main()
