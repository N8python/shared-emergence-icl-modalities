"""Program evaluation harness using curated DSL transformations.

This script loads a language model and evaluates its ability to reproduce
bitstring transformations described by DSL programs. Prompts are generated
using few-shot examples drawn from the DSL itself, and results are written
per task with aggregate statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from tqdm import tqdm

try:
    from evo2 import Evo2
except ImportError:  # pragma: no cover - handled at runtime
    Evo2 = None

try:
    from mlx_lm import batch_generate, load
except ImportError:  # pragma: no cover - optional for the evo backend
    batch_generate = None
    load = None
from dsl import few_shot


EVO2_CANONICAL_MODEL_NAMES = {
    "evo2_40b",
    "evo2_7b",
    "evo2_20b",
    "evo2_40b_base",
    "evo2_7b_base",
    "evo2_1b_base",
    "evo2_7b_262k",
    "evo2_7b_microviridae",
}


def resolve_evo2_model_name(model_name: str) -> str:
    """Map public HF repo ids to the short names accepted by the evo2 package."""
    if model_name in EVO2_CANONICAL_MODEL_NAMES:
        return model_name
    if model_name.startswith("arcinstitute/"):
        short_name = model_name.rsplit("/", 1)[-1]
        if short_name in EVO2_CANONICAL_MODEL_NAMES:
            return short_name
    return model_name


@dataclass
class TrialResult:
    prompt: str
    query: str
    expected: str
    prediction_raw: str
    prediction: str
    correct: bool
    edit_distance: int

    def to_json(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "query": self.query,
            "expected": self.expected,
            "prediction_raw": self.prediction_raw,
            "prediction": self.prediction,
            "correct": self.correct,
            "edit_distance": self.edit_distance,
        }


@dataclass
class TaskResult:
    index: int
    program: Sequence[str]
    description: Optional[str]
    trials: List[TrialResult]

    def accuracy(self) -> float:
        if not self.trials:
            return 0.0
        correct = sum(1 for trial in self.trials if trial.correct)
        return correct / len(self.trials)

    def average_edit_distance(self) -> float:
        if not self.trials:
            return 0.0
        return statistics.mean(trial.edit_distance for trial in self.trials)

    def to_json(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "program": list(self.program),
            "description": self.description,
            "trials": [trial.to_json() for trial in self.trials],
            "total_trials": len(self.trials),
            "correct": sum(1 for trial in self.trials if trial.correct),
            "accuracy": self.accuracy(),
            "average_edit_distance": self.average_edit_distance(),
        }


NUCLEOTIDES = ("A", "C", "G", "T")


@dataclass
class PromptConfig:
    newline_sep: str
    apply_sep: str
    map_fn: Optional[Callable[[str], str]] = None
    decode_map: Optional[Dict[str, str]] = None
    token_size: int = 1


def default_prompt_config() -> PromptConfig:
    return PromptConfig(newline_sep="\n", apply_sep="->")


def make_evo_prompt_config(rng: random.Random) -> PromptConfig:
    zero_code, one_code, newline_sep = rng.sample(NUCLEOTIDES, 3)
    apply_sep = ""

    encode_map = {"0": zero_code, "1": one_code}
    decode_map = {value: key for key, value in encode_map.items()}

    def map_fn(bits: str, *, _encode=encode_map) -> str:
        try:
            return "".join(_encode[char] for char in bits)
        except KeyError as exc:
            raise ValueError("map_fn can only encode bitstrings containing '0' and '1'") from exc

    return PromptConfig(
        newline_sep=newline_sep,
        apply_sep=apply_sep,
        map_fn=map_fn,
        decode_map=decode_map,
        token_size=1,
    )

# --- Global cache of working batch sizes, keyed by model identity ---
# Thread-safe because some users hit this from multiple workers.
import threading
_EVO2_BS_CACHE: Dict[str, int] = {}
_EVO2_BS_LOCK = threading.Lock()

def _model_key(evo_model: Optional["Evo2"]) -> str:
    # Try to get a stable identifier for the loaded checkpoint.
    # Fallback to id() to keep it working even if no name attr exists.
    for attr in ("model_name", "name", "checkpoint", "ckpt_name"):
        if evo_model is not None and hasattr(evo_model, attr):
            v = getattr(evo_model, attr)
            if isinstance(v, str) and v:
                return f"{attr}:{v}"
    return f"id:{id(evo_model)}"

def _maybe_empty_cuda_cache() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        # Best-effort; ignore if torch not present or any other issue.
        pass


def patch_torch_load_for_evo2() -> None:
    """Allow official Evo2 .pt checkpoints to load under newer torch defaults."""
    try:
        import torch
    except Exception:
        return

    original_load = torch.load
    if getattr(original_load, "_evo2_weights_patch", False):
        return

    def load_with_legacy_default(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    load_with_legacy_default._evo2_weights_patch = True  # type: ignore[attr-defined]
    torch.load = load_with_legacy_default  # type: ignore[assignment]

def _is_oom_exception(exc: BaseException) -> bool:
    msg = (str(exc) or "").lower()
    # Cover common OOM signatures across PyTorch/CUDA/cuFFT/FA/TransformerEngine
    return any(
        key in msg
        for key in [
            "out of memory",
            "oom",
            "cuda error 2",               # cudaErrorMemoryAllocation
            "cudamemoryerror",
            "cufft",
            "cublas",
            "resourceexhausted",
            "failed to allocate",
            "std::bad_alloc",
        ]
    )

def _run_evo_generate(
    evo_model: "Evo2",
    prompts: List[str],
    n_tokens: int,
    temperature: float,
    top_k: int,
) -> List[str]:
    """Single evo2 generate call for a list of prompts; returns list[str]."""
    output = evo_model.generate(
        prompt_seqs=prompts,
        n_tokens=n_tokens,
        temperature=temperature,
        top_k=top_k,
        verbose=0,
    )

    sequences: Optional[Sequence[str]]
    if hasattr(output, "sequences"):
        sequences = getattr(output, "sequences")
    elif isinstance(output, dict):
        sequences = output.get("sequences")  # type: ignore[assignment]
    elif isinstance(output, tuple) and output:
        sequences = output[0]  # type: ignore[assignment]
    else:
        sequences = output  # type: ignore[assignment]

    if sequences is None:
        raise ValueError("evo2.generate did not return sequences")
    return [str(s) for s in sequences]

def generate_evo_completions(
    evo_model: Optional["Evo2"],
    trial_specs: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[str]:
    """
    OOM-aware batching for Evo2:
      - tries cached working batch size first (if any) per model
      - on OOM: halves batch size and retries, down to 1
      - remembers the last working size on success
      - if even bs=1 OOMs, records error in each spec and returns [""] * N
    """
    if not trial_specs:
        return []
    if evo_model is None:
        raise RuntimeError("Evo backend requires the evo2 library to be installed")

    token_size = max(int(spec.get("token_size", 1)) for spec in trial_specs)
    n_tokens = max_tokens * max(token_size, 1)
    prompts = [spec["prompt"] for spec in trial_specs]
    N = len(prompts)

    # Determine starting batch size: cached (if any) else full N.
    key = _model_key(evo_model)
    with _EVO2_BS_LOCK:
        start_bs = _EVO2_BS_CACHE.get(key, N)
    start_bs = max(1, min(start_bs, N))

    # We’ll attempt batch sizes: start_bs, then halve on OOM until 1.
    # On success, we store the working size (possibly larger than cache if start_bs==N and it worked).
    bs = start_bs
    while bs >= 1:
        try:
            # Process in chunks of size `bs`, preserving order.
            outputs: List[str] = [""] * N
            for start in range(0, N, bs):
                end = min(start + bs, N)
                chunk = prompts[start:end]
                chunk_out = _run_evo_generate(
                    evo_model=evo_model,
                    prompts=chunk,
                    n_tokens=n_tokens,
                    temperature=0.0,
                    top_k=1,
                )
                if len(chunk_out) != len(chunk):
                    raise ValueError("Mismatch between evo2 outputs and prompts")

                outputs[start:end] = chunk_out

            # Success: remember this batch size as known-good.
            with _EVO2_BS_LOCK:
                _EVO2_BS_CACHE[key] = bs
            return outputs

        except Exception as exc:
            if _is_oom_exception(exc):
                if bs == 1:
                    break
                # Halve batch size and retry from the beginning.
                _maybe_empty_cuda_cache()
                bs = math.ceil(bs / 2)
                continue
            else:
                # Non-OOM failure: propagate original behavior (surface to caller).
                for spec in trial_specs:
                    spec["error"] = exc
                return [""] * N

    # If we got here, even bs=1 failed with OOM.
    err = RuntimeError("Out of memory: even batch_size=1 failed")
    for spec in trial_specs:
        spec["error"] = err
    return [""] * N

def decode_evo_prediction(
    raw_text: str,
    *,
    decode_map: Dict[str, str],
    token_size: int,
) -> str:
    filtered = [ch for ch in raw_text.upper() if ch in NUCLEOTIDES]
    if not filtered:
        return ""

    bits: List[str] = []
    for i in range(0, len(filtered) - token_size + 1, token_size):
        chunk = "".join(filtered[i : i + token_size])
        bit = decode_map.get(chunk)
        if bit is None:
            break
        bits.append(bit)

    return "".join(bits)


def permute_labels_for_ablation(outputs: Sequence[str], rng: random.Random) -> List[str]:
    outputs = list(outputs)
    if len(outputs) <= 1:
        return outputs

    best_outputs = list(outputs)
    best_matches = len(outputs)
    indices = list(range(len(outputs)))
    for _ in range(256):
        rng.shuffle(indices)
        candidate = [outputs[idx] for idx in indices]
        matches = sum(
            original == shuffled
            for original, shuffled in zip(outputs, candidate)
        )
        if matches < best_matches:
            best_outputs = candidate
            best_matches = matches
            if best_matches == 0:
                return best_outputs

    if best_outputs == outputs and len(set(outputs)) > 1:
        best_outputs = outputs[1:] + outputs[:1]
    return best_outputs


def ablate_prompt_labels(
    prompt: str,
    *,
    prompt_config: PromptConfig,
    in_context_examples: int,
    query_input: str,
    rng: random.Random,
) -> str:
    if in_context_examples <= 1:
        return prompt

    newline_sep = prompt_config.newline_sep
    apply_sep = prompt_config.apply_sep

    if apply_sep:
        lines = prompt.split(newline_sep) if newline_sep else prompt.splitlines()
        if len(lines) <= 1:
            return prompt
        context_lines = lines[:-1]
        query_line = lines[-1]
        inputs: List[str] = []
        outputs: List[str] = []
        for line in context_lines:
            if apply_sep not in line:
                return prompt
            input_part, output_part = line.split(apply_sep, 1)
            inputs.append(input_part)
            outputs.append(output_part)
        if len(outputs) <= 1:
            return prompt
        outputs = permute_labels_for_ablation(outputs, rng)
        new_context_lines = [
            f"{inp}{apply_sep}{out}" for inp, out in zip(inputs, outputs)
        ]
        if newline_sep:
            return newline_sep.join(new_context_lines) + newline_sep + query_line
        return "".join(new_context_lines) + query_line

    expected_segment_length = len(query_input)
    if expected_segment_length <= 0:
        return prompt

    if newline_sep:
        lines = prompt.split(newline_sep)
        if len(lines) <= 1:
            return prompt
        context_lines = lines[:-1]
        query_line = lines[-1]
        inputs = []
        outputs = []
        for line in context_lines:
            if len(line) < 2 * expected_segment_length:
                split_index = len(line) // 2
            else:
                split_index = expected_segment_length
            if split_index <= 0:
                return prompt
            inputs.append(line[:split_index])
            outputs.append(line[split_index:])
        if len(outputs) <= 1:
            return prompt
        outputs = permute_labels_for_ablation(outputs, rng)
        new_context_lines = [
            f"{inp}{out}" for inp, out in zip(inputs, outputs)
        ]
        return newline_sep.join(new_context_lines) + newline_sep + query_line

    context_total_len = in_context_examples * 2 * expected_segment_length
    if context_total_len <= 0:
        return prompt
    if len(prompt) < context_total_len + expected_segment_length:
        return prompt
    context_blob = prompt[:context_total_len]
    query_line = prompt[context_total_len:]
    inputs = []
    outputs = []
    segment_size = 2 * expected_segment_length
    for idx in range(in_context_examples):
        start = idx * segment_size
        line = context_blob[start : start + segment_size]
        if len(line) != segment_size:
            return prompt
        inputs.append(line[:expected_segment_length])
        outputs.append(line[expected_segment_length:])
    if len(outputs) <= 1:
        return prompt
    outputs = permute_labels_for_ablation(outputs, rng)
    new_context_blob = "".join(
        f"{inp}{out}" for inp, out in zip(inputs, outputs)
    )
    return new_context_blob + query_line

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a language model on DSL programs using few-shot prompts."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model identifier: pass an mlx_lm checkpoint for the mlx backend "
            "or an Evo 2 checkpoint name for the evo backend (default evo2_7b)."
        ),
    )
    parser.add_argument(
        "--programs",
        required=True,
        help="Path to JSONL file with programs ({\"program\": [...]} per line).",
    )
    parser.add_argument(
        "--output",
        default="program_eval_results.json",
        help="Path to write JSON results (default: program_eval_results.json).",
    )
    parser.add_argument(
        "--in-context-examples",
        type=int,
        default=8,
        help="Number of few-shot examples to include per prompt (default: 8).",
    )
    parser.add_argument(
        "--trials-per-program",
        type=int,
        default=128,
        help="How many prompts to sample per program (default: 128).",
    )
    parser.add_argument(
        "--bit-length",
        type=int,
        default=8,
        help="Bitstring length to use when generating examples (default: 8).",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate (default: bit length).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--backend",
        choices=("mlx", "evo"),
        default="mlx",
        help="Generation backend to use (default: mlx).",
    )
    parser.add_argument(
        "--per-trial-random",
        action="store_true",
        help=(
            "Resample the Evo nucleotide encoding for every trial instead of once per task. "
            "Only supported with the evo backend."
        ),
    )
    parser.add_argument(
        "--ablate-labels",
        action="store_true",
        help="Derange the in-context output labels while preserving the query and expected output.",
    )
    return parser.parse_args()


def load_programs(path: Path) -> List[Dict[str, Any]]:
    programs: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {line}") from exc
            if "program" not in entry:
                raise ValueError(f"Missing 'program' field on line {line_no}")
            program = entry["program"]
            if not isinstance(program, list) or not all(
                isinstance(op, str) for op in program
            ):
                raise ValueError(
                    f"Program on line {line_no} must be a list of operation names"
                )
            programs.append(entry)
    if not programs:
        raise ValueError("No programs loaded from provided JSONL file")
    return programs


def normalize_prediction(output: str, bit_length: int) -> str:
    digits = [ch for ch in output if ch in "01"]
    return "".join(digits)[:bit_length]


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    previous_row = list(range(len_b + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i]
        for j, char_b in enumerate(b, start=1):
            substitution_cost = 0 if char_a == char_b else 1
            current_row.append(
                min(
                    current_row[-1] + 1,
                    previous_row[j] + 1,
                    previous_row[j - 1] + substitution_cost,
                )
            )
        previous_row = current_row

    return previous_row[-1]


def evaluate_task(
    model,
    tokenizer,
    program: Sequence[str],
    *,
    description: Optional[str],
    trials_per_program: int,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    backend: str,
    per_trial_random: bool,
    ablate_labels: bool,
    progress: Optional[tqdm] = None,
) -> TaskResult:
    trials: List[TrialResult] = []

    def build_prompt_config() -> PromptConfig:
        if backend == "evo":
            config = make_evo_prompt_config(random)
            if config.decode_map is None:
                raise ValueError("Evo backend requires a decode map")
            return config
        return default_prompt_config()

    if backend == "evo" and per_trial_random:
        prompt_configs = [build_prompt_config() for _ in range(trials_per_program)]
    else:
        base_config = build_prompt_config()
        prompt_configs = [base_config] * trials_per_program

    trial_specs: List[Dict[str, Any]] = []
    for trial_idx in range(trials_per_program):
        prompt_config = prompt_configs[trial_idx]
        map_argument = prompt_config.map_fn
        if per_trial_random and prompt_config.map_fn is not None:
            map_argument = [prompt_config.map_fn] * (in_context_examples + 1)
        prompt, query_input, expected_output, _, _ = few_shot(
            program,
            in_context_examples,
            prompt_config.newline_sep,
            prompt_config.apply_sep,
            bit_length=bit_length,
            map_fn=map_argument,
        )
        if ablate_labels:
            prompt = ablate_prompt_labels(
                prompt,
                prompt_config=prompt_config,
                in_context_examples=in_context_examples,
                query_input=query_input,
                rng=random,
            )
        trial_specs.append(
            {
                "prompt": prompt,
                "query": query_input,
                "expected": expected_output,
            }
        )
        if backend == "evo" and prompt_config.decode_map is not None:
            trial_specs[-1]["decode_map"] = prompt_config.decode_map
            trial_specs[-1]["token_size"] = prompt_config.token_size

    if backend == "evo":
        outputs = generate_evo_completions(
            model,
            trial_specs,
            max_tokens=max_new_tokens,
        )
    else:
        prompts = [tokenizer.encode(spec["prompt"]) for spec in trial_specs]
        if prompts:
            batch = batch_generate(
                model,
                tokenizer,
                prompts,
                verbose=False,
                max_tokens=max_new_tokens,
            )
            outputs = batch.texts
        else:
            outputs = []

    for spec, raw_output in zip(trial_specs, outputs):
        if backend == "evo":
            decode_map = spec.get("decode_map", {})
            token_size = spec.get("token_size", 1)
            decoded_output = decode_evo_prediction(
                raw_output,
                decode_map=decode_map,
                token_size=token_size,
            )
            prediction = normalize_prediction(decoded_output, bit_length)
            error = spec.get("error")
            if error and not raw_output:
                raw_output = f"<error: {error}>"
        else:
            prediction = normalize_prediction(raw_output, bit_length)
        distance = edit_distance(prediction, spec["expected"])
        trials.append(
            TrialResult(
                prompt=spec["prompt"],
                query=spec["query"],
                expected=spec["expected"],
                prediction_raw=raw_output,
                prediction=prediction,
                correct=prediction == spec["expected"],
                edit_distance=distance,
            )
        )
        if progress is not None:
            progress.update(1)

    return TaskResult(
        index=-1,  # will be set by caller
        program=program,
        description=description,
        trials=trials,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    programs_path = Path(args.programs)
    output_path = Path(args.output)

    program_entries = load_programs(programs_path)

    if args.trials_per_program <= 0:
        raise ValueError("trials_per_program must be positive")
    if args.in_context_examples < 0:
        raise ValueError("in_context_examples cannot be negative")
    if args.bit_length <= 0:
        raise ValueError("bit_length must be positive")

    backend = args.backend

    if backend == "mlx":
        if not args.model:
            raise ValueError("--model is required when using the mlx backend")
        if load is None:
            raise ImportError("The mlx backend requires mlx_lm to be installed.")
        model, tokenizer = load(args.model)
    elif backend == "evo":
        if Evo2 is None:
            raise ImportError(
                "The evo backend requires the evo2 package. Install it with `pip install evo2`."
            )
        evo_model_name = args.model or "evo2_7b"
        patch_torch_load_for_evo2()
        model = Evo2(resolve_evo2_model_name(evo_model_name))
        tokenizer = None
        if args.model is None:
            args.model = evo_model_name
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    if args.per_trial_random and backend != "evo":
        raise ValueError("--per-trial-random is only supported with the evo backend")

    max_tokens = args.max_new_tokens or args.bit_length
    if max_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    task_results: List[TaskResult] = []

    total_predictions = len(program_entries) * args.trials_per_program
    with tqdm(total=total_predictions, desc="Evaluations") as progress:
        for idx, entry in enumerate(program_entries):
            program = entry["program"]
            description = entry.get("description")
            task = evaluate_task(
                model,
                tokenizer,
                program,
                description=description,
                trials_per_program=args.trials_per_program,
                in_context_examples=args.in_context_examples,
                bit_length=args.bit_length,
                max_new_tokens=max_tokens,
                backend=backend,
                per_trial_random=args.per_trial_random,
                ablate_labels=args.ablate_labels,
                progress=progress,
            )
            task.index = idx
            task_results.append(task)

    accuracies = [task.accuracy() for task in task_results]
    mean_accuracy = statistics.mean(accuracies) if accuracies else 0.0
    if len(accuracies) > 1:
        std_dev = statistics.stdev(accuracies)
        stderr = std_dev / math.sqrt(len(accuracies))
    else:
        stderr = 0.0

    avg_edit_distances = [task.average_edit_distance() for task in task_results]
    mean_edit_distance = (
        statistics.mean(avg_edit_distances) if avg_edit_distances else 0.0
    )
    if len(avg_edit_distances) > 1:
        edit_std = statistics.stdev(avg_edit_distances)
        edit_stderr = edit_std / math.sqrt(len(avg_edit_distances))
    else:
        edit_stderr = 0.0

    output_data = {
        "config": {
            "model": args.model,
            "programs_file": str(programs_path),
            "in_context_examples": args.in_context_examples,
            "trials_per_program": args.trials_per_program,
            "bit_length": args.bit_length,
            "max_new_tokens": max_tokens,
            "seed": args.seed,
            "per_trial_random": args.per_trial_random,
            "ablate_labels": args.ablate_labels,
        },
        "overall": {
            "mean_accuracy": mean_accuracy,
            "stderr": stderr,
            "mean_edit_distance": mean_edit_distance,
            "edit_distance_stderr": edit_stderr,
        },
        "tasks": [task.to_json() for task in task_results],
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)

    print(f"Wrote results for {len(task_results)} programs to {output_path}")
    print(
        f"Mean accuracy: {mean_accuracy:.4f} (stderr {stderr:.4f})"
    )
    print(
        f"Mean edit distance: {mean_edit_distance:.4f} (stderr {edit_stderr:.4f})"
    )


if __name__ == "__main__":
    main()
