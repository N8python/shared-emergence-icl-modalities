"""Program evaluation harness using curated DSL transformations.

This script loads a language model and evaluates its ability to reproduce
bitstring transformations described by DSL programs. Prompts are generated
using few-shot examples drawn from the DSL itself, and results are written
per task with aggregate statistics.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import random
import re
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from huggingface_hub import snapshot_download
import requests
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import batch_generate, load
from mlx_lm.generate import BatchGenerator
from mlx_lm.models import cache as kv_cache
from mlx_lm.utils import load_model
from dsl import few_shot
from music_backend import generate_music_completions, make_music_trial_spec
from timemoe_mlx import DEFAULT_TIMEMOE_200M_PATH, KVCache, TimeMoeMLX
from timesfm_mlx import DEFAULT_TIMESFM_2_5_PATH, TimesFm2_5MLX


REPO_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass
class TrialResult:
    prompt: str
    query: str
    expected: str
    prediction_raw: Optional[str] = None
    prediction: Optional[str] = None
    correct: Optional[bool] = None
    edit_distance: Optional[int] = None
    perplexity: Optional[float] = None
    bit_accuracy: Optional[float] = None
    few_shot_examples: Optional[List[Dict[str, str]]] = None
    query_unaltered: Optional[str] = None

    def to_json(self) -> Dict[str, Any]:
        data = {
            "prompt": self.prompt,
            "query": self.query,
            "expected": self.expected,
            "prediction_raw": self.prediction_raw,
            "prediction": self.prediction,
            "correct": self.correct,
            "edit_distance": self.edit_distance,
        }
        if self.perplexity is not None:
            data["perplexity"] = self.perplexity
        if self.bit_accuracy is not None:
            data["bit_accuracy"] = self.bit_accuracy
        if self.few_shot_examples is not None:
            data["few_shot_examples"] = self.few_shot_examples
        if self.query_unaltered is not None:
            data["query_unaltered"] = self.query_unaltered
        return data


@dataclass
class TaskResult:
    index: int
    program: Sequence[str]
    description: Optional[str]
    trials: List[TrialResult]

    def accuracy(self) -> Optional[float]:
        scored = [trial for trial in self.trials if trial.correct is not None]
        if not scored:
            return None
        correct = sum(1 for trial in scored if trial.correct)
        return correct / len(scored)

    def average_edit_distance(self) -> Optional[float]:
        distances = [trial.edit_distance for trial in self.trials if trial.edit_distance is not None]
        if not distances:
            return None
        return statistics.mean(distances)

    def average_perplexity(self) -> Optional[float]:
        perplexities = [trial.perplexity for trial in self.trials if trial.perplexity is not None]
        if not perplexities:
            return None
        return statistics.mean(perplexities)

    def average_bit_accuracy(self) -> Optional[float]:
        accuracies = [trial.bit_accuracy for trial in self.trials if trial.bit_accuracy is not None]
        if not accuracies:
            return None
        return statistics.mean(accuracies)

    def to_json(self) -> Dict[str, Any]:
        scored = [trial for trial in self.trials if trial.correct is not None]
        correct_count = (
            sum(1 for trial in scored if trial.correct) if scored else None
        )
        data = {
            "index": self.index,
            "program": list(self.program),
            "description": self.description,
            "trials": [trial.to_json() for trial in self.trials],
            "total_trials": len(self.trials),
            "correct": correct_count,
            "accuracy": self.accuracy(),
            "average_edit_distance": self.average_edit_distance(),
        }
        avg_ppl = self.average_perplexity()
        if avg_ppl is not None:
            data["average_perplexity"] = avg_ppl
        avg_bit_acc = self.average_bit_accuracy()
        if avg_bit_acc is not None:
            data["average_bit_accuracy"] = avg_bit_acc
        return data


NUCLEOTIDES = ("A", "C", "G", "T")
CANONICAL_AMINO_ACIDS = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)
PROGEN2_CANONICAL_AMINO_ACID_TOKEN_IDS = {
    "A": 5,
    "C": 7,
    "D": 8,
    "E": 9,
    "F": 10,
    "G": 11,
    "H": 12,
    "I": 13,
    "K": 14,
    "L": 15,
    "M": 16,
    "N": 17,
    "P": 19,
    "Q": 20,
    "R": 21,
    "S": 22,
    "T": 23,
    "V": 25,
    "W": 26,
    "Y": 28,
}
DIGIT_TOKENS = tuple(str(i) for i in range(10))
IMAGEGPT_ROW_WIDTH = 32
IMAGEGPT_COLOR_VOCAB_SIZE = 512
IMAGEGPT_SOS_TOKEN_ID = 512
MNIST_ROW_WIDTH = 28
MNIST_IMAGE_ROWS = 28
MNIST_PIXEL_VOCAB_SIZE = 32
MNIST_LABEL_OFFSET = 32
MNIST_NUM_LABELS = 10
MNIST_PAD_TOKEN_ID = 0
MNIST_ZERO_TOKEN_ID = 16
MNIST_ONE_TOKEN_ID = 31
MNIST_VOCAB_SIZE = 259
CHESSGPT_CONTEXT_LIMIT = 1023
TIMEMOE_CONTEXT_LIMIT = 4096
TIMEMOE_SEPARATOR_VALUE = 0.0
TIMEMOE_ZERO_VALUE = -1.0
TIMEMOE_ONE_VALUE = 1.0
TIMEMOE_DEFAULT_PULSE_WIDTH = 8
TIMEMOE_DEFAULT_REPEAT_WIDTH = 4
TIMESFM_CONTEXT_LIMIT = 16384
TIMESFM_PATCH_LENGTH = 32
TIMESFM_HORIZON_LENGTH = 128
TIMESFM_SEPARATOR_VALUE = 0.0
TIMESFM_ZERO_VALUE = -1.0
TIMESFM_ONE_VALUE = 1.0
PROTEIN_CONTEXT_LIMIT = 1024
RITA_CONTEXT_LIMIT = PROTEIN_CONTEXT_LIMIT
EVO_API_KEY = os.environ.get("EVO_API_KEY")
EVO_ENDPOINT = "https://health.api.nvidia.com/v1/biology/arc/evo2-40b/generate"
EVO_MAX_WORKERS = 8


@dataclass
class PromptConfig:
    newline_sep: str
    apply_sep: str
    map_fn: Optional[Callable[[str], str]] = None
    decode_map: Optional[Dict[str, str]] = None
    token_size: int = 1


def default_prompt_config() -> PromptConfig:
    return PromptConfig(newline_sep="\n", apply_sep="->")


def raw_bits_prompt_config() -> PromptConfig:
    return PromptConfig(newline_sep="", apply_sep="")


def _random_nucleotide_string(length: int, rng: random.Random) -> str:
    return "".join(rng.choice(NUCLEOTIDES) for _ in range(length))


def make_evo_prompt_config(
    rng: random.Random,
    *,
    drop_nucleotide: Optional[str] = None,
) -> PromptConfig:
    nucleotide_pool = NUCLEOTIDES
    if drop_nucleotide is not None:
        drop_token = drop_nucleotide.upper()
        nucleotide_pool = tuple(
            token for token in NUCLEOTIDES if token != drop_token
        )
    if len(nucleotide_pool) < 3:
        raise ValueError(
            "Nucleotide encoding requires at least 3 nucleotides in the pool"
        )

    zero_code, one_code, newline_sep = rng.sample(nucleotide_pool, 3)
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


def make_rita_prompt_config(rng: random.Random) -> PromptConfig:
    zero_code, one_code, newline_sep = rng.sample(CANONICAL_AMINO_ACIDS, 3)
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


def make_progen2_prompt_config(rng: random.Random) -> PromptConfig:
    zero_code, one_code, newline_sep = rng.sample(
        tuple(PROGEN2_CANONICAL_AMINO_ACID_TOKEN_IDS),
        3,
    )
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


def make_lm_shuffle_prompt_config(rng: random.Random) -> PromptConfig:
    zero_code, one_code, newline_sep = rng.sample(DIGIT_TOKENS, 3)
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


def make_nextterm_digit_prompt_config(rng: random.Random) -> PromptConfig:
    zero_code, one_code = rng.sample(DIGIT_TOKENS[1:], 2)
    newline_sep = ","
    apply_sep = ","

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


def make_symbol_prompt_config(
    rng: random.Random,
    symbols: str,
    *,
    raw_bits: bool = False,
) -> PromptConfig:
    unique_symbols: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        key = symbol.upper()
        if key in seen:
            continue
        seen.add(key)
        unique_symbols.append(symbol)

    if len(unique_symbols) < 3:
        raise ValueError("Symbol encoding requires at least 3 unique symbols")

    if raw_bits:
        zero_code, one_code = rng.sample(unique_symbols, 2)
        newline_sep = ""
    else:
        zero_code, one_code, newline_sep = rng.sample(unique_symbols, 3)
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


def request_evo_completion(prompt: str, *, num_tokens: int) -> str:
    if not EVO_API_KEY:
        raise RuntimeError(
            "Evo2 backend requires EVO_API_KEY in the environment."
        )
    response = requests.post(
        EVO_ENDPOINT,
        headers={"Authorization": f"Bearer {EVO_API_KEY}"},
        json={
            "sequence": prompt,
            "num_tokens": num_tokens,
            "top_k": 1,
        }
    )
    response.raise_for_status()

    if "application/json" in response.headers.get("Content-Type", ""):
        payload = response.json()
        return payload.get("sequence", "")
    return response.text


def generate_evo_completions(
    trial_specs: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[str]:
    if not trial_specs:
        return []

    outputs: List[str] = ["" for _ in trial_specs]
    max_workers = min(EVO_MAX_WORKERS, len(trial_specs))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(request_evo_completion, spec["prompt"], num_tokens=max_tokens): idx
            for idx, spec in enumerate(trial_specs)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                outputs[idx] = future.result()
            except Exception as exc:  # pragma: no cover
                trial_specs[idx]["error"] = exc
                outputs[idx] = ""

    return outputs


def decode_mapped_prediction(
    raw_text: str,
    *,
    decode_map: Dict[str, str],
    token_size: int,
) -> str:
    if token_size <= 0:
        raise ValueError("token_size must be positive")

    if not decode_map:
        return ""

    normalized_text = raw_text.upper()
    allowed_tokens = {token.upper() for token in decode_map.keys()}
    filtered = [ch for ch in normalized_text if ch in allowed_tokens]
    if not filtered:
        return ""

    bits: List[str] = []
    for i in range(0, len(filtered) - token_size + 1, token_size):
        chunk = "".join(filtered[i : i + token_size])
        bit = decode_map.get(chunk)
        if bit is None:
            lower_chunk = chunk.lower()
            bit = decode_map.get(lower_chunk)
        if bit is None:
            break
        bits.append(bit)

    return "".join(bits)


def extract_context_examples(
    prompt: str,
    *,
    newline_sep: str,
    apply_sep: str,
) -> List[Tuple[str, str]]:
    if not apply_sep:
        raise ValueError("Naive backend requires a non-empty apply separator")

    if newline_sep:
        lines = prompt.split(newline_sep)
    else:
        lines = prompt.splitlines()

    if lines and lines[-1].rstrip().endswith(apply_sep):
        candidate_lines = lines[:-1]
    else:
        candidate_lines = lines

    examples: List[Tuple[str, str]] = []
    for line in candidate_lines:
        if not line:
            continue
        if apply_sep not in line:
            continue
        input_part, output_part = line.split(apply_sep, 1)
        examples.append((input_part.strip(), output_part.strip()))

    return examples


def naive_identity_predict(query: str, context_examples: Sequence[Tuple[str, str]]) -> str:
    return query


def naive_modal_predict(query: str, context_examples: Sequence[Tuple[str, str]]) -> str:
    if not context_examples:
        return query

    outputs = [example_output for _, example_output in context_examples]
    if not outputs:
        return query

    most_common_output, _ = Counter(outputs).most_common(1)[0]
    return most_common_output


NAIVE_BASELINES: Dict[str, Callable[[str, Sequence[Tuple[str, str]]], str]] = {
    "identity": naive_identity_predict,
    "modal": naive_modal_predict,
}


def _decode_segment(
    segment: str,
    *,
    decode_map: Optional[Dict[str, str]],
    token_size: int,
) -> str:
    if not segment:
        return segment
    if decode_map is None:
        return segment
    if token_size <= 0:
        token_size = 1

    bits: List[str] = []
    for idx in range(0, len(segment), token_size):
        token = segment[idx : idx + token_size]
        bit = decode_map.get(token)
        if bit is None:
            token_upper = token.upper()
            token_lower = token.lower()
            bit = decode_map.get(token_upper)
            if bit is None:
                bit = decode_map.get(token_lower)
        if bit is None:
            return segment
        bits.append(bit)

    return "".join(bits)


def collect_few_shot_examples(
    prompt: str,
    *,
    prompt_config: PromptConfig,
    bit_length: int,
) -> List[Dict[str, str]]:
    if prompt_config.apply_sep:
        pairs = extract_context_examples(
            prompt,
            newline_sep=prompt_config.newline_sep,
            apply_sep=prompt_config.apply_sep,
        )
        return [{"input": inp, "output": out} for inp, out in pairs]

    if prompt_config.newline_sep:
        lines = prompt.split(prompt_config.newline_sep)
    else:
        lines = prompt.splitlines()

    if not lines:
        return []

    context_lines = lines[:-1]
    if not context_lines:
        return []

    examples: List[Dict[str, str]] = []
    token_size = prompt_config.token_size or 1
    expected_segment_length = bit_length * token_size if bit_length > 0 else 0
    for line in context_lines:
        if not line:
            continue
        if expected_segment_length and len(line) >= 2 * expected_segment_length:
            split_index = expected_segment_length
        else:
            split_index = len(line) // 2
        if split_index == 0:
            continue
        input_segment = line[:split_index]
        output_segment = line[split_index:]
        decoded_input = _decode_segment(
            input_segment,
            decode_map=prompt_config.decode_map,
            token_size=token_size,
        )
        decoded_output = _decode_segment(
            output_segment,
            decode_map=prompt_config.decode_map,
            token_size=token_size,
        )
        examples.append({"input": decoded_input, "output": decoded_output})

    return examples


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

    if apply_sep and newline_sep == apply_sep:
        parts = prompt.split(newline_sep)
        has_trailing_sep = bool(parts) and parts[-1] == ""
        if has_trailing_sep:
            parts = parts[:-1]
        if len(parts) < (2 * in_context_examples + 1):
            return prompt
        context_parts = parts[: 2 * in_context_examples]
        query_parts = parts[2 * in_context_examples :]
        if len(context_parts) % 2 != 0 or not query_parts:
            return prompt
        inputs = context_parts[0::2]
        outputs = context_parts[1::2]
        if len(outputs) <= 1:
            return prompt
        outputs = permute_labels_for_ablation(outputs, rng)
        new_parts: List[str] = []
        for input_part, output_part in zip(inputs, outputs):
            new_parts.extend([input_part, output_part])
        new_parts.extend(query_parts)
        result = newline_sep.join(new_parts)
        if has_trailing_sep:
            result += newline_sep
        return result

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
            "Identifier or path for the model to load "
            "(ignored for evo/naive; optional for timemoe)."
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
        help="Number of few-shot examples to include per prompt (default: 7).",
    )
    parser.add_argument(
        "--trials-per-program",
        type=int,
        default=8,
        help="How many prompts to sample per program (default: 8).",
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
        "--batch-programs",
        action="store_true",
        help=(
            "mlx backend: batch generation across ALL programs at once (one "
            "BatchGenerator over every program's trials) instead of a separate "
            "batch_generate per program. Much faster on GPU. Additive: default "
            "off reproduces the per-program path exactly."
        ),
    )
    parser.add_argument(
        "--batch-size-prefill",
        type=int,
        default=8,
        help="prefill_batch_size for the mlx BatchGenerator (default: 8).",
    )
    parser.add_argument(
        "--batch-size-completion",
        type=int,
        default=32,
        help="completion_batch_size for the mlx BatchGenerator (default: 32).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Evaluate several in-context-example counts in ONE process (the model "
            "is loaded once and reused across shots). Overrides --in-context-examples. "
            "Requires --output to contain '{shot}', e.g. results/cell/clean/cell_ic{shot}.json."
        ),
    )
    parser.add_argument(
        "--ablate-labels",
        action="store_true",
        help="Shuffle outputs among in-context examples to ablate label alignment.",
    )
    parser.add_argument(
        "--invert-binary-logits",
        action="store_true",
        help="Swap the logits for tokens '0' and '1' during perplexity evaluation (mlx only).",
    )
    parser.add_argument(
        "--uniform-binary-logits",
        action="store_true",
        help="Use uniform 50/50 logits for tokens '0' and '1' during perplexity evaluation.",
    )
    parser.add_argument(
        "--ppl-select",
        action="store_true",
        help="Select the answer with lowest perplexity among all bitstrings (mlx only).",
    )
    parser.add_argument(
        "--perplexity-eval",
        action="store_true",
        help="Compute perplexity of the correct answer instead of generating (mlx only).",
    )
    parser.add_argument(
        "--perplexity-batch-size",
        type=int,
        default=32,
        help="Batch size for perplexity evaluation (default: 32).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--backend",
        choices=(
            "mlx",
            "evo",
            "naive",
            "imagegpt",
            "mnist",
            "chess",
            "music",
            "timemoe",
            "timesfm",
            "rita",
            "protein",
        ),
        default="mlx",
        help="Generation backend to use (default: mlx).",
    )
    parser.add_argument(
        "--protein-format",
        choices=("rita", "progen2"),
        default="rita",
        help=(
            "Protein bitstring prompt family. rita uses raw amino-acid prompts; "
            "progen2 prepends the ProGen2 sequence-start token '1' "
            "(default: rita)."
        ),
    )
    parser.add_argument(
        "--imagegpt-row-layout",
        choices=("left-pad", "middle-pad", "dense"),
        default="left-pad",
        help=(
            "ImageGPT bitstring row format: left-pad uses [pad][input][output]; "
            "middle-pad uses [input][pad][output]; dense packs two "
            "[input][output] pairs per 32-token row (default: left-pad)."
        ),
    )
    parser.add_argument(
        "--chess-encoding",
        choices=("toggle", "signal-wipe", "signal2-wipe", "signal2-wipe-sep"),
        default="toggle",
        help=(
            "Chess bit encoding. toggle uses one bit per reversible knight-toggle ply; "
            "signal-wipe uses fixed starting-square signal moves plus deterministic "
            "wipe moves, for 0.5 bits/ply; signal2-wipe encodes two bits on each "
            "signal ply plus deterministic wipe moves, for 1 bit/ply; "
            "signal2-wipe-sep inserts an unadorned signal-wipe 00 separator between "
            "input/output/example segments (default: toggle)."
        ),
    )
    parser.add_argument(
        "--chess-layout",
        choices=(
            "continuous",
            "example-reset",
            "segment-reset",
            "bit-gamelet",
            "bit-gamelet-pair",
        ),
        default="continuous",
        help=(
            "Chess prompt layout. continuous is one game; example-reset starts "
            "a new game for each input/output pair and query; segment-reset "
            "starts separate games for every input and output segment; bit-gamelet "
            "uses one tiny one-ply game per bit; bit-gamelet-pair uses one two-ply "
            "game per bit (default: continuous)."
        ),
    )
    parser.add_argument(
        "--chess-decode",
        choices=("constrained", "greedy"),
        default="greedy",
        help=(
            "Chess generation route. constrained scores the legal bit moves/chunks; "
            "greedy uses normal raw PGN generation and decodes legal toggle/signal "
            "moves afterward (default: greedy)."
        ),
    )
    parser.add_argument(
        "--music-encoding",
        choices=("pitch", "octave", "rhythm", "2bit-pitch", "slice-pitch"),
        default="pitch",
        help=(
            "Music bit encoding. pitch uses two random melodic pitches for 0/1; "
            "octave uses the same pitch class an octave apart; rhythm uses "
            "short/long durations on a fixed pitch; 2bit-pitch uses four pitches "
            "for two bits per note; slice-pitch targets ROLL/time-slice models "
            "(one 8-slice line per bitstring) (default: pitch)."
        ),
    )
    parser.add_argument(
        "--music-decode",
        choices=("constrained", "greedy"),
        default="constrained",
        help=(
            "Music generation route. constrained teacher-forces the canonical "
            "note bytes and argmaxes logits at the discriminating byte; greedy "
            "free-runs PNO bytes and parses notes tolerantly (default: constrained)."
        ),
    )
    parser.add_argument(
        "--naive-baseline",
        choices=tuple(NAIVE_BASELINES.keys()),
        default="identity",
        help="Naive baseline variant to use when the backend is naive (default: identity).",
    )
    parser.add_argument(
        "--lmshuffle",
        action="store_true",
        help="Randomize digit encodings for 0/1/newline when using the mlx backend.",
    )
    parser.add_argument(
        "--nucleotide-bits",
        action="store_true",
        help=(
            "Use randomized nucleotide encodings for 0/1/newline "
            "(default for evo; optional for mlx)."
        ),
    )
    parser.add_argument(
        "--symbol-bits",
        type=str,
        default=None,
        help=(
            "Use randomized symbols for 0/1/newline from the supplied single-character pool "
            "(mlx only)."
        ),
    )
    parser.add_argument(
        "--drop-nucleotide",
        type=str.lower,
        choices=("a", "c", "t", "g"),
        default=None,
        help=(
            "Exclude one nucleotide from the randomized nucleotide encoding pool "
            "(choices: a/c/t/g)."
        ),
    )
    parser.add_argument(
        "--per-trial-random",
        action="store_true",
        help="Resample the encoding for every trial instead of once per task.",
    )
    parser.add_argument(
        "--raw-bits",
        action="store_true",
        help="Concatenate raw input/output bitstrings with no separators in the prompt.",
    )
    parser.add_argument(
        "--force-binary-tokens",
        action="store_true",
        help="Constrain MLX generation to the active 0/1 tokens for each prompt.",
    )
    parser.add_argument(
        "--nextterm-digits",
        action="store_true",
        help=(
            "Use NextTerm-native comma-separated integer prompts. For each "
            "few-shot prompt, sample two digits from 1-9 as prompt-local "
            "encodings for bits 0 and 1."
        ),
    )
    parser.add_argument(
        "--timemoe-encoding",
        choices=("symbol", "sine-pulse", "repeat-symbol"),
        default="symbol",
        help=(
            "TimeMoE bit encoding. symbol uses one scalar per bit; sine-pulse "
            "uses signed half-sine chunks plus zero separator plateaus; "
            "repeat-symbol repeats each -1/0/+1 scalar chunk-wise "
            "(default: symbol)."
        ),
    )
    parser.add_argument(
        "--timemoe-pulse-width",
        type=int,
        default=TIMEMOE_DEFAULT_PULSE_WIDTH,
        help=(
            "Chunk width for --timemoe-encoding sine-pulse, also used as the "
            "separator plateau width (default: 8)."
        ),
    )
    parser.add_argument(
        "--timemoe-repeat-width",
        type=int,
        default=TIMEMOE_DEFAULT_REPEAT_WIDTH,
        help=(
            "Chunk width for --timemoe-encoding repeat-symbol, also used as the "
            "zero separator plateau width (default: 4)."
        ),
    )
    parser.add_argument(
        "--timesfm-dtype",
        choices=("fp32", "bf16"),
        default="fp32",
        help="TimesFM MLX weight dtype to use (default: fp32).",
    )
    parser.add_argument(
        "--timesfm-encoding",
        choices=("constant-patch", "sine-lobe", "sine-lobe-2bit", "sine-lobe-4bit"),
        default="constant-patch",
        help=(
            "TimesFM bit encoding. constant-patch uses flat -1/+1 patches; "
            "sine-lobe uses signed half-sine patches with exact zero endpoints; "
            "sine-lobe-2bit and sine-lobe-4bit pack multiple signed sub-lobes "
            "into each patch "
            "(default: constant-patch)."
        ),
    )
    parser.add_argument(
        "--timesfm-layout",
        choices=("sep-input-output", "input-zero-output-zero"),
        default="sep-input-output",
        help=(
            "TimesFM prompt layout. sep-input-output uses [zero][input][output] "
            "per example and [zero][query input]; input-zero-output-zero uses "
            "[input][zero][output][zero] examples and [query input][zero] before "
            "generation (default: sep-input-output)."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Include the few-shot examples for each trial in the JSON output.",
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


def invert_binary_logits(logits, zero_id: int, one_id: int):
    if zero_id == one_id:
        return logits
    low_id = min(zero_id, one_id)
    high_id = max(zero_id, one_id)
    parts = [
        logits[..., :low_id],
        logits[..., high_id : high_id + 1],
    ]
    if low_id + 1 < high_id:
        parts.append(logits[..., low_id + 1 : high_id])
    parts.extend(
        [
            logits[..., low_id : low_id + 1],
            logits[..., high_id + 1 :],
        ]
    )
    return mx.concatenate(parts, axis=-1)


def resolve_allowed_generation_token_ids(
    tokenizer,
    decode_map: Optional[Dict[str, str]],
) -> List[int]:
    symbols = list(decode_map.keys()) if decode_map else ["0", "1"]
    token_ids: List[int] = []
    for symbol in symbols:
        encoded = tokenizer.encode(symbol, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Constrained decoding requires single-token symbols, got {symbol!r} -> {encoded}"
            )
        token_ids.append(int(encoded[0]))
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("Constrained decoding requires distinct token ids for the allowed symbols")
    return token_ids


def validate_progen2_amino_acid_tokens(tokenizer) -> None:
    for residue, expected_id in PROGEN2_CANONICAL_AMINO_ACID_TOKEN_IDS.items():
        encoded = tokenizer.encode(residue, add_special_tokens=False)
        if encoded != [expected_id]:
            raise ValueError(
                "ProGen2 amino-acid token mismatch: "
                f"{residue!r} encoded as {encoded}, expected [{expected_id}]"
            )


def progen2_allowed_token_ids(decode_map: Optional[Dict[str, str]]) -> List[int]:
    if not decode_map:
        raise ValueError("ProGen2 protein backend requires a decode map")
    token_ids: List[int] = []
    for residue in decode_map.keys():
        try:
            token_ids.append(PROGEN2_CANONICAL_AMINO_ACID_TOKEN_IDS[residue])
        except KeyError as exc:
            raise ValueError(
                f"ProGen2 backend cannot constrain non-canonical residue {residue!r}"
            ) from exc
    return token_ids


def make_allowed_token_logits_processor(
    allowed_token_ids: List[int],
    vocab_size: int,
):
    allowed_bias_by_size = {}

    def processor(_, logits):
        live_vocab_size = int(logits.shape[-1])
        if max(allowed_token_ids) >= live_vocab_size:
            raise ValueError(
                "Allowed token id is outside the live logits vocabulary: "
                f"allowed={allowed_token_ids}, logits_vocab={live_vocab_size}"
            )
        allowed_bias = allowed_bias_by_size.get(live_vocab_size)
        if allowed_bias is None:
            allowed_bias = mx.full((1, live_vocab_size), -1e9, dtype=mx.float32)
            allowed_bias[:, allowed_token_ids] = 0.0
            allowed_bias_by_size[live_vocab_size] = allowed_bias
        return logits + allowed_bias

    return processor


def bits_to_imagegpt_tokens(
    bits: str,
    *,
    zero_token: int,
    one_token: int,
) -> List[int]:
    tokens: List[int] = []
    for bit in bits:
        if bit == "0":
            tokens.append(zero_token)
        elif bit == "1":
            tokens.append(one_token)
        else:
            raise ValueError(f"ImageGPT bit encoding only supports 0/1, got {bit!r}")
    return tokens


def decode_imagegpt_tokens(
    tokens: Sequence[int],
    *,
    zero_token: int,
    one_token: int,
) -> str:
    bits: List[str] = []
    for token in tokens:
        token = int(token)
        if token == zero_token:
            bits.append("0")
        elif token == one_token:
            bits.append("1")
    return "".join(bits)


def make_model_file_class_loader(model_path: Path) -> Callable[[dict], Tuple[Any, Any]]:
    model_path = model_path.expanduser().resolve()

    def get_model_classes(config: dict) -> Tuple[Any, Any]:
        model_file = config.get("model_file")
        if not model_file:
            raise ValueError(f"{model_path} config does not define model_file")

        module_path = model_path / str(model_file)
        if not module_path.exists():
            raise FileNotFoundError(module_path)

        module_name = f"local_mlx_model_{model_path.name.replace('-', '_')}"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not import custom MLX model from {module_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Model, module.ModelArgs

    return get_model_classes


def resolve_mlx_model_path(model: str | Path) -> Path:
    """Return a local path for either a filesystem model path or a Hub repo ID."""
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path.resolve()
    return Path(
        snapshot_download(
            str(model),
            allow_patterns=[
                "*.json",
                "*.jsonl",
                "*.jinja",
                "*.model",
                "*.py",
                "*.safetensors",
                "*.tiktoken",
                "*.txt",
                "model*.safetensors",
                "tokenizer.model",
                "tiktoken.model",
            ],
        )
    )


def make_imagegpt_trial_spec(
    program: Sequence[str],
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    imagegpt_row_layout: str,
    ablate_labels: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if bit_length <= 0:
        raise ValueError("ImageGPT backend requires positive bit_length")
    if imagegpt_row_layout == "dense" and (4 * bit_length) != IMAGEGPT_ROW_WIDTH:
        raise ValueError(
            "Dense ImageGPT row layout requires 4 * bit_length == 32 so two "
            "input/output pairs exactly fill one row"
        )
    if imagegpt_row_layout != "dense" and bit_length > IMAGEGPT_ROW_WIDTH // 2:
        raise ValueError(
            "ImageGPT row layout requires bit_length <= 16 so input+output fit in one 32-token row"
        )
    if max_new_tokens != bit_length:
        raise ValueError(
            "ImageGPT row layout expects max_new_tokens to equal bit_length"
        )

    row_pad = IMAGEGPT_ROW_WIDTH - (2 * bit_length)
    if imagegpt_row_layout not in {"left-pad", "middle-pad", "dense"}:
        raise ValueError(f"Unsupported ImageGPT row layout: {imagegpt_row_layout}")
    zero_token, one_token, pad_token = random.sample(
        range(IMAGEGPT_COLOR_VOCAB_SIZE),
        3,
    )

    bit_prompt, query_input, expected_output, _, query_input_raw = few_shot(
        program,
        in_context_examples,
        "\n",
        "->",
        bit_length=bit_length,
    )
    lines = bit_prompt.split("\n") if bit_prompt else []
    context_lines = lines[:-1] if lines else []
    context_examples: List[Tuple[str, str]] = []
    for line in context_lines:
        if "->" not in line:
            raise ValueError(f"Malformed few-shot line for ImageGPT backend: {line!r}")
        input_bits, output_bits = line.split("->", 1)
        if len(input_bits) != bit_length or len(output_bits) != bit_length:
            raise ValueError("ImageGPT backend expected fixed-length context bitstrings")
        context_examples.append((input_bits, output_bits))

    if ablate_labels and len(context_examples) > 1:
        inputs = [inp for inp, _ in context_examples]
        outputs = [out for _, out in context_examples]
        outputs = permute_labels_for_ablation(outputs, random)
        context_examples = list(zip(inputs, outputs))

    def append_imagegpt_example(input_bits: str, output_bits: str) -> None:
        prompt_tokens.extend(
            bits_to_imagegpt_tokens(
                input_bits,
                zero_token=zero_token,
                one_token=one_token,
            )
        )
        prompt_tokens.extend(
            bits_to_imagegpt_tokens(
                output_bits,
                zero_token=zero_token,
                one_token=one_token,
            )
        )

    prompt_tokens: List[int] = [IMAGEGPT_SOS_TOKEN_ID]
    query_left_pad = 0

    if imagegpt_row_layout == "dense":
        pair_width = 2 * bit_length
        context_idx = 0
        while context_idx + 1 < len(context_examples):
            append_imagegpt_example(*context_examples[context_idx])
            append_imagegpt_example(*context_examples[context_idx + 1])
            context_idx += 2

        if context_idx < len(context_examples):
            append_imagegpt_example(*context_examples[context_idx])
        else:
            prompt_tokens.extend([pad_token] * pair_width)
            query_left_pad = pair_width
    else:
        for input_bits, output_bits in context_examples:
            if imagegpt_row_layout == "left-pad":
                prompt_tokens.extend([pad_token] * row_pad)
            prompt_tokens.extend(
                bits_to_imagegpt_tokens(
                    input_bits,
                    zero_token=zero_token,
                    one_token=one_token,
                )
            )
            if imagegpt_row_layout == "middle-pad":
                prompt_tokens.extend([pad_token] * row_pad)
            prompt_tokens.extend(
                bits_to_imagegpt_tokens(
                    output_bits,
                    zero_token=zero_token,
                    one_token=one_token,
                )
            )

        if imagegpt_row_layout == "left-pad":
            prompt_tokens.extend([pad_token] * row_pad)
            query_left_pad = row_pad
    prompt_tokens.extend(
        bits_to_imagegpt_tokens(
            query_input,
            zero_token=zero_token,
            one_token=one_token,
        )
    )
    if imagegpt_row_layout == "middle-pad":
        prompt_tokens.extend([pad_token] * row_pad)

    if len(prompt_tokens) + max_new_tokens - 1 > 1024:
        raise ValueError(
            "ImageGPT prompt exceeds 32x32 context: "
            f"prompt={len(prompt_tokens)}, max_new_tokens={max_new_tokens}"
        )

    display_lines = [f"{inp}->{out}" for inp, out in context_examples]
    display_lines.append(f"{query_input}->")
    prompt_display = (
        f"<imagegpt zero={zero_token} one={one_token} pad={pad_token} "
        f"row_layout={imagegpt_row_layout} row_pad={row_pad}>\n"
        + "\n".join(display_lines)
    )

    spec: Dict[str, Any] = {
        "prompt": prompt_display,
        "imagegpt_prompt_tokens": prompt_tokens,
        "imagegpt_zero_token": zero_token,
        "imagegpt_one_token": one_token,
        "imagegpt_pad_token": pad_token,
        "imagegpt_row_layout": imagegpt_row_layout,
        "imagegpt_row_pad": row_pad,
        "imagegpt_row_left_pad": query_left_pad,
        "imagegpt_dense_examples_per_row": 2 if imagegpt_row_layout == "dense" else None,
        "query": query_input,
        "expected": expected_output,
        "expected_mapped": expected_output,
        "query_unaltered": query_input_raw,
    }
    if verbose:
        spec["few_shot_examples"] = [
            {"input": inp, "output": out} for inp, out in context_examples
        ]
    return spec


def generate_imagegpt_completions(
    model,
    trial_specs: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[str]:
    if not trial_specs:
        return []

    outputs: List[Optional[str]] = [None] * len(trial_specs)
    by_prompt_len: Dict[int, List[int]] = {}
    for idx, spec in enumerate(trial_specs):
        by_prompt_len.setdefault(len(spec["imagegpt_prompt_tokens"]), []).append(idx)

    for indices in by_prompt_len.values():
        grouped_specs = [trial_specs[idx] for idx in indices]
        prompts = [spec["imagegpt_prompt_tokens"] for spec in grouped_specs]
        prompt_array = mx.array(prompts, dtype=mx.int32)

        allowed_bias_np = np.full(
            (len(grouped_specs), IMAGEGPT_COLOR_VOCAB_SIZE),
            -1.0e9,
            dtype=np.float32,
        )
        for row_idx, spec in enumerate(grouped_specs):
            allowed_bias_np[row_idx, int(spec["imagegpt_zero_token"])] = 0.0
            allowed_bias_np[row_idx, int(spec["imagegpt_one_token"])] = 0.0
        allowed_bias = mx.array(allowed_bias_np)

        cache = kv_cache.make_prompt_cache(model)
        logits = model(prompt_array, cache=cache)[:, -1, :]
        generated: List[List[int]] = [[] for _ in grouped_specs]

        for step_idx in range(max_tokens):
            constrained_logits = logits.astype(mx.float32) + allowed_bias
            next_tokens = mx.argmax(constrained_logits, axis=-1).astype(mx.int32)
            mx.eval(next_tokens)
            next_token_list = [int(token) for token in next_tokens.tolist()]
            for row_idx, token in enumerate(next_token_list):
                generated[row_idx].append(token)

            if step_idx + 1 < max_tokens:
                logits = model(next_tokens.reshape(-1, 1), cache=cache)[:, -1, :]

        for row_idx, spec in enumerate(grouped_specs):
            outputs[indices[row_idx]] = decode_imagegpt_tokens(
                generated[row_idx],
                zero_token=spec["imagegpt_zero_token"],
                one_token=spec["imagegpt_one_token"],
            )

    if any(output is None for output in outputs):
        raise RuntimeError("Internal error: incomplete ImageGPT generation outputs")
    return outputs


def bits_to_mnist_tokens(bits: str) -> List[int]:
    tokens: List[int] = []
    for bit in bits:
        if bit == "0":
            tokens.append(MNIST_ZERO_TOKEN_ID)
        elif bit == "1":
            tokens.append(MNIST_ONE_TOKEN_ID)
        else:
            raise ValueError(f"MNIST bit encoding only supports 0/1, got {bit!r}")
    return tokens


def decode_mnist_tokens(tokens: Sequence[int]) -> str:
    bits: List[str] = []
    for token in tokens:
        token = int(token)
        if token == MNIST_ZERO_TOKEN_ID:
            bits.append("0")
        elif token == MNIST_ONE_TOKEN_ID:
            bits.append("1")
    return "".join(bits)


def make_mnist_trial_spec(
    program: Sequence[str],
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    ablate_labels: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if bit_length <= 0:
        raise ValueError("MNIST backend requires positive bit_length")
    if bit_length != 8:
        raise ValueError("MNIST row layout currently expects bit_length == 8")
    if max_new_tokens != bit_length:
        raise ValueError("MNIST row layout expects max_new_tokens to equal bit_length")

    row_pad = MNIST_ROW_WIDTH - (2 * bit_length)
    if row_pad < 0:
        raise ValueError("MNIST row layout requires input+output to fit in one 28-token row")
    if in_context_examples > MNIST_IMAGE_ROWS - 1:
        raise ValueError(
            "MNIST row layout stays within a 28-row image, so shots must be <= 27"
        )

    digit_label = random.randrange(MNIST_NUM_LABELS)
    label_token = MNIST_LABEL_OFFSET + digit_label

    bit_prompt, query_input, expected_output, _, query_input_raw = few_shot(
        program,
        in_context_examples,
        "\n",
        "->",
        bit_length=bit_length,
    )
    lines = bit_prompt.split("\n") if bit_prompt else []
    context_lines = lines[:-1] if lines else []
    context_examples: List[Tuple[str, str]] = []
    for line in context_lines:
        if "->" not in line:
            raise ValueError(f"Malformed few-shot line for MNIST backend: {line!r}")
        input_bits, output_bits = line.split("->", 1)
        if len(input_bits) != bit_length or len(output_bits) != bit_length:
            raise ValueError("MNIST backend expected fixed-length context bitstrings")
        context_examples.append((input_bits, output_bits))

    if ablate_labels and len(context_examples) > 1:
        inputs = [inp for inp, _ in context_examples]
        outputs = [out for _, out in context_examples]
        outputs = permute_labels_for_ablation(outputs, random)
        context_examples = list(zip(inputs, outputs))

    prompt_tokens: List[int] = [label_token]
    for input_bits, output_bits in context_examples:
        prompt_tokens.extend([MNIST_PAD_TOKEN_ID] * row_pad)
        prompt_tokens.extend(bits_to_mnist_tokens(input_bits))
        prompt_tokens.extend(bits_to_mnist_tokens(output_bits))

    prompt_tokens.extend([MNIST_PAD_TOKEN_ID] * row_pad)
    prompt_tokens.extend(bits_to_mnist_tokens(query_input))

    max_canvas_tokens = 1 + (MNIST_IMAGE_ROWS * MNIST_ROW_WIDTH)
    if len(prompt_tokens) + max_new_tokens - 1 > max_canvas_tokens:
        raise ValueError(
            "MNIST prompt exceeds the 28x28 training canvas: "
            f"prompt={len(prompt_tokens)}, max_new_tokens={max_new_tokens}"
        )

    display_lines = [f"{inp}->{out}" for inp, out in context_examples]
    display_lines.append(f"{query_input}->")
    prompt_display = (
        f"<mnist label={digit_label} label_token={label_token} "
        f"zero={MNIST_ZERO_TOKEN_ID} one={MNIST_ONE_TOKEN_ID} "
        f"pad={MNIST_PAD_TOKEN_ID} row_pad={row_pad}>\n"
        + "\n".join(display_lines)
    )

    spec: Dict[str, Any] = {
        "prompt": prompt_display,
        "mnist_prompt_tokens": prompt_tokens,
        "mnist_label": digit_label,
        "mnist_label_token": label_token,
        "mnist_row_pad": row_pad,
        "query": query_input,
        "expected": expected_output,
        "expected_mapped": expected_output,
        "query_unaltered": query_input_raw,
    }
    if verbose:
        spec["few_shot_examples"] = [
            {"input": inp, "output": out} for inp, out in context_examples
        ]
    return spec


def generate_mnist_completions(
    model,
    trial_specs: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[str]:
    if not trial_specs:
        return []

    prompts = [spec["mnist_prompt_tokens"] for spec in trial_specs]
    logits_processors = [
        [
            make_allowed_token_logits_processor(
                [MNIST_ZERO_TOKEN_ID, MNIST_ONE_TOKEN_ID],
                MNIST_VOCAB_SIZE,
            )
        ]
        for _ in trial_specs
    ]

    gen = BatchGenerator(model, stop_tokens=None)
    uids = gen.insert(
        prompts,
        [max_tokens] * len(prompts),
        logits_processors=logits_processors,
    )
    token_results: Dict[int, List[int]] = {uid: [] for uid in uids}
    while responses := gen.next_generated():
        for response in responses:
            if response.finish_reason != "stop":
                token_results[response.uid].append(int(response.token))
    gen.close()

    outputs: List[str] = []
    for uid in uids:
        outputs.append(decode_mnist_tokens(token_results[uid]))
    return outputs


def bits_to_timemoe_values(bits: str) -> List[float]:
    values: List[float] = []
    for bit in bits:
        if bit == "0":
            values.append(TIMEMOE_ZERO_VALUE)
        elif bit == "1":
            values.append(TIMEMOE_ONE_VALUE)
        else:
            raise ValueError(f"TimeMoE bit encoding only supports 0/1, got {bit!r}")
    return values


def decode_timemoe_values(values: Sequence[float]) -> str:
    return "".join("1" if float(value) > 0.0 else "0" for value in values)


def timemoe_half_sine_pulse(width: int) -> List[float]:
    if width <= 0:
        raise ValueError("TimeMoE pulse width must be positive")
    return [math.sin(math.pi * (idx + 0.5) / width) for idx in range(width)]


def bits_to_timemoe_pulse_values(bits: str, *, pulse_width: int) -> List[float]:
    pulse = timemoe_half_sine_pulse(pulse_width)
    values: List[float] = []
    for bit in bits:
        if bit == "0":
            values.extend(-value for value in pulse)
        elif bit == "1":
            values.extend(pulse)
        else:
            raise ValueError(f"TimeMoE bit encoding only supports 0/1, got {bit!r}")
    return values


def bits_to_timemoe_repeat_values(bits: str, *, repeat_width: int) -> List[float]:
    if repeat_width <= 0:
        raise ValueError("TimeMoE repeat width must be positive")
    values: List[float] = []
    for bit in bits:
        if bit == "0":
            values.extend([TIMEMOE_ZERO_VALUE] * repeat_width)
        elif bit == "1":
            values.extend([TIMEMOE_ONE_VALUE] * repeat_width)
        else:
            raise ValueError(f"TimeMoE bit encoding only supports 0/1, got {bit!r}")
    return values


def decode_timemoe_chunk_mean_values(
    values: Sequence[float],
    *,
    bit_length: int,
    chunk_width: int,
) -> str:
    if chunk_width <= 0:
        raise ValueError("TimeMoE chunk width must be positive")
    bits: List[str] = []
    expected_values = bit_length * chunk_width
    usable_values = list(values)[:expected_values]
    for start in range(0, len(usable_values), chunk_width):
        chunk = usable_values[start : start + chunk_width]
        if len(chunk) < chunk_width:
            break
        bits.append("1" if sum(float(value) for value in chunk) > 0.0 else "0")
    return "".join(bits)


def timemoe_max_new_values(
    *,
    bit_length: int,
    timemoe_encoding: str,
    pulse_width: int,
    repeat_width: int,
) -> int:
    if timemoe_encoding == "symbol":
        return bit_length
    if timemoe_encoding == "sine-pulse":
        return bit_length * pulse_width
    if timemoe_encoding == "repeat-symbol":
        return bit_length * repeat_width
    raise ValueError(f"Unsupported TimeMoE encoding: {timemoe_encoding}")


def timemoe_prompt_value_count(
    *,
    in_context_examples: int,
    bit_length: int,
    timemoe_encoding: str,
    pulse_width: int,
    repeat_width: int,
) -> int:
    if timemoe_encoding == "symbol":
        unit = 1
        separator_width = 1
    elif timemoe_encoding == "sine-pulse":
        unit = pulse_width
        separator_width = pulse_width
    elif timemoe_encoding == "repeat-symbol":
        unit = repeat_width
        separator_width = repeat_width
    else:
        raise ValueError(f"Unsupported TimeMoE encoding: {timemoe_encoding}")
    return (
        in_context_examples * (separator_width + (2 * bit_length * unit))
        + separator_width
        + (bit_length * unit)
    )


def flatten_timemoe_cache(cache: Optional[KVCache]) -> List[mx.array]:
    if cache is None:
        return []
    arrays: List[mx.array] = []
    for item in cache:
        if item is None:
            continue
        arrays.extend(item)
    return arrays


def make_timemoe_trial_spec(
    program: Sequence[str],
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    timemoe_encoding: str,
    pulse_width: int,
    repeat_width: int,
    ablate_labels: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if bit_length <= 0:
        raise ValueError("TimeMoE backend requires positive bit_length")
    if pulse_width <= 0:
        raise ValueError("TimeMoE pulse width must be positive")
    if repeat_width <= 0:
        raise ValueError("TimeMoE repeat width must be positive")
    expected_new_values = timemoe_max_new_values(
        bit_length=bit_length,
        timemoe_encoding=timemoe_encoding,
        pulse_width=pulse_width,
        repeat_width=repeat_width,
    )
    if max_new_tokens != expected_new_values:
        raise ValueError(
            "TimeMoE backend expects max_new_tokens to equal generated scalar "
            f"values for the encoding ({expected_new_values})"
        )

    bit_prompt, query_input, expected_output, _, query_input_raw = few_shot(
        program,
        in_context_examples,
        "\n",
        "->",
        bit_length=bit_length,
    )
    lines = bit_prompt.split("\n") if bit_prompt else []
    context_lines = lines[:-1] if lines else []
    context_examples: List[Tuple[str, str]] = []
    for line in context_lines:
        if "->" not in line:
            raise ValueError(f"Malformed few-shot line for TimeMoE backend: {line!r}")
        input_bits, output_bits = line.split("->", 1)
        if len(input_bits) != bit_length or len(output_bits) != bit_length:
            raise ValueError("TimeMoE backend expected fixed-length context bitstrings")
        context_examples.append((input_bits, output_bits))

    if ablate_labels and len(context_examples) > 1:
        inputs = [inp for inp, _ in context_examples]
        outputs = [out for _, out in context_examples]
        outputs = permute_labels_for_ablation(outputs, random)
        context_examples = list(zip(inputs, outputs))

    prompt_values: List[float] = []
    for input_bits, output_bits in context_examples:
        if timemoe_encoding == "symbol":
            prompt_values.append(TIMEMOE_SEPARATOR_VALUE)
            prompt_values.extend(bits_to_timemoe_values(input_bits))
            prompt_values.extend(bits_to_timemoe_values(output_bits))
        elif timemoe_encoding == "sine-pulse":
            prompt_values.extend([TIMEMOE_SEPARATOR_VALUE] * pulse_width)
            prompt_values.extend(
                bits_to_timemoe_pulse_values(input_bits, pulse_width=pulse_width)
            )
            prompt_values.extend(
                bits_to_timemoe_pulse_values(output_bits, pulse_width=pulse_width)
            )
        elif timemoe_encoding == "repeat-symbol":
            prompt_values.extend([TIMEMOE_SEPARATOR_VALUE] * repeat_width)
            prompt_values.extend(
                bits_to_timemoe_repeat_values(input_bits, repeat_width=repeat_width)
            )
            prompt_values.extend(
                bits_to_timemoe_repeat_values(output_bits, repeat_width=repeat_width)
            )
        else:
            raise ValueError(f"Unsupported TimeMoE encoding: {timemoe_encoding}")

    if timemoe_encoding == "symbol":
        prompt_values.append(TIMEMOE_SEPARATOR_VALUE)
        prompt_values.extend(bits_to_timemoe_values(query_input))
    elif timemoe_encoding == "sine-pulse":
        prompt_values.extend([TIMEMOE_SEPARATOR_VALUE] * pulse_width)
        prompt_values.extend(
            bits_to_timemoe_pulse_values(query_input, pulse_width=pulse_width)
        )
    elif timemoe_encoding == "repeat-symbol":
        prompt_values.extend([TIMEMOE_SEPARATOR_VALUE] * repeat_width)
        prompt_values.extend(
            bits_to_timemoe_repeat_values(query_input, repeat_width=repeat_width)
        )
    else:
        raise ValueError(f"Unsupported TimeMoE encoding: {timemoe_encoding}")

    display_lines = [
        f"0 {input_bits} {output_bits}" for input_bits, output_bits in context_examples
    ]
    display_lines.append(f"0 {query_input} <generate {bit_length}>")
    prompt_display = (
        f"<timemoe zero={TIMEMOE_ZERO_VALUE:g} one={TIMEMOE_ONE_VALUE:g} "
        f"sep={TIMEMOE_SEPARATOR_VALUE:g} layout=sep-input-output "
        f"encoding={timemoe_encoding} pulse_width={pulse_width} "
        f"repeat_width={repeat_width}>\n"
        + "\n".join(display_lines)
    )

    spec: Dict[str, Any] = {
        "prompt": prompt_display,
        "timemoe_prompt_values": prompt_values,
        "timemoe_encoding": timemoe_encoding,
        "timemoe_pulse_width": pulse_width,
        "timemoe_repeat_width": repeat_width,
        "query": query_input,
        "expected": expected_output,
        "expected_mapped": expected_output,
        "query_unaltered": query_input_raw,
    }
    if verbose:
        spec["few_shot_examples"] = [
            {"input": inp, "output": out} for inp, out in context_examples
        ]
    return spec


def generate_timemoe_completions(
    model: TimeMoeMLX,
    trial_specs: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[str]:
    if not trial_specs:
        return []

    encodings = {spec["timemoe_encoding"] for spec in trial_specs}
    if len(encodings) != 1:
        raise ValueError("TimeMoE batched generation requires matching encodings")
    timemoe_encoding = next(iter(encodings))
    pulse_widths = {int(spec["timemoe_pulse_width"]) for spec in trial_specs}
    if len(pulse_widths) != 1:
        raise ValueError("TimeMoE batched generation requires matching pulse widths")
    pulse_width = next(iter(pulse_widths))
    repeat_widths = {int(spec["timemoe_repeat_width"]) for spec in trial_specs}
    if len(repeat_widths) != 1:
        raise ValueError("TimeMoE batched generation requires matching repeat widths")
    repeat_width = next(iter(repeat_widths))

    prompt_lengths = {
        len(spec["timemoe_prompt_values"]) for spec in trial_specs
    }
    if len(prompt_lengths) != 1:
        raise ValueError("TimeMoE batched generation requires equal prompt lengths")

    prompt_batch = [
        [[value] for value in spec["timemoe_prompt_values"]]
        for spec in trial_specs
    ]
    cur = mx.array(prompt_batch, dtype=mx.bfloat16)
    mx.eval(cur)

    cache: Optional[KVCache] = None
    raw_outputs: List[mx.array] = []
    if timemoe_encoding == "symbol":
        for step in range(max_tokens):
            step_input = cur if step == 0 else next_value
            predictions, cache = model.forward(
                step_input,
                cache=cache,
                use_cache=True,
                max_horizon_length=1,
            )
            raw_value = predictions[:, -1:, :1]
            next_value = mx.where(
                raw_value > 0.0,
                mx.array(TIMEMOE_ONE_VALUE, dtype=mx.bfloat16),
                mx.array(TIMEMOE_ZERO_VALUE, dtype=mx.bfloat16),
            )
            raw_outputs.append(raw_value)
            mx.eval(raw_value, next_value, *flatten_timemoe_cache(cache))
    elif timemoe_encoding == "repeat-symbol":
        for step in range(max_tokens):
            step_input = cur if step == 0 else next_value
            predictions, cache = model.forward(
                step_input,
                cache=cache,
                use_cache=True,
                max_horizon_length=1,
            )
            raw_value = predictions[:, -1:, :1]
            next_value = mx.where(
                raw_value > 0.0,
                mx.array(TIMEMOE_ONE_VALUE, dtype=mx.bfloat16),
                mx.array(TIMEMOE_ZERO_VALUE, dtype=mx.bfloat16),
            )
            raw_outputs.append(raw_value)
            mx.eval(raw_value, next_value, *flatten_timemoe_cache(cache))
    elif timemoe_encoding == "sine-pulse":
        pulse_values = mx.array(
            timemoe_half_sine_pulse(pulse_width),
            dtype=mx.bfloat16,
        )
        for step in range(max_tokens):
            step_input = cur if step == 0 else next_value
            phase_value = pulse_values[step % pulse_width].reshape(1, 1, 1)
            predictions, cache = model.forward(
                step_input,
                cache=cache,
                use_cache=True,
                max_horizon_length=1,
            )
            raw_value = predictions[:, -1:, :1]
            next_value = mx.where(raw_value > 0.0, phase_value, -phase_value)
            raw_outputs.append(raw_value)
            mx.eval(raw_value, next_value, *flatten_timemoe_cache(cache))
    else:
        raise ValueError(f"Unsupported TimeMoE encoding: {timemoe_encoding}")

    raw_array = mx.concatenate(raw_outputs, axis=1).astype(mx.float32)
    mx.eval(raw_array)
    raw_nested = raw_array.tolist()

    outputs: List[str] = []
    for spec, row in zip(trial_specs, raw_nested):
        values = [float(item[0]) for item in row]
        spec["timemoe_generated_values"] = values
        if timemoe_encoding == "symbol":
            spec["timemoe_feedback_values"] = [
                TIMEMOE_ONE_VALUE if value > 0.0 else TIMEMOE_ZERO_VALUE
                for value in values
            ]
            outputs.append(decode_timemoe_values(values))
        elif timemoe_encoding == "repeat-symbol":
            decoded = decode_timemoe_chunk_mean_values(
                values,
                bit_length=len(spec["expected"]),
                chunk_width=repeat_width,
            )
            spec["timemoe_feedback_values"] = bits_to_timemoe_repeat_values(
                decoded,
                repeat_width=repeat_width,
            )
            outputs.append(decoded)
        elif timemoe_encoding == "sine-pulse":
            decoded = decode_timemoe_chunk_mean_values(
                values,
                bit_length=len(spec["expected"]),
                chunk_width=pulse_width,
            )
            spec["timemoe_feedback_values"] = bits_to_timemoe_pulse_values(
                decoded,
                pulse_width=pulse_width,
            )
            outputs.append(decoded)
        else:
            raise ValueError(f"Unsupported TimeMoE encoding: {timemoe_encoding}")
    return outputs


def timesfm_sine_lobe(patch_length: int = TIMESFM_PATCH_LENGTH) -> List[float]:
    if patch_length <= 1:
        raise ValueError("TimesFM sine-lobe patch length must be greater than 1")
    values = [
        math.sin(math.pi * idx / (patch_length - 1))
        for idx in range(patch_length)
    ]
    values[0] = 0.0
    values[-1] = 0.0
    return values


def timesfm_bits_per_patch(timesfm_encoding: str) -> int:
    if timesfm_encoding == "sine-lobe-2bit":
        return 2
    if timesfm_encoding == "sine-lobe-4bit":
        return 4
    if timesfm_encoding in {"constant-patch", "sine-lobe"}:
        return 1
    raise ValueError(f"Unsupported TimesFM encoding: {timesfm_encoding}")


def timesfm_uses_sublobe_projection(timesfm_encoding: str) -> bool:
    return timesfm_bits_per_patch(timesfm_encoding) > 1


def timesfm_bit_patch_count(bit_length: int, timesfm_encoding: str) -> int:
    if bit_length <= 0:
        raise ValueError("TimesFM backend requires positive bit_length")
    bits_per_patch = timesfm_bits_per_patch(timesfm_encoding)
    if bit_length % bits_per_patch != 0:
        raise ValueError(
            f"TimesFM encoding {timesfm_encoding!r} requires bit_length to be "
            f"a multiple of {bits_per_patch}"
        )
    return bit_length // bits_per_patch


def bits_to_timesfm_patch_values(
    bits: str,
    *,
    timesfm_encoding: str,
    patch_length: int = TIMESFM_PATCH_LENGTH,
) -> List[float]:
    if timesfm_encoding == "sine-lobe":
        one_patch = timesfm_sine_lobe(patch_length)
        zero_patch = [-value for value in one_patch]
    elif timesfm_encoding == "constant-patch":
        one_patch = [TIMESFM_ONE_VALUE] * patch_length
        zero_patch = [TIMESFM_ZERO_VALUE] * patch_length
    elif timesfm_uses_sublobe_projection(timesfm_encoding):
        bits_per_patch = timesfm_bits_per_patch(timesfm_encoding)
        if patch_length % bits_per_patch != 0:
            raise ValueError(
                f"TimesFM {timesfm_encoding} requires patch_length to be divisible "
                f"by {bits_per_patch}"
            )
        if len(bits) % bits_per_patch != 0:
            raise ValueError(
                f"TimesFM {timesfm_encoding} requires bitstring length to be "
                f"a multiple of {bits_per_patch}"
            )
        sub_lobe = timesfm_sine_lobe(patch_length // bits_per_patch)
        values: List[float] = []
        for start in range(0, len(bits), bits_per_patch):
            for bit in bits[start : start + bits_per_patch]:
                if bit == "0":
                    values.extend(-value for value in sub_lobe)
                elif bit == "1":
                    values.extend(sub_lobe)
                else:
                    raise ValueError(
                        f"TimesFM bit encoding only supports 0/1, got {bit!r}"
                    )
        return values
    else:
        raise ValueError(f"Unsupported TimesFM encoding: {timesfm_encoding}")

    values: List[float] = []
    for bit in bits:
        if bit == "0":
            values.extend(zero_patch)
        elif bit == "1":
            values.extend(one_patch)
        else:
            raise ValueError(f"TimesFM bit encoding only supports 0/1, got {bit!r}")
    return values


def decode_timesfm_sublobe_projection_values(
    values: Sequence[float],
    *,
    bit_length: int,
    timesfm_encoding: str,
    patch_length: int = TIMESFM_PATCH_LENGTH,
) -> str:
    bits_per_patch = timesfm_bits_per_patch(timesfm_encoding)
    if bits_per_patch <= 1:
        raise ValueError(
            f"TimesFM encoding {timesfm_encoding!r} does not use sub-lobe decoding"
        )
    if patch_length % bits_per_patch != 0:
        raise ValueError(
            f"TimesFM {timesfm_encoding} requires patch_length to be divisible "
            f"by {bits_per_patch}"
        )
    bit_patches = timesfm_bit_patch_count(bit_length, timesfm_encoding)
    sub_length = patch_length // bits_per_patch
    sub_lobe = timesfm_sine_lobe(sub_length)
    usable_values = list(values)[: bit_patches * patch_length]
    bits: List[str] = []
    for patch_start in range(0, len(usable_values), patch_length):
        patch = usable_values[patch_start : patch_start + patch_length]
        if len(patch) < patch_length:
            break
        for sub_start in range(0, patch_length, sub_length):
            sub_values = patch[sub_start : sub_start + sub_length]
            score = sum(
                float(value) * template
                for value, template in zip(sub_values, sub_lobe)
            )
            bits.append("1" if score > 0.0 else "0")
    return "".join(bits[:bit_length])


def decode_timesfm_patch_mean_values(
    values: Sequence[float],
    *,
    bit_length: int,
    patch_length: int = TIMESFM_PATCH_LENGTH,
) -> str:
    bits: List[str] = []
    expected_values = bit_length * patch_length
    usable_values = list(values)[:expected_values]
    for start in range(0, len(usable_values), patch_length):
        chunk = usable_values[start : start + patch_length]
        if len(chunk) < patch_length:
            break
        bits.append("1" if sum(float(value) for value in chunk) > 0.0 else "0")
    return "".join(bits)


def timesfm_prompt_value_count(
    *,
    in_context_examples: int,
    bit_length: int,
    timesfm_encoding: str,
    timesfm_layout: str,
    patch_length: int = TIMESFM_PATCH_LENGTH,
) -> int:
    bit_patches = timesfm_bit_patch_count(bit_length, timesfm_encoding)
    if timesfm_layout == "sep-input-output":
        prompt_patches = (
            in_context_examples * (1 + (2 * bit_patches))
            + 1
            + bit_patches
        )
    elif timesfm_layout == "input-zero-output-zero":
        prompt_patches = (
            in_context_examples * ((2 * bit_patches) + 2)
            + bit_patches
            + 1
        )
    else:
        raise ValueError(f"Unsupported TimesFM layout: {timesfm_layout}")
    return prompt_patches * patch_length


def make_timesfm_trial_spec(
    program: Sequence[str],
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    timesfm_encoding: str,
    timesfm_layout: str,
    ablate_labels: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if bit_length <= 0:
        raise ValueError("TimesFM backend requires positive bit_length")
    if max_new_tokens != bit_length:
        raise ValueError("--backend timesfm expects max_new_tokens to equal bit_length")
    timesfm_bit_patch_count(bit_length, timesfm_encoding)

    bit_prompt, query_input, expected_output, _, query_input_raw = few_shot(
        program,
        in_context_examples,
        "\n",
        "->",
        bit_length=bit_length,
    )
    lines = bit_prompt.split("\n") if bit_prompt else []
    context_lines = lines[:-1] if lines else []
    context_examples: List[Tuple[str, str]] = []
    for line in context_lines:
        if "->" not in line:
            raise ValueError(f"Malformed few-shot line for TimesFM backend: {line!r}")
        input_bits, output_bits = line.split("->", 1)
        if len(input_bits) != bit_length or len(output_bits) != bit_length:
            raise ValueError("TimesFM backend expected fixed-length context bitstrings")
        context_examples.append((input_bits, output_bits))

    if ablate_labels and len(context_examples) > 1:
        inputs = [inp for inp, _ in context_examples]
        outputs = [out for _, out in context_examples]
        outputs = permute_labels_for_ablation(outputs, random)
        context_examples = list(zip(inputs, outputs))

    prompt_values: List[float] = []
    sep_patch = [TIMESFM_SEPARATOR_VALUE] * TIMESFM_PATCH_LENGTH
    display_lines: List[str] = []
    if timesfm_layout == "sep-input-output":
        for input_bits, output_bits in context_examples:
            prompt_values.extend(sep_patch)
            prompt_values.extend(
                bits_to_timesfm_patch_values(
                    input_bits,
                    timesfm_encoding=timesfm_encoding,
                )
            )
            prompt_values.extend(
                bits_to_timesfm_patch_values(
                    output_bits,
                    timesfm_encoding=timesfm_encoding,
                )
            )
            display_lines.append(f"SEP {input_bits} {output_bits}")
        prompt_values.extend(sep_patch)
        prompt_values.extend(
            bits_to_timesfm_patch_values(
                query_input,
                timesfm_encoding=timesfm_encoding,
            )
        )
        display_lines.append(f"SEP {query_input} <generate {bit_length}>")
    elif timesfm_layout == "input-zero-output-zero":
        for input_bits, output_bits in context_examples:
            prompt_values.extend(
                bits_to_timesfm_patch_values(
                    input_bits,
                    timesfm_encoding=timesfm_encoding,
                )
            )
            prompt_values.extend(sep_patch)
            prompt_values.extend(
                bits_to_timesfm_patch_values(
                    output_bits,
                    timesfm_encoding=timesfm_encoding,
                )
            )
            prompt_values.extend(sep_patch)
            display_lines.append(f"{input_bits} SEP {output_bits} SEP")
        prompt_values.extend(
            bits_to_timesfm_patch_values(
                query_input,
                timesfm_encoding=timesfm_encoding,
            )
        )
        prompt_values.extend(sep_patch)
        display_lines.append(f"{query_input} SEP <generate {bit_length}>")
    else:
        raise ValueError(f"Unsupported TimesFM layout: {timesfm_layout}")
    prompt_display = (
        f"<timesfm zero={TIMESFM_ZERO_VALUE:g} one={TIMESFM_ONE_VALUE:g} "
        f"sep={TIMESFM_SEPARATOR_VALUE:g} patch_length={TIMESFM_PATCH_LENGTH} "
        f"encoding={timesfm_encoding} "
        f"layout={timesfm_layout} "
        f"decode={'sub-lobe-projection' if timesfm_uses_sublobe_projection(timesfm_encoding) else 'patch-mean-sign'}>\n"
        + "\n".join(display_lines)
    )

    spec: Dict[str, Any] = {
        "prompt": prompt_display,
        "timesfm_prompt_values": prompt_values,
        "timesfm_encoding": timesfm_encoding,
        "timesfm_layout": timesfm_layout,
        "query": query_input,
        "expected": expected_output,
        "expected_mapped": expected_output,
        "query_unaltered": query_input_raw,
    }
    if verbose:
        spec["few_shot_examples"] = [
            {"input": inp, "output": out} for inp, out in context_examples
        ]
    return spec


def generate_timesfm_completions(
    model: TimesFm2_5MLX,
    trial_specs: List[Dict[str, Any]],
    *,
    max_tokens: int,
) -> List[str]:
    if not trial_specs:
        return []

    prompt_lengths = {len(spec["timesfm_prompt_values"]) for spec in trial_specs}
    if len(prompt_lengths) != 1:
        raise ValueError("TimesFM batched generation requires equal prompt lengths")
    encodings = {spec["timesfm_encoding"] for spec in trial_specs}
    if len(encodings) != 1:
        raise ValueError("TimesFM batched generation requires matching encodings")
    timesfm_encoding = next(iter(encodings))
    layouts = {spec["timesfm_layout"] for spec in trial_specs}
    if len(layouts) != 1:
        raise ValueError("TimesFM batched generation requires matching layouts")

    required_patches = timesfm_bit_patch_count(max_tokens, timesfm_encoding)
    required_values = required_patches * TIMESFM_PATCH_LENGTH
    values_per_forecast = TIMESFM_HORIZON_LENGTH
    current = np.array(
        [spec["timesfm_prompt_values"] for spec in trial_specs],
        dtype=np.float32,
    )
    generated_blocks: List[np.ndarray] = []
    feedback_blocks: List[np.ndarray] = []
    lobe = np.array(timesfm_sine_lobe(TIMESFM_PATCH_LENGTH), dtype=np.float32)
    bits_per_patch = timesfm_bits_per_patch(timesfm_encoding)
    sub_lobe = np.array(
        timesfm_sine_lobe(TIMESFM_PATCH_LENGTH // bits_per_patch),
        dtype=np.float32,
    )

    while sum(block.shape[1] for block in generated_blocks) < required_values:
        raw_mean, _ = model.forecast(
            current,
            forecast_context_len=min(current.shape[1], TIMESFM_CONTEXT_LIMIT),
            force_flip_invariance=True,
            truncate_negative=False,
        )
        mx.eval(raw_mean)
        raw_block = np.array(raw_mean.astype(mx.float32), dtype=np.float32)
        remaining = required_values - sum(block.shape[1] for block in generated_blocks)
        raw_block = raw_block[:, : min(values_per_forecast, remaining)]
        generated_blocks.append(raw_block)

        feedback = np.empty_like(raw_block)
        for start in range(0, raw_block.shape[1], TIMESFM_PATCH_LENGTH):
            chunk = raw_block[:, start : start + TIMESFM_PATCH_LENGTH]
            if chunk.shape[1] < TIMESFM_PATCH_LENGTH:
                feedback[:, start : start + chunk.shape[1]] = chunk
                continue
            signs = np.where(chunk.mean(axis=1, keepdims=True) > 0.0, TIMESFM_ONE_VALUE, TIMESFM_ZERO_VALUE)
            if timesfm_encoding == "constant-patch":
                feedback[:, start : start + TIMESFM_PATCH_LENGTH] = signs
            elif timesfm_encoding == "sine-lobe":
                feedback[:, start : start + TIMESFM_PATCH_LENGTH] = signs * lobe.reshape(1, -1)
            elif timesfm_uses_sublobe_projection(timesfm_encoding):
                sub_length = TIMESFM_PATCH_LENGTH // bits_per_patch
                for sub_idx in range(bits_per_patch):
                    sub_start = sub_idx * sub_length
                    sub_end = sub_start + sub_length
                    sub_values = chunk[:, sub_start:sub_end]
                    sub_scores = (
                        sub_values * sub_lobe.reshape(1, -1)
                    ).sum(axis=1, keepdims=True)
                    sub_signs = np.where(
                        sub_scores > 0.0,
                        TIMESFM_ONE_VALUE,
                        TIMESFM_ZERO_VALUE,
                    )
                    feedback[:, start + sub_start : start + sub_end] = (
                        sub_signs * sub_lobe.reshape(1, -1)
                    )
            else:
                raise ValueError(f"Unsupported TimesFM encoding: {timesfm_encoding}")
        feedback_blocks.append(feedback)
        current = np.concatenate([current, feedback], axis=1)

    raw_values = np.concatenate(generated_blocks, axis=1)[:, :required_values]
    feedback_values = np.concatenate(feedback_blocks, axis=1)[:, :required_values]

    outputs: List[str] = []
    for spec, raw_row, feedback_row in zip(trial_specs, raw_values, feedback_values):
        raw_list = [float(value) for value in raw_row.tolist()]
        feedback_list = [float(value) for value in feedback_row.tolist()]
        spec["timesfm_generated_values"] = raw_list
        spec["timesfm_feedback_values"] = feedback_list
        if timesfm_uses_sublobe_projection(timesfm_encoding):
            outputs.append(
                decode_timesfm_sublobe_projection_values(
                    raw_list,
                    bit_length=len(spec["expected"]),
                    timesfm_encoding=timesfm_encoding,
                )
            )
        else:
            outputs.append(
                decode_timesfm_patch_mean_values(
                    raw_list,
                    bit_length=len(spec["expected"]),
                )
            )
    return outputs


def _require_chess():
    try:
        import chess  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "The chess backend requires python-chess. Install it in the active environment."
        ) from exc
    return chess


def chess_channel_move_for_bit(board, bit: str):
    chess = _require_chess()
    if bit not in {"0", "1"}:
        raise ValueError(f"Chess bit encoding only supports 0/1, got {bit!r}")

    if board.turn == chess.WHITE:
        candidates = (
            (("b1", "c3"), ("c3", "b1"))
            if bit == "0"
            else (("g1", "f3"), ("f3", "g1"))
        )
    else:
        candidates = (
            (("b8", "c6"), ("c6", "b8"))
            if bit == "0"
            else (("g8", "f6"), ("f6", "g8"))
        )

    for source, target in candidates:
        move = chess.Move.from_uci(source + target)
        if move in board.legal_moves:
            return move
    raise ValueError(
        f"No legal knight-toggle move for bit {bit!r} from board {board.fen()}"
    )


def chess_signal_move_for_bit(board, bit: str):
    chess = _require_chess()
    if bit not in {"0", "1"}:
        raise ValueError(f"Chess bit encoding only supports 0/1, got {bit!r}")

    if board.turn == chess.WHITE:
        source, target = ("b1", "c3") if bit == "0" else ("g1", "f3")
    else:
        source, target = ("b8", "c6") if bit == "0" else ("g8", "f6")

    move = chess.Move.from_uci(source + target)
    if move not in board.legal_moves:
        raise ValueError(
            f"No legal signal move for bit {bit!r} from board {board.fen()}"
        )
    return move


def chess_wipe_move_for_bit(board, bit: str):
    chess = _require_chess()
    if bit not in {"0", "1"}:
        raise ValueError(f"Chess bit encoding only supports 0/1, got {bit!r}")

    if board.turn == chess.WHITE:
        source, target = ("c3", "b1") if bit == "0" else ("f3", "g1")
    else:
        source, target = ("c6", "b8") if bit == "0" else ("f6", "g8")

    move = chess.Move.from_uci(source + target)
    if move not in board.legal_moves:
        raise ValueError(
            f"No legal wipe move for bit {bit!r} from board {board.fen()}"
        )
    return move


def chess_signal2_move_for_bits(board, bits: str):
    chess = _require_chess()
    if len(bits) != 2 or any(bit not in {"0", "1"} for bit in bits):
        raise ValueError(
            f"Signal2 chess encoding expects a two-bit chunk, got {bits!r}"
        )

    if board.turn == chess.WHITE:
        source, target = {
            "00": ("b1", "a3"),
            "01": ("b1", "c3"),
            "10": ("g1", "f3"),
            "11": ("g1", "h3"),
        }[bits]
    else:
        source, target = {
            "00": ("b8", "a6"),
            "01": ("b8", "c6"),
            "10": ("g8", "f6"),
            "11": ("g8", "h6"),
        }[bits]

    move = chess.Move.from_uci(source + target)
    if move not in board.legal_moves:
        raise ValueError(
            f"No legal signal2 move for bits {bits!r} from board {board.fen()}"
        )
    return move


def chess_signal2_wipe_move_for_bits(board, bits: str):
    chess = _require_chess()
    if len(bits) != 2 or any(bit not in {"0", "1"} for bit in bits):
        raise ValueError(
            f"Signal2 chess encoding expects a two-bit chunk, got {bits!r}"
        )

    if board.turn == chess.WHITE:
        source, target = {
            "00": ("a3", "b1"),
            "01": ("c3", "b1"),
            "10": ("f3", "g1"),
            "11": ("h3", "g1"),
        }[bits]
    else:
        source, target = {
            "00": ("a6", "b8"),
            "01": ("c6", "b8"),
            "10": ("f6", "g8"),
            "11": ("h6", "g8"),
        }[bits]

    move = chess.Move.from_uci(source + target)
    if move not in board.legal_moves:
        raise ValueError(
            f"No legal signal2 wipe move for bits {bits!r} from board {board.fen()}"
        )
    return move


def chess_text_for_move(transcript: str, board, move) -> str:
    chess = _require_chess()
    if board.turn == chess.WHITE:
        prefix = "" if transcript.endswith(";") else " "
        prefix += f"{board.fullmove_number}."
    else:
        prefix = "" if transcript.endswith(" ") else " "
    return prefix + board.san(move)


def chess_candidate_text_for_bit(transcript: str, board, bit: str) -> Tuple[str, Any]:
    move = chess_channel_move_for_bit(board, bit)
    return chess_text_for_move(transcript, board, move), move


def chess_signal_text_for_bit(transcript: str, board, bit: str) -> Tuple[str, Any]:
    move = chess_signal_move_for_bit(board, bit)
    return chess_text_for_move(transcript, board, move), move


def chess_wipe_text_for_bit(transcript: str, board, bit: str) -> Tuple[str, Any]:
    move = chess_wipe_move_for_bit(board, bit)
    return chess_text_for_move(transcript, board, move), move


def chess_signal2_text_for_bits(
    transcript: str,
    board,
    bits: str,
) -> Tuple[str, Any]:
    move = chess_signal2_move_for_bits(board, bits)
    return chess_text_for_move(transcript, board, move), move


def chess_signal2_wipe_text_for_bits(
    transcript: str,
    board,
    bits: str,
) -> Tuple[str, Any]:
    move = chess_signal2_wipe_move_for_bits(board, bits)
    return chess_text_for_move(transcript, board, move), move


def append_chess_signal_wipe_bits(transcript: str, board, bits: str) -> Tuple[str, Any]:
    if len(bits) % 2 != 0:
        raise ValueError("Signal/wipe chess encoding requires an even bit count")
    for idx in range(0, len(bits), 2):
        white_bit = bits[idx]
        black_bit = bits[idx + 1]
        for bit in (white_bit, black_bit):
            text, move = chess_signal_text_for_bit(transcript, board, bit)
            transcript += text
            board.push(move)
        for bit in (white_bit, black_bit):
            text, move = chess_wipe_text_for_bit(transcript, board, bit)
            transcript += text
            board.push(move)
    return transcript, board


def append_chess_signal2_wipe_bits(transcript: str, board, bits: str) -> Tuple[str, Any]:
    if len(bits) % 4 != 0:
        raise ValueError(
            "Signal2/wipe chess encoding requires a bit count divisible by 4"
        )
    for idx in range(0, len(bits), 4):
        white_bits = bits[idx : idx + 2]
        black_bits = bits[idx + 2 : idx + 4]
        for chunk in (white_bits, black_bits):
            text, move = chess_signal2_text_for_bits(transcript, board, chunk)
            transcript += text
            board.push(move)
        for chunk in (white_bits, black_bits):
            text, move = chess_signal2_wipe_text_for_bits(
                transcript,
                board,
                chunk,
            )
            transcript += text
            board.push(move)
    return transcript, board


def chess_signal2_sep_layout_prompt(
    context_examples: Sequence[Tuple[str, str]],
    query_input: str,
    *,
    max_new_tokens: int,
) -> Tuple[str, Any, int]:
    chess = _require_chess()
    transcript = ";"
    board = chess.Board()
    separator_bits = "00"

    for input_bits, output_bits in context_examples:
        transcript, board = append_chess_signal2_wipe_bits(
            transcript,
            board,
            input_bits,
        )
        transcript, board = append_chess_signal_wipe_bits(
            transcript,
            board,
            separator_bits,
        )
        transcript, board = append_chess_signal2_wipe_bits(
            transcript,
            board,
            output_bits,
        )
        transcript, board = append_chess_signal_wipe_bits(
            transcript,
            board,
            separator_bits,
        )

    transcript, board = append_chess_signal2_wipe_bits(transcript, board, query_input)
    transcript, board = append_chess_signal_wipe_bits(transcript, board, separator_bits)

    output_board = board.copy(stack=False)
    final_transcript, final_board = append_chess_signal2_wipe_bits(
        transcript,
        board.copy(stack=False),
        "0" * max_new_tokens,
    )
    return transcript, output_board, len(final_transcript)


def bits_to_chess_pgn(bits: str, *, chess_encoding: str) -> Tuple[str, Any]:
    chess = _require_chess()
    board = chess.Board()
    transcript = ";"

    if chess_encoding == "toggle":
        for bit in bits:
            text, move = chess_candidate_text_for_bit(transcript, board, bit)
            transcript += text
            board.push(move)
    elif chess_encoding == "signal-wipe":
        transcript, board = append_chess_signal_wipe_bits(transcript, board, bits)
    elif chess_encoding in {"signal2-wipe", "signal2-wipe-sep"}:
        transcript, board = append_chess_signal2_wipe_bits(transcript, board, bits)
    else:
        raise ValueError(f"Unsupported chess encoding: {chess_encoding}")

    return transcript, board


def chess_pgn_length_for_ply_count(ply_count: int) -> int:
    if ply_count < 0:
        raise ValueError("ply_count cannot be negative")
    if ply_count == 0:
        return 1  # leading game delimiter

    full_moves = ply_count // 2
    digit_count = sum(len(str(move_number)) for move_number in range(1, full_moves + 1))
    length = 1 if full_moves == 0 else digit_count + (9 * full_moves)
    if ply_count % 2:
        next_move_number = full_moves + 1
        prefix_length = 0 if full_moves == 0 else 1
        length += prefix_length + len(str(next_move_number)) + 1 + 3
    return length


def chess_pgn_length_for_bit_count(bit_count: int, *, chess_encoding: str) -> int:
    if bit_count < 0:
        raise ValueError("bit_count cannot be negative")
    if chess_encoding == "toggle":
        return chess_pgn_length_for_ply_count(bit_count)
    if chess_encoding == "signal-wipe":
        if bit_count % 2 != 0:
            raise ValueError("Signal/wipe chess encoding requires an even bit count")
        return chess_pgn_length_for_ply_count(2 * bit_count)
    if chess_encoding in {"signal2-wipe", "signal2-wipe-sep"}:
        if bit_count % 4 != 0:
            raise ValueError(
                "Signal2/wipe chess encoding requires a bit count divisible by 4"
            )
        return chess_pgn_length_for_ply_count(bit_count)
    raise ValueError(f"Unsupported chess encoding: {chess_encoding}")


def chess_gamelet_for_bit(bit: str, *, paired: bool) -> str:
    if bit == "0":
        text = ";1.Nc3"
        if paired:
            text += " Nc6"
        return text
    if bit == "1":
        text = ";1.Nf3"
        if paired:
            text += " Nf6"
        return text
    raise ValueError(f"Chess gamelet encoding only supports 0/1, got {bit!r}")


def bits_to_chess_gamelets(bits: str, *, paired: bool) -> str:
    return "".join(chess_gamelet_for_bit(bit, paired=paired) for bit in bits)


def chess_layout_prompt(
    context_examples: Sequence[Tuple[str, str]],
    query_input: str,
    *,
    max_new_tokens: int,
    chess_encoding: str,
    chess_layout: str,
) -> Tuple[str, Any, int]:
    chess = _require_chess()
    if chess_layout != "continuous" and chess_encoding != "toggle":
        raise ValueError("Non-continuous chess layouts currently require toggle encoding")
    carried_bit_count = (
        sum(len(inp) + len(out) for inp, out in context_examples)
        + len(query_input)
    )

    def conservative_final_context_length(
        prompt_text: str,
        minimum_final_context_length: int,
    ) -> int:
        minimum_generation_chars = minimum_final_context_length - len(prompt_text)
        continuous_generation_chars = (
            chess_pgn_length_for_bit_count(
                carried_bit_count + max_new_tokens,
                chess_encoding=chess_encoding,
            )
            - chess_pgn_length_for_bit_count(
                carried_bit_count,
                chess_encoding=chess_encoding,
            )
        )
        return len(prompt_text) + max(minimum_generation_chars, continuous_generation_chars)

    if chess_layout == "continuous":
        if chess_encoding == "signal2-wipe-sep":
            prompt_text, output_board, final_context_length = chess_signal2_sep_layout_prompt(
                context_examples,
                query_input,
                max_new_tokens=max_new_tokens,
            )
        else:
            prompt_bits = "".join(inp + out for inp, out in context_examples) + query_input
            prompt_text, output_board = bits_to_chess_pgn(
                prompt_bits,
                chess_encoding=chess_encoding,
            )
            final_context_length = chess_pgn_length_for_bit_count(
                len(prompt_bits) + max_new_tokens,
                chess_encoding=chess_encoding,
            )
    elif chess_layout == "example-reset":
        context_text = "".join(
            bits_to_chess_pgn(inp + out, chess_encoding=chess_encoding)[0]
            for inp, out in context_examples
        )
        query_prompt_text, output_board = bits_to_chess_pgn(
            query_input,
            chess_encoding=chess_encoding,
        )
        query_final_text, _ = bits_to_chess_pgn(
            query_input + ("0" * max_new_tokens),
            chess_encoding=chess_encoding,
        )
        prompt_text = context_text + query_prompt_text
        final_context_length = conservative_final_context_length(
            prompt_text,
            len(context_text) + len(query_final_text),
        )
    elif chess_layout == "segment-reset":
        context_text = "".join(
            bits_to_chess_pgn(inp, chess_encoding=chess_encoding)[0]
            + bits_to_chess_pgn(out, chess_encoding=chess_encoding)[0]
            for inp, out in context_examples
        )
        query_prompt_text, _ = bits_to_chess_pgn(
            query_input,
            chess_encoding=chess_encoding,
        )
        output_prefix = ";"
        prompt_text = context_text + query_prompt_text + output_prefix
        output_board = chess.Board()
        final_context_length = conservative_final_context_length(
            prompt_text,
            (
                len(prompt_text)
                + chess_pgn_length_for_bit_count(max_new_tokens, chess_encoding=chess_encoding)
                - len(output_prefix)
            ),
        )
    elif chess_layout in {"bit-gamelet", "bit-gamelet-pair"}:
        paired = chess_layout == "bit-gamelet-pair"
        context_text = "".join(
            bits_to_chess_gamelets(inp, paired=paired)
            + bits_to_chess_gamelets(out, paired=paired)
            for inp, out in context_examples
        )
        output_prefix = ";"
        prompt_text = (
            context_text
            + bits_to_chess_gamelets(query_input, paired=paired)
            + output_prefix
        )
        output_board = chess.Board()
        final_context_length = conservative_final_context_length(
            prompt_text,
            (
                len(prompt_text)
                + len(bits_to_chess_gamelets("0" * max_new_tokens, paired=paired))
                - len(output_prefix)
            ),
        )
    else:
        raise ValueError(f"Unsupported chess layout: {chess_layout}")

    if final_context_length > CHESSGPT_CONTEXT_LIMIT:
        raise ValueError(
            "Chess prompt exceeds Chess-GPT context after generation: "
            f"layout={chess_layout}, chars={final_context_length}, "
            f"limit={CHESSGPT_CONTEXT_LIMIT}"
        )
    return prompt_text, output_board, final_context_length


def chess_layout_max_context_chars(
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    chess_encoding: str,
    chess_layout: str,
) -> int:
    example = "0" * bit_length
    prompt_text, _, final_context_length = chess_layout_prompt(
        [(example, example)] * in_context_examples,
        example,
        max_new_tokens=max_new_tokens,
        chess_encoding=chess_encoding,
        chess_layout=chess_layout,
    )
    if len(prompt_text) > final_context_length:
        raise ValueError("Internal chess layout length calculation underflowed")
    return final_context_length


def make_chess_trial_spec(
    program: Sequence[str],
    *,
    in_context_examples: int,
    bit_length: int,
    max_new_tokens: int,
    chess_encoding: str,
    chess_layout: str,
    ablate_labels: bool,
    verbose: bool,
) -> Dict[str, Any]:
    if bit_length <= 0:
        raise ValueError("Chess backend requires positive bit_length")
    if max_new_tokens != bit_length:
        raise ValueError("Chess backend expects max_new_tokens to equal bit_length")

    bit_prompt, query_input, expected_output, _, query_input_raw = few_shot(
        program,
        in_context_examples,
        "\n",
        "->",
        bit_length=bit_length,
    )
    lines = bit_prompt.split("\n") if bit_prompt else []
    context_lines = lines[:-1] if lines else []
    context_examples: List[Tuple[str, str]] = []
    for line in context_lines:
        if "->" not in line:
            raise ValueError(f"Malformed few-shot line for Chess backend: {line!r}")
        input_bits, output_bits = line.split("->", 1)
        if len(input_bits) != bit_length or len(output_bits) != bit_length:
            raise ValueError("Chess backend expected fixed-length context bitstrings")
        context_examples.append((input_bits, output_bits))

    if ablate_labels and len(context_examples) > 1:
        inputs = [inp for inp, _ in context_examples]
        outputs = [out for _, out in context_examples]
        outputs = permute_labels_for_ablation(outputs, random)
        context_examples = list(zip(inputs, outputs))

    if chess_encoding == "signal2-wipe-sep":
        prompt_bits = "".join(
            inp + "00" + out + "00" for inp, out in context_examples
        ) + query_input + "00"
    else:
        prompt_bits = "".join(inp + out for inp, out in context_examples) + query_input
    prompt_text, board, final_context_length = chess_layout_prompt(
        context_examples,
        query_input,
        max_new_tokens=max_new_tokens,
        chess_encoding=chess_encoding,
        chess_layout=chess_layout,
    )

    spec: Dict[str, Any] = {
        "prompt": prompt_text,
        "chess_encoding": chess_encoding,
        "chess_layout": chess_layout,
        "chess_prompt_bits": prompt_bits,
        "chess_prompt_plies": len(prompt_bits),
        "chess_prompt_chars": len(prompt_text),
        "chess_final_context_chars": final_context_length,
        "chess_board": board,
        "query": query_input,
        "expected": expected_output,
        "expected_mapped": expected_output,
        "query_unaltered": query_input_raw,
    }
    if verbose:
        spec["few_shot_examples"] = [
            {"input": inp, "output": out} for inp, out in context_examples
        ]
    return spec


def encode_chess_text(tokenizer, text: str) -> List[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) != len(text):
        raise ValueError(
            f"Chess-GPT tokenizer did not stay character-level for {text!r}: {token_ids}"
        )
    return [int(token_id) for token_id in token_ids]


def clone_chess_cache(cache: Sequence[Any]) -> List[Any]:
    cloned: List[Any] = []
    for layer_cache in cache:
        new_cache = kv_cache.KVCache()
        if not layer_cache.empty():
            # KVCache.state is sliced to the live offset, so subsequent candidate
            # appends allocate new storage instead of writing into the source cache.
            new_cache.state = layer_cache.state
        cloned.append(new_cache)
    return cloned


def eval_chess_cache_state(cache: Sequence[Any]) -> List[Any]:
    return [layer_cache.state for layer_cache in cache if not layer_cache.empty()]


def feed_chess_token_ids(model, cache: Sequence[Any], token_ids: Sequence[int]) -> mx.array:
    if not token_ids:
        raise ValueError("Cannot feed an empty token sequence")
    tokens = mx.array([list(token_ids)], dtype=mx.uint32)
    logits = model(tokens, cache=cache)
    mx.eval(logits, eval_chess_cache_state(cache))
    return logits[0, -1]


def score_chess_candidate(
    model,
    base_cache: Sequence[Any],
    base_last_logits: mx.array,
    token_ids: Sequence[int],
) -> Tuple[float, List[Any], mx.array]:
    if not token_ids:
        raise ValueError("Cannot score an empty candidate")

    candidate_cache = clone_chess_cache(base_cache)
    tokens = mx.array([list(token_ids)], dtype=mx.uint32)
    logits = model(tokens, cache=candidate_cache)

    first_target = mx.array([int(token_ids[0])], dtype=mx.uint32)
    loss = nn.losses.cross_entropy(
        base_last_logits[None, :].astype(mx.float32),
        first_target,
        reduction="sum",
    )
    if len(token_ids) > 1:
        next_targets = mx.array(list(token_ids[1:]), dtype=mx.uint32)
        next_losses = nn.losses.cross_entropy(
            logits[0, :-1, :].astype(mx.float32),
            next_targets,
            reduction="sum",
        )
        loss = loss + next_losses

    mx.eval(loss, logits, eval_chess_cache_state(candidate_cache))
    return float(loss.item()), candidate_cache, logits[0, -1]


CHESS_KNIGHT_SAN_RE = re.compile(
    r"\d+\.{1,3}|"
    r"N(?:a3|c3|f3|h3|b1|g1|a6|c6|f6|h6|b8|g8)[+#]?|"
    r"1-0|0-1|1/2-1/2|\*"
)


def iter_chess_san_tokens(generated_pgn: str):
    for token in CHESS_KNIGHT_SAN_RE.findall(generated_pgn):
        if token.endswith("."):
            continue
        if token in {"1-0", "0-1", "1/2-1/2", "*"}:
            break
        yield token.rstrip("+#")


def parse_chess_generated_move(board, token: str):
    chess = _require_chess()
    try:
        return board.parse_san(token)
    except chess.IllegalMoveError:
        return None
    except chess.InvalidMoveError:
        return None
    except chess.AmbiguousMoveError:
        return None


def decode_chess_toggle_generated_bits(
    generated_pgn: str,
    *,
    start_board,
    max_bits: int,
) -> str:
    board = start_board.copy(stack=False)
    bits: List[str] = []

    for token in iter_chess_san_tokens(generated_pgn):
        if len(bits) >= max_bits:
            break

        move = parse_chess_generated_move(board, token)
        if move is None:
            break

        decoded_bit = None
        for bit in ("0", "1"):
            try:
                expected_move = chess_channel_move_for_bit(board, bit)
            except ValueError:
                continue
            if move == expected_move:
                decoded_bit = bit
                break
        if decoded_bit is None:
            break

        bits.append(decoded_bit)
        board.push(move)

    return "".join(bits)


def decode_chess_signal2_generated_bits(
    generated_pgn: str,
    *,
    start_board,
    max_bits: int,
) -> str:
    board = start_board.copy(stack=False)
    bits: List[str] = []
    tokens = iter(iter_chess_san_tokens(generated_pgn))

    while len(bits) < max_bits:
        signal_chunks: List[str] = []
        for _ in range(2):
            try:
                token = next(tokens)
            except StopIteration:
                return "".join(bits)[:max_bits]

            move = parse_chess_generated_move(board, token)
            if move is None:
                return "".join(bits)[:max_bits]

            decoded_chunk = None
            for chunk in ("00", "01", "10", "11"):
                try:
                    expected_move = chess_signal2_move_for_bits(board, chunk)
                except ValueError:
                    continue
                if move == expected_move:
                    decoded_chunk = chunk
                    break
            if decoded_chunk is None:
                return "".join(bits)[:max_bits]

            bits.extend(decoded_chunk)
            signal_chunks.append(decoded_chunk)
            board.push(move)
            if len(bits) >= max_bits:
                return "".join(bits)[:max_bits]

        for chunk in signal_chunks:
            try:
                token = next(tokens)
            except StopIteration:
                return "".join(bits)[:max_bits]

            move = parse_chess_generated_move(board, token)
            if move is None:
                return "".join(bits)[:max_bits]

            try:
                expected_move = chess_signal2_wipe_move_for_bits(board, chunk)
            except ValueError:
                return "".join(bits)[:max_bits]
            if move != expected_move:
                return "".join(bits)[:max_bits]
            board.push(move)

    return "".join(bits)[:max_bits]


def decode_chess_gamelet_generated_bits(
    generated_pgn: str,
    *,
    max_bits: int,
    paired: bool,
) -> str:
    chess = _require_chess()
    bits: List[str] = []
    text = ";" + generated_pgn.lstrip(";")

    for chunk in text.split(";")[1:]:
        if len(bits) >= max_bits:
            break
        chunk = chunk.strip()
        if not chunk:
            continue

        match = re.match(
            r"^(?:1\.)?(N(?:c3|f3))(?:\s+(N(?:c6|f6)))?",
            chunk,
        )
        if match is None:
            break

        board = chess.Board()
        try:
            white_move = board.parse_san(match.group(1))
        except chess.IllegalMoveError:
            break
        except chess.InvalidMoveError:
            break
        except chess.AmbiguousMoveError:
            break

        decoded_bit = None
        for bit in ("0", "1"):
            if white_move == chess_channel_move_for_bit(board, bit):
                decoded_bit = bit
                break
        if decoded_bit is None:
            break
        board.push(white_move)

        if paired:
            black_san = match.group(2)
            if black_san is None:
                break
            try:
                black_move = board.parse_san(black_san)
            except chess.IllegalMoveError:
                break
            except chess.InvalidMoveError:
                break
            except chess.AmbiguousMoveError:
                break
            if black_move != chess_channel_move_for_bit(board, decoded_bit):
                break

        bits.append(decoded_bit)

    return "".join(bits)


def generate_chess_completions(
    model,
    tokenizer,
    trial_specs: List[Dict[str, Any]],
    *,
    max_bits: int,
    chess_encoding: str,
) -> List[str]:
    if not trial_specs:
        return []
    if chess_encoding == "signal-wipe" and max_bits % 2 != 0:
        raise ValueError("Signal/wipe chess generation requires an even bit count")
    if chess_encoding in {"signal2-wipe", "signal2-wipe-sep"} and max_bits % 4 != 0:
        raise ValueError("Signal2/wipe chess generation requires a bit count divisible by 4")

    outputs: List[str] = []
    for spec in trial_specs:
        prompt_text = spec["prompt"]
        prompt_ids = encode_chess_text(tokenizer, prompt_text)
        cache = kv_cache.make_prompt_cache(model)
        last_logits = feed_chess_token_ids(model, cache, prompt_ids)
        transcript = prompt_text
        board = spec["chess_board"].copy(stack=False)

        generated_bits: List[str] = []
        generated_text_parts: List[str] = []
        if chess_encoding == "toggle":
            for _ in range(max_bits):
                candidates = []
                for bit in ("0", "1"):
                    candidate_text, move = chess_candidate_text_for_bit(transcript, board, bit)
                    candidate_ids = encode_chess_text(tokenizer, candidate_text)
                    loss, candidate_cache, candidate_last_logits = score_chess_candidate(
                        model,
                        cache,
                        last_logits,
                        candidate_ids,
                    )
                    candidates.append(
                        (loss, bit, candidate_text, move, candidate_cache, candidate_last_logits)
                    )

                _, bit, chosen_text, chosen_move, cache, last_logits = min(
                    candidates,
                    key=lambda item: item[0],
                )
                generated_bits.append(bit)
                generated_text_parts.append(chosen_text)
                transcript += chosen_text
                board.push(chosen_move)
        elif chess_encoding == "signal-wipe":
            while len(generated_bits) < max_bits:
                signal_bits: List[str] = []
                for _ in range(2):
                    candidates = []
                    for bit in ("0", "1"):
                        candidate_text, move = chess_signal_text_for_bit(transcript, board, bit)
                        candidate_ids = encode_chess_text(tokenizer, candidate_text)
                        loss, candidate_cache, candidate_last_logits = score_chess_candidate(
                            model,
                            cache,
                            last_logits,
                            candidate_ids,
                        )
                        candidates.append(
                            (loss, bit, candidate_text, move, candidate_cache, candidate_last_logits)
                        )

                    _, bit, chosen_text, chosen_move, cache, last_logits = min(
                        candidates,
                        key=lambda item: item[0],
                    )
                    generated_bits.append(bit)
                    signal_bits.append(bit)
                    generated_text_parts.append(chosen_text)
                    transcript += chosen_text
                    board.push(chosen_move)

                for bit in signal_bits:
                    wipe_text, wipe_move = chess_wipe_text_for_bit(transcript, board, bit)
                    wipe_ids = encode_chess_text(tokenizer, wipe_text)
                    last_logits = feed_chess_token_ids(model, cache, wipe_ids)
                    generated_text_parts.append(wipe_text)
                    transcript += wipe_text
                    board.push(wipe_move)
        elif chess_encoding in {"signal2-wipe", "signal2-wipe-sep"}:
            while len(generated_bits) < max_bits:
                signal_chunks: List[str] = []
                for _ in range(2):
                    candidates = []
                    for chunk in ("00", "01", "10", "11"):
                        candidate_text, move = chess_signal2_text_for_bits(
                            transcript,
                            board,
                            chunk,
                        )
                        candidate_ids = encode_chess_text(tokenizer, candidate_text)
                        loss, candidate_cache, candidate_last_logits = score_chess_candidate(
                            model,
                            cache,
                            last_logits,
                            candidate_ids,
                        )
                        candidates.append(
                            (
                                loss,
                                chunk,
                                candidate_text,
                                move,
                                candidate_cache,
                                candidate_last_logits,
                            )
                        )

                    _, chunk, chosen_text, chosen_move, cache, last_logits = min(
                        candidates,
                        key=lambda item: item[0],
                    )
                    generated_bits.append(chunk)
                    signal_chunks.append(chunk)
                    generated_text_parts.append(chosen_text)
                    transcript += chosen_text
                    board.push(chosen_move)

                for chunk in signal_chunks:
                    wipe_text, wipe_move = chess_signal2_wipe_text_for_bits(
                        transcript,
                        board,
                        chunk,
                    )
                    wipe_ids = encode_chess_text(tokenizer, wipe_text)
                    last_logits = feed_chess_token_ids(model, cache, wipe_ids)
                    generated_text_parts.append(wipe_text)
                    transcript += wipe_text
                    board.push(wipe_move)
        else:
            raise ValueError(f"Unsupported chess encoding: {chess_encoding}")

        generated_pgn = "".join(generated_text_parts)
        generated_bitstring = "".join(generated_bits)
        spec["chess_generated_pgn"] = generated_pgn
        spec["chess_generated_bits"] = generated_bitstring
        outputs.append(generated_bitstring)

    return outputs


def generate_chess_greedy_completions(
    model,
    tokenizer,
    trial_specs: List[Dict[str, Any]],
    *,
    max_bits: int,
    chess_encoding: str,
) -> List[str]:
    if not trial_specs:
        return []
    if chess_encoding not in {"toggle", "signal2-wipe", "signal2-wipe-sep"}:
        raise ValueError(
            "Greedy chess decoding is currently implemented for toggle and signal2-wipe encodings only"
        )

    prompts = [encode_chess_text(tokenizer, spec["prompt"]) for spec in trial_specs]
    max_chars = max(
        spec["chess_final_context_chars"] - spec["chess_prompt_chars"]
        for spec in trial_specs
    )
    batch = batch_generate(
        model,
        tokenizer,
        prompts,
        verbose=False,
        max_tokens=max_chars,
    )

    outputs: List[str] = []
    for spec, generated_pgn in zip(trial_specs, batch.texts):
        chess_layout = spec.get("chess_layout", "continuous")
        if chess_layout in {"bit-gamelet", "bit-gamelet-pair"}:
            generated_bits = decode_chess_gamelet_generated_bits(
                generated_pgn,
                max_bits=max_bits,
                paired=chess_layout == "bit-gamelet-pair",
            )
        elif chess_encoding in {"signal2-wipe", "signal2-wipe-sep"}:
            generated_bits = decode_chess_signal2_generated_bits(
                generated_pgn,
                start_board=spec["chess_board"],
                max_bits=max_bits,
            )
        else:
            generated_bits = decode_chess_toggle_generated_bits(
                generated_pgn,
                start_board=spec["chess_board"],
                max_bits=max_bits,
            )
        spec["chess_generated_pgn"] = generated_pgn
        spec["chess_generated_bits"] = generated_bits
        outputs.append(generated_bits)

    return outputs


def batch_generate_with_logits_processors(
    model,
    tokenizer,
    prompts: List[List[int]],
    *,
    max_tokens: int,
    verbose: bool = False,
    logits_processors: Optional[
        List[List[Callable[[mx.array, mx.array], mx.array]]]
    ] = None,
    **kwargs,
):
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        **kwargs,
    )
    num_samples = len(prompts)
    fin = 0
    if verbose:
        print(f"[batch_generate] Finished processing 0/{num_samples} ...", end="\r")

    uids = gen.insert(
        prompts,
        [max_tokens] * len(prompts),
        logits_processors=logits_processors,
    )
    results = {uid: [] for uid in uids}

    def drain_generator() -> None:
        nonlocal fin
        next_generated = getattr(gen, "next_generated", gen.next)
        while responses := next_generated():
            for response in responses:
                if response.finish_reason is not None and verbose:
                    fin += 1
                    print(
                        f"[batch_generate] Finished processing {fin}/{num_samples} ...",
                        end="\r",
                    )
                if response.finish_reason != "stop":
                    results[response.uid].append(response.token)

    if verbose:
        with gen.stats() as stats:
            drain_generator()
    else:
        stats = None
        drain_generator()
    gen.close()
    if verbose:
        print(f"[batch_generate] Finished processing {fin}/{num_samples}")
        print(
            f"[batch_generate] Prompt: {stats.prompt_tokens} tokens, {stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {stats.generation_tokens} tokens, "
            f"{stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {stats.peak_memory:.3f} GB")
    return [tokenizer.decode(results[uid]) for uid in uids]


def compute_mlx_perplexities(
    model,
    tokenizer,
    trial_specs: List[Dict[str, Any]],
    *,
    batch_size: int,
    invert_binary_logits_flag: bool,
    uniform_binary_logits_flag: bool,
) -> Tuple[List[float], List[float]]:
    if not trial_specs:
        return [], []

    if uniform_binary_logits_flag and invert_binary_logits_flag:
        raise ValueError("uniform_binary_logits_flag cannot be combined with invert_binary_logits_flag")

    if invert_binary_logits_flag or uniform_binary_logits_flag:
        zero_ids = tokenizer.encode("0", add_special_tokens=False)
        one_ids = tokenizer.encode("1", add_special_tokens=False)
        if len(zero_ids) != 1 or len(one_ids) != 1:
            raise ValueError("invert_binary_logits requires single-token '0' and '1'")
        zero_id = zero_ids[0]
        one_id = one_ids[0]
        if zero_id == one_id:
            raise ValueError("Tokenizer encodes '0' and '1' to the same token")
    else:
        zero_id = None
        one_id = None

    encoded_sequences: List[List[int]] = []
    prompt_lengths: List[int] = []
    answer_lengths: List[int] = []
    for spec in trial_specs:
        full_text = f"{spec['prompt']}{spec['expected_mapped']}"
        full_tokens = tokenizer.encode(full_text)
        if not full_tokens:
            raise ValueError("Perplexity evaluation requires non-empty prompts")
        prompt_tokens = tokenizer.encode(spec["prompt"], add_special_tokens=False)
        prompt_len = len(prompt_tokens)
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is not None and full_tokens[0] == bos_token_id:
            prompt_len += 1
        if prompt_len <= 0:
            raise ValueError("Perplexity evaluation requires non-empty prompts")
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        eos_offset = 1 if eos_token_id is not None and full_tokens[-1] == eos_token_id else 0
        answer_len = len(full_tokens) - prompt_len - eos_offset
        if answer_len <= 0:
            raise ValueError("Perplexity evaluation requires non-empty answers")
        prompt_lengths.append(prompt_len)
        answer_lengths.append(answer_len)
        encoded_sequences.append(full_tokens)

    grouped: Dict[Tuple[int, int], List[int]] = {}
    for idx, (prompt_len, answer_len) in enumerate(zip(prompt_lengths, answer_lengths)):
        grouped.setdefault((prompt_len, answer_len), []).append(idx)

    perplexities: List[float] = [0.0] * len(trial_specs)
    bit_accuracies: List[float] = [0.0] * len(trial_specs)
    for (prompt_len, answer_len), indices in grouped.items():
        start_idx = prompt_len - 1
        end_idx = start_idx + answer_len
        if start_idx < 0:
            raise ValueError("Perplexity evaluation requires non-empty prompts")

        for offset in range(0, len(indices), batch_size):
            batch_indices = indices[offset : offset + batch_size]
            batch_tokens = [encoded_sequences[idx] for idx in batch_indices]
            batch = mx.array(batch_tokens)
            if uniform_binary_logits_flag:
                vocab_size = tokenizer.vocab_size
                base_values = [-1e9] * vocab_size
                base_values[zero_id] = 0.0
                base_values[one_id] = 0.0
                base = mx.array(base_values, dtype=mx.float32)
                seq_len = batch.shape[1] - 1
                logits = mx.broadcast_to(
                    base,
                    (batch.shape[0], seq_len, vocab_size),
                )
            else:
                logits = model(batch[:, :-1]).astype(mx.float32)
                if invert_binary_logits_flag:
                    logits = invert_binary_logits(logits, zero_id, one_id)
            losses = nn.losses.cross_entropy(logits, batch[:, 1:], reduction="none")
            answer_losses = losses[:, start_idx:end_idx]
            #print(batch[:, start_idx:end_idx])
            #print(answer_losses)
            mean_loss = answer_losses.mean(axis=1)
            ppl = mx.exp(mean_loss)
            preds = mx.argmax(logits, axis=-1)
            answer_preds = preds[:, start_idx:end_idx]
            answer_targets = batch[:, 1:][:, start_idx:end_idx]
            accuracy = (answer_preds == answer_targets).astype(mx.float32).mean(axis=1)
            mx.eval(ppl, accuracy)
            for idx, value, acc in zip(batch_indices, ppl.tolist(), accuracy.tolist()):
                perplexities[idx] = float(value)
                bit_accuracies[idx] = float(acc)

    return perplexities, bit_accuracies


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
    naive_baseline: str,
    imagegpt_row_layout: str,
    chess_encoding: str,
    chess_layout: str,
    chess_decode: str,
    music_encoding: str,
    music_decode: str,
    timemoe_encoding: str,
    timemoe_pulse_width: int,
    timemoe_repeat_width: int,
    timesfm_encoding: str,
    timesfm_layout: str,
    protein_format: str,
    shuffle_lm: bool,
    nucleotide_bits: bool,
    symbol_bits: Optional[str],
    drop_nucleotide: Optional[str],
    per_trial_random: bool,
    raw_bits: bool,
    force_binary_tokens: bool,
    nextterm_digits: bool,
    verbose: bool,
    ablate_labels: bool,
    perplexity_eval: bool,
    perplexity_batch_size: int,
    invert_binary_logits_flag: bool,
    uniform_binary_logits_flag: bool,
    ppl_select: bool,
    completion_batch_size: int = 32,
    prefill_batch_size: int = 8,
    precomputed_specs: Optional[List[Dict[str, Any]]] = None,
    precomputed_outputs: Optional[List[str]] = None,
    return_specs_only: bool = False,
    progress: Optional[tqdm] = None,
):
    trials: List[TrialResult] = []

    if backend == "naive":
        predictor = NAIVE_BASELINES.get(naive_baseline)
        if predictor is None:
            raise ValueError(f"Unsupported naive baseline: {naive_baseline}")
    else:
        predictor = None

    def build_prompt_config() -> PromptConfig:
        if backend == "protein" and protein_format == "progen2":
            return make_progen2_prompt_config(random)
        if backend in {"rita", "protein"}:
            return make_rita_prompt_config(random)
        if nextterm_digits:
            return make_nextterm_digit_prompt_config(random)
        if raw_bits and backend == "mlx" and symbol_bits:
            return make_symbol_prompt_config(random, symbol_bits, raw_bits=True)
        if raw_bits:
            return raw_bits_prompt_config()
        if backend == "evo" or (backend == "mlx" and nucleotide_bits):
            config = make_evo_prompt_config(
                random,
                drop_nucleotide=drop_nucleotide,
            )
            if config.decode_map is None:
                raise ValueError("Evo backend requires a decode map")
            return config
        if backend == "mlx" and symbol_bits:
            return make_symbol_prompt_config(random, symbol_bits)
        if backend == "mlx" and shuffle_lm:
            return make_lm_shuffle_prompt_config(random)
        return default_prompt_config()

    if precomputed_specs is not None:
        prompt_configs = []
    elif backend in {"rita", "protein"}:
        prompt_configs = [build_prompt_config() for _ in range(trials_per_program)]
    elif nextterm_digits:
        prompt_configs = [build_prompt_config() for _ in range(trials_per_program)]
    elif per_trial_random:
        prompt_configs = [build_prompt_config() for _ in range(trials_per_program)]
    else:
        base_config = build_prompt_config()
        prompt_configs = [base_config] * trials_per_program

    trial_specs: List[Dict[str, Any]] = []
    for trial_idx in range(0 if precomputed_specs is not None else trials_per_program):
        if backend == "imagegpt":
            trial_specs.append(
                make_imagegpt_trial_spec(
                    program,
                    in_context_examples=in_context_examples,
                    bit_length=bit_length,
                    max_new_tokens=max_new_tokens,
                    imagegpt_row_layout=imagegpt_row_layout,
                    ablate_labels=ablate_labels,
                    verbose=verbose,
                )
            )
            continue
        if backend == "mnist":
            trial_specs.append(
                make_mnist_trial_spec(
                    program,
                    in_context_examples=in_context_examples,
                    bit_length=bit_length,
                    max_new_tokens=max_new_tokens,
                    ablate_labels=ablate_labels,
                    verbose=verbose,
                )
            )
            continue
        if backend == "chess":
            trial_specs.append(
                make_chess_trial_spec(
                    program,
                    in_context_examples=in_context_examples,
                    bit_length=bit_length,
                    max_new_tokens=max_new_tokens,
                    chess_encoding=chess_encoding,
                    chess_layout=chess_layout,
                    ablate_labels=ablate_labels,
                    verbose=verbose,
                )
            )
            continue
        if backend == "music":
            trial_specs.append(
                make_music_trial_spec(
                    program,
                    in_context_examples=in_context_examples,
                    bit_length=bit_length,
                    max_new_tokens=max_new_tokens,
                    music_encoding=music_encoding,
                    ablate_labels=ablate_labels,
                    verbose=verbose,
                    permute_fn=permute_labels_for_ablation,
                )
            )
            continue
        if backend == "timemoe":
            trial_specs.append(
                make_timemoe_trial_spec(
                    program,
                    in_context_examples=in_context_examples,
                    bit_length=bit_length,
                    max_new_tokens=max_new_tokens,
                    timemoe_encoding=timemoe_encoding,
                    pulse_width=timemoe_pulse_width,
                    repeat_width=timemoe_repeat_width,
                    ablate_labels=ablate_labels,
                    verbose=verbose,
                )
            )
            continue
        if backend == "timesfm":
            trial_specs.append(
                make_timesfm_trial_spec(
                    program,
                    in_context_examples=in_context_examples,
                    bit_length=bit_length,
                    max_new_tokens=max_new_tokens,
                    timesfm_encoding=timesfm_encoding,
                    timesfm_layout=timesfm_layout,
                    ablate_labels=ablate_labels,
                    verbose=verbose,
                )
            )
            continue

        prompt_config = prompt_configs[trial_idx]
        map_argument = prompt_config.map_fn
        if per_trial_random and prompt_config.map_fn is not None:
            map_argument = [prompt_config.map_fn] * (in_context_examples + 1)
        prompt, query_input, expected_output, _, query_input_raw = few_shot(
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
        prompt_for_examples = prompt
        if backend == "protein" and protein_format == "progen2":
            prompt = "1" + prompt
        spec: Dict[str, Any] = {
            "prompt": prompt,
            "query": query_input,
            "expected": expected_output,
            "query_unaltered": query_input_raw,
        }
        if map_argument is None:
            expected_mapped = expected_output
        elif callable(map_argument):
            expected_mapped = map_argument(expected_output)
        else:
            expected_mapped = map_argument[-1](expected_output)
        spec["expected_mapped"] = expected_mapped
        if map_argument is None:
            map_fn_for_output = None
        elif callable(map_argument):
            map_fn_for_output = map_argument
        else:
            map_fn_for_output = map_argument[-1]
        spec["map_fn_for_output"] = map_fn_for_output
        if backend == "naive":
            if prompt_config.apply_sep:
                spec["context_examples"] = extract_context_examples(
                    prompt,
                    newline_sep=prompt_config.newline_sep,
                    apply_sep=prompt_config.apply_sep,
                )
            else:
                spec["context_examples"] = []
        if prompt_config.decode_map is not None:
            spec["decode_map"] = prompt_config.decode_map
            spec["token_size"] = prompt_config.token_size
            if backend == "protein" and protein_format == "progen2":
                spec["allowed_token_ids"] = progen2_allowed_token_ids(
                    prompt_config.decode_map
                )
        if verbose:
            spec["few_shot_examples"] = collect_few_shot_examples(
                prompt_for_examples,
                prompt_config=prompt_config,
                bit_length=bit_length,
            )
        trial_specs.append(spec)

    if precomputed_specs is not None:
        trial_specs = precomputed_specs
    if return_specs_only:
        return trial_specs

    if ppl_select:
        if backend != "mlx":
            raise ValueError("--ppl-select is only supported with the mlx backend")
        if bit_length <= 0:
            raise ValueError("--ppl-select requires positive bit_length")
        candidate_bits = [
            format(idx, f"0{bit_length}b") for idx in range(1 << bit_length)
        ]
        all_candidate_specs: List[Dict[str, Any]] = []
        trial_candidate_ranges: List[Tuple[int, int, List[str]]] = []
        for spec in trial_specs:
            map_fn_for_output = spec.get("map_fn_for_output")
            if map_fn_for_output is None:
                candidate_mapped = candidate_bits
            else:
                candidate_mapped = [map_fn_for_output(bits) for bits in candidate_bits]
            start = len(all_candidate_specs)
            all_candidate_specs.extend(
                {"prompt": spec["prompt"], "expected_mapped": mapped}
                for mapped in candidate_mapped
            )
            trial_candidate_ranges.append((start, len(candidate_mapped), candidate_mapped))

        all_ppls, _ = compute_mlx_perplexities(
            model,
            tokenizer,
            all_candidate_specs,
            batch_size=perplexity_batch_size,
            invert_binary_logits_flag=invert_binary_logits_flag,
            uniform_binary_logits_flag=uniform_binary_logits_flag,
        )

        for spec, (start, count, candidate_mapped) in zip(trial_specs, trial_candidate_ranges):
            ppls = all_ppls[start : start + count]
            best_idx = min(range(len(ppls)), key=ppls.__getitem__)
            prediction_bits = candidate_bits[best_idx]
            prediction_raw = candidate_mapped[best_idx]
            distance = edit_distance(prediction_bits, spec["expected"])
            bit_accuracy = 1.0 - (distance / bit_length) if bit_length > 0 else None
            trials.append(
                TrialResult(
                    prompt=spec["prompt"],
                    query=spec["query"],
                    expected=spec["expected"],
                    prediction_raw=prediction_raw,
                    prediction=prediction_bits,
                    correct=prediction_bits == spec["expected"],
                    edit_distance=distance,
                    perplexity=ppls[best_idx],
                    bit_accuracy=bit_accuracy,
                    few_shot_examples=spec.get("few_shot_examples") if verbose else None,
                    query_unaltered=spec.get("query_unaltered") if verbose else None,
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

    if precomputed_outputs is not None:
        outputs = precomputed_outputs
    elif perplexity_eval:
        if backend != "mlx":
            raise ValueError("--perplexity-eval is only supported with the mlx backend")
        outputs = []
        perplexities, bit_accuracies = compute_mlx_perplexities(
            model,
            tokenizer,
            trial_specs,
            batch_size=perplexity_batch_size,
            invert_binary_logits_flag=invert_binary_logits_flag,
            uniform_binary_logits_flag=uniform_binary_logits_flag,
        )
    elif backend == "evo":
        outputs = generate_evo_completions(
            trial_specs,
            max_tokens=max_new_tokens,
        )
    elif backend == "imagegpt":
        outputs = generate_imagegpt_completions(
            model,
            trial_specs,
            max_tokens=max_new_tokens,
        )
    elif backend == "mnist":
        outputs = generate_mnist_completions(
            model,
            trial_specs,
            max_tokens=max_new_tokens,
        )
    elif backend == "chess":
        if chess_decode == "constrained":
            outputs = generate_chess_completions(
                model,
                tokenizer,
                trial_specs,
                max_bits=max_new_tokens,
                chess_encoding=chess_encoding,
            )
        elif chess_decode == "greedy":
            outputs = generate_chess_greedy_completions(
                model,
                tokenizer,
                trial_specs,
                max_bits=max_new_tokens,
                chess_encoding=chess_encoding,
            )
        else:
            raise ValueError(f"Unsupported chess decode mode: {chess_decode}")
    elif backend == "music":
        outputs = generate_music_completions(
            model,
            trial_specs,
            max_bits=max_new_tokens,
            music_decode=music_decode,
        )
    elif backend == "timemoe":
        outputs = generate_timemoe_completions(
            model,
            trial_specs,
            max_tokens=max_new_tokens,
        )
    elif backend == "timesfm":
        outputs = generate_timesfm_completions(
            model,
            trial_specs,
            max_tokens=max_new_tokens,
        )
    elif backend == "naive":
        outputs = [predictor(spec["query"], spec.get("context_examples", [])) for spec in trial_specs]
    else:
        prompts = [tokenizer.encode(spec["prompt"]) for spec in trial_specs]
        if prompts:
            logits_processors = None
            if force_binary_tokens or backend in {"rita", "protein"}:
                logits_processors = [
                    [
                        make_allowed_token_logits_processor(
                            spec.get("allowed_token_ids")
                            or resolve_allowed_generation_token_ids(
                                tokenizer,
                                spec.get("decode_map"),
                            ),
                            tokenizer.vocab_size,
                        )
                    ]
                    for spec in trial_specs
                ]
                outputs = batch_generate_with_logits_processors(
                    model,
                    tokenizer,
                    prompts,
                    verbose=False,
                    max_tokens=max_new_tokens,
                    logits_processors=logits_processors,
                    completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size,
                )
            else:
                batch = batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    verbose=False,
                    max_tokens=max_new_tokens,
                    completion_batch_size=completion_batch_size,
                    prefill_batch_size=prefill_batch_size,
                )
                outputs = batch.texts
        else:
            outputs = []

    if perplexity_eval:
        for spec, perplexity, bit_accuracy in zip(trial_specs, perplexities, bit_accuracies):
            trials.append(
                TrialResult(
                    prompt=spec["prompt"],
                    query=spec["query"],
                    expected=spec["expected"],
                    prediction_raw=None,
                    prediction=None,
                    correct=None,
                    edit_distance=None,
                    perplexity=perplexity,
                    bit_accuracy=bit_accuracy,
                    few_shot_examples=spec.get("few_shot_examples") if verbose else None,
                    query_unaltered=spec.get("query_unaltered") if verbose else None,
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

    for spec, raw_output in zip(trial_specs, outputs):
        decode_map = spec.get("decode_map")
        token_size = spec.get("token_size", 1)
        decoded_output = None

        if decode_map is not None:
            decoded_output = decode_mapped_prediction(
                raw_output,
                decode_map=decode_map,
                token_size=token_size,
            )

        if backend == "evo":
            prediction_source = decoded_output or ""
            error = spec.get("error")
            if error and not raw_output:
                raw_output = f"<error: {error}>"
        elif backend == "chess":
            prediction_source = raw_output
            raw_output = spec.get("chess_generated_pgn", raw_output)
        elif backend == "music":
            prediction_source = raw_output
            raw_output = spec.get("music_generated_pno", raw_output)
        elif backend == "timemoe":
            prediction_source = raw_output
            raw_output = json.dumps(spec.get("timemoe_generated_values", []))
        elif backend == "timesfm":
            prediction_source = raw_output
            raw_output = json.dumps(spec.get("timesfm_generated_values", []))
        elif decoded_output is not None:
            prediction_source = decoded_output
        else:
            prediction_source = raw_output

        prediction = normalize_prediction(prediction_source, bit_length)
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
                few_shot_examples=spec.get("few_shot_examples") if verbose else None,
                query_unaltered=spec.get("query_unaltered") if verbose else None,
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


# Process-lifetime cache so --shots reuses a single model load across shot counts.
_MODEL_CACHE: Dict[Any, Any] = {}


def main() -> None:
    args = parse_args()
    shots = args.shots if args.shots is not None else [args.in_context_examples]
    multi = args.shots is not None
    if multi and "{shot}" not in str(args.output):
        raise ValueError(
            "--shots requires --output to contain '{shot}', "
            "e.g. results/cell/clean/cell_ic{shot}.json"
        )
    for shot in shots:
        shot_args = copy.copy(args)
        shot_args.in_context_examples = shot
        if multi:
            shot_args.output = args.output.format(shot=shot)
        _run_once(shot_args)


def _run_once(args) -> None:
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
    if args.perplexity_batch_size <= 0:
        raise ValueError("perplexity_batch_size must be positive")

    backend = args.backend

    if backend == "mlx":
        if not args.model:
            raise ValueError("--model is required when using the mlx backend")
        cache_key = ("mlx", args.model)
        if cache_key in _MODEL_CACHE:
            model, tokenizer = _MODEL_CACHE[cache_key]
        else:
            model, tokenizer = load(args.model)
            _MODEL_CACHE[cache_key] = (model, tokenizer)
        #model.load_weights("model_step_10000.ckpt.safetensors")
    elif backend == "imagegpt":
        if not args.model:
            raise ValueError("--model is required when using the imagegpt backend")
        model_path = resolve_mlx_model_path(args.model)
        model, _ = load_model(
            model_path,
            lazy=False,
            get_model_classes=make_model_file_class_loader(model_path),
        )
        tokenizer = None
    elif backend == "mnist":
        if not args.model:
            raise ValueError("--model is required when using the mnist backend")
        model, _ = load_model(resolve_mlx_model_path(args.model), lazy=False)
        tokenizer = None
    elif backend == "chess":
        if not args.model:
            raise ValueError("--model is required when using the chess backend")
        model, tokenizer = load(args.model)
    elif backend == "music":
        if not args.model:
            raise ValueError("--model is required when using the music backend")
        model, _ = load_model(resolve_mlx_model_path(args.model), lazy=False)
        tokenizer = None
    elif backend in {"rita", "protein"}:
        if not args.model:
            raise ValueError("--model is required when using the protein backend")
        model, tokenizer = load(
            args.model,
            tokenizer_config={"trust_remote_code": True},
        )
        if backend == "protein" and args.protein_format == "progen2":
            validate_progen2_amino_acid_tokens(tokenizer)
    elif backend == "timemoe":
        requested_model = args.model if args.model else DEFAULT_TIMEMOE_200M_PATH
        timemoe_model_path = resolve_mlx_model_path(requested_model)
        model = TimeMoeMLX.from_pretrained(timemoe_model_path)
        tokenizer = None
        if not args.model:
            args.model = str(timemoe_model_path)
    elif backend == "timesfm":
        requested_model = args.model if args.model else DEFAULT_TIMESFM_2_5_PATH
        timesfm_model_path = resolve_mlx_model_path(requested_model)
        model = TimesFm2_5MLX.from_pretrained(
            timesfm_model_path,
            dtype=args.timesfm_dtype,
        )
        tokenizer = None
        if not args.model:
            args.model = str(timesfm_model_path)
    elif backend in {"evo", "naive"}:
        model = None
        tokenizer = None
    else:  # pragma: no cover
        raise ValueError(f"Unsupported backend: {backend}")

    if args.lmshuffle and backend != "mlx":
        raise ValueError("--lmshuffle is only supported with the mlx backend")
    if args.symbol_bits and backend != "mlx":
        raise ValueError("--symbol-bits is only supported with the mlx backend")
    if args.nucleotide_bits and backend == "naive":
        raise ValueError("--nucleotide-bits is not supported with the naive backend")
    if args.force_binary_tokens and backend not in {"mlx", "rita", "protein"}:
        raise ValueError("--force-binary-tokens is only configurable for mlx/protein; imagegpt/mnist/protein use constrained generation internally")
    if backend != "imagegpt" and args.imagegpt_row_layout != "left-pad":
        raise ValueError("--imagegpt-row-layout is only supported with the imagegpt backend")
    if backend != "chess" and args.chess_encoding != "toggle":
        raise ValueError("--chess-encoding is only supported with the chess backend")
    if backend != "music" and args.music_encoding != "pitch":
        raise ValueError("--music-encoding is only supported with the music backend")
    if backend != "music" and args.music_decode != "constrained":
        raise ValueError("--music-decode is only supported with the music backend")
    if backend != "chess" and args.chess_layout != "continuous":
        raise ValueError("--chess-layout is only supported with the chess backend")
    if backend == "chess" and args.chess_layout != "continuous" and args.chess_encoding != "toggle":
        raise ValueError("Non-continuous --chess-layout values currently require --chess-encoding toggle")
    if (
        backend == "chess"
        and args.chess_decode == "greedy"
        and args.chess_encoding not in {"toggle", "signal2-wipe", "signal2-wipe-sep"}
    ):
        raise ValueError(
            "--chess-decode greedy is currently only supported with --chess-encoding toggle or signal2-wipe"
        )
    if (
        backend == "chess"
        and args.chess_decode == "constrained"
        and args.chess_layout in {"bit-gamelet", "bit-gamelet-pair"}
    ):
        raise ValueError("--chess-decode constrained is not supported for bit-gamelet layouts")
    if args.lmshuffle and args.nucleotide_bits:
        raise ValueError("--lmshuffle cannot be combined with --nucleotide-bits")
    if args.symbol_bits and args.lmshuffle:
        raise ValueError("--symbol-bits cannot be combined with --lmshuffle")
    if args.symbol_bits and args.nucleotide_bits:
        raise ValueError("--symbol-bits cannot be combined with --nucleotide-bits")
    if args.nextterm_digits and backend != "mlx":
        raise ValueError("--nextterm-digits is only supported with the mlx backend")
    if backend not in {"rita", "protein"} and args.protein_format != "rita":
        raise ValueError("--protein-format is only supported with the protein/rita backends")
    if args.nextterm_digits and (args.raw_bits or args.lmshuffle or args.nucleotide_bits or args.symbol_bits):
        raise ValueError(
            "--nextterm-digits cannot be combined with --raw-bits, --lmshuffle, "
            "--nucleotide-bits, or --symbol-bits"
        )
    if args.drop_nucleotide and not (backend == "evo" or args.nucleotide_bits):
        raise ValueError(
            "--drop-nucleotide requires nucleotide encoding (evo backend or --nucleotide-bits)"
        )

    if backend == "imagegpt":
        incompatible_flags = [
            ("--lmshuffle", args.lmshuffle),
            ("--nucleotide-bits", args.nucleotide_bits),
            ("--symbol-bits", bool(args.symbol_bits)),
            ("--raw-bits", args.raw_bits),
            ("--nextterm-digits", args.nextterm_digits),
            ("--drop-nucleotide", bool(args.drop_nucleotide)),
            ("--force-binary-tokens", args.force_binary_tokens),
            ("--perplexity-eval", args.perplexity_eval),
            ("--ppl-select", args.ppl_select),
            ("--invert-binary-logits", args.invert_binary_logits),
            ("--uniform-binary-logits", args.uniform_binary_logits),
        ]
        active = [name for name, enabled in incompatible_flags if enabled]
        if active:
            raise ValueError(
                "ImageGPT backend uses its own row-padded color-token encoding; "
                f"cannot combine with {', '.join(active)}"
            )
        if args.bit_length > IMAGEGPT_ROW_WIDTH // 2:
            raise ValueError("--backend imagegpt requires --bit-length <= 16")

    if backend == "mnist":
        incompatible_flags = [
            ("--lmshuffle", args.lmshuffle),
            ("--nucleotide-bits", args.nucleotide_bits),
            ("--symbol-bits", bool(args.symbol_bits)),
            ("--raw-bits", args.raw_bits),
            ("--nextterm-digits", args.nextterm_digits),
            ("--drop-nucleotide", bool(args.drop_nucleotide)),
            ("--force-binary-tokens", args.force_binary_tokens),
            ("--perplexity-eval", args.perplexity_eval),
            ("--ppl-select", args.ppl_select),
            ("--invert-binary-logits", args.invert_binary_logits),
            ("--uniform-binary-logits", args.uniform_binary_logits),
        ]
        active = [name for name, enabled in incompatible_flags if enabled]
        if active:
            raise ValueError(
                "MNIST backend uses its own row-padded pixel-token encoding; "
                f"cannot combine with {', '.join(active)}"
            )
        if args.bit_length != 8:
            raise ValueError("--backend mnist currently requires --bit-length 8")
        if args.in_context_examples > MNIST_IMAGE_ROWS - 1:
            raise ValueError("--backend mnist requires --in-context-examples <= 27")

    if backend == "chess":
        incompatible_flags = [
            ("--lmshuffle", args.lmshuffle),
            ("--nucleotide-bits", args.nucleotide_bits),
            ("--symbol-bits", bool(args.symbol_bits)),
            ("--raw-bits", args.raw_bits),
            ("--nextterm-digits", args.nextterm_digits),
            ("--drop-nucleotide", bool(args.drop_nucleotide)),
            ("--force-binary-tokens", args.force_binary_tokens),
            ("--perplexity-eval", args.perplexity_eval),
            ("--ppl-select", args.ppl_select),
            ("--invert-binary-logits", args.invert_binary_logits),
            ("--uniform-binary-logits", args.uniform_binary_logits),
        ]
        active = [name for name, enabled in incompatible_flags if enabled]
        if active:
            raise ValueError(
                "Chess backend uses PGN chess generation; "
                f"cannot combine with {', '.join(active)}"
            )

    if backend == "timemoe":
        incompatible_flags = [
            ("--lmshuffle", args.lmshuffle),
            ("--nucleotide-bits", args.nucleotide_bits),
            ("--symbol-bits", bool(args.symbol_bits)),
            ("--raw-bits", args.raw_bits),
            ("--nextterm-digits", args.nextterm_digits),
            ("--drop-nucleotide", bool(args.drop_nucleotide)),
            ("--force-binary-tokens", args.force_binary_tokens),
            ("--perplexity-eval", args.perplexity_eval),
            ("--ppl-select", args.ppl_select),
            ("--invert-binary-logits", args.invert_binary_logits),
            ("--uniform-binary-logits", args.uniform_binary_logits),
        ]
        active = [name for name, enabled in incompatible_flags if enabled]
        if active:
            raise ValueError(
                "TimeMoE backend uses numeric time-series encodings; "
                f"cannot combine with {', '.join(active)}"
            )

    if backend == "timesfm":
        incompatible_flags = [
            ("--lmshuffle", args.lmshuffle),
            ("--nucleotide-bits", args.nucleotide_bits),
            ("--symbol-bits", bool(args.symbol_bits)),
            ("--raw-bits", args.raw_bits),
            ("--nextterm-digits", args.nextterm_digits),
            ("--drop-nucleotide", bool(args.drop_nucleotide)),
            ("--force-binary-tokens", args.force_binary_tokens),
            ("--perplexity-eval", args.perplexity_eval),
            ("--ppl-select", args.ppl_select),
            ("--invert-binary-logits", args.invert_binary_logits),
            ("--uniform-binary-logits", args.uniform_binary_logits),
        ]
        active = [name for name, enabled in incompatible_flags if enabled]
        if active:
            raise ValueError(
                "TimesFM backend uses numeric patch encodings; "
                f"cannot combine with {', '.join(active)}"
            )

    if backend in {"rita", "protein"}:
        incompatible_flags = [
            ("--lmshuffle", args.lmshuffle),
            ("--nucleotide-bits", args.nucleotide_bits),
            ("--symbol-bits", bool(args.symbol_bits)),
            ("--raw-bits", args.raw_bits),
            ("--nextterm-digits", args.nextterm_digits),
            ("--drop-nucleotide", bool(args.drop_nucleotide)),
            ("--perplexity-eval", args.perplexity_eval),
            ("--ppl-select", args.ppl_select),
            ("--invert-binary-logits", args.invert_binary_logits),
            ("--uniform-binary-logits", args.uniform_binary_logits),
        ]
        active = [name for name, enabled in incompatible_flags if enabled]
        if active:
            raise ValueError(
                "Protein backend uses prompt-local amino-acid encodings; "
                f"cannot combine with {', '.join(active)}"
            )

    if args.raw_bits:
        if backend == "evo":
            raise ValueError("--raw-bits is not supported with the evo backend")
        if args.lmshuffle:
            raise ValueError("--raw-bits cannot be used with --lmshuffle")
        if args.nucleotide_bits:
            raise ValueError("--raw-bits cannot be used with --nucleotide-bits")

    if args.per_trial_random:
        if backend == "naive":
            raise ValueError("--per-trial-random is not supported with the naive backend")
        if backend == "mlx" and not (args.lmshuffle or args.nucleotide_bits or args.symbol_bits or args.nextterm_digits):
            raise ValueError(
                "--per-trial-random requires --lmshuffle, --nucleotide-bits, "
                "--symbol-bits, or --nextterm-digits when using the mlx backend"
            )
    if args.perplexity_eval and backend != "mlx":
        raise ValueError("--perplexity-eval is only supported with the mlx backend")
    if args.invert_binary_logits and not args.perplexity_eval:
        if not args.ppl_select:
            raise ValueError("--invert-binary-logits requires --perplexity-eval or --ppl-select")
    if args.uniform_binary_logits and not args.perplexity_eval:
        if not args.ppl_select:
            raise ValueError("--uniform-binary-logits requires --perplexity-eval or --ppl-select")
    if args.uniform_binary_logits and args.invert_binary_logits:
        raise ValueError("--uniform-binary-logits cannot be combined with --invert-binary-logits")
    if args.ppl_select and args.perplexity_eval:
        raise ValueError("--ppl-select cannot be combined with --perplexity-eval")
    if args.ppl_select and backend != "mlx":
        raise ValueError("--ppl-select is only supported with the mlx backend")
    if args.force_binary_tokens and (args.ppl_select or args.perplexity_eval):
        raise ValueError("--force-binary-tokens only applies to generation, not perplexity modes")

    max_tokens = args.max_new_tokens or args.bit_length
    if backend == "timemoe" and args.max_new_tokens is None:
        max_tokens = timemoe_max_new_values(
            bit_length=args.bit_length,
            timemoe_encoding=args.timemoe_encoding,
            pulse_width=args.timemoe_pulse_width,
            repeat_width=args.timemoe_repeat_width,
        )
    if max_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if backend == "imagegpt" and max_tokens != args.bit_length:
        raise ValueError("--backend imagegpt requires max_new_tokens to equal bit_length")
    if backend == "mnist" and max_tokens != args.bit_length:
        raise ValueError("--backend mnist requires max_new_tokens to equal bit_length")
    if backend == "chess" and max_tokens != args.bit_length:
        raise ValueError("--backend chess requires max_new_tokens to equal bit_length")
    if backend == "timemoe":
        expected_timemoe_new_values = timemoe_max_new_values(
            bit_length=args.bit_length,
            timemoe_encoding=args.timemoe_encoding,
            pulse_width=args.timemoe_pulse_width,
            repeat_width=args.timemoe_repeat_width,
        )
        if max_tokens != expected_timemoe_new_values:
            raise ValueError(
                "--backend timemoe requires max_new_tokens to equal generated scalar "
                f"values for the encoding ({expected_timemoe_new_values})"
            )
    if backend == "timesfm" and max_tokens != args.bit_length:
        raise ValueError("--backend timesfm requires max_new_tokens to equal bit_length")
    if backend in {"rita", "protein"} and max_tokens != args.bit_length:
        raise ValueError("--backend protein requires max_new_tokens to equal bit_length")
    if backend == "imagegpt":
        row_pad = IMAGEGPT_ROW_WIDTH - (2 * args.bit_length)
        if args.imagegpt_row_layout == "dense":
            pair_width = 2 * args.bit_length
            query_prefix = pair_width if args.in_context_examples % 2 == 0 else 0
            prompt_token_count = (
                1
                + (args.in_context_examples * pair_width)
                + query_prefix
                + args.bit_length
            )
        else:
            prompt_token_count = (
                1
                + (args.in_context_examples * IMAGEGPT_ROW_WIDTH)
                + row_pad
                + args.bit_length
            )
        if prompt_token_count + max_tokens - 1 > IMAGEGPT_ROW_WIDTH * IMAGEGPT_ROW_WIDTH:
            raise ValueError(
                "ImageGPT row layout exceeds the 32x32 context: "
                f"prompt_tokens={prompt_token_count}, max_new_tokens={max_tokens}"
            )
    if backend == "mnist":
        row_pad = MNIST_ROW_WIDTH - (2 * args.bit_length)
        prompt_token_count = (
            1
            + (args.in_context_examples * MNIST_ROW_WIDTH)
            + row_pad
            + args.bit_length
        )
        max_canvas_tokens = 1 + (MNIST_IMAGE_ROWS * MNIST_ROW_WIDTH)
        if prompt_token_count + max_tokens - 1 > max_canvas_tokens:
            raise ValueError(
                "MNIST row layout exceeds the 28x28 training canvas: "
                f"prompt_tokens={prompt_token_count}, max_new_tokens={max_tokens}, "
                f"canvas_tokens={max_canvas_tokens}"
            )
    if backend == "chess":
        final_chars = chess_layout_max_context_chars(
            in_context_examples=args.in_context_examples,
            bit_length=args.bit_length,
            max_new_tokens=max_tokens,
            chess_encoding=args.chess_encoding,
            chess_layout=args.chess_layout,
        )
        if final_chars > CHESSGPT_CONTEXT_LIMIT:
            raise ValueError(
                "Chess layout exceeds Chess-GPT context: "
                f"shots={args.in_context_examples}, bit_length={args.bit_length}, "
                f"encoding={args.chess_encoding}, layout={args.chess_layout}, "
                f"chars={final_chars}, "
                f"limit={CHESSGPT_CONTEXT_LIMIT}"
            )
    if backend == "timemoe":
        prompt_value_count = timemoe_prompt_value_count(
            in_context_examples=args.in_context_examples,
            bit_length=args.bit_length,
            timemoe_encoding=args.timemoe_encoding,
            pulse_width=args.timemoe_pulse_width,
            repeat_width=args.timemoe_repeat_width,
        )
        context_limit = int(
            getattr(model, "config", {}).get(
                "max_position_embeddings",
                TIMEMOE_CONTEXT_LIMIT,
            )
        )
        if prompt_value_count + max_tokens - 1 > context_limit:
            raise ValueError(
                "TimeMoE layout exceeds model context: "
                f"prompt_values={prompt_value_count}, max_new_tokens={max_tokens}, "
                f"limit={context_limit}"
            )
    if backend == "timesfm":
        prompt_value_count = timesfm_prompt_value_count(
            in_context_examples=args.in_context_examples,
            bit_length=args.bit_length,
            timesfm_encoding=args.timesfm_encoding,
            timesfm_layout=args.timesfm_layout,
        )
        context_limit = int(
            getattr(model, "context_length", TIMESFM_CONTEXT_LIMIT)
        )
        generated_values = (
            timesfm_bit_patch_count(max_tokens, args.timesfm_encoding)
            * TIMESFM_PATCH_LENGTH
        )
        if prompt_value_count + generated_values - 1 > context_limit:
            raise ValueError(
                "TimesFM layout exceeds model context: "
                f"prompt_values={prompt_value_count}, generated_values={generated_values}, "
                f"limit={context_limit}"
            )
    if backend in {"rita", "protein"}:
        prompt_token_count = (
            args.in_context_examples * ((2 * args.bit_length) + 1)
            + args.bit_length
        )
        if backend == "protein" and args.protein_format == "progen2":
            prompt_token_count += 1
        if prompt_token_count + max_tokens - 1 > PROTEIN_CONTEXT_LIMIT:
            raise ValueError(
                "Protein amino-acid layout exceeds model context: "
                f"prompt_tokens={prompt_token_count}, max_new_tokens={max_tokens}, "
                f"limit={PROTEIN_CONTEXT_LIMIT}"
            )

    task_results: List[TaskResult] = []

    total_predictions = len(program_entries) * args.trials_per_program

    def _eval(program, description, *, progress=None, **extra):
        return evaluate_task(
            model,
            tokenizer,
            program,
            description=description,
            trials_per_program=args.trials_per_program,
            in_context_examples=args.in_context_examples,
            bit_length=args.bit_length,
            max_new_tokens=max_tokens,
            backend=backend,
            naive_baseline=args.naive_baseline,
            imagegpt_row_layout=args.imagegpt_row_layout,
            chess_encoding=args.chess_encoding,
            chess_layout=args.chess_layout,
            chess_decode=args.chess_decode,
            music_encoding=args.music_encoding,
            music_decode=args.music_decode,
            timemoe_encoding=args.timemoe_encoding,
            timemoe_pulse_width=args.timemoe_pulse_width,
            timemoe_repeat_width=args.timemoe_repeat_width,
            timesfm_encoding=args.timesfm_encoding,
            timesfm_layout=args.timesfm_layout,
            protein_format=args.protein_format,
            shuffle_lm=args.lmshuffle,
            nucleotide_bits=args.nucleotide_bits,
            symbol_bits=args.symbol_bits,
            drop_nucleotide=args.drop_nucleotide,
            per_trial_random=args.per_trial_random,
            raw_bits=args.raw_bits,
            force_binary_tokens=args.force_binary_tokens,
            nextterm_digits=args.nextterm_digits,
            verbose=args.verbose,
            ablate_labels=args.ablate_labels,
            perplexity_eval=args.perplexity_eval,
            perplexity_batch_size=args.perplexity_batch_size,
            invert_binary_logits_flag=args.invert_binary_logits,
            uniform_binary_logits_flag=args.uniform_binary_logits,
            ppl_select=args.ppl_select,
            completion_batch_size=args.batch_size_completion,
            prefill_batch_size=args.batch_size_prefill,
            progress=progress,
            **extra,
        )

    if args.batch_programs:
        if backend != "mlx" or args.perplexity_eval or args.ppl_select:
            raise ValueError(
                "--batch-programs requires the standard mlx generation path "
                "(backend=mlx, no --perplexity-eval/--ppl-select)."
            )
        with tqdm(total=total_predictions, desc="Evaluations") as progress:
            # Phase 1: build every program's specs first. Spec building is the only
            # consumer of the RNG and runs in program order, so the prompts are
            # byte-identical to the per-program path; generation does not touch RNG.
            built: List[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
            for entry in program_entries:
                specs = _eval(
                    entry["program"],
                    entry.get("description"),
                    return_specs_only=True,
                )
                built.append((entry, specs))
            # Phase 2: a single batched generation over every program's prompts.
            flat_specs = [spec for _, specs in built for spec in specs]
            prompts = [tokenizer.encode(spec["prompt"]) for spec in flat_specs]
            if not prompts:
                outputs_all: List[str] = []
            elif args.force_binary_tokens:
                logits_processors = [
                    [
                        make_allowed_token_logits_processor(
                            spec.get("allowed_token_ids")
                            or resolve_allowed_generation_token_ids(
                                tokenizer, spec.get("decode_map")
                            ),
                            tokenizer.vocab_size,
                        )
                    ]
                    for spec in flat_specs
                ]
                outputs_all = batch_generate_with_logits_processors(
                    model,
                    tokenizer,
                    prompts,
                    verbose=False,
                    max_tokens=max_tokens,
                    logits_processors=logits_processors,
                    completion_batch_size=args.batch_size_completion,
                    prefill_batch_size=args.batch_size_prefill,
                )
            else:
                outputs_all = batch_generate(
                    model,
                    tokenizer,
                    prompts,
                    verbose=False,
                    max_tokens=max_tokens,
                    completion_batch_size=args.batch_size_completion,
                    prefill_batch_size=args.batch_size_prefill,
                ).texts
            # Phase 3: score each program against its slice of the outputs.
            pos = 0
            for idx, (entry, specs) in enumerate(built):
                n = len(specs)
                task = _eval(
                    entry["program"],
                    entry.get("description"),
                    precomputed_specs=specs,
                    precomputed_outputs=outputs_all[pos : pos + n],
                )
                pos += n
                task.index = idx
                task_results.append(task)
                progress.update(n)
    else:
        with tqdm(total=total_predictions, desc="Evaluations") as progress:
            for idx, entry in enumerate(program_entries):
                task = _eval(
                    entry["program"],
                    entry.get("description"),
                    progress=progress,
                )
                task.index = idx
                task_results.append(task)

    accuracies: List[float] = []
    for task in task_results:
        value = task.accuracy()
        if value is not None:
            accuracies.append(value)
    if accuracies:
        mean_accuracy = statistics.mean(accuracies)
        if len(accuracies) > 1:
            std_dev = statistics.stdev(accuracies)
            stderr = std_dev / math.sqrt(len(accuracies))
        else:
            stderr = 0.0
    else:
        mean_accuracy = None
        stderr = None

    avg_edit_distances: List[float] = []
    for task in task_results:
        value = task.average_edit_distance()
        if value is not None:
            avg_edit_distances.append(value)
    if avg_edit_distances:
        mean_edit_distance = statistics.mean(avg_edit_distances)
        if len(avg_edit_distances) > 1:
            edit_std = statistics.stdev(avg_edit_distances)
            edit_stderr = edit_std / math.sqrt(len(avg_edit_distances))
        else:
            edit_stderr = 0.0
    else:
        mean_edit_distance = None
        edit_stderr = None

    avg_perplexities: List[float] = []
    for task in task_results:
        value = task.average_perplexity()
        if value is not None:
            avg_perplexities.append(value)
    if avg_perplexities:
        mean_perplexity = statistics.mean(avg_perplexities)
        if len(avg_perplexities) > 1:
            ppl_std = statistics.stdev(avg_perplexities)
            perplexity_stderr = ppl_std / math.sqrt(len(avg_perplexities))
        else:
            perplexity_stderr = 0.0
    else:
        mean_perplexity = None
        perplexity_stderr = None

    avg_bit_accuracies: List[float] = []
    for task in task_results:
        value = task.average_bit_accuracy()
        if value is not None:
            avg_bit_accuracies.append(value)
    if avg_bit_accuracies:
        mean_bit_accuracy = statistics.mean(avg_bit_accuracies)
        if len(avg_bit_accuracies) > 1:
            bit_acc_std = statistics.stdev(avg_bit_accuracies)
            bit_accuracy_stderr = bit_acc_std / math.sqrt(len(avg_bit_accuracies))
        else:
            bit_accuracy_stderr = 0.0
    else:
        mean_bit_accuracy = None
        bit_accuracy_stderr = None

    output_data = {
        "config": {
            "model": args.model,
            "programs_file": portable_path(programs_path),
            "in_context_examples": args.in_context_examples,
            "trials_per_program": args.trials_per_program,
            "backend": backend,
            "naive_baseline": args.naive_baseline if backend == "naive" else None,
            "lmshuffle": args.lmshuffle,
            "nucleotide_bits": args.nucleotide_bits,
            "symbol_bits": args.symbol_bits,
            "drop_nucleotide": args.drop_nucleotide,
            "per_trial_random": args.per_trial_random,
            "raw_bits": args.raw_bits,
            "nextterm_digits": args.nextterm_digits,
            "imagegpt_row_width": IMAGEGPT_ROW_WIDTH if backend == "imagegpt" else None,
            "imagegpt_row_layout": args.imagegpt_row_layout if backend == "imagegpt" else None,
            "imagegpt_color_vocab_size": IMAGEGPT_COLOR_VOCAB_SIZE if backend == "imagegpt" else None,
            "imagegpt_sos_token_id": IMAGEGPT_SOS_TOKEN_ID if backend == "imagegpt" else None,
            "imagegpt_per_trial_random_colors": backend == "imagegpt",
            "imagegpt_constrained_binary_generation": backend == "imagegpt",
            "mnist_row_width": MNIST_ROW_WIDTH if backend == "mnist" else None,
            "mnist_image_rows": MNIST_IMAGE_ROWS if backend == "mnist" else None,
            "mnist_pixel_vocab_size": MNIST_PIXEL_VOCAB_SIZE if backend == "mnist" else None,
            "mnist_label_offset": MNIST_LABEL_OFFSET if backend == "mnist" else None,
            "mnist_pad_token_id": MNIST_PAD_TOKEN_ID if backend == "mnist" else None,
            "mnist_zero_token_id": MNIST_ZERO_TOKEN_ID if backend == "mnist" else None,
            "mnist_one_token_id": MNIST_ONE_TOKEN_ID if backend == "mnist" else None,
            "mnist_per_trial_random_label": backend == "mnist",
            "mnist_constrained_binary_generation": backend == "mnist",
            "chess_context_limit": CHESSGPT_CONTEXT_LIMIT if backend == "chess" else None,
            "chess_encoding": (
                args.chess_encoding
                if backend == "chess"
                else None
            ),
            "chess_layout": args.chess_layout if backend == "chess" else None,
            "chess_decode": args.chess_decode if backend == "chess" else None,
            "chess_constrained_binary_generation": (
                backend == "chess" and args.chess_decode == "constrained"
            ),
            "timemoe_context_limit": (
                int(
                    getattr(model, "config", {}).get(
                        "max_position_embeddings",
                        TIMEMOE_CONTEXT_LIMIT,
                    )
                )
                if backend == "timemoe"
                else None
            ),
            "timemoe_separator_value": (
                TIMEMOE_SEPARATOR_VALUE if backend == "timemoe" else None
            ),
            "timemoe_zero_value": (
                TIMEMOE_ZERO_VALUE if backend == "timemoe" else None
            ),
            "timemoe_one_value": (
                TIMEMOE_ONE_VALUE if backend == "timemoe" else None
            ),
            "timemoe_encoding": args.timemoe_encoding if backend == "timemoe" else None,
            "timemoe_pulse_width": (
                args.timemoe_pulse_width if backend == "timemoe" else None
            ),
            "timemoe_repeat_width": (
                args.timemoe_repeat_width if backend == "timemoe" else None
            ),
            "timemoe_decode": (
                "chunk-mean-sign"
                if backend == "timemoe"
                and args.timemoe_encoding in {"sine-pulse", "repeat-symbol"}
                else ("sign" if backend == "timemoe" else None)
            ),
            "timemoe_one_step_ar": backend == "timemoe",
            "timesfm_context_limit": (
                int(getattr(model, "context_length", TIMESFM_CONTEXT_LIMIT))
                if backend == "timesfm"
                else None
            ),
            "timesfm_patch_length": TIMESFM_PATCH_LENGTH if backend == "timesfm" else None,
            "timesfm_horizon_length": TIMESFM_HORIZON_LENGTH if backend == "timesfm" else None,
            "timesfm_separator_value": (
                TIMESFM_SEPARATOR_VALUE if backend == "timesfm" else None
            ),
            "timesfm_zero_value": TIMESFM_ZERO_VALUE if backend == "timesfm" else None,
            "timesfm_one_value": TIMESFM_ONE_VALUE if backend == "timesfm" else None,
            "timesfm_dtype": args.timesfm_dtype if backend == "timesfm" else None,
            "timesfm_encoding": args.timesfm_encoding if backend == "timesfm" else None,
            "timesfm_layout": args.timesfm_layout if backend == "timesfm" else None,
            "timesfm_decode": (
                (
                    "sub-lobe-projection"
                    if timesfm_uses_sublobe_projection(args.timesfm_encoding)
                    else "patch-mean-sign"
                )
                if backend == "timesfm"
                else None
            ),
            "timesfm_force_flip_invariance": backend == "timesfm",
            "timesfm_feedback": (
                (
                    "quantized-sub-lobes"
                    if timesfm_uses_sublobe_projection(args.timesfm_encoding)
                    else (
                        "quantized-signed-lobe"
                        if args.timesfm_encoding == "sine-lobe"
                        else "quantized-patch-sign"
                    )
                )
                if backend == "timesfm"
                else None
            ),
            "protein_context_limit": (
                PROTEIN_CONTEXT_LIMIT if backend in {"rita", "protein"} else None
            ),
            "protein_format": (
                args.protein_format if backend in {"rita", "protein"} else None
            ),
            "protein_amino_acid_pool": (
                list(
                    PROGEN2_CANONICAL_AMINO_ACID_TOKEN_IDS
                    if backend == "protein" and args.protein_format == "progen2"
                    else CANONICAL_AMINO_ACIDS
                )
                if backend in {"rita", "protein"}
                else None
            ),
            "protein_progen2_amino_acid_token_ids": (
                PROGEN2_CANONICAL_AMINO_ACID_TOKEN_IDS
                if backend == "protein" and args.protein_format == "progen2"
                else None
            ),
            "protein_prompt_local_random_amino_acids": backend in {"rita", "protein"},
            "protein_constrained_binary_generation": backend in {"rita", "protein"},
            "rita_context_limit": RITA_CONTEXT_LIMIT if backend == "rita" else None,
            "rita_amino_acid_pool": (
                list(CANONICAL_AMINO_ACIDS) if backend == "rita" else None
            ),
            "rita_prompt_local_random_amino_acids": backend == "rita",
            "rita_constrained_binary_generation": backend == "rita",
            "bit_length": args.bit_length,
            "max_new_tokens": max_tokens,
            "seed": args.seed,
            "verbose": args.verbose,
            "ablate_labels": args.ablate_labels,
            "invert_binary_logits": args.invert_binary_logits,
            "uniform_binary_logits": args.uniform_binary_logits,
            "ppl_select": args.ppl_select,
            "perplexity_eval": args.perplexity_eval,
            "perplexity_batch_size": args.perplexity_batch_size,
        },
        "overall": {
            "mean_accuracy": mean_accuracy,
            "stderr": stderr,
            "mean_edit_distance": mean_edit_distance,
            "edit_distance_stderr": edit_stderr,
            "mean_perplexity": mean_perplexity,
            "perplexity_stderr": perplexity_stderr,
            "mean_bit_accuracy": mean_bit_accuracy,
            "bit_accuracy_stderr": bit_accuracy_stderr,
        },
        "tasks": [task.to_json() for task in task_results],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_data, handle, indent=2)

    print(f"Wrote results for {len(task_results)} programs to {output_path}")
    if args.perplexity_eval:
        if mean_perplexity is None:
            print("Mean perplexity: n/a")
        elif perplexity_stderr is None:
            print(f"Mean perplexity: {mean_perplexity:.4f}")
        else:
            print(
                f"Mean perplexity: {mean_perplexity:.4f} (stderr {perplexity_stderr:.4f})"
            )
        if mean_bit_accuracy is None:
            print("Mean bit accuracy: n/a")
        elif bit_accuracy_stderr is None:
            print(f"Mean bit accuracy: {mean_bit_accuracy * 100:.2f}%")
        else:
            print(
                f"Mean bit accuracy: {mean_bit_accuracy * 100:.2f}% "
                f"(stderr {bit_accuracy_stderr * 100:.2f}%)"
            )
    else:
        if mean_accuracy is None:
            print("Mean accuracy: n/a")
        elif stderr is None:
            print(f"Mean accuracy: {mean_accuracy:.4f}")
        else:
            print(
                f"Mean accuracy: {mean_accuracy:.4f} (stderr {stderr:.4f})"
            )
        if mean_edit_distance is None:
            print("Mean edit distance: n/a")
        elif edit_stderr is None:
            print(f"Mean edit distance: {mean_edit_distance:.4f}")
        else:
            print(
                f"Mean edit distance: {mean_edit_distance:.4f} (stderr {edit_stderr:.4f})"
            )


if __name__ == "__main__":
    main()
