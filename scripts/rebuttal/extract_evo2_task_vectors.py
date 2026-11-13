#!/usr/bin/env python3
"""Task-vector readout for Evo2 (genome), matching program_synth.py --dump-hidden-states.

Builds the exact prompts of the paper's Evo2 runner (run_evo2_h100_clean_deranged.py:
per-trial random nucleotides for 0, 1 and the separator; seed re-set per condition),
then runs one forward pass per prompt with passive forward hooks on every block and
keeps the residual stream at the last prompt position. With --paired-ablation the
label derangement draws from a separate generator, so each deranged prompt is the
twin of the clean prompt of the same seed.

Output (.npz), same layout as task_vector_extraction.py:
  half_means      float32 [tasks, 2, blocks + 1, hidden]  (index 0 is left at zero)
  half_counts     int32   [tasks, 2]
  prompt_tokens   int32   [prompts, length]
  task_of_prompt  int32   [prompts]
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import program_synth_evo_smartbatch as runner  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="evo2_7b")
    parser.add_argument("--local-path", default=None, help="Path to the .pt checkpoint (optional)")
    parser.add_argument("--programs", type=Path, default=REPO_ROOT / "src/curated_transformations.jsonl")
    parser.add_argument("--in-context-examples", type=int, default=64)
    parser.add_argument("--trials-per-program", type=int, default=128)
    parser.add_argument("--bit-length", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablate-labels", action="store_true")
    parser.add_argument("--paired-ablation", action="store_true")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-programs", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # 1. Prompts, exactly as the paper's runner builds them; generation is replaced by capture.
    captured: list = []

    def capture(model, trial_specs, *, max_tokens):
        captured.append([spec["prompt"] for spec in trial_specs])
        return ["" for _ in trial_specs]

    runner.generate_evo_completions = capture
    if args.paired_ablation:
        ablation_rng = random.Random(args.seed + 1_000_003)
        original_permute = runner.permute_labels_for_ablation
        runner.permute_labels_for_ablation = lambda outputs, rng: original_permute(outputs, ablation_rng)

    programs = runner.load_programs(args.programs)
    if args.limit_programs:
        programs = programs[: args.limit_programs]
    random.seed(args.seed)
    for entry in programs:
        runner.evaluate_task(
            None, None, entry["program"], description=entry.get("description"),
            trials_per_program=args.trials_per_program, in_context_examples=args.in_context_examples,
            bit_length=args.bit_length, max_new_tokens=args.bit_length, backend="evo",
            per_trial_random=True, ablate_labels=args.ablate_labels,
        )
    task_sizes = [len(task) for task in captured]
    prompts = [prompt for task in captured for prompt in task]
    lengths = {len(p) for p in prompts}
    if len(lengths) != 1:
        raise ValueError(f"Prompts differ in length: {sorted(lengths)}")

    # 2. Model and hooks.
    runner.patch_torch_load_for_evo2()
    from evo2 import Evo2

    evo = Evo2(args.model, local_path=args.local_path) if args.local_path else Evo2(args.model)
    blocks = list(evo.model.blocks)
    states: dict = {}

    def make_hook(index):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            states[index] = hidden[:, -1, :].float().detach().cpu().numpy()
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(blocks)]
    device = next(evo.model.parameters()).device
    tokens = np.array([evo.tokenizer.tokenize(p) for p in prompts], dtype=np.int32)
    print(f"[hidden-states] {len(prompts)} prompts of {tokens.shape[1]} tokens, {len(blocks)} blocks, device {device}")

    task_of_prompt = np.repeat(np.arange(len(task_sizes)), task_sizes)
    trial_of_prompt = np.concatenate([np.arange(n) for n in task_sizes])
    sums = None
    counts = np.zeros((len(task_sizes), 2), dtype=np.int32)
    with torch.inference_mode():
        for start in tqdm(range(0, len(prompts), args.batch_size), desc="Forward"):
            batch = torch.tensor(tokens[start : start + args.batch_size], dtype=torch.long, device=device)
            evo(batch)
            stacked = np.stack([states[i] for i in range(len(blocks))], axis=1)  # [batch, blocks, hidden]
            if sums is None:
                sums = np.zeros((len(task_sizes), 2, len(blocks) + 1, stacked.shape[-1]), dtype=np.float64)
            for row in range(stacked.shape[0]):
                i = start + row
                half = int(trial_of_prompt[i] % 2)
                sums[task_of_prompt[i], half, 1:] += stacked[row]
                counts[task_of_prompt[i], half] += 1
    for handle in handles:
        handle.remove()
    half_means = (sums / np.maximum(counts, 1)[:, :, None, None]).astype(np.float32)
    if not np.isfinite(half_means).all():
        raise RuntimeError("Non-finite hidden states")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, half_means=half_means, half_counts=counts, prompt_tokens=tokens,
                        task_of_prompt=task_of_prompt.astype(np.int32))
    print(f"[hidden-states] wrote {args.output} half_means={half_means.shape}")


if __name__ == "__main__":
    main()
