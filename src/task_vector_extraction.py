"""Hidden-state readout for task-vector analyses (ICLR rebuttal).

Given the exact token-id prompts that program_synth.py evaluates, run one
forward pass per prompt and read the residual stream at the last prompt
position after the embedding and after every transformer block. Per task, the
states are averaged over trials separately for even and odd trial indices, so
the file holds a full-trial mean (average the two halves) and a split-half
reliability estimate without storing per-trial activations.

Output (.npz):
  half_means      float32 [tasks, 2, layers + 1, hidden]  (index 0 = embeddings)
  half_counts     int32   [tasks, 2]
  prompt_tokens   int32   [prompts, max_len], right-padded with -1
  task_of_prompt  int32   [prompts]
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import mlx.core as mx
import numpy as np


def _embed_and_blocks(model, x):
    """(initial residual stream, blocks, final norm) for the supported model families."""
    transformer = getattr(model, "transformer", None)
    if transformer is not None and hasattr(transformer, "wpe"):  # ImageGPT
        positions = mx.arange(x.shape[1])[None, :]
        return transformer.wte(x) + transformer.wpe(positions), transformer.h, transformer.ln_f
    if transformer is not None:  # ProGen2 (rotary positions, no position table)
        return transformer.wte(x), transformer.h, transformer.ln_f
    inner = model.model  # mlx-lm decoder-only language models
    return inner.embed_tokens(x), inner.layers, inner.norm


def last_position_states(model, token_ids: Sequence[Sequence[int]]) -> np.ndarray:
    """[batch, layers + 1, hidden] float32 states at the last position.

    All prompts in the batch must have the same length (no padding is used).
    """
    from mlx_lm.models.base import create_attention_mask

    x = mx.array(token_ids)
    h, blocks, _ = _embed_and_blocks(model, x)
    mask = create_attention_mask(h, None)
    states = [h[:, -1, :].astype(mx.float32)]
    for layer in blocks:
        h = layer(h, mask, None)
        states.append(h[:, -1, :].astype(mx.float32))
    out = mx.stack(states, axis=1)
    mx.eval(out)
    return np.array(out)


def check_against_model(model, token_ids: Sequence[int]) -> float:
    """Max gap between the manual block loop and the model's own forward pass.

    Compared on the final-norm hidden state at the last position (relative to its
    largest entry), which is what the model feeds its output head.
    """
    from mlx_lm.models.base import create_attention_mask

    x = mx.array([list(token_ids)])
    h, blocks, norm = _embed_and_blocks(model, x)
    mask = create_attention_mask(h, None)
    for layer in blocks:
        h = layer(h, mask, None)
    manual = norm(h)[:, -1].astype(mx.float32)
    transformer = getattr(model, "transformer", None)
    reference = transformer(x) if transformer is not None else model.model(x)
    reference = reference[:, -1].astype(mx.float32)
    gap = mx.max(mx.abs(manual - reference)) / mx.maximum(mx.max(mx.abs(reference)), 1e-6)
    return float(gap.item())


def timesfm_last_patch_states(model, values: np.ndarray, context_limit: int, check: bool = False):
    """[batch, layers + 1, hidden] float32 states at the last input patch of TimesFM 2.5.

    Mirrors TimesFm2_5MLX.forecast up to the transformer stack (global normalisation,
    patching, input residual block), for the un-flipped series only; the model's
    forecast additionally averages a sign-flipped pass, which has no single state.
    Index 0 is the patch embedding entering the first transformer layer.
    """
    context_len = min(values.shape[1], context_limit)
    input_ts, input_padding, _ = model._prepare_batch(values, forecast_context_len=context_len)
    mu_global, sigma_global = model._global_mean_std(input_ts)
    past_values = model._revin(input_ts, mu_global, sigma_global, reverse=False)
    batch_size, seq_len = past_values.shape
    num_patches = seq_len // model.patch_length
    patched_inputs = past_values.reshape(batch_size, num_patches, model.patch_length)
    patched_masks = input_padding[:, :seq_len].reshape(batch_size, num_patches, model.patch_length)
    patched_masks_bool = patched_masks >= 0.5
    context_mu, context_sigma = model._patch_running_stats(patched_inputs, patched_masks_bool)
    normed_inputs = model._revin(patched_inputs, context_mu, context_sigma, reverse=False, mask=patched_masks_bool)
    tokenizer_inputs = mx.concatenate([normed_inputs, patched_masks_bool.astype(normed_inputs.dtype)], axis=-1)
    x = model._residual_block(tokenizer_inputs, "model.input_ff_layer", input_bias=True)
    patch_padding = patched_masks_bool[..., -1]
    num_masked = mx.sum(patch_padding.astype(mx.int32), axis=-1, keepdims=True)
    position_ids = mx.arange(num_patches).reshape(1, -1) - num_masked
    attention_mask = model._attention_mask(patch_padding, x.dtype)
    position_embeddings = model._position_embeddings(position_ids, x.dtype)
    states = [x[:, -1, :].astype(mx.float32)]
    for layer_idx in range(model.num_layers):
        x = model.layer(x, layer_idx, position_embeddings, attention_mask)
        states.append(x[:, -1, :].astype(mx.float32))
    out = mx.stack(states, axis=1)
    mx.eval(out)
    if check:
        reference, _, _ = model.model_forward(past_values, input_padding)
        reference = reference[:, -1].astype(mx.float32)
        gap = mx.max(mx.abs(states[-1] - reference)) / mx.maximum(mx.max(mx.abs(reference)), 1e-6)
        return np.array(out), float(gap.item())
    return np.array(out)


def dump_task_hidden_states(
    model,
    prompts: List[List],
    task_sizes: Sequence[int],
    output_path: Path,
    *,
    batch_size: int = 16,
    progress=None,
    timesfm_context_limit: int | None = None,
) -> None:
    """Token-id prompts by default; real-valued series when timesfm_context_limit is set."""
    if sum(task_sizes) != len(prompts):
        raise ValueError("task_sizes must partition the prompts")
    is_series = timesfm_context_limit is not None
    if is_series:
        _, gap = timesfm_last_patch_states(model, np.array([prompts[0]], dtype=np.float32), timesfm_context_limit, check=True)
        states_fn = lambda batch: timesfm_last_patch_states(model, np.array(batch, dtype=np.float32), timesfm_context_limit)
    else:
        gap = check_against_model(model, prompts[0])
        states_fn = lambda batch: last_position_states(model, batch)
    print(f"[hidden-states] manual-vs-model last-position relative gap: {gap:.3g}")
    if gap > 1e-3:
        raise RuntimeError("Manual block loop does not reproduce the model's final hidden state")

    task_of_prompt = np.repeat(np.arange(len(task_sizes)), task_sizes)
    trial_of_prompt = np.concatenate([np.arange(n) for n in task_sizes])
    sums = None
    counts = np.zeros((len(task_sizes), 2), dtype=np.int32)
    by_length: dict = {}
    for index, prompt in enumerate(prompts):
        by_length.setdefault(len(prompt), []).append(index)
    for indices in by_length.values():
        for start in range(0, len(indices), batch_size):
            chunk = indices[start : start + batch_size]
            states = states_fn([prompts[i] for i in chunk])
            if sums is None:
                sums = np.zeros((len(task_sizes), 2) + states.shape[1:], dtype=np.float64)
            for row, i in enumerate(chunk):
                half = int(trial_of_prompt[i] % 2)
                sums[task_of_prompt[i], half] += states[row]
                counts[task_of_prompt[i], half] += 1
            if progress is not None:
                progress.update(len(chunk))
    half_means = (sums / np.maximum(counts, 1)[:, :, None, None]).astype(np.float32)
    max_len = max(len(p) for p in prompts)
    tokens = np.full((len(prompts), max_len), -1, dtype=np.float32 if is_series else np.int32)
    for i, prompt in enumerate(prompts):
        tokens[i, : len(prompt)] = prompt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        half_means=half_means,
        half_counts=counts,
        prompt_tokens=tokens,
        task_of_prompt=task_of_prompt.astype(np.int32),
    )
    print(f"[hidden-states] wrote {output_path} half_means={half_means.shape}")
