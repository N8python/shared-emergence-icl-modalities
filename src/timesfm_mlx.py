#!/usr/bin/env python3
"""Minimal MLX inference port for TimesFM 2.5 Transformers checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

from huggingface_hub import snapshot_download
import mlx.core as mx
import numpy as np


DEFAULT_TIMESFM_2_5_PATH = "google/timesfm-2.5-200m-transformers"


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def softplus(x: mx.array) -> mx.array:
    return mx.log1p(mx.exp(-mx.abs(x))) + mx.maximum(x, 0)


def linear(x: mx.array, weight: mx.array, bias: Optional[mx.array] = None) -> mx.array:
    y = x @ weight.T
    if bias is not None:
        y = y + bias
    return y


def rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def load_config(model_path: Path) -> Dict[str, Any]:
    with (model_path / "config.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_weights(
    model_path: Path,
    *,
    dtype: str = "fp32",
    eval_weights: bool = True,
) -> Dict[str, mx.array]:
    weights = mx.load(str(model_path / "model.safetensors"))
    if dtype == "bf16":
        weights = {name: value.astype(mx.bfloat16) for name, value in weights.items()}
    elif dtype != "fp32":
        raise ValueError(f"Unsupported TimesFM MLX dtype: {dtype!r}")
    if eval_weights:
        mx.eval(list(weights.values()))
    return weights


def resolve_model_path(model_path: Path | str) -> Path:
    path = Path(model_path).expanduser()
    if path.exists():
        return path.resolve()
    return Path(
        snapshot_download(
            str(model_path),
            allow_patterns=["*.json", "model*.safetensors"],
        )
    )


class TimesFm2_5MLX:
    def __init__(
        self,
        config: Dict[str, Any],
        weights: Dict[str, mx.array],
        *,
        dtype: str = "fp32",
    ):
        self.config = config
        self.weights = weights
        self.dtype = dtype
        self.context_length = int(config["context_length"])
        self.patch_length = int(config["patch_length"])
        self.horizon_length = int(config["horizon_length"])
        self.output_quantile_len = int(config["output_quantile_len"])
        self.hidden_size = int(config["hidden_size"])
        self.intermediate_size = int(config["intermediate_size"])
        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.num_heads))
        self.num_quantiles = len(config["quantiles"]) + 1
        self.decode_index = int(config["decode_index"])
        self.rms_norm_eps = float(config["rms_norm_eps"])
        self.rope_theta = float(config.get("rope_parameters", {}).get("rope_theta", 10000.0))
        self.tolerance = 1e-6
        if self.num_kv_heads != self.num_heads:
            raise ValueError("TimesFM 2.5 MLX port currently expects no GQA")

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path | str = DEFAULT_TIMESFM_2_5_PATH,
        *,
        dtype: str = "fp32",
        eval_weights: bool = True,
    ) -> "TimesFm2_5MLX":
        path = resolve_model_path(model_path)
        return cls(
            load_config(path),
            load_weights(path, dtype=dtype, eval_weights=eval_weights),
            dtype=dtype,
        )

    def _w(self, name: str) -> mx.array:
        return self.weights[name]

    def _linear(self, x: mx.array, weight_name: str, bias_name: Optional[str] = None) -> mx.array:
        bias = self._w(bias_name) if bias_name is not None and bias_name in self.weights else None
        return linear(x, self._w(weight_name), bias)

    def _rms_norm_weight(self, x: mx.array, weight: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, weight, self.rms_norm_eps)

    def _rms_norm(self, x: mx.array, weight_name: str) -> mx.array:
        return self._rms_norm_weight(x, self._w(weight_name))

    def _revin(
        self,
        x: mx.array,
        loc: mx.array,
        scale: mx.array,
        *,
        reverse: bool = False,
        mask: Optional[mx.array] = None,
    ) -> mx.array:
        if len(loc.shape) == len(x.shape) - 1:
            loc = mx.expand_dims(loc, -1)
            scale = mx.expand_dims(scale, -1)
        elif len(loc.shape) == len(x.shape) - 2:
            loc = mx.expand_dims(mx.expand_dims(loc, -1), -1)
            scale = mx.expand_dims(mx.expand_dims(scale, -1), -1)

        safe_scale = mx.where(scale < self.tolerance, mx.ones_like(scale), scale)
        if reverse:
            return x * scale + loc
        y = (x - loc) / safe_scale
        if mask is not None:
            y = mx.where(mask, mx.zeros_like(y), y)
        return y

    def _residual_block(
        self,
        x: mx.array,
        prefix: str,
        *,
        input_bias: bool = False,
    ) -> mx.array:
        x = x.astype(self._w(f"{prefix}.input_layer.weight").dtype)
        hidden = self._linear(
            x,
            f"{prefix}.input_layer.weight",
            f"{prefix}.input_layer.bias" if input_bias else None,
        )
        hidden = silu(hidden)
        output = self._linear(
            hidden,
            f"{prefix}.output_layer.weight",
            f"{prefix}.output_layer.bias" if input_bias else None,
        )
        residual = self._linear(
            x,
            f"{prefix}.residual_layer.weight",
            f"{prefix}.residual_layer.bias" if input_bias else None,
        )
        return output + residual

    def _prepare_batch(
        self,
        past_values: Sequence[np.ndarray] | np.ndarray | mx.array,
        *,
        forecast_context_len: Optional[int] = None,
    ) -> Tuple[mx.array, mx.array, float]:
        context_len = int(forecast_context_len or self.context_length)
        arrays: list[np.ndarray] = []
        mins: list[float] = []

        if isinstance(past_values, mx.array):
            raw = np.array(past_values.astype(mx.float32))
            values = [raw[i] for i in range(raw.shape[0])] if raw.ndim == 2 else [raw]
        elif isinstance(past_values, np.ndarray):
            values = [past_values[i] for i in range(past_values.shape[0])] if past_values.ndim == 2 else [past_values]
        else:
            values = list(past_values)

        padding_arrays: list[np.ndarray] = []
        for value in values:
            ts = np.asarray(value, dtype=np.float32).reshape(-1)
            ts = ts[-context_len:]
            mins.append(float(ts.min()))
            padding = np.zeros(ts.shape[0] + self.horizon_length, dtype=np.float32)
            if ts.shape[0] < context_len:
                front_pad = context_len - ts.shape[0]
                ts = np.concatenate([np.zeros(front_pad, dtype=np.float32), ts])
                padding = np.concatenate(
                    [np.ones(front_pad, dtype=np.float32), padding],
                    axis=0,
                )
            elif ts.shape[0] > context_len:
                ts = ts[-context_len:]
                padding = padding[-(context_len + self.horizon_length) :]
            arrays.append(ts)
            padding_arrays.append(padding)

        return (
            mx.array(np.stack(arrays, axis=0), dtype=mx.float32),
            mx.array(np.stack(padding_arrays, axis=0), dtype=mx.float32),
            min(mins),
        )

    def _global_mean_std(self, x: mx.array) -> Tuple[mx.array, mx.array]:
        mean = mx.mean(x, axis=1, keepdims=True)
        centered = x - mean
        denom = max(x.shape[1] - 1, 1)
        std = mx.sqrt(mx.sum(mx.square(centered), axis=1, keepdims=True) / denom)
        return mean, std

    def _patch_running_stats(self, patched_inputs: mx.array, patched_masks_bool: mx.array) -> Tuple[mx.array, mx.array]:
        valid = (~patched_masks_bool).astype(patched_inputs.dtype)
        count = mx.cumsum(mx.sum(valid, axis=-1), axis=1)
        value_sum = mx.cumsum(mx.sum(patched_inputs * valid, axis=-1), axis=1)
        value_sq_sum = mx.cumsum(mx.sum(mx.square(patched_inputs) * valid, axis=-1), axis=1)
        count_safe = mx.where(count == 0, mx.ones_like(count), count)
        mean = mx.where(count == 0, mx.zeros_like(value_sum), value_sum / count_safe)
        var = value_sq_sum / count_safe - mx.square(mean)
        var = mx.where(count == 0, mx.zeros_like(var), mx.maximum(var, 0))
        std = mx.sqrt(var)
        return mean, std

    def _position_embeddings(self, position_ids: mx.array, dtype: mx.Dtype) -> Tuple[mx.array, mx.array]:
        inv_idx = mx.arange(0, self.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.rope_theta ** (inv_idx / self.head_dim))
        freqs = mx.expand_dims(position_ids.astype(mx.float32), -1) * inv_freq
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    def _apply_rope(
        self,
        q: mx.array,
        k: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> Tuple[mx.array, mx.array]:
        cos = mx.expand_dims(cos, 1)
        sin = mx.expand_dims(sin, 1)
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)

    def _attention_mask(self, patch_padding: mx.array, dtype: mx.Dtype) -> mx.array | str:
        seq_len = patch_padding.shape[1]
        if not bool(np.array(mx.any(patch_padding))):
            return "causal"
        positions = mx.arange(seq_len)
        causal = positions[None, :] > positions[:, None]
        key_padding = mx.expand_dims(patch_padding, 1)
        mask = mx.expand_dims(causal, (0, 1)) | mx.expand_dims(key_padding, 1)
        return mx.where(mask, mx.array(-1e9, dtype=dtype), mx.array(0.0, dtype=dtype))

    def attention(
        self,
        x: mx.array,
        layer_idx: int,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: mx.array | str,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        prefix = f"model.layers.{layer_idx}.self_attn"

        q = self._linear(x, f"{prefix}.q_proj.weight")
        k = self._linear(x, f"{prefix}.k_proj.weight")
        v = self._linear(x, f"{prefix}.v_proj.weight")

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        q, k = self._apply_rope(q, k, *position_embeddings)
        q = self._rms_norm_weight(q, self._w(f"{prefix}.q_norm.weight"))
        k = self._rms_norm_weight(k, self._w(f"{prefix}.k_norm.weight"))

        scale = softplus(self._w(f"{prefix}.scaling")) * (1.442695041 / math.sqrt(self.head_dim))
        q = q * scale.reshape(1, 1, 1, self.head_dim)

        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0,
            mask=attention_mask,
        )
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        return self._linear(y, f"{prefix}.o_proj.weight")

    def mlp(self, x: mx.array, layer_idx: int) -> mx.array:
        prefix = f"model.layers.{layer_idx}.mlp"
        hidden = self._linear(x, f"{prefix}.ff0.weight")
        hidden = silu(hidden)
        return self._linear(hidden, f"{prefix}.ff1.weight")

    def layer(
        self,
        x: mx.array,
        layer_idx: int,
        position_embeddings: Tuple[mx.array, mx.array],
        attention_mask: mx.array | str,
    ) -> mx.array:
        residual = x
        x_norm = self._rms_norm(x, f"model.layers.{layer_idx}.input_layernorm.weight")
        x_attn = self.attention(x_norm, layer_idx, position_embeddings, attention_mask)
        x = self._rms_norm(x_attn, f"model.layers.{layer_idx}.post_attention_layernorm.weight") + residual

        residual = x
        x_norm = self._rms_norm(x, f"model.layers.{layer_idx}.pre_feedforward_layernorm.weight")
        x_mlp = self.mlp(x_norm, layer_idx)
        x = self._rms_norm(x_mlp, f"model.layers.{layer_idx}.post_feedforward_layernorm.weight") + residual
        return x

    def model_forward(self, past_values: mx.array, past_values_padding: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
        batch_size, seq_len = past_values.shape
        num_patches = seq_len // self.patch_length
        patched_inputs = past_values.reshape(batch_size, num_patches, self.patch_length)
        patched_masks = past_values_padding[:, :seq_len].reshape(batch_size, num_patches, self.patch_length)
        patched_masks_bool = patched_masks >= 0.5

        context_mu, context_sigma = self._patch_running_stats(patched_inputs, patched_masks_bool)
        normed_inputs = self._revin(
            patched_inputs,
            context_mu,
            context_sigma,
            reverse=False,
            mask=patched_masks_bool,
        )
        tokenizer_inputs = mx.concatenate(
            [normed_inputs, patched_masks_bool.astype(normed_inputs.dtype)],
            axis=-1,
        )
        x = self._residual_block(
            tokenizer_inputs,
            "model.input_ff_layer",
            input_bias=True,
        )

        patch_padding = patched_masks_bool[..., -1]
        num_masked = mx.sum(patch_padding.astype(mx.int32), axis=-1, keepdims=True)
        position_ids = mx.arange(num_patches).reshape(1, -1) - num_masked
        attention_mask = self._attention_mask(patch_padding, x.dtype)
        position_embeddings = self._position_embeddings(position_ids, x.dtype)

        for layer_idx in range(self.num_layers):
            x = self.layer(x, layer_idx, position_embeddings, attention_mask)

        return x, context_mu, context_sigma

    def _decode_and_project(
        self,
        normalized_ts: mx.array,
        input_padding: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        hidden_states, context_mu, context_sigma = self.model_forward(normalized_ts, input_padding)
        point_output = self._revin(
            self._residual_block(hidden_states, "output_projection_point"),
            context_mu,
            context_sigma,
            reverse=True,
        )
        quantile_output = self._revin(
            self._residual_block(hidden_states, "output_projection_quantiles"),
            context_mu,
            context_sigma,
            reverse=True,
        )

        batch_size, num_patches = point_output.shape[:2]
        point_forecast = point_output.reshape(
            batch_size,
            num_patches,
            self.horizon_length,
            self.num_quantiles,
        )[:, -1, :, :]
        quantile_spreads = quantile_output.reshape(
            batch_size,
            num_patches,
            self.output_quantile_len,
            self.num_quantiles,
        )[:, -1, :, :]
        return point_forecast, quantile_spreads, hidden_states

    def forecast(
        self,
        past_values: Sequence[np.ndarray] | np.ndarray | mx.array,
        *,
        forecast_context_len: Optional[int] = None,
        force_flip_invariance: Optional[bool] = None,
        truncate_negative: Optional[bool] = None,
    ) -> Tuple[mx.array, mx.array]:
        input_ts, input_padding, input_min = self._prepare_batch(
            past_values,
            forecast_context_len=forecast_context_len,
        )
        mu_global, sigma_global = self._global_mean_std(input_ts)
        normalized_ts = self._revin(input_ts, mu_global, sigma_global, reverse=False)

        if force_flip_invariance is None:
            force_flip_invariance = bool(self.config.get("force_flip_invariance", False))
        if truncate_negative is None:
            truncate_negative = bool(self.config.get("infer_is_positive", False))

        pf_outputs, quantile_spreads, _ = self._decode_and_project(normalized_ts, input_padding)

        if force_flip_invariance:
            flipped_pf, flipped_qs, _ = self._decode_and_project(-normalized_ts, input_padding)

            def flip_quantiles(x: mx.array) -> mx.array:
                reverse_idx = mx.arange(x.shape[-1] - 1, 0, -1)
                return mx.concatenate([x[..., :1], mx.take(x, reverse_idx, axis=-1)], axis=-1)

            pf_outputs = (pf_outputs - flip_quantiles(flipped_pf)) / 2
            quantile_spreads = (quantile_spreads - flip_quantiles(flipped_qs)) / 2

        horizon = min(self.horizon_length, pf_outputs.shape[1])
        full_forecast = pf_outputs[:, :horizon, :]
        median_index = min(self.decode_index, full_forecast.shape[-1] - 1)

        if bool(self.config.get("use_continuous_quantile_head", False)):
            q = quantile_spreads[:, :horizon, :]
            adjusted = q - q[..., median_index : median_index + 1] + full_forecast[..., median_index : median_index + 1]
            parts = []
            for idx in range(self.num_quantiles):
                if idx == 0 or idx == median_index:
                    parts.append(full_forecast[..., idx : idx + 1])
                else:
                    parts.append(adjusted[..., idx : idx + 1])
            full_forecast = mx.concatenate(parts, axis=-1)

        full_predictions = self._revin(full_forecast, mu_global, sigma_global, reverse=True)
        mean_predictions = full_predictions[:, :, median_index]

        if truncate_negative and input_min >= 0:
            mean_predictions = mx.maximum(mean_predictions, 0)
            full_predictions = mx.maximum(full_predictions, 0)

        return mean_predictions, full_predictions


def make_toy_series(batch_size: int, context_len: int, *, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(context_len, dtype=np.float32)
    series = []
    for idx in range(batch_size):
        if idx % 2 == 0:
            period = 48.0 + 4.0 * idx
            values = np.sin(2 * np.pi * t / period)
        else:
            values = 0.01 * t + 0.35 * np.sin(2 * np.pi * t / 64.0)
        values = values + 0.01 * rng.normal(size=context_len)
        series.append(values.astype(np.float32))
    return np.stack(series, axis=0)
