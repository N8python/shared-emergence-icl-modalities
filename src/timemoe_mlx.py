#!/usr/bin/env python3
"""Minimal MLX inference port for Maple728/TimeMoE checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from huggingface_hub import snapshot_download
import mlx.core as mx
from mlx_lm.models.switch_layers import SwitchGLU


DEFAULT_TIMEMOE_200M_PATH = "Maple728/TimeMoE-200M"


KVCache = List[Optional[Tuple[mx.array, mx.array]]]


def silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


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


def load_weights(model_path: Path, *, eval_weights: bool = True) -> Dict[str, mx.array]:
    weights = mx.load(str(model_path / "model.safetensors"))
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
            allow_patterns=["*.json", "*.py", "model*.safetensors"],
        )
    )


class TimeMoeMLX:
    def __init__(
        self,
        config: Dict[str, Any],
        weights: Dict[str, mx.array],
        *,
        moe_impl: str = "switch",
    ):
        self.config = config
        self.weights = weights
        self.moe_impl = moe_impl
        self.hidden_size = int(config["hidden_size"])
        self.input_size = int(config.get("input_size", 1))
        self.num_layers = int(config["num_hidden_layers"])
        self.num_heads = int(config["num_attention_heads"])
        self.num_kv_heads = int(config["num_key_value_heads"])
        self.head_dim = self.hidden_size // self.num_heads
        self.num_experts = int(config["num_experts"])
        self.top_k = int(config["num_experts_per_tok"])
        self.intermediate_size = int(config["intermediate_size"])
        self.expert_intermediate_size = self.intermediate_size // self.top_k
        self.rope_theta = float(config.get("rope_theta", 10000))
        self.rms_norm_eps = float(config.get("rms_norm_eps", 1e-6))
        self.horizon_lengths = [int(x) for x in config["horizon_lengths"]]
        if self.moe_impl not in {"switch", "all"}:
            raise ValueError(f"unknown MoE implementation: {self.moe_impl!r}")
        self.expert_switches = (
            [self._make_switch_glu(layer_idx) for layer_idx in range(self.num_layers)]
            if self.moe_impl == "switch"
            else []
        )

    @classmethod
    def from_pretrained(
        cls,
        model_path: Path | str = DEFAULT_TIMEMOE_200M_PATH,
        *,
        eval_weights: bool = True,
        moe_impl: str = "switch",
    ) -> "TimeMoeMLX":
        path = resolve_model_path(model_path)
        return cls(
            load_config(path),
            load_weights(path, eval_weights=eval_weights),
            moe_impl=moe_impl,
        )

    def _w(self, name: str) -> mx.array:
        return self.weights[name]

    def _linear(self, x: mx.array, weight_name: str, bias_name: Optional[str] = None) -> mx.array:
        bias = self._w(bias_name) if bias_name is not None and bias_name in self.weights else None
        return linear(x, self._w(weight_name), bias)

    def _rms_norm(self, x: mx.array, weight_name: str) -> mx.array:
        return mx.fast.rms_norm(x, self._w(weight_name), self.rms_norm_eps)

    def _make_switch_glu(self, layer_idx: int) -> SwitchGLU:
        prefix = f"model.layers.{layer_idx}.ffn_layer"
        switch = SwitchGLU(
            self.hidden_size,
            self.expert_intermediate_size,
            self.num_experts,
            bias=False,
        )
        switch.gate_proj.weight = mx.stack(
            [
                self._w(f"{prefix}.experts.{expert_idx}.gate_proj.weight")
                for expert_idx in range(self.num_experts)
            ]
        )
        switch.up_proj.weight = mx.stack(
            [
                self._w(f"{prefix}.experts.{expert_idx}.up_proj.weight")
                for expert_idx in range(self.num_experts)
            ]
        )
        switch.down_proj.weight = mx.stack(
            [
                self._w(f"{prefix}.experts.{expert_idx}.down_proj.weight")
                for expert_idx in range(self.num_experts)
            ]
        )
        mx.eval(
            switch.gate_proj.weight,
            switch.up_proj.weight,
            switch.down_proj.weight,
        )
        return switch

    def apply_rope(self, x: mx.array, offset: int) -> mx.array:
        seq_len = x.shape[-2]
        positions = mx.arange(offset, offset + seq_len, dtype=mx.float32)
        freq_idx = mx.arange(0, self.head_dim, 2, dtype=mx.float32)
        inv_freq = 1.0 / (self.rope_theta ** (freq_idx / self.head_dim))
        freqs = positions[:, None] * inv_freq[None, :]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb).astype(x.dtype).reshape(1, 1, seq_len, self.head_dim)
        sin = mx.sin(emb).astype(x.dtype).reshape(1, 1, seq_len, self.head_dim)
        return (x * cos) + (rotate_half(x) * sin)

    def embed(self, x: mx.array) -> mx.array:
        if x.ndim == 2:
            x = x[..., None]
        x = x.astype(self._w("model.embed_layer.emb_layer.weight").dtype)
        gate = self._linear(x, "model.embed_layer.gate_layer.weight")
        emb = self._linear(x, "model.embed_layer.emb_layer.weight")
        return silu(gate) * emb

    def temporal_block(self, x: mx.array, prefix: str) -> mx.array:
        gate = self._linear(x, f"{prefix}.gate_proj.weight")
        up = self._linear(x, f"{prefix}.up_proj.weight")
        return self._linear(silu(gate) * up, f"{prefix}.down_proj.weight")

    def sparse_experts_all(self, x: mx.array, layer_idx: int) -> mx.array:
        batch_size, seq_len, hidden_dim = x.shape
        flat = x.reshape(batch_size * seq_len, hidden_dim)
        prefix = f"model.layers.{layer_idx}.ffn_layer"

        router_logits = self._linear(flat, f"{prefix}.gate.weight")
        routing_probs = mx.softmax(router_logits, axis=-1, precise=True)
        selected = mx.argpartition(-routing_probs, kth=self.top_k - 1, axis=-1)[:, : self.top_k]
        top_weights = mx.take_along_axis(routing_probs, selected, axis=-1).astype(flat.dtype)

        out = mx.zeros_like(flat)
        zero_weights = mx.zeros_like(top_weights)
        for expert_idx in range(self.num_experts):
            expert_prefix = f"{prefix}.experts.{expert_idx}"
            expert_weight = mx.sum(
                mx.where(selected == expert_idx, top_weights, zero_weights),
                axis=-1,
                keepdims=True,
            )
            out = out + self.temporal_block(flat, expert_prefix) * expert_weight

        shared = self.temporal_block(flat, f"{prefix}.shared_expert")
        shared_gate = mx.sigmoid(self._linear(flat, f"{prefix}.shared_expert_gate.weight"))
        out = out + shared * shared_gate
        return out.reshape(batch_size, seq_len, hidden_dim)

    def sparse_experts_switch(self, x: mx.array, layer_idx: int) -> mx.array:
        batch_size, seq_len, hidden_dim = x.shape
        flat = x.reshape(batch_size * seq_len, hidden_dim)
        prefix = f"model.layers.{layer_idx}.ffn_layer"

        router_logits = self._linear(flat, f"{prefix}.gate.weight")
        routing_probs = mx.softmax(router_logits, axis=-1, precise=True)
        selected = mx.argpartition(-routing_probs, kth=self.top_k - 1, axis=-1)[
            :, : self.top_k
        ]
        selected = mx.stop_gradient(selected)
        top_weights = mx.take_along_axis(routing_probs, selected, axis=-1).astype(flat.dtype)

        out = self.expert_switches[layer_idx](flat, selected)
        out = mx.sum(out * top_weights[..., None], axis=-2)

        shared = self.temporal_block(flat, f"{prefix}.shared_expert")
        shared_gate = mx.sigmoid(self._linear(flat, f"{prefix}.shared_expert_gate.weight"))
        out = out + shared * shared_gate
        return out.reshape(batch_size, seq_len, hidden_dim)

    def sparse_experts(self, x: mx.array, layer_idx: int) -> mx.array:
        if self.moe_impl == "switch":
            return self.sparse_experts_switch(x, layer_idx)
        return self.sparse_experts_all(x, layer_idx)

    def attention(
        self,
        x: mx.array,
        layer_idx: int,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
    ) -> Tuple[mx.array, Tuple[mx.array, mx.array]]:
        batch_size, seq_len, _ = x.shape
        prefix = f"model.layers.{layer_idx}.self_attn"
        q = self._linear(x, f"{prefix}.q_proj.weight", f"{prefix}.q_proj.bias")
        k = self._linear(x, f"{prefix}.k_proj.weight", f"{prefix}.k_proj.bias")
        v = self._linear(x, f"{prefix}.v_proj.weight", f"{prefix}.v_proj.bias")

        q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        past_len = 0 if cache is None else cache[0].shape[2]
        q = self.apply_rope(q, past_len)
        k = self.apply_rope(k, past_len)

        if cache is not None:
            k = mx.concatenate([cache[0], k], axis=2)
            v = mx.concatenate([cache[1], v], axis=2)

        mask = "causal" if cache is None and seq_len > 1 else None
        y = mx.fast.scaled_dot_product_attention(
            q,
            k,
            v,
            scale=1.0 / math.sqrt(self.head_dim),
            mask=mask,
        )
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        y = self._linear(y, f"{prefix}.o_proj.weight")
        return y, (k, v)

    def layer(
        self,
        x: mx.array,
        layer_idx: int,
        cache: Optional[Tuple[mx.array, mx.array]] = None,
        *,
        use_cache: bool = False,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        residual = x
        x_norm = self._rms_norm(x, f"model.layers.{layer_idx}.input_layernorm.weight")
        attn, new_cache = self.attention(x_norm, layer_idx, cache)
        x = residual + attn

        residual = x
        x_norm = self._rms_norm(x, f"model.layers.{layer_idx}.post_attention_layernorm.weight")
        x = residual + self.sparse_experts(x_norm, layer_idx)
        return x, new_cache if use_cache else None

    def forward(
        self,
        input_ids: mx.array,
        *,
        cache: Optional[KVCache] = None,
        use_cache: bool = False,
        max_horizon_length: Optional[int] = 1,
    ) -> Tuple[mx.array, Optional[KVCache]]:
        x = self.embed(input_ids)
        next_cache: KVCache = []
        if cache is None:
            cache = [None] * self.num_layers

        for layer_idx in range(self.num_layers):
            x, layer_cache = self.layer(
                x,
                layer_idx,
                cache[layer_idx],
                use_cache=use_cache,
            )
            if use_cache:
                next_cache.append(layer_cache)

        x = self._rms_norm(x, "model.norm.weight")

        if max_horizon_length is None:
            horizon = self.horizon_lengths[0]
            max_horizon_length = horizon
        else:
            horizon = self.horizon_lengths[0]
            for candidate in self.horizon_lengths[1:]:
                if candidate > max_horizon_length:
                    break
                horizon = candidate

        head_index = self.horizon_lengths.index(horizon)
        predictions = self._linear(x, f"lm_heads.{head_index}.out_layer.weight")
        if horizon > max_horizon_length:
            predictions = predictions[:, :, : self.input_size * max_horizon_length]
        if self.input_size == 1:
            predictions = predictions.reshape(predictions.shape[0], predictions.shape[1], -1)
        return predictions, next_cache if use_cache else None

    def generate_one_step_ar(self, input_ids: mx.array, steps: int) -> mx.array:
        if input_ids.ndim == 2:
            input_ids = input_ids[..., None]
        cache: Optional[KVCache] = None
        cur = input_ids
        outputs = [input_ids]
        for step in range(steps):
            x = cur if step == 0 else outputs[-1]
            predictions, cache = self.forward(
                x,
                cache=cache,
                use_cache=True,
                max_horizon_length=1,
            )
            next_value = predictions[:, -1:, : self.input_size]
            outputs.append(next_value)
            mx.eval(next_value)
        return mx.concatenate(outputs, axis=1)


def make_random_series(
    batch_size: int,
    prompt_len: int,
    *,
    seed: int = 0,
    dtype: mx.Dtype = mx.bfloat16,
) -> mx.array:
    mx.random.seed(seed)
    x = mx.random.normal((batch_size, prompt_len, 1)).astype(dtype)
    mx.eval(x)
    return x
