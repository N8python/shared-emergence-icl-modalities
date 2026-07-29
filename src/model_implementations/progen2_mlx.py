# Copyright (c) 2026
"""MLX-LM architecture for ProGen2 causal protein LMs."""

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    scaled_dot_product_attention,
)


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    vocab_size_emb: int
    vocab_size_lm_head: int
    n_positions: int
    embed_dim: int
    n_layer: int
    n_head: int
    rotary_dim: int = 64
    n_inner: Optional[int] = None
    activation_function: str = "gelu_new"
    layer_norm_epsilon: float = 1e-5
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


def gelu_new(x: mx.array) -> mx.array:
    return 0.5 * x * (
        1.0
        + mx.tanh(0.7978845608028654 * (x + 0.044715 * mx.power(x, 3)))
    )


def rotate_every_two(x: mx.array) -> mx.array:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    stacked = mx.stack((-x2, x1), axis=-1)
    return stacked.reshape(*x.shape)


class PartialRotaryEmbedding(nn.Module):
    def __init__(self, rotary_dim: int):
        super().__init__()
        self.rotary_dim = rotary_dim
        inv_freq = 1.0 / (
            10000
            ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim)
        )
        self.inv_freq = inv_freq

    def __call__(self, x: mx.array, offset: int | mx.array = 0) -> mx.array:
        seq_len = x.shape[-2]
        offset = mx.array(offset, dtype=mx.float32)
        positions = mx.arange(seq_len, dtype=mx.float32)

        if offset.ndim == 0:
            positions = positions + offset
            freqs = positions[:, None] * self.inv_freq[None, :]
            emb = mx.repeat(freqs, 2, axis=-1)
            cos = mx.cos(emb).astype(x.dtype).reshape(
                1,
                1,
                seq_len,
                self.rotary_dim,
            )
            sin = mx.sin(emb).astype(x.dtype).reshape(
                1,
                1,
                seq_len,
                self.rotary_dim,
            )
        else:
            positions = positions[None, :] + offset[:, None]
            freqs = positions[:, :, None] * self.inv_freq[None, None, :]
            emb = mx.repeat(freqs, 2, axis=-1)
            cos = mx.cos(emb).astype(x.dtype).reshape(
                x.shape[0],
                1,
                seq_len,
                self.rotary_dim,
            )
            sin = mx.sin(emb).astype(x.dtype).reshape(
                x.shape[0],
                1,
                seq_len,
                self.rotary_dim,
            )

        return (x * cos) + (rotate_every_two(x) * sin)


class ProGenAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.embed_dim % args.n_head != 0:
            raise ValueError("embed_dim must be divisible by n_head")
        self.embed_dim = args.embed_dim
        self.num_heads = args.n_head
        self.head_dim = args.embed_dim // args.n_head
        self.mp_num = 8
        self.mp_part = args.embed_dim // self.mp_num
        self.scale = self.head_dim**-0.5
        self.rotary_dim = args.rotary_dim

        self.qkv_proj = nn.Linear(args.embed_dim, args.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(args.embed_dim, args.embed_dim, bias=False)
        self.rotary = PartialRotaryEmbedding(self.rotary_dim)

    def _split_heads_from_mp(self, x: mx.array) -> mx.array:
        batch_size, seq_len = x.shape[:2]
        x = x.reshape(batch_size, seq_len, self.embed_dim)
        return x.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim,
        ).transpose(0, 2, 1, 3)

    def _apply_partial_rotary(self, x: mx.array, offset: int | mx.array) -> mx.array:
        x_rot = x[..., : self.rotary_dim]
        x_pass = x[..., self.rotary_dim :]
        return mx.concatenate([self.rotary(x_rot, offset=offset), x_pass], axis=-1)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, self.mp_num, -1)

        query, value, key = mx.split(qkv, 3, axis=-1)
        query = self._split_heads_from_mp(query)
        key = self._split_heads_from_mp(key)
        value = self._split_heads_from_mp(value)

        offset = 0 if cache is None else cache.offset
        query = self._apply_partial_rotary(query, offset=offset)
        key = self._apply_partial_rotary(key, offset=offset)

        if cache is not None:
            key, value = cache.update_and_fetch(key, value)

        attn_output = scaled_dot_product_attention(
            query,
            key,
            value,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size,
            seq_len,
            self.embed_dim,
        )
        return self.out_proj(attn_output)


class ProGenMLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        inner_dim = args.n_inner if args.n_inner is not None else 4 * args.embed_dim
        self.fc_in = nn.Linear(args.embed_dim, inner_dim, bias=True)
        self.fc_out = nn.Linear(inner_dim, args.embed_dim, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.fc_out(gelu_new(self.fc_in(hidden_states)))


class ProGenBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(
            args.embed_dim,
            eps=args.layer_norm_epsilon,
            affine=True,
            bias=True,
        )
        self.attn = ProGenAttention(args)
        self.mlp = ProGenMLP(args)

    def __call__(
        self,
        hidden_states: mx.array,
        mask: Optional[Any] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        normed = self.ln_1(hidden_states)
        attn_output = self.attn(normed, mask=mask, cache=cache)
        mlp_output = self.mlp(normed)
        return residual + attn_output + mlp_output


class ProGenModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.wte = nn.Embedding(args.vocab_size_emb, args.embed_dim)
        self.h = [ProGenBlock(args) for _ in range(args.n_layer)]
        self.ln_f = nn.LayerNorm(
            args.embed_dim,
            eps=args.layer_norm_epsilon,
            affine=True,
            bias=True,
        )

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        hidden_states = self.wte(inputs)
        if cache is None:
            cache = [None] * len(self.h)
        mask = create_attention_mask(hidden_states, cache[0])
        for block, c in zip(self.h, cache):
            hidden_states = block(hidden_states, mask=mask, cache=c)
        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = ProGenModel(args)
        self.lm_head = nn.Linear(args.embed_dim, args.vocab_size_lm_head, bias=True)

    def __call__(self, inputs: mx.array, cache=None) -> mx.array:
        return self.lm_head(self.transformer(inputs, cache=cache))

    def sanitize(self, weights):
        weights = dict(weights)
        inv_freq = 1.0 / (
            10000
            ** (
                mx.arange(0, self.args.rotary_dim, 2, dtype=mx.float32)
                / self.args.rotary_dim
            )
        )
        for layer_idx in range(self.args.n_layer):
            weights[f"transformer.h.{layer_idx}.attn.rotary.inv_freq"] = inv_freq
        return weights

    @property
    def layers(self):
        return self.transformer.h
