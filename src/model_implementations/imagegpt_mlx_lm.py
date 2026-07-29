# Copyright © 2023 - 2024 Apple Inc.
#
# Adapted from mlx_lm.models.gpt2 for OpenAI ImageGPT checkpoints.

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
    n_embd: int
    n_head: int
    n_layer: int
    n_positions: int
    layer_norm_epsilon: float
    vocab_size: int
    activation_function: str = "quick_gelu"
    scale_attn_weights: bool = True
    tie_word_embeddings: bool = False
    num_key_value_heads: int = None

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.n_head


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        assert args.n_embd % args.n_head == 0, "n_embd must be divisible by n_head"

        self.n_embd = args.n_embd
        self.n_head = args.n_head
        self.head_dim = self.n_embd // self.n_head
        self.scale = self.head_dim**-0.5 if args.scale_attn_weights else 1.0

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.c_attn(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        queries = queries.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.c_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.activation_function != "quick_gelu":
            raise ValueError(f"Unsupported activation {args.activation_function!r}")

        self.n_embd = args.n_embd
        self.c_fc = nn.Linear(self.n_embd, 4 * self.n_embd)
        self.c_proj = nn.Linear(4 * self.n_embd, self.n_embd)

    def __call__(self, x) -> mx.array:
        h = self.c_fc(x)
        return self.c_proj(h * mx.sigmoid(mx.array(1.702, dtype=h.dtype) * h))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.layer_norm_epsilon = args.layer_norm_epsilon
        self.attn = Attention(args)
        self.mlp = MLP(args)
        self.ln_1 = nn.RMSNorm(self.n_embd, eps=self.layer_norm_epsilon)
        self.ln_2 = nn.RMSNorm(self.n_embd, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.attn(self.ln_1(x), mask, cache)
        return h + self.mlp(self.ln_2(h))


class ImageGPTModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_embd = args.n_embd
        self.n_positions = args.n_positions
        self.vocab_size = args.vocab_size
        self.n_layer = args.n_layer
        self.layer_norm_epsilon = args.layer_norm_epsilon
        assert self.vocab_size > 0
        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        self.wpe = nn.Embedding(self.n_positions, self.n_embd)
        self.h = [TransformerBlock(args=args) for _ in range(self.n_layer)]
        self.ln_f = nn.RMSNorm(self.n_embd, eps=self.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape
        hidden_states = self.wte(inputs)

        if cache is None:
            cache = [None] * len(self.h)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        offset = mx.array(offset)
        position_ids = mx.arange(L) + offset[..., None]
        hidden_states += self.wpe(position_ids)

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.h, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        return self.ln_f(hidden_states)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = ImageGPTModel(args)
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size - 1, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        weights = dict(weights)
        for key in list(weights):
            if key.endswith(".attn.bias") or key.endswith(".attn.masked_bias"):
                del weights[key]

        for i in range(self.args.n_layer):
            for key in (
                f"transformer.h.{i}.attn.c_attn.weight",
                f"transformer.h.{i}.attn.c_proj.weight",
                f"transformer.h.{i}.mlp.c_fc.weight",
                f"transformer.h.{i}.mlp.c_proj.weight",
            ):
                if key in weights:
                    weights[key] = weights[key].transpose(1, 0)

        return weights

    @property
    def layers(self):
        return self.transformer.h
