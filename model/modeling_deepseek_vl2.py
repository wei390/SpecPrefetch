from __future__ import annotations

import logging
import math
import os
import weakref
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as safetensors_load_file
from torch import Tensor

from .configuration_deepseek_vl2 import DeepseekVL2DraftRouterConfig

logger = logging.getLogger(__name__)


# ============================================================
# Utility functions
# ============================================================

def _is_finite_tensor(x: Tensor | None) -> bool:
    if x is None:
        return True
    return bool(torch.isfinite(x).all().item())


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    weight = mask.to(device=values.device, dtype=values.dtype)
    if weight.shape != values.shape:
        weight = torch.broadcast_to(weight, values.shape)
    denom = weight.sum().clamp_min(1.0)
    return (values * weight).sum() / denom


def build_valid_token_mask(labels: Tensor, attention_mask: Tensor | None) -> Tensor:
    if labels.ndim != 2:
        raise ValueError(f"labels must be [B,S], got {tuple(labels.shape)}")
    valid = labels != -100
    if attention_mask is None:
        return valid
    if attention_mask.ndim != 2 or attention_mask.shape[0] != labels.shape[0]:
        raise ValueError(
            f"attention_mask must be [B,S] and match batch, got {tuple(attention_mask.shape)}"
        )
    if attention_mask.shape[1] == labels.shape[1]:
        return valid & attention_mask.bool()
    n = min(int(labels.shape[1]), int(attention_mask.shape[1]))
    return valid[:, -n:] & attention_mask[:, -n:].bool()


# ============================================================
# Loss functions
# ============================================================

def _future_expert_kl_loss(
    pred_probs: Tensor, teacher_probs: Tensor, valid_mask: Tensor,
) -> Tensor:
    pred = pred_probs.to(torch.float32).clamp(min=1e-8)
    teacher = teacher_probs.to(torch.float32).clamp(min=1e-8)
    kl = (teacher * (teacher.log() - pred.log())).sum(dim=-1)
    return _masked_mean(kl, valid_mask)


def compute_topk_recall(pred_indices: Tensor, teacher_indices: Tensor, valid_mask: Tensor) -> Tensor:
    if pred_indices.ndim != 3 or teacher_indices.ndim != 3:
        raise ValueError(f"must be [B,S,K], got {tuple(pred_indices.shape)} / {tuple(teacher_indices.shape)}")
    seq = min(int(pred_indices.shape[1]), int(teacher_indices.shape[1]), int(valid_mask.shape[1]))
    pred = pred_indices[:, -seq:, :]
    teacher = teacher_indices[:, -seq:, :]
    mask = valid_mask[:, -seq:]
    matches = (pred.unsqueeze(-1) == teacher.unsqueeze(-2)).any(dim=-2).to(torch.float32)
    return _masked_mean(matches.mean(dim=-1), mask)


def compute_topk_exact(pred_indices: Tensor, teacher_indices: Tensor, valid_mask: Tensor) -> Tensor:
    if pred_indices.ndim != 3 or teacher_indices.ndim != 3:
        raise ValueError(f"must be [B,S,K], got {tuple(pred_indices.shape)} / {tuple(teacher_indices.shape)}")
    seq = min(int(pred_indices.shape[1]), int(teacher_indices.shape[1]), int(valid_mask.shape[1]))
    pred = pred_indices[:, -seq:, :]
    teacher = teacher_indices[:, -seq:, :]
    mask = valid_mask[:, -seq:]
    if pred.shape[-1] != teacher.shape[-1]:
        zeros = torch.zeros((pred.shape[0], pred.shape[1]), device=pred.device, dtype=torch.float32)
        return _masked_mean(zeros, mask)
    token_exact = (torch.sort(pred, dim=-1).values == torch.sort(teacher, dim=-1).values).all(dim=-1).float()
    return _masked_mean(token_exact, mask)


# ============================================================
# Language Model Primitives
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight * x).to(dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_seq_len_cached = max_position_embeddings

    def forward(self, x: Tensor, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DeepseekV2Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 max_position_embeddings: int, rope_theta: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings, rope_theta)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None,
                position_ids: Tensor | None = None,
                past_key_value: tuple[Tensor, Tensor] | None = None,
                use_cache: bool = False) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        bsz, q_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, q_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = past_key_value[0].shape[2] if past_key_value is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + q_len, device=hidden_states.device).unsqueeze(0).expand(bsz, -1)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos.unsqueeze(1), sin.unsqueeze(1))
        q = q.to(v.dtype)
        k = k.to(v.dtype)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)

        new_past = (k, v) if use_cache else None

        if self.num_kv_groups > 1:
            k_exp = k.repeat_interleave(self.num_kv_groups, dim=1)
            v_exp = v.repeat_interleave(self.num_kv_groups, dim=1)
        else:
            k_exp, v_exp = k, v

        attn_output = F.scaled_dot_product_attention(
            q, k_exp, v_exp, attn_mask=attention_mask,
            is_causal=(attention_mask is None and past_key_value is None),
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, -1)
        return self.o_proj(attn_output), new_past


class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DeepseekV2MoEGate(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int,
                 routed_scaling_factor: float = 1.0):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, hidden_dim = hidden_states.shape
        h = hidden_states.view(-1, hidden_dim)
        logits = F.linear(h.float(), self.weight.float())
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):
    def __init__(self, hidden_size: int, moe_intermediate_size: int,
                 num_experts: int, top_k: int, n_shared_experts: int,
                 routed_scaling_factor: float = 1.0):
        super().__init__()
        self.gate = DeepseekV2MoEGate(hidden_size, num_experts, top_k, routed_scaling_factor)
        self.experts = nn.ModuleList([
            DeepseekV2MLP(hidden_size, moe_intermediate_size)
            for _ in range(num_experts)
        ])
        if n_shared_experts > 0:
            self.shared_experts = DeepseekV2MLP(hidden_size, moe_intermediate_size * n_shared_experts)
        else:
            self.shared_experts = None
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, hidden_states: Tensor) -> Tensor:
        bsz, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states

        topk_idx, topk_weight = self.gate(hidden_states)
        h_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = h_flat.shape[0]

        out = torch.zeros_like(h_flat)
        for i in range(self.num_experts):
            mask = (topk_idx == i).any(dim=-1)
            if not mask.any():
                continue
            token_indices = mask.nonzero(as_tuple=True)[0]
            expert_input = h_flat[token_indices]
            expert_output = self.experts[i](expert_input)

            weight_mask = (topk_idx[token_indices] == i)
            expert_weight = (topk_weight[token_indices] * weight_mask.float()).sum(dim=-1, keepdim=True)
            out[token_indices] += expert_output * expert_weight

        out = out.view(bsz, seq_len, hidden_dim)
        if self.shared_experts is not None:
            out = out + self.shared_experts(residual)
        return out


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekVL2DraftRouterConfig, layer_idx: int):
        super().__init__()
        lang = config.language
        self.self_attn = DeepseekV2Attention(
            hidden_size=lang.hidden_size,
            num_heads=lang.num_attention_heads,
            num_kv_heads=lang.num_key_value_heads,
            max_position_embeddings=lang.max_position_embeddings,
            rope_theta=lang.rope_theta,
        )
        if layer_idx < lang.first_k_dense_replace:
            self.mlp = DeepseekV2MLP(lang.hidden_size, lang.intermediate_size)
        else:
            self.mlp = DeepseekV2MoE(
                hidden_size=lang.hidden_size,
                moe_intermediate_size=lang.moe_intermediate_size,
                num_experts=lang.n_routed_experts,
                top_k=lang.num_experts_per_tok,
                n_shared_experts=lang.n_shared_experts,
                routed_scaling_factor=lang.routed_scaling_factor,
            )
        self.input_layernorm = RMSNorm(lang.hidden_size, eps=lang.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(lang.hidden_size, eps=lang.rms_norm_eps)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None,
                position_ids: Tensor | None = None,
                past_key_value: tuple[Tensor, Tensor] | None = None,
                use_cache: bool = False) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, new_past = self.self_attn(
            hidden_states, attention_mask, position_ids,
            past_key_value=past_key_value, use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, new_past


class DeepseekV2Model(nn.Module):
    def __init__(self, config: DeepseekVL2DraftRouterConfig):
        super().__init__()
        lang = config.language
        self.embed_tokens = nn.Embedding(lang.vocab_size, lang.hidden_size)
        self.layers = nn.ModuleList([
            DeepseekV2DecoderLayer(config, i) for i in range(lang.num_hidden_layers)
        ])
        self.norm = RMSNorm(lang.hidden_size, eps=lang.rms_norm_eps)

    def forward(self, input_ids: Tensor | None = None,
                inputs_embeds: Tensor | None = None,
                attention_mask: Tensor | None = None,
                position_ids: Tensor | None = None,
                past_key_values: list | None = None,
                use_cache: bool = False) -> Tensor | tuple[Tensor, list]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        causal_mask = None
        if attention_mask is not None and attention_mask.ndim == 2:
            bsz, seq_len = hidden_states.shape[:2]
            causal_mask = torch.full(
                (seq_len, seq_len), float("-inf"), device=hidden_states.device, dtype=hidden_states.dtype
            )
            causal_mask = causal_mask.triu(diagonal=1).unsqueeze(0).unsqueeze(0)
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask.masked_fill(padding_mask, float("-inf"))

        new_past_key_values = [] if use_cache else None
        for i, layer in enumerate(self.layers):
            layer_past = past_key_values[i] if past_key_values is not None else None
            hidden_states, new_past = layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_value=layer_past, use_cache=use_cache,
            )
            if use_cache:
                new_past_key_values.append(new_past)

        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, new_past_key_values
        return hidden_states


class DeepseekV2ForCausalLM(nn.Module):
    """Language model wrapper. Weight prefix: language.model.* and language.lm_head.*"""
    def __init__(self, config: DeepseekVL2DraftRouterConfig):
        super().__init__()
        self.model = DeepseekV2Model(config)
        self.lm_head = nn.Linear(config.language.hidden_size, config.language.vocab_size, bias=False)

    def forward(self, input_ids: Tensor | None = None,
                inputs_embeds: Tensor | None = None,
                attention_mask: Tensor | None = None,
                position_ids: Tensor | None = None,
                labels: Tensor | None = None,
                past_key_values: list | None = None,
                use_cache: bool = False) -> tuple[Tensor | None, Tensor] | tuple[Tensor | None, Tensor, list]:
        model_out = self.model(
            input_ids=input_ids, inputs_embeds=inputs_embeds,
            attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=use_cache,
        )
        if use_cache:
            hidden_states, new_past_key_values = model_out
        else:
            hidden_states = model_out
            new_past_key_values = None
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # DEBUG: check for out-of-range labels
            num_classes = shift_logits.size(-1)
            flat_labels = shift_labels.view(-1)
            valid_mask = flat_labels != -100
            if valid_mask.any():
                valid_labels = flat_labels[valid_mask]
                max_label = valid_labels.max().item()
                min_label = valid_labels.min().item()
                if max_label >= num_classes or min_label < 0:
                    raise ValueError(
                        f"[DEBUG] Labels out of range! "
                        f"min_label={min_label}, max_label={max_label}, "
                        f"num_classes(vocab_size)={num_classes}, "
                        f"logits.shape={shift_logits.shape}, labels.shape={shift_labels.shape}, "
                        f"logits.device={shift_logits.device}, labels.device={flat_labels.device}"
                    )
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if use_cache:
            return loss, logits, new_past_key_values
        return loss, logits


# ============================================================
# Vision Model (SigLIP)
# ============================================================

class SigLIPAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.qkv = nn.Linear(width, width * 3, bias=True)
        self.proj = nn.Linear(width, width, bias=True)
        self.head_dim = 64
        self.num_heads = width // self.head_dim

    def forward(self, x: Tensor) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn_output = F.scaled_dot_product_attention(q, k, v)
        return self.proj(attn_output.transpose(1, 2).reshape(B, N, C))


class SigLIPMLP(nn.Module):
    def __init__(self, width: int, mlp_ratio: float):
        super().__init__()
        mlp_width = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, mlp_width, bias=True)
        self.fc2 = nn.Linear(mlp_width, width, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SigLIPBlock(nn.Module):
    def __init__(self, width: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = SigLIPAttention(width)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = SigLIPMLP(width, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SigLIPAttentionPooling(nn.Module):
    def __init__(self, width: int, mlp_ratio: float = 3.7362):
        super().__init__()
        self.latent = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.q = nn.Linear(width, width, bias=True)
        self.kv = nn.Linear(width, width * 2, bias=True)
        self.proj = nn.Linear(width, width, bias=True)
        self.norm = nn.LayerNorm(width)
        self.mlp = SigLIPMLP(width, mlp_ratio=mlp_ratio)
        self.num_heads = width // 64
        self.head_dim = 64

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        latent = self.latent.expand(B, -1, -1)
        q = self.q(latent).reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(x).reshape(B, x.shape[1], 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, 1, -1)
        out = latent + self.proj(attn_out)
        out = out + self.mlp(self.norm(out))
        return out


class SigLIPVisionEncoder(nn.Module):
    """Weight prefix: vision.*"""
    def __init__(self, config: DeepseekVL2DraftRouterConfig):
        super().__init__()
        vis = config.vision
        width = vis.width
        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, width, kernel_size=vis.patch_size, stride=vis.patch_size, bias=True)
        grid_size = vis.image_size // vis.patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, grid_size * grid_size, width))
        self.blocks = nn.ModuleList([SigLIPBlock(width, vis.mlp_ratio) for _ in range(vis.layers)])
        self.norm = nn.LayerNorm(width)
        self.attn_pool = SigLIPAttentionPooling(width, mlp_ratio=vis.mlp_ratio)
        self.width = width
        self.grid_size = grid_size

    def forward(self, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
        x = self.patch_embed.proj(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :x.shape[1], :]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        pooled = self.attn_pool(x)
        return x, pooled


# ============================================================
# VL Model
# ============================================================

class MLPProjector(nn.Module):
    """Weight prefix: projector.*"""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, out_features, bias=True),
            nn.GELU(),
            nn.Linear(out_features, out_features, bias=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DeepseekVL2ForConditionalGeneration(nn.Module):
    def __init__(self, config: DeepseekVL2DraftRouterConfig):
        super().__init__()
        self.config = config
        self.language = DeepseekV2ForCausalLM(config)
        self.vision = SigLIPVisionEncoder(config)

        vis_width = config.vision.width
        grid_size = config.vision.image_size // config.vision.patch_size
        ds_ratio = config.downsample_ratio
        ds_grid = grid_size // ds_ratio
        proj_in = vis_width * (ds_ratio ** 2)
        self.projector = MLPProjector(proj_in, config.language.hidden_size)

        self.image_newline = nn.Parameter(torch.zeros(config.language.hidden_size))
        self.view_seperator = nn.Parameter(torch.zeros(config.language.hidden_size))

        self._ds_ratio = ds_ratio
        self._ds_grid = ds_grid
        self._grid_size = grid_size
        self._image_token_id = config.image_token_id
        self._tokens_per_tile = ds_grid * (ds_grid + 1)

    def _process_vision_features(self, pixel_values: Tensor) -> Tensor:
        patch_features, _ = self.vision(pixel_values)

        B, N, C = patch_features.shape
        h = w = self._grid_size
        x = patch_features.view(B, h, w, C)

        ds = self._ds_ratio
        dh = h // ds
        dw = w // ds
        x = x[:, :dh*ds, :dw*ds, :].reshape(B, dh, ds, dw, ds, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, dh, dw, ds*ds*C)

        x = self.projector(x)

        newline = self.image_newline.view(1, 1, 1, -1).expand(B, dh, 1, -1)
        x = torch.cat([x, newline], dim=2)
        x = x.reshape(B, -1, x.shape[-1])
        return x

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        pixel_values: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor | None, Tensor]:
        inputs_embeds = self.language.model.embed_tokens(input_ids)

        if pixel_values is not None and pixel_values.numel() > 0:
            image_features = self._process_vision_features(pixel_values)

            image_mask = input_ids == self._image_token_id
            num_image_tokens = image_mask.sum()

            if num_image_tokens > 0:
                image_features_flat = image_features.reshape(-1, image_features.shape[-1])
                actual_count = min(num_image_tokens, image_features_flat.shape[0])
                if actual_count > 0:
                    inputs_embeds[image_mask] = image_features_flat[:actual_count].to(inputs_embeds.dtype)

        return self.language(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
        )


# ============================================================
# Draft Router / Gate-Base Residual Predictor
# ============================================================

class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_size: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.fc2 = nn.Linear(hidden_size * 4, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class FutureExpertPredictor(nn.Module):
    def __init__(self, config: DeepseekVL2DraftRouterConfig, correction_mode: bool = False):
        super().__init__()
        self.input_hidden_size = config.language.hidden_size
        self.predictor_hidden_size = config.future_expert_predictor_hidden_size
        self.num_experts = config.language.n_routed_experts
        self.num_layers = config.language.num_hidden_layers
        self.use_layer_embedding = config.future_expert_predictor_use_layer_embedding
        self.correction_mode = correction_mode

        self.input_proj = (
            nn.Identity()
            if self.input_hidden_size == self.predictor_hidden_size
            else nn.Linear(self.input_hidden_size, self.predictor_hidden_size, bias=False)
        )
        self.layer_embedding = (
            nn.Embedding(self.num_layers, self.predictor_hidden_size)
            if self.use_layer_embedding else None
        )
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(self.predictor_hidden_size,
                             config.future_expert_predictor_dropout)
            for _ in range(config.future_expert_predictor_num_layers)
        ])
        self.final_norm = nn.LayerNorm(self.predictor_hidden_size)
        self.next1_head = nn.Linear(self.predictor_hidden_size, self.num_experts, bias=False)
        self.next2_head = nn.Linear(self.predictor_hidden_size, self.num_experts, bias=False)

        if correction_mode:
            nn.init.zeros_(self.next1_head.weight)
            nn.init.zeros_(self.next2_head.weight)

    def forward(self, current_hidden_states: Tensor, current_layer_idx: int,
                attention_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        x = self.input_proj(current_hidden_states)
        if self.layer_embedding is not None:
            layer_vec = self.layer_embedding(
                torch.tensor(int(current_layer_idx), device=x.device, dtype=torch.long)
            ).view(1, 1, -1)
            x = x + layer_vec
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        logits_n1 = self.next1_head(x).float()
        logits_n2 = self.next2_head(x).float()
        if self.correction_mode:
            return logits_n1, logits_n2
        return logits_n1.softmax(dim=-1), logits_n2.softmax(dim=-1)


@dataclass
class FutureExpertPredictionContext:
    batch_size: int
    seq_len: int
    attention_mask: Tensor | None
    anchor_layers: set[int]
    collect_teacher_topk: bool = True
    run_predictor: bool = True
    teacher_topk_by_layer: dict[int, Tensor] = field(default_factory=dict)
    teacher_scores_by_layer: dict[int, Tensor] = field(default_factory=dict)
    probs_next1_by_anchor: dict[int, Tensor] = field(default_factory=dict)
    probs_next2_by_anchor: dict[int, Tensor] = field(default_factory=dict)


class TeacherRouterRecorderGate(nn.Module):
    def __init__(self, teacher_gate: DeepseekV2MoEGate, layer_idx: int,
                 owner: "DeepseekVL2DraftRouterForConditionalGeneration"):
        super().__init__()
        self.weight = teacher_gate.weight
        self.top_k = teacher_gate.top_k
        self.num_experts = teacher_gate.num_experts
        self.routed_scaling_factor = teacher_gate.routed_scaling_factor
        self.layer_idx = layer_idx
        self._owner_ref = weakref.ref(owner)

    def set_owner(self, owner: "DeepseekVL2DraftRouterForConditionalGeneration") -> None:
        self._owner_ref = weakref.ref(owner)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len, hidden_dim = hidden_states.shape
        h = hidden_states.view(-1, hidden_dim)
        logits = F.linear(h.float(), self.weight.float())
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.routed_scaling_factor

        owner = self._owner_ref()
        if owner is not None:
            owner._on_gate_forward(
                layer_idx=self.layer_idx,
                hidden_states=hidden_states,
                topk_idx=topk_idx,
                gate_scores=scores,
            )
        return topk_idx, topk_weight


class DeepseekVL2DraftRouterForConditionalGeneration(DeepseekVL2ForConditionalGeneration):
    def __init__(self, config: DeepseekVL2DraftRouterConfig):
        super().__init__(config)

        self._future_context: FutureExpertPredictionContext | None = None
        self._last_known_global_step: int = 0
        self._metrics_output_dir: str | None = None
        self._future_log_accum = self._make_empty_log_accum()

        self._moe_layer_indices = self._install_router_recorders()
        self._anchor_layers = self._resolve_anchor_layers(config.future_expert_predictor_anchor_layers)

        self._fusion_mode = config.future_expert_fusion_mode
        is_correction = self._fusion_mode == "correction"
        self.future_expert_predictor = FutureExpertPredictor(config, correction_mode=is_correction)
        self._horizons = sorted(set(int(h) for h in config.future_expert_predictor_horizons))

        if self._fusion_mode == "alpha":
            init_val = float(config.future_expert_fusion_init_alpha)
            self.fusion_alphas = nn.ParameterDict()
            for layer_idx in sorted(self._anchor_layers):
                for h in self._horizons:
                    key = f"layer{layer_idx}_h{h}"
                    self.fusion_alphas[key] = nn.Parameter(torch.tensor(init_val))

        elif self._fusion_mode == "lora":
            rank = config.future_expert_lora_rank
            hidden_size = config.language.hidden_size
            num_experts = config.language.n_routed_experts
            self.lora_A = nn.ParameterDict()
            self.lora_B = nn.ParameterDict()
            for layer_idx in sorted(self._anchor_layers):
                for h in self._horizons:
                    target_layer = layer_idx + h
                    if target_layer in self._moe_layer_indices:
                        key = f"layer{layer_idx}_h{h}"
                        self.lora_A[key] = nn.Parameter(
                            torch.randn(rank, hidden_size) * (1.0 / rank))
                        self.lora_B[key] = nn.Parameter(
                            torch.zeros(num_experts, rank))

        elif self._fusion_mode == "reinit_gate":
            hidden_size = config.language.hidden_size
            num_experts = config.language.n_routed_experts
            self.reinit_gate_weights = nn.ParameterDict()
            for layer_idx in sorted(self._anchor_layers):
                for h in self._horizons:
                    target_layer = layer_idx + h
                    if target_layer in self._moe_layer_indices:
                        key = f"layer{layer_idx}_h{h}"
                        w = torch.empty(num_experts, hidden_size)
                        nn.init.kaiming_uniform_(w, a=math.sqrt(5))
                        self.reinit_gate_weights[key] = nn.Parameter(w)

        self._apply_freeze_policy()

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        logger.info(
            "FutureExpertPredictor created. trainable=%d total=%d ratio=%.6f anchors=%s horizons=%s fusion_mode=%s",
            n_trainable, n_total,
            float(n_trainable) / float(max(1, n_total)),
            sorted(self._anchor_layers), self._horizons,
            self._fusion_mode,
        )

    @staticmethod
    def _make_empty_log_accum() -> dict[str, float]:
        return {
            "n": 0.0, "loss_lm": 0.0, "valid_lm_tokens": 0.0,
            "loss_future_expert": 0.0, "loss_future_expert_next1": 0.0,
            "loss_future_expert_next2": 0.0,
            "loss_next1_kl": 0.0, "loss_next2_kl": 0.0,
            "loss_total": 0.0, "recall_next1": 0.0, "recall_next2": 0.0,
            "exact_next1": 0.0, "exact_next2": 0.0,
            "token_acc": 0.0,
            "fusion_alpha_h1_mean": 0.0, "fusion_alpha_h2_mean": 0.0,
        }

    def _resolve_anchor_layers(self, configured: list[int] | None) -> set[int]:
        if configured is None:
            anchors = sorted(self._moe_layer_indices)
        else:
            anchors = sorted(set(int(x) for x in configured))
            missing = [x for x in anchors if x not in self._moe_layer_indices]
            if missing:
                raise ValueError(f"Anchor layers not MoE: {missing}, available: {sorted(self._moe_layer_indices)}")
        return set(anchors)

    def _install_router_recorders(self) -> set[int]:
        layers = self.language.model.layers
        moe_layers: list[int] = []
        for layer_idx, decoder_layer in enumerate(layers):
            mlp = getattr(decoder_layer, "mlp", None)
            if not isinstance(mlp, DeepseekV2MoE):
                continue
            gate = mlp.gate
            moe_layers.append(layer_idx)
            if isinstance(gate, TeacherRouterRecorderGate):
                gate.set_owner(self)
                continue
            mlp.gate = TeacherRouterRecorderGate(gate, layer_idx, self)
        if not moe_layers:
            raise RuntimeError("No MoE layers found in language model")
        return set(moe_layers)

    def _apply_freeze_policy(self) -> None:
        cfg = self.config
        if cfg.freeze_backbone:
            for p in self.language.parameters():
                p.requires_grad_(False)
            for p in self.vision.parameters():
                p.requires_grad_(False)
            for p in self.projector.parameters():
                p.requires_grad_(False)
            self.image_newline.requires_grad_(False)
            self.view_seperator.requires_grad_(False)

        if cfg.freeze_lm_head:
            for p in self.language.lm_head.parameters():
                p.requires_grad_(False)

        if cfg.freeze_input_embedding:
            for p in self.language.model.embed_tokens.parameters():
                p.requires_grad_(False)

        if cfg.freeze_experts:
            for layer in self.language.model.layers:
                mlp = getattr(layer, "mlp", None)
                if not isinstance(mlp, DeepseekV2MoE):
                    continue
                for p in mlp.experts.parameters():
                    p.requires_grad_(False)
                if mlp.shared_experts is not None:
                    for p in mlp.shared_experts.parameters():
                        p.requires_grad_(False)

        if cfg.freeze_router_teacher:
            for layer in self.language.model.layers:
                mlp = getattr(layer, "mlp", None)
                if not isinstance(mlp, DeepseekV2MoE):
                    continue
                gate = mlp.gate
                if isinstance(gate, TeacherRouterRecorderGate):
                    gate.weight.requires_grad_(False)
                else:
                    for p in gate.parameters():
                        p.requires_grad_(False)

        if self._fusion_mode in ("lora", "reinit_gate"):
            for p in self.future_expert_predictor.parameters():
                p.requires_grad_(False)
        else:
            for p in self.future_expert_predictor.parameters():
                p.requires_grad_(True)

        if hasattr(self, "fusion_alphas"):
            for p in self.fusion_alphas.parameters():
                p.requires_grad_(True)

        if hasattr(self, "lora_A"):
            for p in self.lora_A.parameters():
                p.requires_grad_(True)
        if hasattr(self, "lora_B"):
            for p in self.lora_B.parameters():
                p.requires_grad_(True)

        if hasattr(self, "reinit_gate_weights"):
            for p in self.reinit_gate_weights.parameters():
                p.requires_grad_(True)

    def _get_gate_weight(self, layer_idx: int) -> Tensor | None:
        if layer_idx not in self._moe_layer_indices:
            return None
        layers = self.language.model.layers
        if layer_idx >= len(layers):
            return None
        gate = layers[layer_idx].mlp.gate
        if isinstance(gate, TeacherRouterRecorderGate):
            return gate.weight
        return getattr(gate, "weight", None)

    def _on_gate_forward(self, *, layer_idx: int, hidden_states: Tensor,
                         topk_idx: Tensor, gate_scores: Tensor) -> None:
        ctx = self._future_context
        if ctx is None:
            return

        bsz, seq_len, hidden_dim = hidden_states.shape
        if bsz != ctx.batch_size or seq_len != ctx.seq_len:
            return

        if ctx.run_predictor and layer_idx in ctx.anchor_layers:
            if self._fusion_mode == "lora":
                output_n1 = None
                output_n2 = None
            elif self._fusion_mode == "reinit_gate":
                output_n1 = None
                output_n2 = None
            else:
                output_n1, output_n2 = self.future_expert_predictor(
                    current_hidden_states=hidden_states,
                    current_layer_idx=layer_idx,
                    attention_mask=ctx.attention_mask,
                )

            if self._fusion_mode == "alpha" and hasattr(self, "fusion_alphas"):
                if 1 in self._horizons:
                    gate_w_n1 = self._get_gate_weight(layer_idx + 1)
                    if gate_w_n1 is not None:
                        h_flat = hidden_states.reshape(-1, hidden_dim).float()
                        pred2_n1 = F.linear(h_flat, gate_w_n1.float()).softmax(dim=-1)
                        pred2_n1 = pred2_n1.reshape(bsz, seq_len, -1)
                        alpha_key = f"layer{layer_idx}_h1"
                        a = torch.sigmoid(self.fusion_alphas[alpha_key])
                        output_n1 = a * output_n1 + (1.0 - a) * pred2_n1

                if 2 in self._horizons:
                    gate_w_n2 = self._get_gate_weight(layer_idx + 2)
                    if gate_w_n2 is not None:
                        h_flat = hidden_states.reshape(-1, hidden_dim).float()
                        pred2_n2 = F.linear(h_flat, gate_w_n2.float()).softmax(dim=-1)
                        pred2_n2 = pred2_n2.reshape(bsz, seq_len, -1)
                        alpha_key = f"layer{layer_idx}_h2"
                        a = torch.sigmoid(self.fusion_alphas[alpha_key])
                        output_n2 = a * output_n2 + (1.0 - a) * pred2_n2

            elif self._fusion_mode == "correction":
                if 1 in self._horizons:
                    gate_w = self._get_gate_weight(layer_idx + 1)
                    if gate_w is not None:
                        h_flat = hidden_states.reshape(-1, hidden_dim).float()
                        pred2_logits = F.linear(h_flat, gate_w.float())
                        corr_n1 = output_n1.reshape(-1, output_n1.shape[-1])
                        fused_logits = pred2_logits + corr_n1
                        output_n1 = fused_logits.softmax(dim=-1).reshape(bsz, seq_len, -1)
                    else:
                        output_n1 = output_n1.softmax(dim=-1)

                if 2 in self._horizons:
                    gate_w = self._get_gate_weight(layer_idx + 2)
                    if gate_w is not None:
                        h_flat = hidden_states.reshape(-1, hidden_dim).float()
                        pred2_logits = F.linear(h_flat, gate_w.float())
                        corr_n2 = output_n2.reshape(-1, output_n2.shape[-1])
                        fused_logits = pred2_logits + corr_n2
                        output_n2 = fused_logits.softmax(dim=-1).reshape(bsz, seq_len, -1)
                    else:
                        output_n2 = output_n2.softmax(dim=-1)

            elif self._fusion_mode == "lora":
                h_flat = hidden_states.reshape(-1, hidden_dim).float()
                if 1 in self._horizons:
                    gate_w = self._get_gate_weight(layer_idx + 1)
                    key = f"layer{layer_idx}_h1"
                    if gate_w is not None and hasattr(self, "lora_A") and key in self.lora_A:
                        lora_correction = self.lora_B[key] @ self.lora_A[key]
                        effective_w = gate_w.float() + lora_correction.float()
                        output_n1 = F.linear(h_flat, effective_w).softmax(dim=-1).reshape(bsz, seq_len, -1)
                    elif gate_w is not None:
                        output_n1 = F.linear(h_flat, gate_w.float()).softmax(dim=-1).reshape(bsz, seq_len, -1)

                if 2 in self._horizons:
                    gate_w = self._get_gate_weight(layer_idx + 2)
                    key = f"layer{layer_idx}_h2"
                    if gate_w is not None and hasattr(self, "lora_A") and key in self.lora_A:
                        lora_correction = self.lora_B[key] @ self.lora_A[key]
                        effective_w = gate_w.float() + lora_correction.float()
                        output_n2 = F.linear(h_flat, effective_w).softmax(dim=-1).reshape(bsz, seq_len, -1)
                    elif gate_w is not None:
                        output_n2 = F.linear(h_flat, gate_w.float()).softmax(dim=-1).reshape(bsz, seq_len, -1)

            elif self._fusion_mode == "reinit_gate":
                h_flat = hidden_states.reshape(-1, hidden_dim).float()
                if 1 in self._horizons:
                    key = f"layer{layer_idx}_h1"
                    if hasattr(self, "reinit_gate_weights") and key in self.reinit_gate_weights:
                        output_n1 = F.linear(h_flat, self.reinit_gate_weights[key].float()).softmax(dim=-1).reshape(bsz, seq_len, -1)

                if 2 in self._horizons:
                    key = f"layer{layer_idx}_h2"
                    if hasattr(self, "reinit_gate_weights") and key in self.reinit_gate_weights:
                        output_n2 = F.linear(h_flat, self.reinit_gate_weights[key].float()).softmax(dim=-1).reshape(bsz, seq_len, -1)

            if output_n1 is not None:
                ctx.probs_next1_by_anchor[layer_idx] = output_n1
            if output_n2 is not None:
                ctx.probs_next2_by_anchor[layer_idx] = output_n2

        if ctx.collect_teacher_topk:
            topk_view = topk_idx.reshape(ctx.batch_size, ctx.seq_len, -1).detach()
            ctx.teacher_topk_by_layer[layer_idx] = topk_view

            scores_view = gate_scores.reshape(ctx.batch_size, ctx.seq_len, -1).detach()
            ctx.teacher_scores_by_layer[layer_idx] = scores_view

    def _zero_predictor_anchor_loss(self, *, device: torch.device) -> Tensor:
        for p in self.future_expert_predictor.parameters():
            if p.requires_grad:
                return torch.nan_to_num(p.view(-1)[0].float(), nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        if hasattr(self, "lora_A"):
            for p in self.lora_A.parameters():
                if p.requires_grad:
                    return torch.nan_to_num(p.view(-1)[0].float(), nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        if hasattr(self, "reinit_gate_weights"):
            for p in self.reinit_gate_weights.parameters():
                if p.requires_grad:
                    return torch.nan_to_num(p.view(-1)[0].float(), nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        if hasattr(self, "fusion_alphas"):
            for p in self.fusion_alphas.parameters():
                if p.requires_grad:
                    return torch.nan_to_num(p.view(-1)[0].float(), nan=0.0, posinf=0.0, neginf=0.0) * 0.0
        return torch.zeros((), device=device, dtype=torch.float32)

    @staticmethod
    def _align_tensors(pred_probs, teacher_probs, teacher_topk, valid_mask):
        seq = min(pred_probs.shape[1], teacher_probs.shape[1], teacher_topk.shape[1], valid_mask.shape[1])
        if seq <= 0:
            raise ValueError("Aligned sequence length must be > 0")
        return (pred_probs[:, -seq:, :], teacher_probs[:, -seq:, :],
                teacher_topk[:, -seq:, :], valid_mask[:, -seq:])

    def _compute_single_horizon_objective(self, *, pred_probs, teacher_probs, teacher_topk,
                                          valid_mask):
        pred_probs, teacher_probs, teacher_topk, valid_mask = self._align_tensors(
            pred_probs, teacher_probs, teacher_topk, valid_mask
        )
        zero = pred_probs.new_zeros(())
        if int(valid_mask.to(torch.int64).sum().item()) <= 0:
            return zero, zero.detach(), zero.detach()

        if not _is_finite_tensor(pred_probs) or not _is_finite_tensor(teacher_probs):
            return zero, zero.detach(), zero.detach()

        loss_kl = _future_expert_kl_loss(pred_probs, teacher_probs, valid_mask)
        if not _is_finite_tensor(loss_kl):
            return zero, zero.detach(), zero.detach()

        num_experts = int(pred_probs.shape[-1])
        active_top_k = int(teacher_topk.shape[-1])
        pred_k = min(active_top_k, num_experts)
        pred_topk = torch.topk(pred_probs, k=pred_k, dim=-1).indices
        recall = compute_topk_recall(pred_topk, teacher_topk[..., :pred_k], valid_mask)
        exact = compute_topk_exact(pred_topk, teacher_topk[..., :pred_k], valid_mask)
        return loss_kl, recall, exact

    def _compute_future_expert_loss_and_metrics(self, *, labels, attention_mask,
                                                teacher_topk_by_layer, teacher_scores_by_layer,
                                                probs_next1_by_anchor, probs_next2_by_anchor,
                                                device):
        valid_mask = build_valid_token_mask(labels, attention_mask)

        n1_losses, n2_losses = [], []
        n1_recalls, n2_recalls, n1_exacts, n2_exacts = [], [], [], []

        for layer_idx in sorted(self._anchor_layers):
            if 1 in self._horizons:
                tt = teacher_topk_by_layer.get(layer_idx + 1)
                ts = teacher_scores_by_layer.get(layer_idx + 1)
                pp = probs_next1_by_anchor.get(layer_idx)
                if tt is not None and ts is not None and pp is not None:
                    l, rc, ex = self._compute_single_horizon_objective(
                        pred_probs=pp, teacher_probs=ts, teacher_topk=tt,
                        valid_mask=valid_mask)
                    n1_losses.append(l)
                    n1_recalls.append(rc); n1_exacts.append(ex)

            if 2 in self._horizons:
                tt = teacher_topk_by_layer.get(layer_idx + 2)
                ts = teacher_scores_by_layer.get(layer_idx + 2)
                pp = probs_next2_by_anchor.get(layer_idx)
                if tt is not None and ts is not None and pp is not None:
                    l, rc, ex = self._compute_single_horizon_objective(
                        pred_probs=pp, teacher_probs=ts, teacher_topk=tt,
                        valid_mask=valid_mask)
                    n2_losses.append(l)
                    n2_recalls.append(rc); n2_exacts.append(ex)

        zero = self._zero_predictor_anchor_loss(device=device)
        loss_n1 = torch.stack(n1_losses).mean() if n1_losses else zero
        loss_n2 = torch.stack(n2_losses).mean() if n2_losses else zero

        loss_future = (
            float(self.config.future_expert_predictor_kl_coef_next1) * loss_n1
            + float(self.config.future_expert_predictor_kl_coef_next2) * loss_n2
        )
        return (
            loss_future,
            loss_n1, loss_n2,
            torch.stack(n1_recalls).mean() if n1_recalls else zero.detach(),
            torch.stack(n2_recalls).mean() if n2_recalls else zero.detach(),
            torch.stack(n1_exacts).mean() if n1_exacts else zero.detach(),
            torch.stack(n2_exacts).mean() if n2_exacts else zero.detach(),
        )

    def consume_future_expert_predictor_log_accum(self) -> dict[str, float]:
        out = dict(self._future_log_accum)
        self._future_log_accum = self._make_empty_log_accum()
        return out

    consume_draft_router_log_accum = consume_future_expert_predictor_log_accum

    def set_future_expert_predictor_output_dir(self, output_dir: str | None) -> None:
        if output_dir and str(output_dir).strip():
            os.makedirs(str(output_dir), exist_ok=True)
            self._metrics_output_dir = str(output_dir)

    def set_future_expert_predictor_global_step(self, global_step: int) -> None:
        try:
            self._last_known_global_step = int(global_step)
        except Exception:
            pass

    def _accumulate_logs(self, *, loss_lm, valid_lm_tokens, loss_future_expert,
                         loss_future_expert_next1, loss_future_expert_next2,
                         loss_next1_kl, loss_next2_kl,
                         loss_total, recall_next1, recall_next2, exact_next1, exact_next2,
                         token_acc: float = 0.0):
        a = self._future_log_accum
        a["n"] += 1.0
        a["loss_lm"] += float(loss_lm.detach().item())
        a["valid_lm_tokens"] += float(valid_lm_tokens)
        a["loss_future_expert"] += float(loss_future_expert.detach().item())
        a["loss_future_expert_next1"] += float(loss_future_expert_next1.detach().item())
        a["loss_future_expert_next2"] += float(loss_future_expert_next2.detach().item())
        a["loss_next1_kl"] += float(loss_next1_kl.detach().item())
        a["loss_next2_kl"] += float(loss_next2_kl.detach().item())
        a["loss_total"] += float(loss_total.detach().item())
        a["recall_next1"] += float(recall_next1.detach().item())
        a["recall_next2"] += float(recall_next2.detach().item())
        a["exact_next1"] += float(exact_next1.detach().item())
        a["exact_next2"] += float(exact_next2.detach().item())
        a["token_acc"] += float(token_acc)

        if hasattr(self, "fusion_alphas"):
            h1_vals, h2_vals = [], []
            for key, param in self.fusion_alphas.items():
                val = float(torch.sigmoid(param).item())
                if "_h1" in key:
                    h1_vals.append(val)
                elif "_h2" in key:
                    h2_vals.append(val)
            a["fusion_alpha_h1_mean"] += (sum(h1_vals) / len(h1_vals)) if h1_vals else 0.0
            a["fusion_alpha_h2_mean"] += (sum(h2_vals) / len(h2_vals)) if h2_vals else 0.0

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        pixel_values: Tensor | None = None,
        labels: Tensor | None = None,
        **kwargs,
    ) -> tuple[Tensor | None, Tensor]:
        predictor_enabled = bool(self.config.future_expert_predictor_enabled)

        if not predictor_enabled or labels is None:
            return super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, pixel_values=pixel_values, labels=labels, **kwargs
            )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        norm_mask = None
        if attention_mask is not None:
            norm_mask = attention_mask.to(device=device).bool()

        context = FutureExpertPredictionContext(
            batch_size=batch_size, seq_len=seq_len,
            attention_mask=norm_mask, anchor_layers=set(self._anchor_layers),
        )

        self._future_context = context
        try:
            loss_lm, logits = super().forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, pixel_values=pixel_values, labels=labels, **kwargs
            )
        finally:
            captured = self._future_context
            self._future_context = None

        if captured is None or labels is None:
            return loss_lm, logits

        valid_lm_tokens = float((labels != -100).float().sum().item())

        # Token accuracy: compare predicted tokens with labels on valid positions
        token_acc = 0.0
        if valid_lm_tokens > 0:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            valid_mask = shift_labels != -100
            if valid_mask.any():
                preds = shift_logits.argmax(dim=-1)
                token_acc = float((preds[valid_mask] == shift_labels[valid_mask]).float().mean().item())

        if loss_lm is None:
            loss_lm = self._zero_predictor_anchor_loss(device=logits.device)
        elif torch.is_tensor(loss_lm) and loss_lm.ndim > 0:
            loss_lm = loss_lm.mean()

        (
            loss_future, loss_n1, loss_n2,
            recall_n1, recall_n2, exact_n1, exact_n2,
        ) = self._compute_future_expert_loss_and_metrics(
            labels=labels, attention_mask=attention_mask,
            teacher_topk_by_layer=captured.teacher_topk_by_layer,
            teacher_scores_by_layer=captured.teacher_scores_by_layer,
            probs_next1_by_anchor=captured.probs_next1_by_anchor,
            probs_next2_by_anchor=captured.probs_next2_by_anchor,
            device=logits.device,
        )

        loss_total = loss_lm + loss_future
        if loss_total.ndim > 0:
            loss_total = loss_total.mean()

        self._accumulate_logs(
            loss_lm=loss_lm, valid_lm_tokens=valid_lm_tokens,
            loss_future_expert=loss_future,
            loss_future_expert_next1=loss_n1, loss_future_expert_next2=loss_n2,
            loss_next1_kl=loss_n1, loss_next2_kl=loss_n2,
            loss_total=loss_total,
            recall_next1=recall_n1, recall_next2=recall_n2,
            exact_next1=exact_n1, exact_next2=exact_n2,
            token_acc=token_acc,
        )

        return loss_total, logits


# ============================================================
# Weight Loading
# ============================================================

def load_pretrained_weights(model: nn.Module, model_dir: str) -> tuple[list[str], list[str]]:
    import glob
    safetensor_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    state_dict = {}
    for sf in safetensor_files:
        state_dict.update(safetensors_load_file(sf, device="cpu"))

    model_keys = set(n for n, _ in model.named_parameters())
    model_keys.update(n for n, _ in model.named_buffers())

    loaded_keys = set(state_dict.keys())
    missing = sorted(model_keys - loaded_keys)
    unexpected = sorted(loaded_keys - model_keys)

    result = model.load_state_dict(state_dict, strict=False)

    if missing:
        logger.info("Missing keys (%d): %s", len(missing), missing[:20])
    if unexpected:
        logger.info("Unexpected keys (%d): %s", len(unexpected), unexpected[:20])

    return missing, unexpected


__all__ = [
    "DeepseekVL2ForConditionalGeneration",
    "DeepseekVL2DraftRouterForConditionalGeneration",
    "FutureExpertPredictor",
    "load_pretrained_weights",
]
