from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any


def _parse_int_list(value: Any, *, none_if_empty: bool) -> list[int] | None:
    if value is None:
        return None if none_if_empty else []
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None if none_if_empty else []
        parsed = [int(x.strip()) for x in raw.split(",") if x.strip()]
    elif isinstance(value, (list, tuple)):
        parsed = [int(x) for x in value]
    else:
        raise TypeError(f"Unsupported list field type: {type(value)!r}")
    if not parsed and none_if_empty:
        return None
    return parsed


@dataclass
class LanguageConfig:
    hidden_size: int = 1280
    intermediate_size: int = 6848
    moe_intermediate_size: int = 896
    num_hidden_layers: int = 12
    num_attention_heads: int = 10
    num_key_value_heads: int = 10
    vocab_size: int = 129280
    max_position_embeddings: int = 4096
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    num_experts_per_tok: int = 6
    first_k_dense_replace: int = 1
    topk_method: str = "greedy"
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    bos_token_id: int = 0
    eos_token_id: int = 1
    torch_dtype: str = "bfloat16"


@dataclass
class VisionConfig:
    model_name: str = "siglip_so400m_patch14_384"
    width: int = 1152
    layers: int = 27
    patch_size: int = 14
    mlp_ratio: float = 3.7362
    image_size: int = 384


@dataclass
class ProjectorConfig:
    n_embed: int = 1280


@dataclass
class DeepseekVL2DraftRouterConfig:
    language: LanguageConfig = field(default_factory=LanguageConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)

    candidate_resolutions: list[list[int]] = field(default_factory=lambda: [[384, 384]])
    tile_tag: str = "2D"
    global_view_pos: str = "head"
    downsample_ratio: int = 2
    image_token_id: int = 128815

    future_expert_predictor_enabled: bool = True
    future_expert_predictor_hidden_size: int = 1280
    future_expert_predictor_num_layers: int = 2
    future_expert_predictor_dropout: float = 0.1
    future_expert_predictor_use_layer_embedding: bool = True
    future_expert_predictor_anchor_layers: list[int] | None = None
    future_expert_predictor_horizons: list[int] = field(default_factory=lambda: [1, 2])

    future_expert_predictor_kl_coef_next1: float = 2.0
    future_expert_predictor_kl_coef_next2: float = 1.0

    future_expert_fusion_mode: str = "none"
    future_expert_fusion_init_alpha: float = 0.0
    future_expert_lora_rank: int = 64

    freeze_backbone: bool = True
    freeze_experts: bool = True
    freeze_router_teacher: bool = True
    freeze_lm_head: bool = True
    freeze_input_embedding: bool = True

    def __post_init__(self):
        horizons = sorted(set(int(h) for h in self.future_expert_predictor_horizons))
        if any(h <= 0 for h in horizons):
            raise ValueError(f"horizons must be positive, got {horizons}")
        self.future_expert_predictor_horizons = horizons

        valid_modes = ("none", "alpha", "correction", "lora", "reinit_gate")
        if self.future_expert_fusion_mode not in valid_modes:
            raise ValueError(
                f"future_expert_fusion_mode must be one of {valid_modes}, "
                f"got {self.future_expert_fusion_mode!r}"
            )

    @property
    def future_expert_fusion_enabled(self) -> bool:
        return self.future_expert_fusion_mode in ("alpha", "correction")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_pretrained(cls, model_dir: str, **overrides) -> "DeepseekVL2DraftRouterConfig":
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        lang_raw = raw.get("language_config", {})
        lang = LanguageConfig(
            hidden_size=lang_raw.get("hidden_size", 1280),
            intermediate_size=lang_raw.get("intermediate_size", 6848),
            moe_intermediate_size=lang_raw.get("moe_intermediate_size", 896),
            num_hidden_layers=lang_raw.get("num_hidden_layers", 12),
            num_attention_heads=lang_raw.get("num_attention_heads", 10),
            num_key_value_heads=lang_raw.get("num_key_value_heads", 10),
            vocab_size=lang_raw.get("vocab_size", 129280),
            max_position_embeddings=lang_raw.get("max_position_embeddings", 4096),
            n_routed_experts=lang_raw.get("n_routed_experts", 64),
            n_shared_experts=lang_raw.get("n_shared_experts", 2),
            num_experts_per_tok=lang_raw.get("num_experts_per_tok", 6),
            first_k_dense_replace=lang_raw.get("first_k_dense_replace", 1),
            topk_method=lang_raw.get("topk_method", "greedy"),
            n_group=lang_raw.get("n_group", 1),
            topk_group=lang_raw.get("topk_group", 1),
            routed_scaling_factor=lang_raw.get("routed_scaling_factor", 1.0),
            rms_norm_eps=lang_raw.get("rms_norm_eps", 1e-6),
            rope_theta=lang_raw.get("rope_theta", 10000.0),
            bos_token_id=lang_raw.get("bos_token_id", 0),
            eos_token_id=lang_raw.get("eos_token_id", 1),
            torch_dtype=lang_raw.get("torch_dtype", "bfloat16"),
        )

        vis_raw = raw.get("vision_config", {})
        vision = VisionConfig(
            model_name=vis_raw.get("model_name", "siglip_so400m_patch14_384"),
            width=vis_raw.get("width", 1152),
            layers=vis_raw.get("layers", 27),
            patch_size=vis_raw.get("patch_size", 14),
            mlp_ratio=vis_raw.get("mlp_ratio", 3.7362),
            image_size=384,
        )

        proj_raw = raw.get("projector_config", {})
        projector = ProjectorConfig(n_embed=proj_raw.get("n_embed", 1280))

        candidate_resolutions = raw.get("candidate_resolutions", [[384, 384]])

        proc_path = os.path.join(model_dir, "processor_config.json")
        downsample_ratio = 2
        image_token_id = 128815
        if os.path.exists(proc_path):
            with open(proc_path, "r", encoding="utf-8") as f:
                proc = json.load(f)
            downsample_ratio = proc.get("downsample_ratio", 2)

        cfg = cls(
            language=lang,
            vision=vision,
            projector=projector,
            candidate_resolutions=candidate_resolutions,
            tile_tag=raw.get("tile_tag", "2D"),
            global_view_pos=raw.get("global_view_pos", "head"),
            downsample_ratio=downsample_ratio,
            image_token_id=image_token_id,
        )

        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
        cfg.__post_init__()
        return cfg


__all__ = [
    "DeepseekVL2DraftRouterConfig",
    "LanguageConfig",
    "VisionConfig",
    "ProjectorConfig",
]
