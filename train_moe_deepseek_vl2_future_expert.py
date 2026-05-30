"""
Train DeepSeek-VL2-Tiny with frozen teacher MoE and a lightweight
Future-2-Layer TopK Expert Recall Predictor.

Uses standalone HuggingFace Trainer (not ms-swift) because the mid_train env
has transformers>=5.x, incompatible with ms-swift's deepseek_vl2 support.

Core behavior:
1) Main DeepSeek-VL2 forward keeps native teacher router execution.
2) Predictor consumes current-layer hidden states and predicts future l+1 / l+2 TopK expert sets.
3) Training objective: loss_total = loss_lm + loss_future_expert (BCE + ranking loss).
4) By default, freeze backbone/expert/router/lm_head/embeddings and train predictor only.
"""

from __future__ import annotations

import argparse
import bisect
import json
import multiprocessing
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import torch
from datasets import concatenate_datasets
from datasets import load_dataset as hf_load_dataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from callbacks_deepseek_vl2_future_expert import FutureExpertPredictorMetricsCallback
from model.configuration_deepseek_vl2 import DeepseekVL2DraftRouterConfig
from model.modeling_deepseek_vl2 import (
    DeepseekVL2DraftRouterForConditionalGeneration,
    load_pretrained_weights,
)

import logging

logger = logging.getLogger(__name__)


def _is_master() -> bool:
    return os.environ.get("RANK", "0") == "0" and os.environ.get("LOCAL_RANK", "0") == "0"


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    raw = value.strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value


# ----------------------------
# Image processing
# ----------------------------

def _resize_and_pad(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    w, h = image.size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new("RGB", (target_w, target_h), (127, 127, 127))
    padded.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padded


def _select_best_resolution(image_size: tuple[int, int],
                             candidate_resolutions: list[list[int]]) -> tuple[int, int]:
    w, h = image_size
    aspect = w / max(h, 1)
    best_res = candidate_resolutions[0]
    best_diff = float("inf")
    for res in candidate_resolutions:
        rw, rh = res[1], res[0]
        res_aspect = rw / max(rh, 1)
        diff = abs(aspect - res_aspect) + abs(rw * rh - w * h) / max(w * h, 1)
        if diff < best_diff:
            best_diff = diff
            best_res = res
    return int(best_res[0]), int(best_res[1])


def process_image(
    image: Image.Image,
    candidate_resolutions: list[list[int]],
    patch_size: int = 14,
    image_mean: float = 0.5,
    image_std: float = 0.5,
) -> tuple[torch.Tensor, int, int]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    target_h, target_w = _select_best_resolution(image.size, candidate_resolutions)
    image = _resize_and_pad(image, target_h, target_w)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - image_mean) / image_std
    pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).float()

    tile_h = target_h // patch_size
    tile_w = target_w // patch_size

    num_tiles_h = target_h // 384
    num_tiles_w = target_w // 384
    if num_tiles_h < 1:
        num_tiles_h = 1
    if num_tiles_w < 1:
        num_tiles_w = 1

    tiles = []
    tile_size = 384
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            top = i * tile_size
            left = j * tile_size
            tile = pixel_values[:, top:top + tile_size, left:left + tile_size]
            if tile.shape[1] == tile_size and tile.shape[2] == tile_size:
                tiles.append(tile)

    if not tiles:
        tiles = [pixel_values[:, :tile_size, :tile_size]]

    pixel_values_tiled = torch.stack(tiles, dim=0)
    return pixel_values_tiled, num_tiles_h, num_tiles_w


def compute_num_image_tokens(
    num_tiles_h: int, num_tiles_w: int,
    patch_size: int = 14, image_size: int = 384, downsample_ratio: int = 2,
) -> int:
    grid = image_size // patch_size
    ds_grid = grid // downsample_ratio
    tokens_per_tile = ds_grid * (ds_grid + 1)
    return num_tiles_h * num_tiles_w * tokens_per_tile


# ----------------------------
# Dataset classes (same pattern as Qwen3)
# ----------------------------

def _load_single_dataset(task: tuple[str, float, str]):
    jsonl_path, ratio, recipe_dir = task
    if os.path.isabs(jsonl_path):
        abs_jsonl_path = jsonl_path
    else:
        abs_jsonl_path = os.path.abspath(os.path.join(recipe_dir, jsonl_path))
    if not os.path.exists(abs_jsonl_path):
        return None
    sub_ds = hf_load_dataset("json", data_files=abs_jsonl_path, split="train", keep_in_memory=False)
    current_data_dir = os.path.dirname(abs_jsonl_path)
    return sub_ds, current_data_dir, float(ratio), len(sub_ds)


class DynamicSamplingDataset:
    def __init__(self, recipe_path: str, max_workers: int | None = None):
        with open(recipe_path, "r", encoding="utf-8") as f:
            recipe: dict[str, float] = json.load(f)

        recipe_dir = os.path.dirname(os.path.abspath(recipe_path))
        tasks = [(jsonl_path, ratio, recipe_dir) for jsonl_path, ratio in recipe.items()]

        if max_workers is None:
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, len(tasks)))

        self.file_ranges: list[dict[str, Any]] = []
        self.dataset_root_dirs: list[dict[str, Any]] = []
        self.full_hf_dataset = None

        all_hf_datasets = []
        current_offset = 0

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_load_single_dataset, task): task for task in tasks}
            for future in as_completed(future_to_task):
                result = future.result()
                if result is None:
                    continue
                sub_ds, current_data_dir, ratio, real_count = result
                all_hf_datasets.append(sub_ds)
                self.dataset_root_dirs.append(
                    {"start": current_offset, "end": current_offset + real_count, "root_dir": current_data_dir}
                )
                self.file_ranges.append(
                    {"start": current_offset, "end": current_offset + real_count, "ratio": ratio, "count": real_count}
                )
                current_offset += real_count

        if not all_hf_datasets:
            raise RuntimeError(f"No dataset loaded from recipe: {recipe_path}")

        self.full_hf_dataset = concatenate_datasets(all_hf_datasets)
        self.sorted_root_dirs = sorted(self.dataset_root_dirs, key=lambda x: x["start"])

    def get_root_dir_for_idx(self, idx: int) -> str:
        starts = [item["start"] for item in self.sorted_root_dirs]
        pos = bisect.bisect_right(starts, idx) - 1
        if 0 <= pos < len(self.sorted_root_dirs):
            item = self.sorted_root_dirs[pos]
            if item["start"] <= idx < item["end"]:
                return item["root_dir"]
        return ""

    def get_sampled_indices(self) -> list[int]:
        sampled_indices = []
        for info in self.file_ranges:
            sample_count = max(1, int(info["count"] * info["ratio"]))
            start, end = int(info["start"]), int(info["end"])
            available_indices = np.arange(start, min(end, len(self.full_hf_dataset)))
            if len(available_indices) <= 0:
                continue
            selected = (
                available_indices
                if sample_count >= len(available_indices)
                else np.random.choice(available_indices, size=sample_count, replace=False)
            )
            sampled_indices.extend(selected.tolist())
        random.shuffle(sampled_indices)
        return sampled_indices


class EpochResampleDataset(Dataset):
    def __init__(
        self,
        dynamic_dataset: DynamicSamplingDataset,
        base_seed: int,
        mid_training_path: str,
        max_images_per_sample: int,
    ):
        self.dynamic_dataset = dynamic_dataset
        self.base_seed = int(base_seed)
        self.mid_training_path = mid_training_path
        self.max_images_per_sample = int(max_images_per_sample)
        self.current_indices: list[int] = []
        self.resample_for_epoch(epoch=0)

    def resample_for_epoch(self, epoch: int = 0):
        seed = self.base_seed + int(epoch)
        random.seed(seed)
        np.random.seed(seed)
        self.current_indices = self.dynamic_dataset.get_sampled_indices()
        if _is_master():
            logger.info("Epoch %d sampled %d examples", epoch, len(self.current_indices))

    def __len__(self):
        return len(self.current_indices)

    def __getitem__(self, idx: int):
        real_idx = int(self.current_indices[idx])
        row = self.dynamic_dataset.full_hf_dataset[real_idx]

        images = row.get("images")
        processed_images: list[str] = []
        if isinstance(images, list):
            for img in images:
                if os.path.isabs(img):
                    processed_images.append(img)
                else:
                    processed_images.append(os.path.join(self.mid_training_path, img.lstrip("/")))
            if self.max_images_per_sample > 0:
                processed_images = processed_images[: self.max_images_per_sample]
        row["images"] = processed_images

        messages = row.get("messages", [])
        if messages and isinstance(messages, list):
            raw_content = messages[0].get("content", "")
            if "<seg>" in raw_content:
                question, answer = raw_content.split("<seg>", 1)
            else:
                question, answer = "", raw_content
            question = question.strip()
            answer = answer.strip()
            row["question"] = question
            row["answer"] = answer
        else:
            row["question"] = ""
            row["answer"] = ""

        return row


class ResampleCallback(TrainerCallback):
    def __init__(self, dataset_instance: EpochResampleDataset):
        self.dataset_instance = dataset_instance

    def on_epoch_begin(self, args, state, control, **kwargs):
        current_epoch = int(state.epoch)
        self.dataset_instance.resample_for_epoch(current_epoch)


# ----------------------------
# Data collation
# ----------------------------

class DeepseekVL2DataCollator:
    def __init__(
        self,
        tokenizer,
        config: DeepseekVL2DraftRouterConfig,
        max_length: int = 2048,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.image_token_id = config.image_token_id
        self.candidate_resolutions = config.candidate_resolutions
        self.patch_size = config.vision.patch_size
        self.image_size = config.vision.image_size
        self.downsample_ratio = config.downsample_ratio

        self.bos_token_id = tokenizer.bos_token_id or 0
        self.eos_token_id = tokenizer.eos_token_id or 1
        self.pad_token_id = tokenizer.pad_token_id or 2

        self._user_token = "<|User|>"
        self._assistant_token = "<|Assistant|>"

        self._user_token_ids = tokenizer.encode(self._user_token, add_special_tokens=False)
        self._assistant_token_ids = tokenizer.encode(self._assistant_token, add_special_tokens=False)
        self._eos_ids = [self.eos_token_id]
        self._nl_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    def _build_conversation_ids(
        self,
        question: str,
        answer: str,
        num_image_tokens: int,
        has_image: bool,
    ) -> tuple[list[int], list[int]]:
        if has_image and num_image_tokens > 0:
            image_placeholder_ids = [self.image_token_id] * num_image_tokens
        else:
            image_placeholder_ids = []

        question_ids = self.tokenizer.encode(question, add_special_tokens=False) if question else []
        answer_ids = self.tokenizer.encode(answer, add_special_tokens=False) if answer else []

        input_ids = [self.bos_token_id]
        input_ids.extend(self._user_token_ids)

        if image_placeholder_ids:
            input_ids.extend(image_placeholder_ids)
        if question_ids:
            input_ids.extend(question_ids)

        input_ids.extend(self._nl_ids)
        input_ids.extend(self._assistant_token_ids)
        prompt_len = len(input_ids)

        input_ids.extend(answer_ids)
        input_ids.extend(self._eos_ids)

        labels = [-100] * prompt_len + input_ids[prompt_len:]

        return input_ids, labels

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch_input_ids = []
        batch_labels = []
        batch_pixel_values = []

        for sample in features:
            images = sample.get("images", [])
            question = sample.get("question", "")
            answer = sample.get("answer", "")

            pixel_values_list = []
            total_image_tokens = 0
            has_image = False

            if images:
                for img_path in images:
                    try:
                        pil_image = Image.open(img_path)
                        pv, tiles_h, tiles_w = process_image(
                            pil_image,
                            self.candidate_resolutions,
                            self.patch_size,
                        )
                        pixel_values_list.append(pv)
                        total_image_tokens += compute_num_image_tokens(
                            tiles_h, tiles_w,
                            self.patch_size, self.image_size, self.downsample_ratio,
                        )
                        has_image = True
                    except Exception as e:
                        if _is_master():
                            logger.warning("Failed to load image %s: %s", img_path, e)

            input_ids, labels = self._build_conversation_ids(
                question, answer, total_image_tokens, has_image
            )

            if len(input_ids) > self.max_length:
                input_ids = input_ids[:self.max_length]
                labels = labels[:self.max_length]

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            if pixel_values_list:
                batch_pixel_values.append(torch.cat(pixel_values_list, dim=0))
            else:
                batch_pixel_values.append(None)

        max_len = max(len(ids) for ids in batch_input_ids)

        padded_input_ids = []
        padded_labels = []
        padded_attention_mask = []
        for ids, lab in zip(batch_input_ids, batch_labels):
            pad_len = max_len - len(ids)
            padded_input_ids.append(ids + [self.pad_token_id] * pad_len)
            padded_labels.append(lab + [-100] * pad_len)
            padded_attention_mask.append([1] * len(ids) + [0] * pad_len)

        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }

        has_any_images = any(pv is not None for pv in batch_pixel_values)
        if has_any_images:
            valid_pvs = [pv for pv in batch_pixel_values if pv is not None]
            result["pixel_values"] = torch.cat(valid_pvs, dim=0)
        else:
            result["pixel_values"] = torch.zeros((0, 3, self.image_size, self.image_size))

        return result


# ----------------------------
# Custom Trainer
# ----------------------------

class DeepseekVL2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs.pop("pixel_values", None)
        labels = inputs.get("labels")

        loss, logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            pixel_values=pixel_values,
            labels=labels,
        )

        if loss is None:
            loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss, {"logits": logits}) if return_outputs else loss


# ----------------------------
# CLI and main
# ----------------------------

def _resolve_attn_impl(attn_impl: str) -> str:
    if attn_impl != "auto":
        return attn_impl
    try:
        import flash_attn  # noqa: F401
        return "flash_attn"
    except Exception:
        return "sdpa"


def _resolve_default_deepspeed(zero_stage: str) -> str | None:
    stage = "3" if zero_stage == "3" else "2"
    candidates: list[str] = []
    if stage == "3":
        candidates.extend([
            "/mnt/data/kjw_code/Qwen_moe/ms_swift/config/deepspeed/deepspeed_bf16_zero_3.json",
            "/mnt/data/kjw_code/Qwen_moe/ms_swift/swift/config/deepspeed/deepspeed_bf16_zero_3.json",
        ])
    else:
        candidates.extend([
            "/mnt/data/kjw_code/Qwen_moe/ms_swift/config/deepspeed/deepspeed_bf16_zero_2.json",
            "/mnt/data/kjw_code/Qwen_moe/ms_swift/swift/config/deepspeed/deepspeed_bf16_zero_2.json",
        ])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if stage == "3":
        candidates.append(os.path.join(script_dir, "deepspeed_bf16_zero3_stable.json"))
    else:
        candidates.extend([
            os.path.join(script_dir, "deepspeed_bf16_zero2_stable.json"),
            os.path.join(script_dir, "deepspeed_bf16_zero2_fast.json"),
        ])
    return next((p for p in candidates if os.path.exists(p)), None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train DeepSeek-VL2 Future-2-Layer TopK Expert Recall Predictor"
    )

    parser.add_argument("--model_path", type=str, default="/mnt/data/kjw/deepseek/deepseek-vl2-t")
    parser.add_argument("--mid_training_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--recipe_path", type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--attn_impl", type=str, default="auto", choices=["auto", "flash_attn", "sdpa", "eager"])
    parser.add_argument("--deepspeed", type=str, default=None)
    parser.add_argument("--gradient_checkpointing", type=_str2bool, default=False)
    parser.add_argument("--dataset_num_proc", type=int, default=4)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--dataloader_pin_memory", action="store_true", default=False)
    parser.add_argument("--max_images_per_sample", type=int, default=1)
    parser.add_argument("--run_name", type=str, default="deepseek_vl2_future_expert_predictor_sft_v2")
    parser.add_argument("--no_wandb", action="store_true", default=False)

    parser.add_argument("--future_expert_predictor_enabled", type=_str2bool, default=True)
    parser.add_argument("--future_expert_predictor_hidden_size", type=int, default=1280)
    parser.add_argument("--future_expert_predictor_num_layers", type=int, default=2)
    parser.add_argument("--future_expert_predictor_dropout", type=float, default=0.1)
    parser.add_argument("--future_expert_predictor_use_layer_embedding", type=_str2bool, default=True)
    parser.add_argument(
        "--future_expert_predictor_anchor_layers", type=str, default="",
        help="Comma-separated anchor layer ids. Empty means all available MoE layers.",
    )
    parser.add_argument(
        "--future_expert_predictor_horizons", type=str, default="1,2",
        help="Comma-separated future horizons.",
    )
    parser.add_argument("--future_expert_predictor_kl_coef_next1", type=float, default=2.0)
    parser.add_argument("--future_expert_predictor_kl_coef_next2", type=float, default=1.0)

    parser.add_argument("--future_expert_fusion_mode", type=str, default="none",
                        choices=["none", "alpha", "correction", "lora", "reinit_gate"])
    parser.add_argument("--future_expert_fusion_init_alpha", type=float, default=0.0)
    parser.add_argument("--future_expert_lora_rank", type=int, default=64)

    parser.add_argument("--freeze_backbone", type=_str2bool, default=True)
    parser.add_argument("--freeze_experts", type=_str2bool, default=True)
    parser.add_argument("--freeze_router_teacher", type=_str2bool, default=True)
    parser.add_argument("--freeze_lm_head", type=_str2bool, default=True)
    parser.add_argument("--freeze_input_embedding", type=_str2bool, default=True)

    parser.add_argument("--ddp_timeout", type=int, default=1800)
    return parser.parse_args()


def _parse_int_list(text: str) -> list[int] | None:
    text = text.strip()
    if not text:
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_horizons(text: str) -> list[int]:
    text = text.strip()
    if not text:
        return [1, 2]
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    args = parse_args()

    if (
        args.gradient_checkpointing
        and args.freeze_backbone
        and args.freeze_experts
        and args.freeze_router_teacher
        and args.freeze_lm_head
        and args.freeze_input_embedding
    ):
        if _is_master():
            logger.warning(
                "Detected predictor-only training with all backbone parts frozen; "
                "force disable gradient_checkpointing to avoid useless checkpoint warnings."
            )
        args.gradient_checkpointing = False

    # Resolve paths
    model_path = os.path.abspath(args.model_path)
    output_dir = os.path.abspath(args.output_dir)
    mid_training_path = os.path.abspath(args.mid_training_path)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.recipe_path is not None:
        recipe_path = os.path.abspath(os.path.expanduser(args.recipe_path))
    else:
        recipe_path = os.path.join(script_dir, "data_recipe/final_recipe.json")

    if not os.path.exists(recipe_path):
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Resolve DeepSpeed config
    zero_stage = _get_env("ZERO_STAGE", "2")
    if args.deepspeed is None:
        deepspeed_config = _resolve_default_deepspeed(zero_stage)
    else:
        raw_ds = args.deepspeed.strip().lower()
        if raw_ds in ("none", "off", "0"):
            deepspeed_config = None
        else:
            deepspeed_config = os.path.abspath(os.path.expanduser(args.deepspeed))

    attn_impl = _resolve_attn_impl(args.attn_impl)

    if _is_master():
        logger.info("model_path=%s", model_path)
        logger.info("output_dir=%s", output_dir)
        logger.info("recipe_path=%s", recipe_path)
        logger.info("attn_impl=%s", attn_impl)
        logger.info("deepspeed=%s", deepspeed_config)
        logger.info(
            "Future predictor settings: hidden=%s num_layers=%s dropout=%s anchors='%s' horizons='%s' "
            "kl_next1=%s kl_next2=%s fusion_mode=%s init_alpha=%s lora_rank=%s",
            args.future_expert_predictor_hidden_size,
            args.future_expert_predictor_num_layers,
            args.future_expert_predictor_dropout,
            args.future_expert_predictor_anchor_layers,
            args.future_expert_predictor_horizons,
            args.future_expert_predictor_kl_coef_next1,
            args.future_expert_predictor_kl_coef_next2,
            args.future_expert_fusion_mode,
            args.future_expert_fusion_init_alpha,
            args.future_expert_lora_rank,
        )

    # Build config
    anchor_layers = _parse_int_list(args.future_expert_predictor_anchor_layers)
    horizons = _parse_horizons(args.future_expert_predictor_horizons)

    config = DeepseekVL2DraftRouterConfig.from_pretrained(
        model_path,
        future_expert_predictor_enabled=args.future_expert_predictor_enabled,
        future_expert_predictor_hidden_size=args.future_expert_predictor_hidden_size,
        future_expert_predictor_num_layers=args.future_expert_predictor_num_layers,
        future_expert_predictor_dropout=args.future_expert_predictor_dropout,
        future_expert_predictor_use_layer_embedding=args.future_expert_predictor_use_layer_embedding,
        future_expert_predictor_anchor_layers=anchor_layers,
        future_expert_predictor_horizons=horizons,
        future_expert_predictor_kl_coef_next1=args.future_expert_predictor_kl_coef_next1,
        future_expert_predictor_kl_coef_next2=args.future_expert_predictor_kl_coef_next2,
        future_expert_fusion_mode=args.future_expert_fusion_mode,
        future_expert_fusion_init_alpha=args.future_expert_fusion_init_alpha,
        future_expert_lora_rank=args.future_expert_lora_rank,
        freeze_backbone=args.freeze_backbone,
        freeze_experts=args.freeze_experts,
        freeze_router_teacher=args.freeze_router_teacher,
        freeze_lm_head=args.freeze_lm_head,
        freeze_input_embedding=args.freeze_input_embedding,
    )

    if _is_master():
        logger.info("Config built: hidden=%d layers=%d experts=%d topk=%d",
                     config.language.hidden_size, config.language.num_hidden_layers,
                     config.language.n_routed_experts, config.language.num_experts_per_tok)

    # Build model
    if _is_master():
        logger.info("Building model...")
    model = DeepseekVL2DraftRouterForConditionalGeneration(config)

    if _is_master():
        logger.info("Loading pretrained weights from %s...", model_path)
    missing, unexpected = load_pretrained_weights(model, model_path)
    if _is_master():
        logger.info("Weight loading done. missing=%d unexpected=%d", len(missing), len(unexpected))

    model = model.to(dtype=torch.bfloat16)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    if _is_master():
        logger.info("Model params: trainable=%d total=%d ratio=%.6f",
                     n_trainable, n_total, n_trainable / max(1, n_total))

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 2

    # Build dataset
    if _is_master():
        logger.info("Loading datasets from recipe: %s", recipe_path)
    dynamic_ds = DynamicSamplingDataset(recipe_path, max_workers=args.dataset_num_proc)
    train_dataset = EpochResampleDataset(
        dynamic_ds,
        base_seed=42,
        mid_training_path=mid_training_path,
        max_images_per_sample=args.max_images_per_sample,
    )
    if _is_master():
        logger.info("Dataset loaded: %d samples", len(train_dataset))

    # Data collator
    data_collator = DeepseekVL2DataCollator(
        tokenizer=tokenizer,
        config=config,
        max_length=args.max_length,
    )

    # Training arguments
    report_to = [] if args.no_wandb else ["wandb"]
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        report_to=report_to,
        run_name=args.run_name,
        ddp_timeout=args.ddp_timeout,
        deepspeed=deepspeed_config,
        remove_unused_columns=False,
        seed=42,
    )

    # Callbacks
    resample_cb = ResampleCallback(train_dataset)
    metrics_cb = FutureExpertPredictorMetricsCallback(training_args, None)

    # Build trainer
    trainer = DeepseekVL2Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[resample_cb, metrics_cb],
    )
    metrics_cb.trainer = trainer

    if _is_master():
        logger.info("Starting training...")
    trainer.train()

    if _is_master():
        logger.info("Training complete. Saving final model...")
    trainer.save_model(output_dir)
    if _is_master():
        logger.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
