"""
Evaluate expert prediction accuracy of a trained FutureExpertPredictor model
on OCRBench during autoregressive decoding.

Method:
  1. Feed only the prompt (image + question) into the model
  2. Generate tokens autoregressively (greedy)
  3. Measure predictor recall ONLY on decode-phase tokens

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_expert_prediction_ocrbench.py \
        --checkpoint /path/to/checkpoint \
        --dataset ./datasets/OCRBench.tsv \
        --fusion_mode correction \
        --num_samples 0 \
        --max_new_tokens 100
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.configuration_deepseek_vl2 import DeepseekVL2DraftRouterConfig
from model.modeling_deepseek_vl2 import (
    DeepseekV2MoE,
    DeepseekVL2DraftRouterForConditionalGeneration,
    FutureExpertPredictionContext,
    TeacherRouterRecorderGate,
    load_pretrained_weights,
)

BASE_MODEL_PATH = "/mnt/data/kjw/deepseek/deepseek-vl2-t"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate expert prediction on OCRBench (decode phase)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint directory")
    parser.add_argument("--dataset", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "datasets", "OCRBench.tsv"))
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--fusion_mode", type=str, default="correction",
                        choices=["none", "alpha", "correction", "lora"])
    parser.add_argument("--num_samples", type=int, default=0,
                        help="Number of samples to evaluate. 0 = all")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_ocrbench_tsv(path: str, num_samples: int = 0):
    csv.field_size_limit(sys.maxsize)
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            samples.append({
                "index": row.get("index", ""),
                "image_b64": row.get("image", ""),
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "category": row.get("category", ""),
            })
    if num_samples > 0:
        samples = samples[:num_samples]
    return samples


def decode_base64_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")


def _resize_and_pad(image: Image.Image, target_h: int, target_w: int) -> Image.Image:
    w, h = image.size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new("RGB", (target_w, target_h), (127, 127, 127))
    padded.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padded


def process_image_for_model(
    image: Image.Image,
    candidate_resolutions: list[list[int]],
    patch_size: int = 14,
) -> tuple[torch.Tensor, int, int]:
    if image.mode != "RGB":
        image = image.convert("RGB")

    w, h = image.size
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

    target_h, target_w = int(best_res[0]), int(best_res[1])
    image = _resize_and_pad(image, target_h, target_w)

    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).float()

    num_tiles_h = max(1, target_h // 384)
    num_tiles_w = max(1, target_w // 384)

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


def compute_num_image_tokens(num_tiles_h: int, num_tiles_w: int,
                             image_size: int = 384, downsample_ratio: int = 2,
                             patch_size: int = 14) -> int:
    grid = image_size // patch_size
    ds_grid = grid // downsample_ratio
    tokens_per_tile = ds_grid * (ds_grid + 1)
    return num_tiles_h * num_tiles_w * tokens_per_tile


def load_model(args):
    print(f"Loading config from {args.base_model}")
    config = DeepseekVL2DraftRouterConfig.from_pretrained(
        args.base_model,
        future_expert_predictor_enabled=True,
        future_expert_fusion_mode=args.fusion_mode,
    )

    print(f"Creating model (fusion_mode={args.fusion_mode})...")
    model = DeepseekVL2DraftRouterForConditionalGeneration(config)

    print(f"Loading checkpoint weights from {args.checkpoint}")
    missing, unexpected = load_pretrained_weights(model, args.checkpoint)
    pred_keys = [k for k in missing if "future_expert_predictor" in k or "lora_" in k or "fusion_alpha" in k]
    if pred_keys:
        print(f"WARNING: predictor/fusion missing keys ({len(pred_keys)}): {pred_keys[:10]}")

    model = model.to(dtype=torch.bfloat16, device=args.device)
    model.eval()
    return model, config


def load_tokenizer(base_model: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)


@dataclass
class DecodeRecallAccumulator:
    """Accumulate per-token recall during decode."""
    n_tokens: int = 0
    recall_n1_sum: float = 0.0
    recall_n2_sum: float = 0.0
    exact_n1_sum: float = 0.0
    exact_n2_sum: float = 0.0
    n_anchors_n1: int = 0
    n_anchors_n2: int = 0

    def add(self, recall_n1: float, recall_n2: float, exact_n1: float, exact_n2: float,
            has_n1: bool = True, has_n2: bool = True):
        self.n_tokens += 1
        if has_n1:
            self.recall_n1_sum += recall_n1
            self.exact_n1_sum += exact_n1
            self.n_anchors_n1 += 1
        if has_n2:
            self.recall_n2_sum += recall_n2
            self.exact_n2_sum += exact_n2
            self.n_anchors_n2 += 1

    @property
    def recall_n1(self) -> float:
        return self.recall_n1_sum / max(self.n_anchors_n1, 1)

    @property
    def recall_n2(self) -> float:
        return self.recall_n2_sum / max(self.n_anchors_n2, 1)

    @property
    def exact_n1(self) -> float:
        return self.exact_n1_sum / max(self.n_anchors_n1, 1)

    @property
    def exact_n2(self) -> float:
        return self.exact_n2_sum / max(self.n_anchors_n2, 1)


@torch.no_grad()
def generate_and_measure_recall(model, input_ids: torch.Tensor, pixel_values: torch.Tensor | None,
                                config, max_new_tokens: int, eos_token_id: int, device: str):
    """
    Autoregressive generation with KV cache and per-decode-step recall measurement.
    Returns: generated_ids, DecodeRecallAccumulator
    """
    top_k = config.language.num_experts_per_tok
    anchor_layers = model._anchor_layers
    horizons = model._horizons

    accum = DecodeRecallAccumulator()
    generated_ids = []

    bsz, prompt_len = input_ids.shape

    # === Prefill: process full prompt, get KV cache ===
    context = FutureExpertPredictionContext(
        batch_size=bsz,
        seq_len=prompt_len,
        attention_mask=None,
        anchor_layers=set(anchor_layers),
        collect_teacher_topk=True,
        run_predictor=True,
    )
    model._future_context = context

    try:
        inputs_embeds = model.language.model.embed_tokens(input_ids)

        if pixel_values is not None:
            image_features = model._process_vision_features(pixel_values)
            image_mask = input_ids == model._image_token_id
            num_image_tokens = image_mask.sum()
            if num_image_tokens > 0:
                image_features_flat = image_features.reshape(-1, image_features.shape[-1])
                actual_count = min(num_image_tokens, image_features_flat.shape[0])
                if actual_count > 0:
                    inputs_embeds[image_mask] = image_features_flat[:actual_count].to(inputs_embeds.dtype)

        hidden_states, past_key_values = model.language.model(
            inputs_embeds=inputs_embeds, use_cache=True
        )
        logits = model.language.lm_head(hidden_states[:, -1:, :])
    finally:
        model._future_context = None

    next_token = logits[0, 0, :].argmax(dim=-1).item()
    generated_ids.append(next_token)

    if next_token == eos_token_id:
        return generated_ids, accum

    # === Decode loop: one token at a time with KV cache ===
    for step in range(max_new_tokens - 1):
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=device)

        context = FutureExpertPredictionContext(
            batch_size=bsz,
            seq_len=1,
            attention_mask=None,
            anchor_layers=set(anchor_layers),
            collect_teacher_topk=True,
            run_predictor=True,
        )
        model._future_context = context

        try:
            token_embed = model.language.model.embed_tokens(next_token_t)
            hidden_states, past_key_values = model.language.model(
                inputs_embeds=token_embed, past_key_values=past_key_values, use_cache=True
            )
            logits = model.language.lm_head(hidden_states)
        finally:
            captured = model._future_context
            model._future_context = None

        next_token = logits[0, 0, :].argmax(dim=-1).item()
        generated_ids.append(next_token)

        if next_token == eos_token_id:
            break

        # Compute recall for this decode step (position 0 in seq_len=1 context)
        token_recall_n1_list = []
        token_recall_n2_list = []
        token_exact_n1_list = []
        token_exact_n2_list = []

        for anchor_idx in sorted(anchor_layers):
            if 1 in horizons:
                target_layer = anchor_idx + 1
                pred_probs = captured.probs_next1_by_anchor.get(anchor_idx)
                teacher_topk = captured.teacher_topk_by_layer.get(target_layer)

                if pred_probs is not None and teacher_topk is not None:
                    pred_last = pred_probs[0, 0, :]  # [num_experts]
                    teacher_last = teacher_topk[0, 0, :]  # [K]

                    pred_topk_indices = pred_last.topk(top_k).indices
                    matches = (pred_topk_indices.unsqueeze(-1) == teacher_last.unsqueeze(0)).any(dim=0)
                    recall = matches.float().mean().item()
                    exact = float(torch.sort(pred_topk_indices).values.equal(
                        torch.sort(teacher_last).values))

                    token_recall_n1_list.append(recall)
                    token_exact_n1_list.append(exact)

            if 2 in horizons:
                target_layer = anchor_idx + 2
                pred_probs = captured.probs_next2_by_anchor.get(anchor_idx)
                teacher_topk = captured.teacher_topk_by_layer.get(target_layer)

                if pred_probs is not None and teacher_topk is not None:
                    pred_last = pred_probs[0, 0, :]
                    teacher_last = teacher_topk[0, 0, :]

                    pred_topk_indices = pred_last.topk(top_k).indices
                    matches = (pred_topk_indices.unsqueeze(-1) == teacher_last.unsqueeze(0)).any(dim=0)
                    recall = matches.float().mean().item()
                    exact = float(torch.sort(pred_topk_indices).values.equal(
                        torch.sort(teacher_last).values))

                    token_recall_n2_list.append(recall)
                    token_exact_n2_list.append(exact)

        has_n1 = len(token_recall_n1_list) > 0
        has_n2 = len(token_recall_n2_list) > 0
        avg_recall_n1 = np.mean(token_recall_n1_list) if has_n1 else 0.0
        avg_recall_n2 = np.mean(token_recall_n2_list) if has_n2 else 0.0
        avg_exact_n1 = np.mean(token_exact_n1_list) if has_n1 else 0.0
        avg_exact_n2 = np.mean(token_exact_n2_list) if has_n2 else 0.0

        accum.add(avg_recall_n1, avg_recall_n2, avg_exact_n1, avg_exact_n2, has_n1, has_n2)

    return generated_ids, accum


@torch.no_grad()
def evaluate_samples(model, tokenizer, samples, config, args):
    device = args.device
    image_token_id = config.image_token_id
    candidate_resolutions = config.candidate_resolutions
    eos_token_id = tokenizer.eos_token_id or 1

    all_metrics = []
    per_category = defaultdict(list)

    for si, sample in enumerate(tqdm(samples, desc="Evaluating")):
        has_image = bool(sample.get("image_b64", "").strip())

        pixel_t = None
        if has_image:
            try:
                image = decode_base64_image(sample["image_b64"])
            except Exception as e:
                print(f"  [SKIP] sample {si}: image decode error: {e}")
                continue

            pixel_values, tiles_h, tiles_w = process_image_for_model(
                image, candidate_resolutions
            )
            n_img_tokens = compute_num_image_tokens(tiles_h, tiles_w)
            pixel_t = pixel_values.to(dtype=torch.bfloat16, device=device)

        question = sample["question"]

        user_ids = tokenizer.encode("<|User|>", add_special_tokens=False)
        asst_ids = tokenizer.encode("<|Assistant|>", add_special_tokens=False)
        nl_ids = tokenizer.encode("\n\n", add_special_tokens=False)
        bos_id = tokenizer.bos_token_id or 0

        if has_image:
            img_placeholder = [image_token_id] * n_img_tokens
            input_ids = [bos_id] + user_ids + img_placeholder + tokenizer.encode(question, add_special_tokens=False) + nl_ids + asst_ids
        else:
            input_ids = [bos_id] + user_ids + tokenizer.encode(question, add_special_tokens=False) + nl_ids + asst_ids

        if len(input_ids) > args.max_length - args.max_new_tokens:
            input_ids = input_ids[:args.max_length - args.max_new_tokens]

        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated_ids, accum = generate_and_measure_recall(
            model, input_t, pixel_t, config,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id,
            device=device,
        )

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        metrics = {
            "index": sample["index"],
            "category": sample["category"],
            "recall_n1": accum.recall_n1,
            "recall_n2": accum.recall_n2,
            "exact_n1": accum.exact_n1,
            "exact_n2": accum.exact_n2,
            "decode_tokens": accum.n_tokens,
            "generated": generated_text[:80],
            "answer": sample["answer"][:80],
        }
        all_metrics.append(metrics)
        per_category[sample["category"]].append(metrics)

    return all_metrics, per_category


def print_results(all_metrics, per_category, args):
    print(f"\n{'='*80}")
    print(f"  Expert Prediction Accuracy (Decode Phase Only)")
    print(f"  Model: {args.checkpoint}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Fusion mode: {args.fusion_mode}")
    print(f"  Samples evaluated: {len(all_metrics)}")
    total_decode_tokens = sum(m["decode_tokens"] for m in all_metrics)
    print(f"  Total decode tokens: {total_decode_tokens}")
    print(f"  Avg decode tokens/sample: {total_decode_tokens / max(len(all_metrics), 1):.1f}")
    print(f"{'='*80}")

    if not all_metrics:
        print("No samples evaluated!")
        return

    avg_recall_n1 = np.mean([m["recall_n1"] for m in all_metrics])
    avg_recall_n2 = np.mean([m["recall_n2"] for m in all_metrics])
    avg_exact_n1 = np.mean([m["exact_n1"] for m in all_metrics])
    avg_exact_n2 = np.mean([m["exact_n2"] for m in all_metrics])

    print(f"\n  Overall Results:")
    print(f"  {'Metric':<25s} {'Next-1':<12s} {'Next-2':<12s}")
    print(f"  {'-'*49}")
    print(f"  {'Top-K Recall':<25s} {avg_recall_n1:<12.4f} {avg_recall_n2:<12.4f}")
    print(f"  {'Exact Match':<25s} {avg_exact_n1:<12.4f} {avg_exact_n2:<12.4f}")

    if per_category:
        print(f"\n  Per-Category Results (Top-K Recall):")
        print(f"  {'Category':<30s} {'Count':<8s} {'Recall-N1':<12s} {'Recall-N2':<12s} {'Exact-N1':<12s}")
        print(f"  {'-'*74}")
        for cat in sorted(per_category.keys()):
            ms = per_category[cat]
            r1 = np.mean([m["recall_n1"] for m in ms])
            r2 = np.mean([m["recall_n2"] for m in ms])
            e1 = np.mean([m["exact_n1"] for m in ms])
            print(f"  {cat:<30s} {len(ms):<8d} {r1:<12.4f} {r2:<12.4f} {e1:<12.4f}")

    # Print a few generation examples
    print(f"\n  Sample Generations (first 10):")
    for m in all_metrics[:10]:
        print(f"    [{m['category'][:20]}] tokens={m['decode_tokens']} "
              f"recall_n1={m['recall_n1']:.4f} recall_n2={m['recall_n2']:.4f}")
        print(f"      gen: {m['generated']}")
        print(f"      ref: {m['answer']}")

    print(f"\n{'='*80}")


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"Loading dataset from {args.dataset} ...")
    samples = load_ocrbench_tsv(args.dataset, args.num_samples)
    print(f"  Loaded {len(samples)} samples")

    model, config = load_model(args)
    tokenizer = load_tokenizer(args.base_model)

    t0 = time.time()
    all_metrics, per_category = evaluate_samples(model, tokenizer, samples, config, args)
    elapsed = time.time() - t0

    print_results(all_metrics, per_category, args)
    print(f"\n  Time: {elapsed:.1f}s ({elapsed / max(len(all_metrics), 1):.2f}s/sample)")


if __name__ == "__main__":
    main()
