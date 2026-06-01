"""
Batch evaluation of expert prediction accuracy across multiple datasets.

Metrics:
  - Recall@{3,6,8} for next-1 predictions
  - Exact match@6 for next-1 predictions

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval_expert_prediction_batch.py \
        --checkpoint /path/to/checkpoint \
        --fusion_mode correction \
        --datasets ./datasets/OCRBench.tsv ./datasets/GSM8K.tsv \
        --num_samples 0 \
        --max_new_tokens 200 \
        --output_json results.json
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
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
EVAL_TOP_KS = [3, 6, 8]
TEACHER_K = 6


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", type=str, nargs="+", required=True)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--fusion_mode", type=str, default="correction",
                        choices=["none", "alpha", "correction", "lora", "reinit_gate"])
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def load_tsv(path: str, num_samples: int = 0):
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
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BICUBIC)
    padded = Image.new("RGB", (target_w, target_h), (127, 127, 127))
    padded.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return padded


def process_image_for_model(image: Image.Image, candidate_resolutions):
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    aspect = w / max(h, 1)
    best_res, best_diff = candidate_resolutions[0], float("inf")
    for res in candidate_resolutions:
        rw, rh = res[1], res[0]
        diff = abs(aspect - rw / max(rh, 1)) + abs(rw * rh - w * h) / max(w * h, 1)
        if diff < best_diff:
            best_diff, best_res = diff, res
    target_h, target_w = int(best_res[0]), int(best_res[1])
    image = _resize_and_pad(image, target_h, target_w)
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array - 0.5) / 0.5
    pixel_values = torch.from_numpy(img_array).permute(2, 0, 1).float()
    num_tiles_h, num_tiles_w = max(1, target_h // 384), max(1, target_w // 384)
    tiles = []
    for i in range(num_tiles_h):
        for j in range(num_tiles_w):
            t, l = i * 384, j * 384
            tile = pixel_values[:, t:t+384, l:l+384]
            if tile.shape[1] == 384 and tile.shape[2] == 384:
                tiles.append(tile)
    if not tiles:
        tiles = [pixel_values[:, :384, :384]]
    return torch.stack(tiles, dim=0), num_tiles_h, num_tiles_w


def compute_num_image_tokens(tiles_h, tiles_w, image_size=384, downsample_ratio=2, patch_size=14):
    grid = image_size // patch_size
    ds_grid = grid // downsample_ratio
    return tiles_h * tiles_w * ds_grid * (ds_grid + 1)


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
    missing, _ = load_pretrained_weights(model, args.checkpoint)
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
class MultiTopKAccumulator:
    """Accumulate recall@{3,6,8} and exact@6 for next-1."""
    n_tokens: int = 0

    # keyed by (horizon, top_k) -> list of per-anchor-mean recall values
    recall_sums: dict = field(default_factory=lambda: defaultdict(float))
    recall_counts: dict = field(default_factory=lambda: defaultdict(int))

    # exact@6 for horizon 1
    exact6_sums: dict = field(default_factory=lambda: defaultdict(float))
    exact6_counts: dict = field(default_factory=lambda: defaultdict(int))

    def add_token(self, horizon: int, top_k: int, recall: float, exact6: float | None = None):
        key = (horizon, top_k)
        self.recall_sums[key] += recall
        self.recall_counts[key] += 1
        if top_k == TEACHER_K and exact6 is not None:
            self.exact6_sums[horizon] += exact6
            self.exact6_counts[horizon] += 1

    def get_recall(self, horizon: int, top_k: int) -> float:
        key = (horizon, top_k)
        return self.recall_sums[key] / max(self.recall_counts[key], 1)

    def get_exact6(self, horizon: int) -> float:
        return self.exact6_sums[horizon] / max(self.exact6_counts[horizon], 1)


def compute_recall_at_k(pred_probs: torch.Tensor, teacher_topk: torch.Tensor, k: int) -> float:
    """pred_probs: [num_experts], teacher_topk: [teacher_K]. Returns recall = |pred_topk ∩ teacher| / teacher_K."""
    pred_topk = pred_probs.topk(k).indices
    matches = (pred_topk.unsqueeze(-1) == teacher_topk.unsqueeze(0)).any(dim=0)
    return matches.float().mean().item()


def compute_exact_at_k(pred_probs: torch.Tensor, teacher_topk: torch.Tensor, k: int) -> float:
    pred_topk = pred_probs.topk(k).indices
    return float(torch.sort(pred_topk).values.equal(torch.sort(teacher_topk).values))


@torch.no_grad()
def generate_and_measure(model, input_ids, pixel_values, config, max_new_tokens, eos_token_id, device):
    anchor_layers = model._anchor_layers
    horizons = model._horizons

    accum = MultiTopKAccumulator()
    generated_ids = []
    bsz, prompt_len = input_ids.shape

    # === Prefill ===
    context = FutureExpertPredictionContext(
        batch_size=bsz, seq_len=prompt_len, attention_mask=None,
        anchor_layers=set(anchor_layers), collect_teacher_topk=True, run_predictor=True,
    )
    model._future_context = context
    try:
        inputs_embeds = model.language.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image_features = model._process_vision_features(pixel_values)
            image_mask = input_ids == model._image_token_id
            n_img = image_mask.sum()
            if n_img > 0:
                feats = image_features.reshape(-1, image_features.shape[-1])
                cnt = min(n_img, feats.shape[0])
                if cnt > 0:
                    inputs_embeds[image_mask] = feats[:cnt].to(inputs_embeds.dtype)
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

    # === Decode loop ===
    for step in range(max_new_tokens - 1):
        next_token_t = torch.tensor([[next_token]], dtype=torch.long, device=device)
        context = FutureExpertPredictionContext(
            batch_size=bsz, seq_len=1, attention_mask=None,
            anchor_layers=set(anchor_layers), collect_teacher_topk=True, run_predictor=True,
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

        accum.n_tokens += 1

        for anchor_idx in sorted(anchor_layers):
            for horizon in sorted(horizons):
                if horizon != 1:
                    continue
                target_layer = anchor_idx + horizon
                pred_probs = captured.probs_next1_by_anchor.get(anchor_idx)
                teacher_topk = captured.teacher_topk_by_layer.get(target_layer)

                if pred_probs is None or teacher_topk is None:
                    continue

                pred_vec = pred_probs[0, 0, :]
                teacher_vec = teacher_topk[0, 0, :]

                for k in EVAL_TOP_KS:
                    recall = compute_recall_at_k(pred_vec, teacher_vec, k)
                    exact6 = None
                    if k == TEACHER_K:
                        exact6 = compute_exact_at_k(pred_vec, teacher_vec, k)
                    accum.add_token(horizon, k, recall, exact6)

    return generated_ids, accum


@torch.no_grad()
def evaluate_dataset(model, tokenizer, samples, config, args):
    device = args.device
    image_token_id = config.image_token_id
    candidate_resolutions = config.candidate_resolutions
    eos_token_id = tokenizer.eos_token_id or 1

    per_sample_accums = []

    for si, sample in enumerate(tqdm(samples, desc="Evaluating")):
        has_image = bool(sample.get("image_b64", "").strip())
        pixel_t = None
        if has_image:
            try:
                image = decode_base64_image(sample["image_b64"])
            except Exception as e:
                print(f"  [SKIP] sample {si}: image decode error: {e}")
                continue
            pv, th, tw = process_image_for_model(image, candidate_resolutions)
            n_img_tokens = compute_num_image_tokens(th, tw)
            pixel_t = pv.to(dtype=torch.bfloat16, device=device)

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

        generated_ids, accum = generate_and_measure(
            model, input_t, pixel_t, config,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=eos_token_id, device=device,
        )
        per_sample_accums.append(accum)

    # aggregate
    agg = {}
    for horizon in [1]:
        h_key = f"next{horizon}"
        agg[h_key] = {}
        for k in EVAL_TOP_KS:
            recalls = [a.get_recall(horizon, k) for a in per_sample_accums if a.recall_counts.get((horizon, k), 0) > 0]
            agg[h_key][f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        exact_vals = [a.get_exact6(horizon) for a in per_sample_accums if a.exact6_counts.get(horizon, 0) > 0]
        agg[h_key]["exact@6"] = float(np.mean(exact_vals)) if exact_vals else 0.0

    total_tokens = sum(a.n_tokens for a in per_sample_accums)
    agg["n_samples"] = len(per_sample_accums)
    agg["total_decode_tokens"] = total_tokens
    return agg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    model, config = load_model(args)
    tokenizer = load_tokenizer(args.base_model)

    all_results = {}

    for ds_path in args.datasets:
        ds_name = os.path.splitext(os.path.basename(ds_path))[0]
        print(f"\n{'='*60}")
        print(f"  Dataset: {ds_name} ({ds_path})")
        print(f"{'='*60}")

        samples = load_tsv(ds_path, args.num_samples)
        print(f"  Loaded {len(samples)} samples")

        t0 = time.time()
        agg = evaluate_dataset(model, tokenizer, samples, config, args)
        elapsed = time.time() - t0

        agg["time_s"] = round(elapsed, 1)
        agg["time_per_sample"] = round(elapsed / max(agg["n_samples"], 1), 2)
        all_results[ds_name] = agg

        print(f"\n  {ds_name}: {agg['n_samples']} samples, {agg['total_decode_tokens']} decode tokens, {elapsed:.1f}s")
        for h_key in ["next1"]:
            d = agg[h_key]
            print(f"    {h_key}: recall@3={d['recall@3']:.4f}  recall@6={d['recall@6']:.4f}  recall@8={d['recall@8']:.4f}  exact@6={d['exact@6']:.4f}")

    # Save JSON
    out_json = args.output_json
    if not out_json:
        out_json = f"eval_results_{args.fusion_mode}.json"
    with open(out_json, "w") as f:
        json.dump({"fusion_mode": args.fusion_mode, "checkpoint": args.checkpoint, "results": all_results}, f, indent=2)
    print(f"\nResults saved to {out_json}")


if __name__ == "__main__":
    main()
