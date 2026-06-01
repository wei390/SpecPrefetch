"""
Verify FutureExpertPredictor accuracy during real inference and simulate
expert cache behavior under different GPU memory budgets.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 verify_expert_prefetch.py [--max-tokens 64] [--num-samples 20]
"""
import argparse
import json
import os
import random
import sys
from collections import OrderedDict, defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling_deepseek_vl2 import (
    DeepseekVL2DraftRouterForConditionalGeneration,
    DeepseekV2MoE,
    FutureExpertPredictor,
    load_pretrained_weights,
)
from model.configuration_deepseek_vl2 import DeepseekVL2DraftRouterConfig

BASE_MODEL_PATH = "/mnt/data/kjw/deepseek/deepseek-vl2-t"
CHECKPOINT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "deepseek_vl2_output_future_expert_no_compress_v1",
    "checkpoint-454",
)
DATA_PATH = "/mnt/data/kjw/data_qwen/deepseek-vl2-t/DocVQA/intermediate/DocVQA-QwenVL72b-passed.jsonl"


# ============================================================
# LRU Cache simulator
# ============================================================

class LRUExpertCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict[int, bool] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def access(self, expert_ids: list[int]) -> int:
        cold_loads = 0
        for eid in expert_ids:
            if eid in self.cache:
                self.cache.move_to_end(eid)
                self.hits += 1
            else:
                self.misses += 1
                cold_loads += 1
                self.cache[eid] = True
                while len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)
        return cold_loads

    def prefetch(self, expert_ids: list[int]):
        for eid in expert_ids:
            if eid in self.cache:
                self.cache.move_to_end(eid)
            else:
                self.cache[eid] = True
                while len(self.cache) > self.capacity:
                    self.cache.popitem(last=False)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ============================================================
# Core
# ============================================================

def load_model(device: str = "cuda"):
    print(f"Loading config from {BASE_MODEL_PATH}")
    config = DeepseekVL2DraftRouterConfig.from_pretrained(BASE_MODEL_PATH)
    config.future_expert_predictor_enabled = True
    config.future_expert_predictor_intermediate_size = 3456

    print("Creating model...")
    model = DeepseekVL2DraftRouterForConditionalGeneration(config)

    print(f"Loading weights from {CHECKPOINT_DIR}")
    missing, unexpected = load_pretrained_weights(model, CHECKPOINT_DIR)
    pred_missing = [k for k in missing if "future_expert_predictor" in k]
    if pred_missing:
        print(f"WARNING: predictor missing keys: {pred_missing}")

    model = model.to(dtype=torch.bfloat16, device=device)
    model.eval()
    return model, config


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)


def load_samples(n: int, seed: int = 42):
    random.seed(seed)
    with open(DATA_PATH) as f:
        lines = f.readlines()
    indices = random.sample(range(len(lines)), min(n, len(lines)))
    samples = []
    for i in indices:
        d = json.loads(lines[i])
        content = d["messages"][0]["content"]
        q, a = content.split("<seg>", 1)
        samples.append({"question": q.strip(), "answer": a.strip(), "line": i})
    return samples


@torch.no_grad()
def run_teacher_forcing(model, tokenizer, samples, device):
    """Run teacher-forcing evaluation using the model's built-in predictor path.
    Returns per-sample metrics and per-token gate/prediction data for cache simulation."""

    config = model.config
    n_experts = config.language.n_routed_experts
    top_k = config.language.num_experts_per_tok
    first_dense = config.language.first_k_dense_replace
    n_layers = config.language.num_hidden_layers
    moe_layer_indices = list(range(first_dense, n_layers))

    bos_id = tokenizer.bos_token_id or 0
    eos_id = tokenizer.eos_token_id or 1
    user_ids = tokenizer.encode("<|User|>", add_special_tokens=False)
    asst_ids = tokenizer.encode("<|Assistant|>", add_special_tokens=False)
    nl_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    # Hook gates to capture per-position expert selections
    gate_records = {}

    def make_gate_hook(li):
        def hook_fn(module, args, output):
            topk_idx, _ = output
            h = args[0]
            b, s, _ = h.shape
            gate_records[li] = topk_idx.reshape(b, s, -1).detach().cpu()
        return hook_fn

    hooks = []
    for li in moe_layer_indices:
        layer = model.language.model.layers[li]
        if isinstance(layer.mlp, DeepseekV2MoE):
            h = layer.mlp.gate.register_forward_hook(make_gate_hook(li))
            hooks.append(h)

    predictor = model.future_expert_predictor

    # Also hook hidden states for our own predictor evaluation
    hidden_records = {}

    def make_hidden_hook(li):
        def hook_fn(module, args, output):
            hidden_records[li] = args[0].detach()
        return hook_fn

    for li in moe_layer_indices:
        layer = model.language.model.layers[li]
        if isinstance(layer.mlp, DeepseekV2MoE):
            h = layer.mlp.gate.register_forward_hook(make_hidden_hook(li))
            hooks.append(h)

    all_sample_metrics = []
    all_token_data = []  # for cache simulation: per-token gate decisions and predictions

    for si, s in enumerate(samples):
        q_ids = tokenizer.encode(s["question"], add_special_tokens=False)
        a_ids = tokenizer.encode(s["answer"], add_special_tokens=False)
        input_ids = [bos_id] + user_ids + q_ids + nl_ids + asst_ids + a_ids + [eos_id]
        prompt_len = len([bos_id] + user_ids + q_ids + nl_ids + asst_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_t = torch.tensor([input_ids], dtype=torch.long, device=device)
        labels_t = torch.tensor([labels], dtype=torch.long, device=device)

        gate_records.clear()
        hidden_records.clear()

        # Use the built-in forward path which invokes the predictor
        loss, logits = model(input_ids=input_t, labels=labels_t)
        accum = model.consume_future_expert_predictor_log_accum()

        all_sample_metrics.append({
            "recall_n1": accum["recall_next1"],
            "exact_n1": accum["exact_next1"],
            "seq_len": len(input_ids),
            "question": s["question"][:50],
        })

        # Also capture per-position data for cache simulation
        # For each position in the sequence, record gate decisions and predictions
        seq_len = len(input_ids)
        for pos in range(seq_len):
            token_gates = {}
            token_preds = {}

            for li in moe_layer_indices:
                if li in gate_records and pos < gate_records[li].shape[1]:
                    token_gates[li] = gate_records[li][0, pos, :].tolist()

                if li in hidden_records and pos < hidden_records[li].shape[1]:
                    h = hidden_records[li][:, pos:pos+1, :]
                    ln1 = predictor(h, li)
                    pred_n1 = torch.topk(ln1[0, 0], k=top_k).indices.cpu().tolist()
                    token_preds[li] = {"next1": pred_n1}

            all_token_data.append({
                "gates": token_gates,
                "predictions": token_preds,
            })

    for h in hooks:
        h.remove()

    return all_sample_metrics, all_token_data, moe_layer_indices


def simulate_expert_cache(all_token_data, moe_layer_indices, n_experts, top_k, cache_sizes):
    results = {}

    for cs in cache_sizes:
        baseline_caches = {li: LRUExpertCache(cs) for li in moe_layer_indices}
        prefetch_caches = {li: LRUExpertCache(cs) for li in moe_layer_indices}
        baseline_cold = 0
        prefetch_cold = 0
        total_accesses = 0

        for td in all_token_data:
            gates = td["gates"]
            preds = td["predictions"]

            # Prefetch: use predictions from layer i to prefetch for layer i+1
            for anchor_li in sorted(preds.keys()):
                t1 = anchor_li + 1
                if t1 in prefetch_caches:
                    prefetch_caches[t1].prefetch(preds[anchor_li]["next1"])

            # Actual access
            for li in moe_layer_indices:
                if li not in gates:
                    continue
                experts = gates[li]
                total_accesses += len(experts)
                baseline_cold += baseline_caches[li].access(experts)
                prefetch_cold += prefetch_caches[li].access(experts)

        bl_hit = sum(c.hit_rate for c in baseline_caches.values()) / len(baseline_caches)
        pf_hit = sum(c.hit_rate for c in prefetch_caches.values()) / len(prefetch_caches)

        results[cs] = {
            "baseline_cold": baseline_cold,
            "prefetch_cold": prefetch_cold,
            "baseline_hit_rate": bl_hit,
            "prefetch_hit_rate": pf_hit,
            "total_accesses": total_accesses,
            "reduction": (baseline_cold - prefetch_cold) / max(baseline_cold, 1),
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model, config = load_model(args.device)
    tokenizer = load_tokenizer()
    samples = load_samples(args.num_samples)

    n_experts = config.language.n_routed_experts
    top_k = config.language.num_experts_per_tok

    print(f"\n{'='*70}")
    print(f"=== Expert Predictor Verification ===")
    print(f"{'='*70}")
    print(f"Model: DeepSeek VL2-tiny + FutureExpertPredictor")
    print(f"Config: {n_experts} experts/layer, top-{top_k}, 11 MoE layers")
    print(f"Samples: {len(samples)} from DocVQA training data (text-only, no images)")
    print(f"Checkpoint: {CHECKPOINT_DIR}")

    sample_metrics, token_data, moe_layers = run_teacher_forcing(
        model, tokenizer, samples, args.device,
    )

    # Part 1: Predictor accuracy
    print(f"\n{'='*70}")
    print(f"=== Part 1: Predictor Accuracy (teacher-forcing, answer positions) ===")
    print(f"{'='*70}")
    for i, m in enumerate(sample_metrics):
        print(
            f"  Sample {i:2d}: seq={m['seq_len']:3d}, "
            f"recall_n1={m['recall_n1']:.4f}, "
            f"exact_n1={m['exact_n1']:.4f}  Q: {m['question']}..."
        )

    avg_r1 = sum(m["recall_n1"] for m in sample_metrics) / len(sample_metrics)
    avg_e1 = sum(m["exact_n1"] for m in sample_metrics) / len(sample_metrics)
    print(f"\n  Average recall_n1={avg_r1:.4f}")
    print(f"  Average exact_n1={avg_e1:.4f}")
    print(f"  Gap explanation: text-only inference (no image tokens in training data)")

    # Part 2: Cache simulation
    print(f"\n{'='*70}")
    print(f"=== Part 2: Expert Cache Simulation ({len(token_data)} token positions) ===")
    print(f"{'='*70}")
    expert_bytes_q4 = 3 * 896 * 1280 * 0.5  # gate+up+down, Q4_K_M ~0.5 bytes/param
    expert_mb = expert_bytes_q4 / 1024 / 1024
    print(f"Per expert (Q4_K_M): ~{expert_mb:.1f} MB | All {n_experts} experts: ~{n_experts * expert_mb:.0f} MB/layer")

    cache_sizes = [8, 12, 16, 24, 32, 48]
    results = simulate_expert_cache(token_data, moe_layers, n_experts, top_k, cache_sizes)

    print(f"\n{'Cache':>8s} | {'Budget':>8s} | {'Base hit':>9s} | {'Pred hit':>9s} | {'Base cold':>10s} | {'Pred cold':>10s} | {'Reduction':>9s}")
    print("-" * 80)
    for cs in cache_sizes:
        r = results[cs]
        budget = cs * expert_mb
        print(
            f"{cs:>3d}/{n_experts:<3d} | "
            f"{budget:>5.0f} MB | "
            f"{r['baseline_hit_rate']:>8.1%} | "
            f"{r['prefetch_hit_rate']:>8.1%} | "
            f"{r['baseline_cold']:>10d} | "
            f"{r['prefetch_cold']:>10d} | "
            f"{r['reduction']:>8.1%}"
        )

    # Part 3: Latency estimation
    print(f"\n{'='*70}")
    print(f"=== Part 3: Latency Estimation ===")
    print(f"{'='*70}")
    print(f"Assumptions for mobile (Adreno 829):")
    print(f"  Expert load CPU->GPU: ~0.3 ms (DMA transfer ~1.7 MB)")
    print(f"  MoE layer GPU compute (6 experts): ~0.5 ms")
    print(f"  With prefetch, loads overlap with prev layer compute (hidden latency)")

    load_ms = 0.3
    compute_ms = 0.5
    n_moe = len(moe_layers)
    n_tokens = len(token_data)

    print(f"\n{'Cache':>8s} | {'No prefetch':>14s} | {'With prefetch':>14s} | {'Speedup':>8s}")
    print("-" * 55)
    for cs in cache_sizes:
        r = results[cs]
        # Without prefetch: cold loads are synchronous (block on load)
        bl_cold_per_tok = r["baseline_cold"] / max(n_tokens, 1)
        # With prefetch: most loads are overlapped, only mispredictions cause sync load
        pf_cold_per_tok = r["prefetch_cold"] / max(n_tokens, 1)

        base_lat = n_moe * compute_ms + bl_cold_per_tok * load_ms
        # With prefetch: correctly predicted experts are loaded during prev layer compute (free)
        # Only mispredicted experts cause additional sync latency
        pred_lat = n_moe * compute_ms + pf_cold_per_tok * load_ms
        speedup = base_lat / pred_lat if pred_lat > 0 else float("inf")

        print(
            f"{cs:>3d}/{n_experts:<3d} | "
            f"{base_lat:>10.2f} ms | "
            f"{pred_lat:>10.2f} ms | "
            f"{speedup:>7.2f}x"
        )

    # Part 4: Memory savings
    print(f"\n{'='*70}")
    print(f"=== Part 4: Memory Savings Summary ===")
    print(f"{'='*70}")
    full_expert_mb = n_experts * expert_mb * n_moe
    print(f"Full model expert weights (Q4_K_M): {full_expert_mb:.0f} MB ({n_experts} experts x {n_moe} layers)")
    for cs in cache_sizes:
        cached_mb = cs * expert_mb * n_moe
        savings = (full_expert_mb - cached_mb) / full_expert_mb
        r = results[cs]
        print(
            f"  Cache {cs:>2d}/64: {cached_mb:>5.0f} MB GPU "
            f"(save {savings:.0%}), "
            f"predictor hit rate: {r['prefetch_hit_rate']:.1%}"
        )

    print(f"\nDone.")


if __name__ == "__main__":
    main()
