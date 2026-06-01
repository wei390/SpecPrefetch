"""
Compare deepseek-vl2-t model output with data answers to diagnose token_acc.
Two modes:
  1) vLLM generation: compare generated text vs expected answer
  2) Forward pass token_acc: compute exact token-level accuracy (same metric as training)
"""
import json
import random
import os
import torch
import numpy as np
from PIL import Image

MODEL_PATH = "/mnt/data/kjw/deepseek/deepseek-vl2-t"
DATA_PATH = "/mnt/data/kjw/data_qwen/deepseek-vl2-t/DocVQA/intermediate/DocVQA-QwenVL72b-passed.jsonl"
MID_TRAINING_PATH = "/data/aiso-data/processed_data/mid_training"
NUM_SAMPLES = 5
SEED = 42


def load_samples(path, n, seed):
    random.seed(seed)
    with open(path) as f:
        lines = f.readlines()
    indices = random.sample(range(len(lines)), min(n, len(lines)))
    samples = []
    for i in indices:
        d = json.loads(lines[i])
        content = d["messages"][0]["content"]
        q, a = content.split("<seg>", 1)
        img_path = d["images"][0]
        if not os.path.isabs(img_path):
            img_path = os.path.join(MID_TRAINING_PATH, img_path.lstrip("/"))
        samples.append({"question": q.strip(), "answer": a.strip(), "image": img_path, "line": i})
    return samples


def test_vllm_generation(samples):
    from vllm import LLM, SamplingParams

    print("=" * 60)
    print("Loading model with vLLM...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    from vllm import TextPrompt
    from vllm.multimodal import MultiModalDataDict

    results = []
    for s in samples:
        print(f"\n--- Line {s['line']} ---")
        print(f"Q: {s['question']}")
        print(f"Expected A: {s['answer']}")
        print(f"IMG: {s['image']}")

        try:
            img = Image.open(s["image"]).convert("RGB")
        except Exception as e:
            print(f"  [skip] cannot open image: {e}")
            continue

        prompt = f"<|User|><image>\n{s['question']}\n\n<|Assistant|>"

        output = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": img},
            },
            sampling_params=sampling_params,
        )
        generated = output[0].outputs[0].text.strip()
        print(f"Model A: {generated}")
        match = generated == s["answer"]
        print(f"Exact match: {match}")
        results.append({"expected": s["answer"], "generated": generated, "match": match})

    print("\n" + "=" * 60)
    print(f"Exact match: {sum(r['match'] for r in results)}/{len(results)}")
    return results


def test_token_acc_forward(samples):
    """Compute token_acc the same way training does: teacher-forced next-token prediction."""
    from transformers import AutoTokenizer

    print("=" * 60)
    print("Computing token_acc via forward pass (same as training metric)...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    bos_id = tokenizer.bos_token_id or 0
    eos_id = tokenizer.eos_token_id or 1

    user_ids = tokenizer.encode("<|User|>", add_special_tokens=False)
    asst_ids = tokenizer.encode("<|Assistant|>", add_special_tokens=False)
    nl_ids = tokenizer.encode("\n\n", add_special_tokens=False)

    print(f"Tokenizer vocab_size={tokenizer.vocab_size}")
    print(f"BOS={bos_id}, EOS={eos_id}")
    print(f"<|User|> ids={user_ids}, <|Assistant|> ids={asst_ids}")

    for s in samples:
        q_ids = tokenizer.encode(s["question"], add_special_tokens=False)
        a_ids = tokenizer.encode(s["answer"], add_special_tokens=False)
        a_text_roundtrip = tokenizer.decode(a_ids)
        print(f"\n--- Line {s['line']} ---")
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"A tokens ({len(a_ids)}): {a_ids}")
        print(f"A roundtrip: '{a_text_roundtrip}'")

    # Load model for forward pass (without images for now, text-only token_acc)
    print("\nLoading model for forward pass (text-only, no image tokens)...")

    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.modeling_deepseek_vl2 import DeepseekVL2DraftRouterForConditionalGeneration
    from model.configuration_deepseek_vl2 import DeepseekVL2DraftRouterConfig

    config = DeepseekVL2DraftRouterConfig.from_pretrained(MODEL_PATH)
    model = DeepseekVL2DraftRouterForConditionalGeneration(config)

    from model.modeling_deepseek_vl2 import load_pretrained_weights
    load_pretrained_weights(model, MODEL_PATH)
    model = model.to(dtype=torch.bfloat16, device="cuda")
    model.eval()

    total_correct = 0
    total_tokens = 0

    for s in samples:
        q_ids = tokenizer.encode(s["question"], add_special_tokens=False)
        a_ids = tokenizer.encode(s["answer"], add_special_tokens=False)

        # Build input same as training (no image)
        input_ids = [bos_id] + user_ids + q_ids + nl_ids + asst_ids + a_ids + [eos_id]
        prompt_len = len([bos_id] + user_ids + q_ids + nl_ids + asst_ids)
        labels = [-100] * prompt_len + input_ids[prompt_len:]

        input_t = torch.tensor([input_ids], dtype=torch.long, device="cuda")
        labels_t = torch.tensor([labels], dtype=torch.long, device="cuda")

        with torch.no_grad():
            loss, logits = model.language(input_ids=input_t)
            # logits shape: [batch, seq_len, vocab_size]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels_t[..., 1:].contiguous()
        valid_mask = shift_labels != -100
        preds = shift_logits.argmax(dim=-1)
        correct = (preds[valid_mask] == shift_labels[valid_mask]).sum().item()
        total_valid = valid_mask.sum().item()

        acc = correct / max(total_valid, 1)
        total_correct += correct
        total_tokens += total_valid

        pred_ids = preds[0][valid_mask[0]].tolist()
        label_ids = shift_labels[0][valid_mask[0]].tolist()

        print(f"\n--- Line {s['line']} ---")
        print(f"Q: {s['question']}")
        print(f"A: {s['answer']}")
        print(f"Label tokens ({total_valid}): {label_ids}")
        print(f"Pred  tokens ({total_valid}): {pred_ids}")
        print(f"Label text: '{tokenizer.decode(label_ids)}'")
        print(f"Pred  text: '{tokenizer.decode(pred_ids)}'")
        print(f"Token acc: {correct}/{total_valid} = {acc:.4f}")

    overall = total_correct / max(total_tokens, 1)
    print(f"\n{'='*60}")
    print(f"Overall token_acc: {total_correct}/{total_tokens} = {overall:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["vllm", "forward", "both"], default="forward")
    parser.add_argument("--num_samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    samples = load_samples(DATA_PATH, args.num_samples, SEED)
    print(f"Loaded {len(samples)} samples")

    if args.mode in ("vllm", "both"):
        test_vllm_generation(samples)
    if args.mode in ("forward", "both"):
        test_token_acc_forward(samples)
