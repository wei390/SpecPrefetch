"""
Fast evaluation: run all 5 benchmarks with a small per-dataset sample cap and
compare the resulting Recall@{3,6,8} / Exact@6 numbers against a baseline JSON
(default: ./eval_results_lora.json) for sanity-checking.

The whole thing reuses model loading, sample loading, and metric computation
from `eval_expert_prediction_batch.py` — only the dataset selection, sample
cap, and baseline-comparison reporting are new here.

Why this is "fast":
  Each sample contributes (decode_tokens × anchor_layers × horizons × top_ks)
  per-token recall measurements (~5000 per sample), so averages stabilize
  quickly. ~50 samples per dataset is usually enough to hit ±0.01 of the
  full-dataset recall numbers.

Usage:
    CUDA_VISIBLE_DEVICES=0 python eval/fast_eval.py \\
        --checkpoint /path/to/lora/checkpoint \\
        --fusion_mode lora \\
        --num_samples 50

Exit code: 0 if every metric is within --tolerance of the baseline, 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

# Allow importing the existing batch-eval helpers from the same directory,
# and the model package from the project root.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from eval_expert_prediction_batch import (
    BASE_MODEL_PATH,
    evaluate_dataset,
    load_model,
    load_tokenizer,
    load_tsv,
)

DATASETS_DIR = os.path.join(_THIS_DIR, "datasets")

# (display_name, tsv_filename) — order kept consistent with eval_results_lora.json
DATASETS = [
    ("ChartQA_TEST", "ChartQA_TEST.tsv"),
    ("OCRBench", "OCRBench.tsv"),
    ("HallusionBench", "HallusionBench.tsv"),
    ("GSM8K", "GSM8K.tsv"),
    ("openai_humaneval", "openai_humaneval.tsv"),
]

DEFAULT_BASELINE = os.path.join(_THIS_DIR, "eval_results.json")

METRIC_KEYS = ["recall@3", "recall@6", "recall@8", "exact@6"]
HORIZON_KEYS = ["next1"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast eval over all benchmarks with sample cap + baseline diff.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoint directory.")
    parser.add_argument("--fusion_mode", type=str, default="lora",
                        choices=["none", "alpha", "correction", "lora", "reinit_gate"])
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH)
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Sample cap per dataset (0 = full).")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--only", type=str, nargs="*", default=None,
                        help="Restrict to these dataset names (default: all).")
    parser.add_argument("--baseline", type=str, default=DEFAULT_BASELINE,
                        help="Reference JSON to compare against. Pass empty string to skip.")
    parser.add_argument("--tolerance", type=float, default=0.05,
                        help="Max allowed |current - baseline| for any metric to be considered OK.")
    parser.add_argument("--output_json", type=str, default="",
                        help="Where to dump current results. Defaults to fast_eval_results_<mode>.json")
    return parser.parse_args()


def load_baseline(path: str):
    if not path:
        return None
    if not os.path.exists(path):
        print(f"[INFO] baseline not found at {path}; skipping comparison.")
        return None
    with open(path) as f:
        return json.load(f)


def diff_against_baseline(curr_results: dict, baseline: dict, tolerance: float):
    """Return (rows, all_ok). rows is a list of dicts ready for printing."""
    base_results = baseline.get("results", {})
    rows = []
    all_ok = True
    for ds_name, ds_curr in curr_results.items():
        ds_base = base_results.get(ds_name)
        if ds_base is None:
            rows.append({"ds": ds_name, "horizon": "-", "metric": "-",
                         "curr": None, "base": None, "diff": None, "ok": False,
                         "note": "no baseline"})
            all_ok = False
            continue
        for horizon in HORIZON_KEYS:
            for metric in METRIC_KEYS:
                try:
                    curr = float(ds_curr[horizon][metric])
                    base = float(ds_base[horizon][metric])
                except (KeyError, TypeError):
                    rows.append({"ds": ds_name, "horizon": horizon, "metric": metric,
                                 "curr": None, "base": None, "diff": None, "ok": False,
                                 "note": "missing"})
                    all_ok = False
                    continue
                diff = curr - base
                ok = abs(diff) <= tolerance
                if not ok:
                    all_ok = False
                rows.append({"ds": ds_name, "horizon": horizon, "metric": metric,
                             "curr": curr, "base": base, "diff": diff, "ok": ok,
                             "note": ""})
    return rows, all_ok


def print_comparison(rows, tolerance: float):
    print("\n" + "=" * 96)
    print(f"  Comparison vs baseline   (tolerance ±{tolerance:.3f})")
    print("=" * 96)
    print(f"  {'Dataset':<18} {'Horizon':<7} {'Metric':<10} "
          f"{'Current':>10} {'Baseline':>10} {'Diff':>10}  Status")
    print("-" * 96)
    for r in rows:
        if r["curr"] is None:
            print(f"  {r['ds']:<18} {r['horizon']:<7} {r['metric']:<10} "
                  f"{'-':>10} {'-':>10} {'-':>10}  {'!! ' + r['note']}")
            continue
        status = "OK" if r["ok"] else "!!"
        print(f"  {r['ds']:<18} {r['horizon']:<7} {r['metric']:<10} "
              f"{r['curr']:>10.4f} {r['base']:>10.4f} {r['diff']:>+10.4f}  {status}")
    print("=" * 96)


def select_datasets(only_filter):
    if not only_filter:
        return DATASETS
    sel = []
    missing = []
    for name in only_filter:
        for ds_name, fname in DATASETS:
            if name == ds_name:
                sel.append((ds_name, fname))
                break
        else:
            missing.append(name)
    if missing:
        print(f"[WARN] unknown datasets in --only: {missing}; "
              f"valid names are {[n for n, _ in DATASETS]}")
    return sel


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    print(f"\n>>> Fast eval — fusion_mode={args.fusion_mode}, "
          f"num_samples={args.num_samples} per dataset")
    print(f"    checkpoint = {args.checkpoint}")
    if args.baseline:
        print(f"    baseline   = {args.baseline}")

    selected = select_datasets(args.only)
    if not selected:
        print("[ERROR] no datasets selected.")
        sys.exit(2)

    # Verify all datasets exist before paying the model-load cost.
    missing_paths = []
    for ds_name, fname in selected:
        path = os.path.join(DATASETS_DIR, fname)
        if not os.path.exists(path):
            missing_paths.append(path)
    if missing_paths:
        print("[ERROR] missing dataset files:")
        for p in missing_paths:
            print(f"  - {p}")
        sys.exit(2)

    model, config = load_model(args)
    tokenizer = load_tokenizer(args.base_model)

    all_results = {}
    overall_t0 = time.time()
    for ds_name, fname in selected:
        ds_path = os.path.join(DATASETS_DIR, fname)
        print(f"\n{'=' * 60}\n  Dataset: {ds_name}\n{'=' * 60}")

        samples = load_tsv(ds_path, args.num_samples)
        print(f"  loaded {len(samples)} samples from {ds_path}")

        t0 = time.time()
        agg = evaluate_dataset(model, tokenizer, samples, config, args)
        elapsed = time.time() - t0

        agg["time_s"] = round(elapsed, 1)
        agg["time_per_sample"] = round(elapsed / max(agg["n_samples"], 1), 2)
        all_results[ds_name] = agg

        print(f"\n  {ds_name}: {agg['n_samples']} samples, "
              f"{agg['total_decode_tokens']} decode tokens, {elapsed:.1f}s")
        for h in HORIZON_KEYS:
            d = agg[h]
            print(f"    {h}: recall@3={d['recall@3']:.4f}  "
                  f"recall@6={d['recall@6']:.4f}  recall@8={d['recall@8']:.4f}  "
                  f"exact@6={d['exact@6']:.4f}")

    overall_elapsed = time.time() - overall_t0

    out_json = args.output_json or os.path.join(
        _THIS_DIR, f"fast_eval_results_{args.fusion_mode}.json"
    )
    payload = {
        "fusion_mode": args.fusion_mode,
        "checkpoint": args.checkpoint,
        "num_samples_per_dataset": args.num_samples,
        "results": all_results,
        "total_time_s": round(overall_elapsed, 1),
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults written to {out_json}  (total {overall_elapsed:.1f}s)")

    baseline = load_baseline(args.baseline)
    if baseline is None:
        print("\n[INFO] no baseline → skipping comparison.")
        return

    rows, all_ok = diff_against_baseline(all_results, baseline, args.tolerance)
    print_comparison(rows, args.tolerance)

    if all_ok:
        print(f"\nOK  every metric within ±{args.tolerance:.3f} of baseline.")
        sys.exit(0)
    else:
        bad = sum(1 for r in rows if r["curr"] is None or not r["ok"])
        print(f"\n!!  {bad} metric(s) deviate by more than ±{args.tolerance:.3f} "
              f"(or are missing). Increase --tolerance or --num_samples to triage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
