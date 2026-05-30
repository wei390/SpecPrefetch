# SpeMoe — DeepSeek-VL2 Future-Expert Predictor

A lightweight predictor head trained on top of a frozen **DeepSeek-VL2-Tiny** MoE
model. The predictor consumes the hidden state at an anchor layer `l` and
predicts the **Top-K expert set** that the *teacher router* will select two
layers later (`l+1` and `l+2`). The predictions are used at inference time to
**prefetch experts** before the router gets there, hiding cold-load latency on
GPUs with limited expert cache.

The training objective is BCE + ranking loss against the teacher's TopK.
The evaluation metric is **Recall@{3,6,8}** and **Exact-match@6** (against the
teacher's TopK=6) on standard multimodal benchmarks during autoregressive
decoding.

---

## Repository layout

```
SpeMoe/
├── model/                                  # Model + config (frozen teacher + predictor head)
│   ├── configuration_deepseek_vl2.py
│   ├── modeling_deepseek_vl2.py
│   └── __init__.py
├── train_moe_deepseek_vl2_future_expert.py # Training entry (HF Trainer)
├── callbacks_deepseek_vl2_future_expert.py # Predictor metrics callback
├── run_moe_train.sh                        # Distributed training launcher
├── deepspeed_bf16_zero{2_fast,2_stable,3_stable}.json
└── eval/
    ├── fast_eval.py                        # Sampled multi-bmk eval + baseline diff
    ├── eval_expert_prediction_batch.py     # Full multi-bmk Recall eval
    ├── eval_expert_prediction_ocrbench.py  # Full OCRBench-only Recall eval
    ├── verify_expert_prefetch.py           # LRU-cache simulation for prefetch
    ├── test_token_acc_compare.py           # Token-level accuracy compare
    ├── eval_results.json                   # res
    └── datasets/                           # NOT shipped — download separately (see "Datasets")
        ├── ChartQA_TEST.tsv
        ├── GSM8K.tsv
        ├── HallusionBench.tsv
        ├── OCRBench.tsv
        └── openai_humaneval.tsv
```

## Datasets

The 5 evaluation `.tsv` files are **not** included in this repo (the two
largest exceed GitHub's 100 MB per-file limit). Download them from the
standard [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) sources and
place them under `eval/datasets/` with the exact filenames shown above before
running any eval script. The data recipe + training JSONLs are likewise not
shipped (see Training §2).

---

## Environment

```bash
# A working conda env with `torch`, `transformers>=5.x`, `datasets`,
# `Pillow`, `tqdm`, `numpy`, `deepspeed`, optionally `flash_attn`.
# The launcher will activate `mid_train` by default; override with CONDA_ENV.
conda activate mid_train
```

Required external paths:

| Variable          | Default                                              | Meaning                                  |
| ----------------- | ---------------------------------------------------- | ---------------------------------------- |
| `MODEL_PATH`      | `./deepseek-vl2-t`                                   | Pretrained DeepSeek-VL2-Tiny weights.    |
| `MID_TRAINING_PATH` | `./mid_training`                                   | Root dir for relative image paths.       |
| `RECIPE_PATH`     | `$SCRIPT_DIR/data_recipe/final_recipe.json`          | Sampling recipe (see below).             |
| `OUTPUT_DIR`      | `$SCRIPT_DIR/output`                                 | Checkpoints + tokenizer.                 |

---

## Training

### 1. Training-data format

The dataset is a list of **JSONL files** referenced by a recipe file. Each
line is one training sample. The schema is intentionally minimal:

```json
{"messages": [{"role": "assistant", "content": "what is the date mentioned ?<seg>14 AUG 1980"}], "images": ["/data/.../14167_p1.jpg"]}
```

Two fields:

| Field      | Type              | Description                                                                                                       |
| ---------- | ----------------- | ----------------------------------------------------------------------------------------------------------------- |
| `messages` | `list[{role, content}]` | Only `messages[0].content` is read. The string is split on `<seg>` into `(question, answer)`. Answer is the loss target; question + image are the prompt. |
| `images`   | `list[str]`       | 0 or more image paths. Absolute paths are used as-is; relative paths are joined to `$MID_TRAINING_PATH`. Capped at `MAX_IMAGES_PER_SAMPLE` (default 1). |

The collator builds the prompt as

```
<bos> <|User|> [image tokens]* <question> \n\n <|Assistant|> <answer> <eos>
```

Loss is computed only on `<answer>` and `<eos>` tokens (everything before
`<|Assistant|>` is masked with `-100`).

**Pure-text sample** (no image): set `"images": []` and write the same
`Q<seg>A` content. The image-token block is omitted automatically.

### 2. Data recipe

`run_moe_train.sh` expects a recipe at `$SCRIPT_DIR/data_recipe/final_recipe.json`,
which is a `{jsonl_path: sampling_ratio}` map:

```json
{
  "/.../DocVQA-QwenVL72b-passed.jsonl": 1,
  "/.../FigureQA-qwen3.5-passed.jsonl": 0.5,
  "/.../OCR-VQA-qwen3.5-passed.jsonl": 1
}
```

`ratio` is the per-epoch fraction of that JSONL to draw (1 = use it all,
0.5 = sample half, etc.). Datasets are concatenated then resampled per epoch
via `EpochResampleDataset`.

> The recipe is **not** included in this repo — drop your own
> `data_recipe/final_recipe.json` (and the JSONL files it points to) into
> place before launching, or set `RECIPE_PATH=...` to point elsewhere.

### 3. Launch training

Single-node:

```bash
# Defaults: fusion_mode=none (predictor only, no fusion path).
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_moe_train.sh
```

With env-var overrides (any fusion mode, custom run name, custom output dir):

```bash
FUTURE_EXPERT_FUSION_MODE=<mode> \
FUTURE_EXPERT_RANK=<rank> \
RUN_NAME=<run_name> \
OUTPUT_DIR=$PWD/<output_dir> \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_moe_train.sh
```

Multi-node (torchrun-style; the launcher reads `WORLD_SIZE`, `RANK`,
`MASTER_ADDR`, `MASTER_PORT` from the env):

```bash
WORLD_SIZE=2 RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_moe_train.sh   # node 0
WORLD_SIZE=2 RANK=1 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash run_moe_train.sh   # node 1
```

Set `MOE_DRY_RUN=1` to print the resolved command without launching.

### 4. Notable env-var knobs

| Variable                                  | Default   | Purpose                                                                                  |
| ----------------------------------------- | --------- | ---------------------------------------------------------------------------------------- |
| `FUTURE_EXPERT_FUSION_MODE`               | `none`    | Controls how predictor probs are fused into the routing path; see `python train_moe_deepseek_vl2_future_expert.py --help` for the full list of supported values. |
| `FUTURE_EXPERT_RANK`                      | `32`      | Predictor adapter rank, used by fusion modes that involve a low-rank adapter. Ignored otherwise. |
| `FUTURE_EXPERT_PREDICTOR_HORIZONS`        | `1,2`     | Which future layers to predict (`l+1`, `l+2`).                                           |
| `FUTURE_EXPERT_COEF_NEXT1` / `_NEXT2`     | `2.0/1.0` | KL-loss weights for next-1 / next-2 horizons.                                            |
| `FREEZE_BACKBONE` / `_EXPERTS` / `_ROUTER_TEACHER` / `_LM_HEAD` / `_INPUT_EMBEDDING` | `true`    | Frozen by default — only the predictor (and any fusion-mode adapter) trains. |
| `ZERO_STAGE`                              | `2`       | DeepSpeed ZeRO stage. `3` selects the zero3_stable config automatically.                 |
| `PER_DEVICE_TRAIN_BATCH_SIZE`             | `12`      | Micro-batch per GPU.                                                                     |
| `GRADIENT_ACCUMULATION_STEPS`             | `8`       | Accumulation; effective batch = bs × accum × world_size.                                  |
| `MAX_LENGTH`                              | `4096`    | Max prompt+answer tokens; longer samples are right-truncated.                            |
| `LEARNING_RATE`                           | `1e-4`    | AdamW lr.                                                                                |
| `NUM_TRAIN_EPOCHS`                        | `1`       | Epochs over the resampled set.                                                           |
| `ATTN_IMPL`                               | `auto`    | `auto / flash_attn / sdpa`.                                                              |
| `NO_WANDB`                                | `0`       | `1` to disable W&B logging.                                                              |

Everything else is in the `# 3) Hyper-parameters` block of `run_moe_train.sh`
and is overridable via env vars with the same name.

### 5. Outputs

- `OUTPUT_DIR/checkpoint-N/` — HF-style checkpoint directory (loadable via
  `load_pretrained_weights(model, checkpoint_dir)`).
- `OUTPUT_DIR/future_expert_predictor_metrics.jsonl` — per-step predictor
  recall + exact match (written by `FutureExpertPredictorMetricsCallback`).
- `LOG_DIR/train_rank_${NODE_RANK}.log` — per-rank stdout/stderr.

---

## Evaluation

### Fast sanity check (recommended for verifying a checkpoint)

`eval/fast_eval.py` runs all 5 benchmarks with a small per-dataset sample
cap and diffs the result against `eval/eval_results.json`. The
predictor's recall is averaged over `decode_tokens × anchor_layers ×
horizons × top_ks` measurements per sample, so 50 samples per dataset
typically already lands within ±0.01 of the full-set numbers.

```bash
CUDA_VISIBLE_DEVICES=0 python eval/fast_eval.py \
    --checkpoint /path/to/checkpoint-N \
    --fusion_mode <mode> \
    --num_samples 50
```

What you get:

```
============================================================
  Dataset: openai_humaneval
============================================================
  loaded 50 samples from .../eval/datasets/openai_humaneval.tsv
Evaluating: 100%|██████████| 50/50 [04:12<00:00,  ...]
  openai_humaneval: 50 samples, 1734 decode tokens, 252.4s
    next1: recall@3=0.4760  recall@6=0.8147  recall@8=0.9061  exact@6=0.2649
    next2: recall@3=0.4499  recall@6=0.7558  recall@8=0.8504  exact@6=0.1572
...
================================================================================================
  Comparison vs baseline   (tolerance ±0.050)
================================================================================================
  Dataset            Horizon Metric        Current   Baseline       Diff  Status
------------------------------------------------------------------------------------------------
  openai_humaneval   next1   recall@3       0.4760     0.4761    -0.0001  OK
  openai_humaneval   next1   recall@6       0.8147     0.8151    -0.0004  OK
  ...
================================================================================================

OK  every metric within ±0.050 of baseline.
```

Useful flags:

| Flag            | Default                       | Notes                                                          |
| --------------- | ----------------------------- | -------------------------------------------------------------- |
| `--checkpoint`  | required                      | Path to `checkpoint-N/` produced by training.                  |
| `--fusion_mode` | (required match) | Must match how the checkpoint was trained.                     |
| `--num_samples` | `50`                          | Per-dataset cap. `0` = full set.                               |
| `--only`        | all 5                         | E.g. `--only OCRBench GSM8K` to restrict to two benchmarks.    |
| `--baseline`    | `eval/eval_results.json`      | JSON to diff against. Pass `""` to skip.                       |
| `--tolerance`   | `0.05`                        | Max `|current − baseline|` allowed per metric.                 |
| `--output_json` | `eval/fast_eval_results_<mode>.json` | Where to write the current run's results.               |

Exit codes: `0` = all metrics within tolerance, `1` = at least one deviates,
`2` = bad args / missing dataset files. Easy to wire into CI.

Approximate runtime on a single A800: ~12 min for `--num_samples 50` over
all 5 benchmarks (incl. ~30–60s model load).

### Full evaluation (~4 hours on A800)

For full-set numbers, drop `--num_samples` (or set to `0`):

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_expert_prediction_batch.py \
    --checkpoint /path/to/checkpoint-N \
    --fusion_mode <mode> \
    --datasets eval/datasets/ChartQA_TEST.tsv \
               eval/datasets/OCRBench.tsv \
               eval/datasets/HallusionBench.tsv \
               eval/datasets/GSM8K.tsv \
               eval/datasets/openai_humaneval.tsv \
    --max_new_tokens 200 \
    --output_json eval/eval_results_repro.json
```

OCRBench-only with the per-OCRBench script:

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_expert_prediction_ocrbench.py \
    --checkpoint /path/to/checkpoint-N \
    --fusion_mode <mode> \
    --max_new_tokens 100
```

### Reference numbers (full-set)

`eval/eval_results.json` is the baseline `fast_eval.py` compares against.
Quoting the headline numbers (n_samples / `next2 recall@3`):

| Dataset           | n     | next1 r@3 | next1 r@6 | next1 r@8 | next2 r@3 | next2 r@6 | next2 r@8 |
| ----------------- | ----- | --------- | --------- | --------- | --------- | --------- | --------- |
| ChartQA_TEST      | 2500  | 0.4963    | 0.8961    | 0.9683    | 0.4880    | 0.8447    | 0.9296    |
| OCRBench          | 1000  | 0.4936    | 0.8929    | 0.9645    | 0.4838    | 0.8433    | 0.9275    |
| HallusionBench    | 1129  | 0.4925    | 0.8872    | 0.9596    | 0.4824    | 0.8381    | 0.9220    |
| GSM8K             | 1319  | 0.4836    | 0.8435    | 0.9247    | 0.4632    | 0.7815    | 0.8718    |
| openai_humaneval  | 164   | 0.4761    | 0.8151    | 0.9065    | 0.4500    | 0.7562    | 0.8508    |

A reproduction run should land within ±0.02 of these
numbers on every cell when `fast_eval.py --num_samples 50` is run on the
trained checkpoint.

### Other eval scripts

- `eval/verify_expert_prefetch.py` — runs an LRU expert-cache simulator at
  several memory budgets to estimate cold-load reduction from the predictor.
- `eval/test_token_acc_compare.py` — token-level greedy decode equality check
  against the teacher.

---

## Tips & gotchas

- **`fusion_mode` must match the checkpoint** at eval time. Loading a
  checkpoint trained with one fusion mode under a different `--fusion_mode`
  will silently skip the unmatched fusion path — recall will look
  catastrophically wrong.
- **`max_new_tokens` matters** for the eval scripts — recall is averaged
  over decode-phase tokens only, so very small `max_new_tokens` gives a
  noisier estimate. The defaults (100 for OCRBench, 200 for batch) are tuned
  to match the baseline run.
- **No `data_recipe/` or `eval/datasets/` shipped** — provide your own data
  recipe + JSONLs before training, and download the 5 eval `.tsv` files
  before running any eval script.
