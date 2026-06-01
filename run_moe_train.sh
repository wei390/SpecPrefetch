#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --------------------------
# 0) Conda / Python
# --------------------------
CONDA_ENV="${CONDA_ENV:-mid_train}"

if [ "${SKIP_CONDA_ACTIVATE:-0}" = "1" ]; then
  echo "SKIP_CONDA_ACTIVATE=1: use current python from PATH"
elif command -v python >/dev/null 2>&1 && python -c "import torch" >/dev/null 2>&1; then
  echo "Current python already has torch, skip conda activate"
else
  echo "Current python has no torch, trying conda activate ${CONDA_ENV}"
  if [ -n "${CONDA_ROOT:-}" ] && [ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source /opt/miniconda3/etc/profile.d/conda.sh
  elif [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
    source /root/miniconda3/etc/profile.d/conda.sh
  elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source /opt/conda/etc/profile.d/conda.sh
  else
    echo "ERROR: cannot find conda.sh, set SKIP_CONDA_ACTIVATE=1 or CONDA_ROOT"
    exit 1
  fi
  conda activate "$CONDA_ENV"
fi

command -v python
python -V
if ! python -c "import torch" >/dev/null 2>&1; then
  echo "ERROR: torch not found in current python: $(command -v python)"
  exit 1
fi
python -c "import torch; print('torch=', torch.__version__, 'cuda_available=', torch.cuda.is_available(), 'cuda=', torch.version.cuda)"

# --------------------------
# 1) Distributed bootstrap
# --------------------------
NNODES="${WORLD_SIZE:-1}"
NODE_RANK="${RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [ -z "${NPROC_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    NPROC_PER_NODE="$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L | wc -l | tr -d ' ')"
  else
    NPROC_PER_NODE=1
  fi
fi

if [ "$NPROC_PER_NODE" = "0" ]; then
  echo "ERROR: NPROC_PER_NODE resolved to 0"
  exit 1
fi

if [ "$NPROC_PER_NODE" = "1" ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES=0
fi

# --------------------------
# 2) Paths / data / output
# --------------------------
MODEL_PATH="${MODEL_PATH:-./deepseek-vl2-t}"
MID_TRAINING_PATH="${MID_TRAINING_PATH:-./mid_training}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output}"
RECIPE_PATH="${RECIPE_PATH:-$SCRIPT_DIR/data_recipe/final_recipe.json}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model path not found: $MODEL_PATH"
  exit 1
fi
if [ ! -d "$MID_TRAINING_PATH" ]; then
  echo "ERROR: mid training path not found: $MID_TRAINING_PATH"
  exit 1
fi
if [ ! -f "$RECIPE_PATH" ]; then
  echo "ERROR: recipe path not found: $RECIPE_PATH"
  exit 1
fi

# --------------------------
# 3) Hyper-parameters (override by env)
# --------------------------
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-12}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
ZERO_STAGE="${ZERO_STAGE:-2}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
SAVE_STRATEGY="${SAVE_STRATEGY:-epoch}"
ATTN_IMPL="${ATTN_IMPL:-auto}"
RUN_NAME="${RUN_NAME:-deepseek_vl2_predictor_sft}"
DDP_TIMEOUT="${DDP_TIMEOUT:-1800}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
MAX_IMAGES_PER_SAMPLE="${MAX_IMAGES_PER_SAMPLE:-1}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-0}"

FUTURE_EXPERT_PREDICTOR_ENABLED="${FUTURE_EXPERT_PREDICTOR_ENABLED:-true}"
FUTURE_EXPERT_PREDICTOR_HIDDEN_SIZE="${FUTURE_EXPERT_PREDICTOR_HIDDEN_SIZE:-1280}"
FUTURE_EXPERT_PREDICTOR_NUM_LAYERS="${FUTURE_EXPERT_PREDICTOR_NUM_LAYERS:-2}"
FUTURE_EXPERT_PREDICTOR_DROPOUT="${FUTURE_EXPERT_PREDICTOR_DROPOUT:-0.1}"
FUTURE_EXPERT_PREDICTOR_USE_LAYER_EMBEDDING="${FUTURE_EXPERT_PREDICTOR_USE_LAYER_EMBEDDING:-true}"
FUTURE_EXPERT_PREDICTOR_ANCHOR_LAYERS="${FUTURE_EXPERT_PREDICTOR_ANCHOR_LAYERS:-}"
FUTURE_EXPERT_PREDICTOR_HORIZONS="${FUTURE_EXPERT_PREDICTOR_HORIZONS:-1}"
FUTURE_EXPERT_COEF_NEXT1="${FUTURE_EXPERT_COEF_NEXT1:-2.0}"

FUTURE_EXPERT_FUSION_MODE="${FUTURE_EXPERT_FUSION_MODE:-none}"
FUTURE_EXPERT_FUSION_INIT_ALPHA="${FUTURE_EXPERT_FUSION_INIT_ALPHA:-0.0}"
FUTURE_EXPERT_RANK="${FUTURE_EXPERT_RANK:-32}"

FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
FREEZE_EXPERTS="${FREEZE_EXPERTS:-true}"
FREEZE_ROUTER_TEACHER="${FREEZE_ROUTER_TEACHER:-true}"
FREEZE_LM_HEAD="${FREEZE_LM_HEAD:-true}"
FREEZE_INPUT_EMBEDDING="${FREEZE_INPUT_EMBEDDING:-true}"

# --------------------------
# 4) DeepSpeed config
# --------------------------
DEEPSPEED_ARG=()
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"
if [ -z "$DEEPSPEED_CONFIG" ]; then
  if [ "$ZERO_STAGE" = "3" ]; then
    if [ -f "./deepspeed_bf16_zero_3.json" ]; then
      DEEPSPEED_CONFIG="./deepspeed_bf16_zero_3.json"
    elif [ -f "$SCRIPT_DIR/deepspeed_bf16_zero3_stable.json" ]; then
      DEEPSPEED_CONFIG="$SCRIPT_DIR/deepspeed_bf16_zero3_stable.json"
    fi
  else
    if [ -f "./deepspeed_bf16_zero_2.json" ]; then
      DEEPSPEED_CONFIG="./deepspeed_bf16_zero_2.json"
    elif [ -f "$SCRIPT_DIR/deepspeed_bf16_zero2_stable.json" ]; then
      DEEPSPEED_CONFIG="$SCRIPT_DIR/deepspeed_bf16_zero2_stable.json"
    elif [ -f "$SCRIPT_DIR/deepspeed_bf16_zero2_fast.json" ]; then
      DEEPSPEED_CONFIG="$SCRIPT_DIR/deepspeed_bf16_zero2_fast.json"
    fi
  fi
fi
if [ -n "$DEEPSPEED_CONFIG" ]; then
  case "${DEEPSPEED_CONFIG,,}" in
    none|off|0) ;;
    *)
      if [ ! -f "$DEEPSPEED_CONFIG" ]; then
        echo "ERROR: deepspeed config not found: $DEEPSPEED_CONFIG"
        exit 1
      fi
      DEEPSPEED_ARG=(--deepspeed "$DEEPSPEED_CONFIG")
      ;;
  esac
fi

# --------------------------
# 5) Optional flags
# --------------------------
NO_WANDB_ARG=()
if [ "${NO_WANDB:-0}" = "1" ]; then
  NO_WANDB_ARG=(--no_wandb)
  export WANDB_MODE="${WANDB_MODE:-offline}"
fi

PIN_MEMORY_ARG=()
if [ "$DATALOADER_PIN_MEMORY" = "1" ]; then
  PIN_MEMORY_ARG=(--dataloader_pin_memory)
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ZERO_STAGE="$ZERO_STAGE"

if [ "$ATTN_IMPL" = "auto" ]; then
  if python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
  then
    ATTN_IMPL="flash_attn"
  else
    ATTN_IMPL="sdpa"
  fi
  echo "[INFO] ATTN_IMPL=auto resolved to: $ATTN_IMPL"
elif [ "$ATTN_IMPL" = "flash_attn" ]; then
  if ! python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
  then
    echo "[WARN] flash_attn import failed, fallback to sdpa"
    ATTN_IMPL="sdpa"
  fi
fi

echo "============= DeepSeek-VL2 Gate Predictor Training ==================="
echo "MODEL_PATH=$MODEL_PATH"
echo "MID_TRAINING_PATH=$MID_TRAINING_PATH"
echo "RECIPE_PATH=$RECIPE_PATH"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "NPROC_PER_NODE=$NPROC_PER_NODE NNODES=$NNODES NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "ATTN_IMPL=$ATTN_IMPL DEEPSPEED=${DEEPSPEED_CONFIG:-disabled}"
echo "ZERO_STAGE=$ZERO_STAGE MAX_LENGTH=$MAX_LENGTH GRADIENT_CHECKPOINTING=$GRADIENT_CHECKPOINTING"
echo "PREDICTOR anchors='${FUTURE_EXPERT_PREDICTOR_ANCHOR_LAYERS}' horizons='${FUTURE_EXPERT_PREDICTOR_HORIZONS}'"
echo "PREDICTOR KL_COEF next1=$FUTURE_EXPERT_COEF_NEXT1"
echo "FUSION mode=$FUTURE_EXPERT_FUSION_MODE rank=$FUTURE_EXPERT_RANK"
echo "========================================================================"

if [ "${MOE_DRY_RUN:-0}" = "1" ]; then
  echo "MOE_DRY_RUN=1, skip training launch"
  exit 0
fi

CMD=(
  python -m torch.distributed.run
  --nnodes "$NNODES"
  --node_rank "$NODE_RANK"
  --master_addr "$MASTER_ADDR"
  --master_port "$MASTER_PORT"
  --max_restarts 0
  --nproc_per_node "$NPROC_PER_NODE"
  train_moe_deepseek_vl2_future_expert.py
  --model_path "$MODEL_PATH"
  --mid_training_path "$MID_TRAINING_PATH"
  --output_dir "$OUTPUT_DIR"
  --recipe_path "$RECIPE_PATH"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --learning_rate "$LEARNING_RATE"
  --max_grad_norm "$MAX_GRAD_NORM"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --max_length "$MAX_LENGTH"
  --logging_steps "$LOGGING_STEPS"
  --save_strategy "$SAVE_STRATEGY"
  --attn_impl "$ATTN_IMPL"
  --gradient_checkpointing "$GRADIENT_CHECKPOINTING"
  --run_name "$RUN_NAME"
  --ddp_timeout "$DDP_TIMEOUT"
  --max_images_per_sample "$MAX_IMAGES_PER_SAMPLE"
  --dataset_num_proc "$DATASET_NUM_PROC"
  --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
  --future_expert_predictor_enabled "$FUTURE_EXPERT_PREDICTOR_ENABLED"
  --future_expert_predictor_hidden_size "$FUTURE_EXPERT_PREDICTOR_HIDDEN_SIZE"
  --future_expert_predictor_num_layers "$FUTURE_EXPERT_PREDICTOR_NUM_LAYERS"
  --future_expert_predictor_dropout "$FUTURE_EXPERT_PREDICTOR_DROPOUT"
  --future_expert_predictor_use_layer_embedding "$FUTURE_EXPERT_PREDICTOR_USE_LAYER_EMBEDDING"
  --future_expert_predictor_anchor_layers "$FUTURE_EXPERT_PREDICTOR_ANCHOR_LAYERS"
  --future_expert_predictor_horizons "$FUTURE_EXPERT_PREDICTOR_HORIZONS"
  --future_expert_predictor_kl_coef_next1 "$FUTURE_EXPERT_COEF_NEXT1"
  --future_expert_fusion_mode "$FUTURE_EXPERT_FUSION_MODE"
  --future_expert_fusion_init_alpha "$FUTURE_EXPERT_FUSION_INIT_ALPHA"
  --future_expert_lora_rank "$FUTURE_EXPERT_RANK"
  --freeze_backbone "$FREEZE_BACKBONE"
  --freeze_experts "$FREEZE_EXPERTS"
  --freeze_router_teacher "$FREEZE_ROUTER_TEACHER"
  --freeze_lm_head "$FREEZE_LM_HEAD"
  --freeze_input_embedding "$FREEZE_INPUT_EMBEDDING"
)

CMD+=("${DEEPSPEED_ARG[@]}")
CMD+=("${PIN_MEMORY_ARG[@]}")
CMD+=("${NO_WANDB_ARG[@]}")

printf 'Launch command:\n%s\n' "${CMD[*]}"
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/train_rank_${NODE_RANK}.log"
