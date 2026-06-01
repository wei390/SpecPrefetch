from __future__ import annotations

import json
import os
from typing import Any

from transformers import TrainerCallback


class FutureExpertPredictorMetricsCallback(TrainerCallback):
    """Log Future-2-Layer TopK expert recall predictor metrics."""

    def __init__(self, args, trainer):
        super().__init__()
        self.trainer = trainer
        self._wandb = None
        self._warned_failure = False
        self._jsonl_path = os.path.join(str(getattr(args, "output_dir", ".")), "future_expert_predictor_metrics.jsonl")

    @staticmethod
    def _is_main_process(state: Any) -> bool:
        return bool(getattr(state, "is_world_process_zero", False))

    @staticmethod
    def _safe_unwrap_model(model: Any) -> Any:
        if model is None:
            return None
        try:
            from transformers.modeling_utils import unwrap_model
            model = unwrap_model(model)
        except Exception:
            pass
        seen = set()
        while hasattr(model, "module") and getattr(model, "module") is not None:
            next_model = getattr(model, "module")
            if next_model is model or id(next_model) in seen:
                break
            seen.add(id(model))
            model = next_model
        return model

    def _get_model(self, model: Any, kwargs: dict[str, Any]) -> Any:
        candidate = model
        if candidate is None:
            candidate = kwargs.get("model")
        if candidate is None and self.trainer is not None:
            candidate = getattr(self.trainer, "model", None)
        return self._safe_unwrap_model(candidate)

    def _warn_once(self, msg: str) -> None:
        if self._warned_failure:
            return
        self._warned_failure = True
        print(f"[future-expert-metrics][warn] {msg}", flush=True)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    @staticmethod
    def _clamp_01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _append_jsonl(self, payload: dict[str, Any]) -> None:
        os.makedirs(os.path.dirname(self._jsonl_path), exist_ok=True)
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _sync_runtime_context(self, real_model: Any, args: Any, state: Any) -> None:
        if real_model is None:
            return
        set_out_dir = getattr(real_model, "set_future_expert_predictor_output_dir", None)
        if callable(set_out_dir):
            try:
                set_out_dir(str(getattr(args, "output_dir", "")))
            except Exception:
                pass
        set_step = getattr(real_model, "set_future_expert_predictor_global_step", None)
        if callable(set_step):
            try:
                set_step(int(getattr(state, "global_step", 0)))
            except Exception:
                pass

    def on_train_begin(self, args, state, control, **kwargs):
        real_model = self._get_model(kwargs.get("model"), kwargs)
        self._sync_runtime_context(real_model, args, state)

        if not self._is_main_process(state):
            return control

        try:
            import wandb
        except Exception:
            self._wandb = None
            return control

        self._wandb = wandb
        if getattr(wandb, "run", None) is None:
            return control

        try:
            wandb.define_metric("train/global_step")
            wandb.define_metric("train/valid_lm_tokens", step_metric="train/global_step")
            wandb.define_metric("train/loss_lm", step_metric="train/global_step")
            wandb.define_metric("train/loss_total", step_metric="train/global_step")
            wandb.define_metric("train/loss_future_expert", step_metric="train/global_step", summary="min")
            wandb.define_metric("train/loss_future_expert_next1", step_metric="train/global_step", summary="min")
            wandb.define_metric("train/loss_next1_kl", step_metric="train/global_step", summary="min")
            wandb.define_metric("train/recall_next1", step_metric="train/global_step", summary="max")
            wandb.define_metric("train/exact_next1", step_metric="train/global_step", summary="max")
            wandb.define_metric("train/token_acc", step_metric="train/global_step", summary="max")
            wandb.define_metric("train/fusion_alpha_h1_mean", step_metric="train/global_step")
        except Exception as e:
            self._warn_once(f"wandb.define_metric failed: {e}")

        return control

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        real_model = self._get_model(model, kwargs)
        self._sync_runtime_context(real_model, args, state)
        return control

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return control

        try:
            real_model = self._get_model(model, kwargs)
            self._sync_runtime_context(real_model, args, state)

            consume_fn = getattr(real_model, "consume_future_expert_predictor_log_accum", None)
            if not callable(consume_fn):
                consume_fn = getattr(real_model, "consume_draft_router_log_accum", None)
            if not callable(consume_fn):
                return control

            accum = consume_fn()
            if not isinstance(accum, dict):
                return control

            n = int(accum.get("n", 0))
            if n <= 0:
                return control

            metrics = {
                "train/global_step": int(getattr(state, "global_step", 0)),
                "train/valid_lm_tokens": self._safe_float(accum.get("valid_lm_tokens", 0.0)) / n,
                "train/loss_lm": self._safe_float(accum.get("loss_lm", 0.0)) / n,
                "train/loss_total": self._safe_float(accum.get("loss_total", 0.0)) / n,
                "train/loss_future_expert": self._safe_float(accum.get("loss_future_expert", 0.0)) / n,
                "train/loss_future_expert_next1": self._safe_float(accum.get("loss_future_expert_next1", 0.0)) / n,
                "train/loss_next1_kl": self._safe_float(accum.get("loss_next1_kl", 0.0)) / n,
                "train/recall_next1": self._clamp_01(self._safe_float(accum.get("recall_next1", 0.0)) / n),
                "train/exact_next1": self._clamp_01(self._safe_float(accum.get("exact_next1", 0.0)) / n),
                "train/token_acc": self._clamp_01(self._safe_float(accum.get("token_acc", 0.0)) / n),
                "train/fusion_alpha_h1_mean": self._safe_float(accum.get("fusion_alpha_h1_mean", 0.0)) / n,
            }

            logs.update(metrics)

            if not self._is_main_process(state):
                return control

            print(
                "[future-expert-metrics] "
                f"step={metrics['train/global_step']} "
                f"valid_lm_tokens={metrics['train/valid_lm_tokens']:.1f} "
                f"loss_lm={metrics['train/loss_lm']:.6f} "
                f"loss_total={metrics['train/loss_total']:.6f} "
                f"loss_future={metrics['train/loss_future_expert']:.6f} "
                f"token_acc={metrics['train/token_acc']:.4f} "
                f"next1={metrics['train/loss_future_expert_next1']:.6f}/"
                f"r{metrics['train/recall_next1']:.6f}/e{metrics['train/exact_next1']:.6f} "
                f"next1_kl={metrics['train/loss_next1_kl']:.6f} "
                f"alpha_h1={metrics['train/fusion_alpha_h1_mean']:.4f}",
                flush=True,
            )

            payload = {
                "global_step": int(getattr(state, "global_step", 0)),
                "epoch": self._safe_float(getattr(state, "epoch", 0.0)),
                **metrics,
            }
            self._append_jsonl(payload)

            wb = self._wandb
            if wb is None:
                try:
                    import wandb as wb
                except Exception:
                    wb = None
                self._wandb = wb
            if wb is not None and getattr(wb, "run", None) is not None:
                wb.log(metrics, step=int(getattr(state, "global_step", 0)))

        except Exception as e:
            self._warn_once(f"on_log failed but training continues: {e}")

        return control


DraftRouterMetricsCallback = FutureExpertPredictorMetricsCallback


__all__ = [
    "FutureExpertPredictorMetricsCallback",
    "DraftRouterMetricsCallback",
]
