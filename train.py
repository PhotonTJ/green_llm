#!/usr/bin/env python3
"""
train.py — CLI entry-point for the Green-LLM pipeline.

Subcommands
-----------
  train     Train (CE baseline or joint carbon-aware loss).
  infer     Run SQuAD + MMLU inference on a trained checkpoint.
  pipeline  Run the full pipeline: calibration → train → infer → plots.

Examples
--------
  python -m green_llm.train train    --model llama_1b
  python -m green_llm.train infer    --model llama_1b
  python -m green_llm.train pipeline --model mistral_7b --loss_mode joint
  python -m green_llm.train pipeline --model qwen_1b    --fast_dev_run
  python -m green_llm.train --list_models
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer

from green_llm.carbon import run_calibration_passes, run_with_carbon_tracking
from green_llm.config import Config, load_model_zoo
from green_llm.data import CausalLMDataCollator, load_squad, setup_tokenizer
from green_llm.evaluate import (
    evaluate_mmlu_subject,
    evaluate_no_answer_distribution,
    generate_squad_predictions,
    mmlu_metrics,
    score_squad_predictions,
)
from green_llm.plots import (
    plot_and_save,
    plot_confidence_distribution,
    plot_confusion,
    plot_emissions_over_time,
    plot_emissions_proxy,
    plot_loss_curves,
    plot_pr_roc,
    plot_subject_bars,
    plot_summary_grid,
    plot_threshold_sweep,
    plot_tradeoff,
)
from green_llm.trainer import run_training
from green_llm.utils import save_df, save_json, seed_everything


# ────────────────────────────────────────────────────────────────────
# HuggingFace auto-login from setup.env / env var
# ────────────────────────────────────────────────────────────────────
def _hf_login() -> None:
    """Log in to HuggingFace Hub if HF_TOKEN is set."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token and token != "hf_YOUR_TOKEN_HERE":
        try:
            from huggingface_hub import login
            login(token=token, add_to_git_credential=False)
            print("[green-llm] Logged in to HuggingFace Hub.")
        except Exception as exc:
            print(f"[green-llm] HF login warning: {exc}")
    else:
        print("[green-llm] No HF_TOKEN set — skipping auto-login. "
              "Set it in green_llm/setup.env for gated models.")


# ────────────────────────────────────────────────────────────────────
# Shared CLI arguments (attached to every subcommand)
# ────────────────────────────────────────────────────────────────────
def _add_common_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("Model")
    g.add_argument("--model", type=str, default="qwen_1b",
                   help="Model alias from model_zoo.json")

    g = p.add_argument_group("Training")
    g.add_argument("--loss_mode", choices=["ce", "joint"], default="ce")
    g.add_argument("--epochs", type=int, default=None)
    g.add_argument("--lr", type=float, default=None)
    g.add_argument("--weight_decay", type=float, default=None)
    g.add_argument("--train_batch_size", type=int, default=None)
    g.add_argument("--eval_batch_size", type=int, default=None)
    g.add_argument("--grad_accum", type=int, default=None)
    g.add_argument("--mu", type=float, default=None,
                   help="LoRA regularisation weight")
    g.add_argument("--lambda_weight", type=float, default=None,
                   help="Carbon surrogate weight for the train subcommand")
    g.add_argument("--f1_tolerance", type=float, default=None)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora_r", type=int, default=None)
    g.add_argument("--lora_alpha", type=int, default=None)
    g.add_argument("--lora_dropout", type=float, default=None)

    g = p.add_argument_group("Sequence")
    g.add_argument("--max_source_len", type=int, default=None)
    g.add_argument("--max_context_tokens", type=int, default=None)
    g.add_argument("--max_new_tokens", type=int, default=None)
    g.add_argument("--use_kv_cache", action=argparse.BooleanOptionalAction, default=None,
                   help="Use KV cache during generation/inference")
    g.add_argument("--kv_cache_energy_reduction", type=float, default=None,
                   help="Expected fractional inference-energy reduction from KV cache")

    g = p.add_argument_group("Evaluation")
    g.add_argument("--val_threshold_steps", type=int, default=None)
    g.add_argument("--no_answer_text", type=str, default=None)

    g = p.add_argument_group("Fast-dev")
    g.add_argument("--fast_dev_run", action="store_true")
    g.add_argument("--fast_dev_squad_train", type=int, default=None)
    g.add_argument("--fast_dev_squad_val", type=int, default=None)
    g.add_argument("--fast_dev_mmlu", type=int, default=None)

    g = p.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--country_iso_code", type=str, default=None)
    g.add_argument("--skip_calibration", action="store_true")
    g.add_argument("--skip_mmlu", action="store_true")


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────
def _print_model_zoo() -> None:
    zoo = load_model_zoo()
    print(f"\n{'Alias':<25} {'HuggingFace Model ID':<50} {'Enabled'}")
    print("-" * 85)
    for alias, entry in sorted(zoo.items()):
        flag = "✓" if entry.get("flag", False) else "✗"
        print(f"{alias:<25} {entry['model_name']:<50} {flag}")
    print()


def _setup(args) -> tuple:
    """Common setup for all subcommands: login, config, tokenizer, data."""
    _hf_login()
    cfg = Config.from_cli(args)
    if args.fast_dev_run:
        cfg.fast_dev_run = True
    seed_everything(cfg.seed)
    print(f"[green-llm] Model  : {cfg.base_model}  (alias: {cfg.model_alias})")
    print(f"[green-llm] Device : "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"[green-llm] Output : {cfg.out_dir.resolve()}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=cfg.use_fast_tokenizer,
    )
    setup_tokenizer(tokenizer)
    collator = CausalLMDataCollator(tokenizer)
    train_raw, val_raw, train_ds, val_ds, val_records = load_squad(tokenizer, cfg)
    return cfg, tokenizer, collator, train_raw, val_raw, train_ds, val_ds, val_records


# ────────────────────────────────────────────────────────────────────
# Subcommand: train
# ────────────────────────────────────────────────────────────────────
def cmd_train(args) -> None:
    cfg, tokenizer, collator, train_raw, _, train_ds, val_ds, _ = _setup(args)
    lambda_weight = 0.0
    carbon_weights = None
    if args.loss_mode == "joint":
        lambda_weight = cfg.lambda_weight
        if args.skip_calibration:
            carbon_weights = [1.0, 1.0, 1.0]
        else:
            fit = run_calibration_passes(train_raw, tokenizer, collator, cfg)
            carbon_weights = fit["normalized_weights"]

    model, history, carbon, ckpt = run_training(
        run_name=f"{cfg.model_alias}_{args.loss_mode}",
        train_dataset=train_ds,
        eval_dataset=val_ds,
        collator=collator,
        cfg=cfg,
        loss_mode=args.loss_mode,
        lambda_weight=lambda_weight,
        carbon_weights=carbon_weights,
    )
    save_df(history, cfg.metrics_dir / f"{cfg.model_alias}_{args.loss_mode}_history.csv")
    tokenizer.save_pretrained(ckpt)
    print(f"\n[green-llm] Training done.  Checkpoint: {ckpt}")
    print(f"[green-llm] Carbon summary: {carbon}")


# ────────────────────────────────────────────────────────────────────
# Subcommand: infer
# ────────────────────────────────────────────────────────────────────
def cmd_infer(args) -> None:
    cfg, tokenizer, collator, _, _, _, val_ds, val_records = _setup(args)

    # Load model from checkpoint
    from green_llm.model import build_lora_model
    model = build_lora_model(cfg, for_training=False)

    # SQuAD inference
    preds, infer_carbon = run_with_carbon_tracking(
        f"{cfg.model_alias}_squad_inference",
        lambda: generate_squad_predictions(model, tokenizer, val_records, cfg),
        cfg=cfg,
    )
    sweep = evaluate_no_answer_distribution(preds, cfg)
    best_thr = float(sweep["best_threshold"])
    metrics = score_squad_predictions(preds, threshold=best_thr)
    metrics["best_threshold"] = best_thr
    save_df(preds, cfg.metrics_dir / "squad_predictions.csv")
    save_json(metrics, cfg.metrics_dir / "squad_metrics.json")
    print(f"\n[green-llm] SQuAD metrics: {metrics}")

    # MMLU inference
    if not args.skip_mmlu:
        for subject in cfg.mmlu_subjects:
            df = evaluate_mmlu_subject(model, tokenizer, subject, cfg)
            save_df(df, cfg.metrics_dir / f"mmlu_{subject}_predictions.csv")
            m = mmlu_metrics(df)
            save_json(m, cfg.metrics_dir / f"mmlu_{subject}_metrics.json")
            print(f"[green-llm] MMLU {subject}: acc={m['accuracy']:.3f}")

    print(f"[green-llm] Inference done.  Results in {cfg.out_dir}")


# ────────────────────────────────────────────────────────────────────
# Subcommand: pipeline (full end-to-end)
# ────────────────────────────────────────────────────────────────────
def cmd_pipeline(args) -> None:
    cfg, tokenizer, collator, train_raw, _, train_ds, val_ds, val_records = _setup(args)

    # ── 1. CE baseline training ───────────────────────────────────────
    ce_model, ce_history, ce_carbon, _ = run_training(
        run_name="ce_baseline",
        train_dataset=train_ds,
        eval_dataset=val_ds,
        collator=collator,
        cfg=cfg,
        loss_mode="ce",
    )
    save_df(ce_history, cfg.metrics_dir / "ce_baseline_history.csv")

    ce_preds, ce_infer_carbon = run_with_carbon_tracking(
        "ce_squad_inference",
        lambda: generate_squad_predictions(ce_model, tokenizer, val_records, cfg),
        cfg=cfg,
    )
    ce_sweep = evaluate_no_answer_distribution(ce_preds, cfg)
    ce_thr = float(ce_sweep["best_threshold"])
    ce_metrics = score_squad_predictions(ce_preds, threshold=ce_thr)
    ce_metrics.update(run_name="ce_baseline", best_threshold=ce_thr)
    print("[green-llm] CE SQuAD:", ce_metrics)

    # ── 2. Calibration + joint training (if requested) ────────────────
    best_lambda = 0.0
    joint_model = ce_model
    joint_history = ce_history
    joint_carbon = ce_carbon

    if args.loss_mode == "joint":
        # Surrogate-weight calibration
        if args.skip_calibration:
            surrogate_weights = [1.0, 1.0, 1.0]
        else:
            fit = run_calibration_passes(train_raw, tokenizer, collator, cfg)
            surrogate_weights = fit["normalized_weights"]
            save_json(
                {
                    k: fit[k]
                    for k in (
                        "raw_weights",
                        "normalized_weights",
                        "scales",
                        "r2",
                        "feature_cols",
                        "input_length_carbon",
                    )
                },
                cfg.metrics_dir / "surrogate_weights.json",
            )

        # Lambda sweep
        from green_llm.carbon import _calibration_dataset
        sweep_subset = train_raw.shuffle(seed=cfg.seed).select(
            range(min(len(train_raw), 1024 if not cfg.fast_dev_run else 128))
        )
        sweep_ds = _calibration_dataset(sweep_subset, tokenizer, cfg, 256)
        lambda_results = []

        for lw in cfg.lambda_grid:
            c_model, _, c_carbon, _ = run_training(
                run_name=f"joint_lambda_{lw}",
                train_dataset=sweep_ds,
                eval_dataset=val_ds,
                collator=collator,
                cfg=cfg,
                loss_mode="joint",
                lambda_weight=lw,
                carbon_weights=surrogate_weights,
                max_steps=40 if not cfg.fast_dev_run else 10,
            )
            c_preds = generate_squad_predictions(
                c_model, tokenizer,
                val_records[:min(len(val_records), 128 if not cfg.fast_dev_run else 32)],
                cfg,
            )
            c_sw = evaluate_no_answer_distribution(c_preds, cfg)["sweep"]
            c_thr = float(c_sw.loc[c_sw["f1"].idxmax(), "threshold"])
            c_m = score_squad_predictions(c_preds, threshold=c_thr)
            lambda_results.append({
                "lambda_weight": lw,
                "val_f1": c_m["f1"],
                "val_em": c_m["exact_match"],
                "emissions_kgco2eq": c_carbon["emissions_kgco2eq"],
            })
            del c_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ldf = pd.DataFrame(lambda_results)
        save_df(ldf, cfg.metrics_dir / "lambda_sweep_results.csv")
        eligible = ldf[ldf["val_f1"] >= ce_metrics["f1"] - cfg.f1_tolerance]
        if eligible.empty:
            eligible = ldf
        best_lambda = float(
            eligible.sort_values(["val_f1", "emissions_kgco2eq"],
                                 ascending=[False, True]).iloc[0]["lambda_weight"]
        )
        print(f"[green-llm] Selected λ = {best_lambda}")

        # Full joint training
        joint_model, joint_history, joint_carbon, _ = run_training(
            run_name=f"joint_lambda_{best_lambda}",
            train_dataset=train_ds,
            eval_dataset=val_ds,
            collator=collator,
            cfg=cfg,
            loss_mode="joint",
            lambda_weight=best_lambda,
            carbon_weights=surrogate_weights,
        )
        save_df(joint_history, cfg.metrics_dir / "joint_loss_history.csv")

    # ── 3. Joint SQuAD inference ──────────────────────────────────────
    joint_preds, joint_infer_carbon = run_with_carbon_tracking(
        "joint_squad_inference",
        lambda: generate_squad_predictions(joint_model, tokenizer, val_records, cfg),
        cfg=cfg,
    )
    joint_sweep = evaluate_no_answer_distribution(joint_preds, cfg)
    joint_thr = float(joint_sweep["best_threshold"])
    joint_metrics = score_squad_predictions(joint_preds, threshold=joint_thr)
    joint_metrics.update(run_name=f"joint_lambda_{best_lambda}", best_threshold=joint_thr)

    squad_df = pd.DataFrame([ce_metrics, joint_metrics])
    save_df(squad_df, cfg.metrics_dir / "squad_metrics_summary.csv")
    save_df(ce_preds, cfg.metrics_dir / "ce_squad_predictions.csv")
    save_df(joint_preds, cfg.metrics_dir / "joint_squad_predictions.csv")

    # ── 4. MMLU evaluation ────────────────────────────────────────────
    mmlu_predictions = {}
    mmlu_summary = pd.DataFrame()
    if not args.skip_mmlu:
        def _run_mmlu(model, tag):
            rows = []
            for subject in cfg.mmlu_subjects:
                df = evaluate_mmlu_subject(model, tokenizer, subject, cfg)
                mmlu_predictions[subject] = df
                save_df(df, cfg.metrics_dir / f"mmlu_{tag}_{subject}_predictions.csv")
                m = mmlu_metrics(df)
                m.update(subject=subject, model=tag)
                rows.append(m)
            return rows

        ce_mmlu, _ = run_with_carbon_tracking(
            "mmlu_ce", lambda: _run_mmlu(ce_model, "ce_baseline"), cfg=cfg,
        )
        jt_mmlu, _ = run_with_carbon_tracking(
            "mmlu_joint", lambda: _run_mmlu(joint_model, f"joint_{best_lambda}"), cfg=cfg,
        )
        mmlu_summary = pd.DataFrame(ce_mmlu + jt_mmlu)
        save_df(mmlu_summary, cfg.metrics_dir / "mmlu_summary.csv")

    # ── 5. Emission tables ────────────────────────────────────────────
    training_summaries = pd.DataFrame([
        {"run_name": "ce_baseline", **ce_carbon,
         "squad_f1": ce_metrics["f1"], "stage": "training"},
        {"run_name": f"joint_lambda_{best_lambda}", **joint_carbon,
         "squad_f1": joint_metrics["f1"], "stage": "training"},
    ])
    save_df(training_summaries, cfg.metrics_dir / "training_summaries.csv")

    # ── 6. Plots ──────────────────────────────────────────────────────
    plot_and_save(
        plot_loss_curves(ce_history, "ce_baseline"),
        cfg.fig_dir / "ce_loss_curves.png")
    plot_and_save(
        plot_loss_curves(joint_history, f"joint_lambda_{best_lambda}"),
        cfg.fig_dir / "joint_loss_curves.png")
    plot_and_save(
        plot_threshold_sweep(ce_sweep["sweep"], "CE threshold sweep"),
        cfg.fig_dir / "ce_threshold_sweep.png")
    plot_and_save(
        plot_threshold_sweep(joint_sweep["sweep"], "Joint threshold sweep"),
        cfg.fig_dir / "joint_threshold_sweep.png")
    plot_and_save(
        plot_confidence_distribution(ce_preds, "CE confidence"),
        cfg.fig_dir / "ce_confidence_distribution.png")
    plot_and_save(
        plot_confidence_distribution(joint_preds, "Joint confidence"),
        cfg.fig_dir / "joint_confidence_distribution.png")
    plot_and_save(
        plot_tradeoff(training_summaries),
        cfg.fig_dir / "carbon_vs_performance.png")
    plot_and_save(
        plot_emissions_proxy(training_summaries),
        cfg.fig_dir / "stage_emissions.png")
    plot_and_save(
        plot_emissions_over_time(training_summaries),
        cfg.fig_dir / "emissions_over_time.png")

    if not mmlu_summary.empty:
        jt_mmlu_df = mmlu_summary[mmlu_summary["model"] == f"joint_{best_lambda}"]
        if not jt_mmlu_df.empty:
            plot_and_save(plot_subject_bars(jt_mmlu_df),
                          cfg.fig_dir / "mmlu_subject_metrics.png")
        for subject, df in mmlu_predictions.items():
            plot_and_save(plot_pr_roc(df, subject),
                          cfg.fig_dir / f"mmlu_{subject}_pr_roc.png")
            plot_and_save(plot_confusion(df, subject),
                          cfg.fig_dir / f"mmlu_{subject}_confusion.png")

    print(f"\n[green-llm] Pipeline complete.  All outputs → {cfg.out_dir.resolve()}")


# ────────────────────────────────────────────────────────────────────
# Argument parser
# ────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="green_llm",
        description="Green-LLM: carbon-aware LoRA fine-tuning",
    )
    root.add_argument("--list_models", action="store_true",
                       help="Print model zoo and exit.")

    sub = root.add_subparsers(dest="command")

    # train
    p_train = sub.add_parser("train", help="Train a model (CE or joint).")
    _add_common_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # infer
    p_infer = sub.add_parser("infer", help="Run SQuAD + MMLU inference.")
    _add_common_args(p_infer)
    p_infer.set_defaults(func=cmd_infer)

    # pipeline
    p_pipe = sub.add_parser("pipeline",
                            help="Full pipeline: calibrate → train → infer → plots.")
    _add_common_args(p_pipe)
    p_pipe.set_defaults(func=cmd_pipeline)

    return root


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        _print_model_zoo()
        sys.exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
