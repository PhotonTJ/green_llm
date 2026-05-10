"""
carbon.py — CodeCarbon tracking wrappers and surrogate-weight calibration.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker
from sklearn.linear_model import LinearRegression
from torch.utils.data import DataLoader

from green_llm.config import Config
from green_llm.model import (
    build_lora_model,
    carbon_proxy_terms,
    compute_task_loss,
    lora_reg_loss,
)
from green_llm.utils import get_device, get_torch_dtype, save_df, save_json


# ---------------------------------------------------------------------------
# Tracker helpers
# ---------------------------------------------------------------------------
def make_tracker(run_name: str, run_dir: Path, cfg: Config) -> OfflineEmissionsTracker:
    run_dir.mkdir(parents=True, exist_ok=True)
    return OfflineEmissionsTracker(
        project_name=run_name,
        country_iso_code=cfg.country_iso_code,
        output_dir=str(run_dir),
        save_to_file=True,
    )


def summarize_tracker(tracker: OfflineEmissionsTracker, run_name: str) -> Dict[str, Any]:
    emissions = float(tracker.stop())
    final = getattr(tracker, "final_emissions_data", None)
    energy = float(getattr(final, "energy_consumed", float("nan"))) if final else float("nan")
    duration = float(getattr(final, "duration", float("nan"))) if final else float("nan")
    kgco2_per_kwh = (
        emissions / energy
        if energy and not math.isnan(energy) and energy > 0
        else float("nan")
    )
    return {
        "run_name": run_name,
        "emissions_kgco2eq": emissions,
        "kgco2e": emissions,
        "energy_consumed_kwh": energy,
        "kgco2_per_kwh": kgco2_per_kwh,
        "duration_s": duration,
    }


def run_with_carbon_tracking(
    run_name: str,
    fn: Callable,
    cfg: Config,
    output_subdir: str | None = None,
) -> Tuple[Any, Dict[str, Any]]:
    run_dir = cfg.carbon_dir / (output_subdir or run_name)
    tracker = make_tracker(run_name, run_dir, cfg)
    tracker.start()
    result = fn()
    summary = summarize_tracker(tracker, run_name)
    summary.update({
        "country_iso_code": cfg.country_iso_code,
        "use_kv_cache": bool(getattr(cfg, "use_kv_cache", True)),
        "kv_cache_expected_energy_reduction": float(
            getattr(cfg, "kv_cache_energy_reduction", 0.1636)
        ),
    })
    save_json(summary, cfg.log_dir / f"{run_name}_carbon_summary.json")
    return result, summary


# ---------------------------------------------------------------------------
# Surrogate-weight calibration
# ---------------------------------------------------------------------------
def _calibration_dataset(raw_subset, tokenizer, cfg: Config, max_context_tokens: int):
    from green_llm.data import encode_squad_example
    from dataclasses import replace
    sub_cfg = Config(
        model_alias=cfg.model_alias,
        max_source_len=cfg.max_source_len,
        max_context_tokens=max_context_tokens,
        no_answer_text=cfg.no_answer_text,
    )
    encode = lambda ex: encode_squad_example(ex, tokenizer, sub_cfg)
    return raw_subset.map(encode, remove_columns=raw_subset.column_names)


def fit_surrogate_weights(calibration_rows: List[Dict[str, float]]) -> Dict[str, Any]:
    df = pd.DataFrame(calibration_rows)
    feature_cols = ["param_norm", "flops_proxy", "memory_proxy"]
    fit_df = (
        df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=feature_cols + ["emissions_kgco2eq"])
        .copy()
    )
    if fit_df.empty:
        raw_weights = np.zeros(len(feature_cols), dtype=float)
        return {
            "raw_weights": raw_weights.tolist(),
            "normalized_weights": raw_weights.tolist(),
            "scales": np.ones(len(feature_cols), dtype=float).tolist(),
            "r2": float("nan"),
            "intercept": 0.0,
            "feature_cols": feature_cols,
            "calibration_table": df,
        }
    X = fit_df[feature_cols].astype(float).to_numpy()
    y = fit_df["emissions_kgco2eq"].astype(float).to_numpy()
    scales = np.maximum(X.mean(axis=0), 1e-9)
    X_scaled = X / scales
    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(X_scaled, y)
    raw_weights = reg.coef_ / scales
    raw_weights = np.maximum(raw_weights, 0.0)
    normalized = raw_weights / raw_weights.sum() if raw_weights.sum() > 0 else raw_weights
    return {
        "raw_weights": raw_weights.tolist(),
        "normalized_weights": normalized.tolist(),
        "scales": scales.tolist(),
        "r2": float(reg.score(X_scaled, y)),
        "intercept": float(getattr(reg, "intercept_", 0.0)),
        "feature_cols": feature_cols,
        "calibration_table": df,
    }


def fit_input_length_carbon_model(calibration_rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Fit C(T) = c0 + c1*T + c2*T^2 from calibration emissions."""
    df = pd.DataFrame(calibration_rows).replace([np.inf, -np.inf], np.nan)
    token_col = "avg_seq_len" if "avg_seq_len" in df.columns else "max_context_tokens"
    fit_df = df.dropna(subset=[token_col, "emissions_kgco2eq"]).copy()
    if len(fit_df) < 2:
        return {
            "c0": float("nan"),
            "c1": float("nan"),
            "c2": float("nan"),
            "r2": float("nan"),
            "token_column": token_col,
        }

    tokens = fit_df[token_col].astype(float).to_numpy()
    emissions = fit_df["emissions_kgco2eq"].astype(float).to_numpy()
    if len(fit_df) >= 3:
        x = np.column_stack([np.ones_like(tokens), tokens, tokens**2])
    else:
        x = np.column_stack([np.ones_like(tokens), tokens])

    reg = LinearRegression(positive=True, fit_intercept=False)
    reg.fit(x, emissions)
    coef = reg.coef_.tolist()
    while len(coef) < 3:
        coef.append(0.0)
    return {
        "c0": float(coef[0]),
        "c1": float(coef[1]),
        "c2": float(coef[2]),
        "r2": float(reg.score(x, emissions)),
        "token_column": token_col,
    }


def estimate_input_length_carbon(seq_len: float, coefficients: Dict[str, float]) -> float:
    return (
        float(coefficients.get("c0", 0.0))
        + float(coefficients.get("c1", 0.0)) * float(seq_len)
        + float(coefficients.get("c2", 0.0)) * float(seq_len) ** 2
    )


def run_calibration_passes(raw_train, tokenizer, collator, cfg: Config) -> Dict[str, Any]:
    specs = [
        {"name": "calib_128", "max_context_tokens": 128, "sample_size": 64},
        {"name": "calib_256", "max_context_tokens": 256, "sample_size": 64},
        {"name": "calib_384", "max_context_tokens": 384, "sample_size": 64},
    ]
    device = get_device()
    dtype = get_torch_dtype()
    rows = []

    for spec in specs:
        sample = raw_train.shuffle(seed=cfg.seed).select(
            range(min(spec["sample_size"], len(raw_train)))
        )
        ds = _calibration_dataset(sample, tokenizer, cfg, spec["max_context_tokens"])
        model = build_lora_model(cfg, device)
        loader = DataLoader(
            ds,
            batch_size=cfg.train_batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        run_dir = cfg.carbon_dir / spec["name"]
        tracker = make_tracker(spec["name"], run_dir, cfg)
        tracker.start()
        model.train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=cfg.lr
        )
        use_scaler = torch.cuda.is_available() and dtype == torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

        proxy_keys = [
            "param_norm",
            "flops_proxy",
            "memory_proxy",
            "attention_quadratic_proxy",
            "attention_linear_proxy",
            "ffn_proxy",
            "normalization_proxy",
            "activation_memory_proxy",
            "kv_cache_proxy",
            "network_proxy",
            "avg_seq_len",
            "token_count",
        ]
        observed = {key: [] for key in proxy_keys}
        from tqdm.auto import tqdm
        for step, batch in enumerate(tqdm(loader, desc=f"calibration {spec['name']}")):
            if step >= 10:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                task_loss = compute_task_loss(model, batch)
                proxies = carbon_proxy_terms(
                    model,
                    batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                total_loss = task_loss + 1e-4 * lora_reg_loss(model)
                scaled = total_loss / cfg.grad_accum
            if use_scaler:
                scaler.scale(scaled).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                scaled.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            for k in observed:
                observed[k].append(float(proxies[k].detach().float().item()))

        summary = summarize_tracker(tracker, spec["name"])
        row = {
            "run_name": spec["name"],
            "max_context_tokens": float(spec["max_context_tokens"]),
            "emissions_kgco2eq": summary["emissions_kgco2eq"],
            "energy_consumed_kwh": summary["energy_consumed_kwh"],
        }
        for key in proxy_keys:
            row[key] = float(np.mean(observed[key])) if observed[key] else float("nan")
        rows.append(row)
        del model, optimizer, scaler
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fit = fit_surrogate_weights(rows)
    length_fit = fit_input_length_carbon_model(rows)
    save_df(fit["calibration_table"], cfg.metrics_dir / "carbon_calibration_table.csv")
    save_json(
        {
            **{
                k: fit[k]
                for k in ("raw_weights", "normalized_weights", "scales", "r2", "intercept", "feature_cols")
            },
            "input_length_carbon": length_fit,
        },
        cfg.metrics_dir / "surrogate_weights.json",
    )
    print("Fitted surrogate weights:", fit["raw_weights"])
    print("Fitted input-length carbon model:", length_fit)
    fit["input_length_carbon"] = length_fit
    return fit
