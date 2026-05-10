"""
trainer.py — Training loop with CE and joint (carbon-aware) loss modes.
"""

from __future__ import annotations

import gc
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from codecarbon import OfflineEmissionsTracker
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from green_llm.carbon import make_tracker, summarize_tracker
from green_llm.config import Config
from green_llm.model import (
    batch_to_device,
    build_lora_model,
    compute_task_loss,
    count_trainable_parameters,
    lora_reg_loss,
    surrogate_carbon_loss,
)
from green_llm.utils import get_device, get_torch_dtype, save_df, save_json


# ---------------------------------------------------------------------------
# Validation loss
# ---------------------------------------------------------------------------
def evaluate_loss(model, loader: DataLoader, cfg: Config) -> float:
    device = get_device()
    dtype = get_torch_dtype()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device)
            with torch.autocast(
                device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()
            ):
                losses.append(compute_task_loss(model, batch).detach().float().item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def run_training(
    run_name: str,
    train_dataset,
    eval_dataset,
    collator,
    cfg: Config,
    loss_mode: str = "ce",
    lambda_weight: float = 0.0,
    carbon_weights: Optional[List[float]] = None,
    num_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    save_subdir: Optional[str] = None,
    limit_val_batches: Optional[int] = None,
) -> Tuple:
    device = get_device()
    dtype = get_torch_dtype()
    num_epochs = num_epochs if num_epochs is not None else cfg.epochs

    model = build_lora_model(cfg, device)
    print(run_name, count_trainable_parameters(model))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    updates_per_epoch = max(1, math.ceil(len(train_loader) / cfg.grad_accum))
    total_updates = num_epochs * updates_per_epoch if max_steps is None else max_steps
    warmup_steps = max(5, int(0.05 * total_updates))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_updates)

    use_scaler = torch.cuda.is_available() and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    run_dir = cfg.carbon_dir / run_name
    tracker = make_tracker(run_name, run_dir, cfg)
    tracker.start()

    history: List[dict] = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()

    for epoch in range(num_epochs):
        epoch_task_losses: List[float] = []

        for batch_idx, batch in enumerate(
            tqdm(train_loader, desc=f"{run_name} epoch {epoch + 1}/{num_epochs}")
        ):
            if max_steps is not None and global_step >= max_steps:
                break

            batch = batch_to_device(batch, device)

            with torch.autocast(device_type="cuda", dtype=dtype, enabled=torch.cuda.is_available()):
                task_loss = compute_task_loss(model, batch)
                carbon_loss = torch.tensor(0.0, device=device)
                reg_loss = torch.tensor(0.0, device=device)

                if loss_mode == "joint":
                    w = carbon_weights or [1.0, 1.0, 1.0]
                    carbon_loss = surrogate_carbon_loss(
                        model,
                        batch["input_ids"],
                        w,
                        attention_mask=batch["attention_mask"],
                    )
                    reg_loss = lora_reg_loss(model)
                    total_loss = task_loss + lambda_weight * carbon_loss + cfg.mu * reg_loss
                else:
                    total_loss = task_loss

                scaled_loss = total_loss / cfg.grad_accum

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            epoch_task_losses.append(float(task_loss.detach().float().item()))
            history.append({
                "run_name": run_name,
                "loss_mode": loss_mode,
                "record_type": "batch",
                "epoch": epoch + 1,
                "step": global_step + 1,
                "task_loss": float(task_loss.detach().float().item()),
                "carbon_loss": float(carbon_loss.detach().float().item()),
                "reg_loss": float(reg_loss.detach().float().item()),
                "total_loss": float(total_loss.detach().float().item()),
                "lambda": float(lambda_weight),
            })

            if ((batch_idx + 1) % cfg.grad_accum == 0) or (batch_idx + 1 == len(train_loader)):
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # End-of-epoch validation
        eval_subset = eval_dataset
        if limit_val_batches is not None:
            eval_subset = eval_dataset.select(range(min(len(eval_dataset), limit_val_batches)))

        val_loss = evaluate_loss(
            model,
            DataLoader(
                eval_subset,
                batch_size=cfg.train_batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            ),
            cfg,
        )
        history.append({
            "run_name": run_name,
            "loss_mode": loss_mode,
            "record_type": "epoch",
            "epoch": epoch + 1,
            "step": global_step,
            "task_loss": float(np.mean(epoch_task_losses)) if epoch_task_losses else float("nan"),
            "val_loss": val_loss,
        })

    carbon_summary = summarize_tracker(tracker, run_name)
    carbon_summary.update({
        "country_iso_code": cfg.country_iso_code,
        "use_kv_cache": False,
        "kv_cache_expected_energy_reduction": float(
            getattr(cfg, "kv_cache_energy_reduction", 0.1636)
        ),
    })
    save_dir = cfg.checkpoint_dir / (save_subdir or run_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)

    history_df = pd.DataFrame(history)
    save_df(history_df, cfg.log_dir / f"{run_name}_history.csv")
    save_json(carbon_summary, cfg.log_dir / f"{run_name}_carbon_summary.json")

    del optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, history_df, carbon_summary, save_dir
