"""
data.py — SQuAD v2 and MMLU dataset loading, encoding, and DataLoader creation.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

from green_llm.config import Config
from green_llm.utils import (
    build_squad_prompt,
    pick_gold_answer,
    window_context_around_answer,
)


# ---------------------------------------------------------------------------
# Tokenizer setup (call once after tokenizer is loaded)
# ---------------------------------------------------------------------------
def setup_tokenizer(tokenizer) -> None:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


# ---------------------------------------------------------------------------
# SQuAD encoding
# ---------------------------------------------------------------------------
def encode_squad_example(
    example: Dict[str, Any],
    tokenizer,
    cfg: Config,
) -> Dict[str, Any]:
    picked = pick_gold_answer(example, no_answer_text=cfg.no_answer_text)
    context_window = window_context_around_answer(
        example["context"],
        picked["answer_start"],
        picked["answer_text"] if not picked["is_impossible"] else None,
        tokenizer,
        max_context_tokens=cfg.max_context_tokens,
    )
    prompt = build_squad_prompt(context_window, example["question"])
    target_text = picked["answer_text"] if not picked["is_impossible"] else cfg.no_answer_text

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(" " + target_text.strip(), add_special_tokens=False)["input_ids"]
    if len(prompt_ids) + len(target_ids) > cfg.max_source_len:
        overflow = len(prompt_ids) + len(target_ids) - cfg.max_source_len
        prompt_ids = prompt_ids[overflow:]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {
        "id": example.get("id", ""),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def build_squad_eval_record(
    example: Dict[str, Any], tokenizer, cfg: Config
) -> Dict[str, Any]:
    picked = pick_gold_answer(example, no_answer_text=cfg.no_answer_text)
    context_window = window_context_around_answer(
        example["context"],
        picked["answer_start"],
        picked["answer_text"] if not picked["is_impossible"] else None,
        tokenizer,
        max_context_tokens=cfg.max_context_tokens,
    )
    prompt = build_squad_prompt(context_window, example["question"])
    return {
        "id": example.get("id", ""),
        "context_window": context_window,
        "question": example["question"],
        "prompt": prompt,
        "gold_texts": picked["gold_texts"],
        "is_impossible": picked["is_impossible"],
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
def load_squad(tokenizer, cfg: Config):
    raw = load_dataset(cfg.squad_dataset)
    train_raw = raw["train"]
    val_raw = raw["validation"]

    if cfg.fast_dev_run:
        train_raw = train_raw.shuffle(seed=cfg.seed).select(
            range(min(cfg.fast_dev_squad_train, len(train_raw)))
        )
        val_raw = val_raw.shuffle(seed=cfg.seed).select(
            range(min(cfg.fast_dev_squad_val, len(val_raw)))
        )

    encode = lambda ex: encode_squad_example(ex, tokenizer, cfg)
    train_ds = train_raw.map(encode, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(encode, remove_columns=val_raw.column_names)
    val_records = [build_squad_eval_record(ex, tokenizer, cfg) for ex in val_raw]

    return train_raw, val_raw, train_ds, val_ds, val_records


def load_mmlu_subject(subject: str, cfg: Config):
    ds = load_dataset(cfg.mmlu_dataset, subject, split="test")
    if cfg.fast_dev_run:
        ds = ds.shuffle(seed=cfg.seed).select(range(min(cfg.fast_dev_mmlu, len(ds))))
    return ds


# ---------------------------------------------------------------------------
# DataCollator
# ---------------------------------------------------------------------------
class CausalLMDataCollator:
    def __init__(self, tokenizer):
        self.pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features):
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_loader(dataset, collator, batch_size: int, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
