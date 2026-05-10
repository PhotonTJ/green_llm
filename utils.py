"""
utils.py
--------
Shared utility helpers used across the pipeline:
  - seeding
  - device / dtype resolution
  - JSON / CSV I/O
  - prompt templates
  - SQuAD text-normalisation and scoring
  - context windowing
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    """Set seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Device / dtype helpers
# ---------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_torch_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
SQUAD_PROMPT_TEMPLATE = (
    "You are a question answering model.\n"
    "Use only the provided context.\n"
    "If the question is not answerable from the context, reply with no answer.\n\n"
    "Context:\n{context}\n\n"
    "Question:\n{question}\n\n"
    "Answer:"
)

MMLU_PROMPT_TEMPLATE = (
    "You are answering a multiple-choice question. "
    "Reply with only the letter A, B, C, or D.\n\n"
    "Question: {question}\n"
    "A. {a}\n"
    "B. {b}\n"
    "C. {c}\n"
    "D. {d}\n"
    "Answer:"
)


def build_squad_prompt(context: str, question: str) -> str:
    return SQUAD_PROMPT_TEMPLATE.format(
        context=context.strip(), question=question.strip()
    )


def build_mmlu_prompt(example: Dict[str, Any]) -> str:
    choices = example["choices"]
    return MMLU_PROMPT_TEMPLATE.format(
        question=example["question"].strip(),
        a=choices[0],
        b=choices[1],
        c=choices[2],
        d=choices[3],
    )


# ---------------------------------------------------------------------------
# SQuAD scoring
# ---------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def squad_f1(prediction: str, reference: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    ref_tokens = normalize_text(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    common: Dict[str, int] = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    num_same = 0
    for tok in ref_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def squad_em(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def squad_score(prediction: str, references: List[str]) -> Dict[str, float]:
    references = references or [""]
    em = max(squad_em(prediction, ref) for ref in references)
    f1 = max(squad_f1(prediction, ref) for ref in references)
    return {"em": em, "f1": f1}


# ---------------------------------------------------------------------------
# SQuAD answer helpers
# ---------------------------------------------------------------------------
def pick_gold_answer(example: Dict[str, Any], no_answer_text: str = "no answer") -> Dict[str, Any]:
    is_impossible = bool(example.get("is_impossible", False))
    answer_texts = example.get("answers", {}).get("text", [])
    answer_starts = example.get("answers", {}).get("answer_start", [])
    texts = [t.strip() for t in answer_texts if t and t.strip()]
    if is_impossible or not texts:
        return {
            "gold_texts": [""],
            "is_impossible": True,
            "answer_text": no_answer_text,
            "answer_start": None,
        }
    return {
        "gold_texts": texts,
        "is_impossible": False,
        "answer_text": texts[0],
        "answer_start": int(answer_starts[0]) if answer_starts else None,
    }


def window_context_around_answer(
    context: str,
    answer_start: Optional[int],
    answer_text: Optional[str],
    tokenizer: Any,
    max_context_tokens: int = 384,
) -> str:
    context = (context or "").strip()
    if not context:
        return context
    if not getattr(tokenizer, "is_fast", False):
        ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        return tokenizer.decode(ids[:max_context_tokens], skip_special_tokens=True).strip()

    encoded = tokenizer(
        context,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    if answer_start is None or not answer_text:
        return tokenizer.decode(ids[:max_context_tokens], skip_special_tokens=True).strip()

    answer_end = answer_start + len(answer_text)
    span_tokens = [
        i for i, (start, end) in enumerate(offsets)
        if not (end <= answer_start or start >= answer_end)
    ]
    center = (
        (span_tokens[0] + span_tokens[-1]) // 2
        if span_tokens
        else min(len(ids) // 2, max(len(ids) - 1, 0))
    )
    left = max(0, center - max_context_tokens // 2)
    right = min(len(ids), left + max_context_tokens)
    left = max(0, right - max_context_tokens)
    return tokenizer.decode(ids[left:right], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def chunked(items: List[Any], size: int):
    for idx in range(0, len(items), size):
        yield items[idx: idx + size]


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def clean_generated_text(text: str) -> str:
    text = text.strip()
    text = re.split(r"\n(?:Question:|Context:|Answer:)", text)[0].strip()
    text = re.sub(r"^[:\-\s]+", "", text).strip()
    return text


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def save_json(obj: Any, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)


def save_df(df, path: Path) -> None:
    import pandas as pd
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
