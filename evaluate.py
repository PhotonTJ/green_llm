"""
evaluate.py — SQuAD and MMLU evaluation routines.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm.auto import tqdm

from green_llm.config import Config
from green_llm.data import load_mmlu_subject
from green_llm.model import kv_cache_context
from green_llm.utils import (
    build_mmlu_prompt,
    chunked,
    clean_generated_text,
    get_device,
    get_torch_dtype,
    squad_score,
)


# ---------------------------------------------------------------------------
# Continuation log-probability (used for both SQuAD no-answer and MMLU)
# ---------------------------------------------------------------------------
def continuation_logprob(model, tokenizer, prompt: str, completion: str) -> float:
    device = get_device()
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    if len(full_ids) <= len(prompt_ids):
        return float("-inf")
    prompt_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(prompt_tensor)
    with torch.no_grad():
        logits = model(prompt_tensor[:, :-1], attention_mask=attn[:, :-1]).logits
        log_probs = F.log_softmax(logits, dim=-1)
    target = prompt_tensor[:, 1:]
    start = max(len(prompt_ids) - 1, 0)
    token_log_probs = (
        log_probs[:, start:, :].gather(-1, target[:, start:].unsqueeze(-1)).squeeze(-1)
    )
    return float(token_log_probs.sum().item())


# ---------------------------------------------------------------------------
# SQuAD generation + scoring
# ---------------------------------------------------------------------------
def generate_squad_predictions(
    model, tokenizer, records: List[Dict[str, Any]], cfg: Config
) -> pd.DataFrame:
    device = get_device()
    rows = []
    model.eval()
    for batch_records in tqdm(
        list(chunked(records, cfg.eval_batch_size)), desc="SQuAD generation"
    ):
        prompts = [r["prompt"] for r in batch_records]
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_source_len,
        ).to(device)
        use_kv_cache = bool(getattr(cfg, "use_kv_cache", True))
        with kv_cache_context(model, use_kv_cache):
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_new_tokens=cfg.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=use_kv_cache,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
        prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
        decoded = [
            tokenizer.decode(gen[i, int(pl):], skip_special_tokens=True)
            for i, pl in enumerate(prompt_lens)
        ]
        for record, prompt, raw_pred in zip(batch_records, prompts, decoded):
            pred_text = clean_generated_text(raw_pred)
            answer_score = (
                continuation_logprob(model, tokenizer, prompt, " " + pred_text)
                if pred_text
                else float("-inf")
            )
            no_answer_score = continuation_logprob(
                model, tokenizer, prompt, " " + cfg.no_answer_text
            )
            rows.append({
                "id": record["id"],
                "question": record["question"],
                "prompt": prompt,
                "gold_texts": json.dumps(record["gold_texts"]),
                "is_impossible": record["is_impossible"],
                "generated_text": pred_text,
                "answer_score": answer_score,
                "no_answer_score": no_answer_score,
                "confidence": answer_score - no_answer_score,
            })
    model.train()
    return pd.DataFrame(rows)


def score_squad_predictions(rows: pd.DataFrame, threshold: float = 0.0) -> Dict[str, float]:
    preds, refs = [], []
    for _, row in rows.iterrows():
        pred = "" if row["confidence"] < threshold else str(row["generated_text"])
        golds = json.loads(row["gold_texts"])
        if row["is_impossible"]:
            golds = [""]
        preds.append(pred)
        refs.append(golds)

    ems, f1s = [], []
    for pred, golds in zip(preds, refs):
        s = squad_score(pred, golds)
        ems.append(s["em"])
        f1s.append(s["f1"])

    answerability_true = np.array([0 if bool(x) else 1 for x in rows["is_impossible"].tolist()])
    answerability_pred = np.array([0 if c < threshold else 1 for c in rows["confidence"].tolist()])
    prec, rec, f1, _ = precision_recall_fscore_support(
        answerability_true, answerability_pred, average="binary", zero_division=0
    )
    return {
        "exact_match": float(np.mean(ems)),
        "f1": float(np.mean(f1s)),
        "answerability_accuracy": float(np.mean(answerability_true == answerability_pred)),
        "answerability_precision": float(prec),
        "answerability_recall": float(rec),
        "answerability_f1": float(f1),
    }


def threshold_sweep(rows: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    scores = rows["confidence"].to_numpy(dtype=float)
    thresholds = np.linspace(float(np.nanmin(scores)), float(np.nanmax(scores)), cfg.val_threshold_steps)
    return pd.DataFrame([
        {"threshold": float(t), **score_squad_predictions(rows, threshold=float(t))}
        for t in thresholds
    ])


def evaluate_no_answer_distribution(rows: pd.DataFrame, cfg: Config) -> Dict[str, Any]:
    sweep_df = threshold_sweep(rows, cfg)
    best_threshold = float(sweep_df.loc[sweep_df["f1"].idxmax(), "threshold"])
    return {"sweep": sweep_df, "best_threshold": best_threshold}


# ---------------------------------------------------------------------------
# MMLU scoring
# ---------------------------------------------------------------------------
def score_mmlu_example(model, tokenizer, example: Dict[str, Any]) -> Dict[str, Any]:
    prompt = build_mmlu_prompt(example)
    labels = ["A", "B", "C", "D"]
    scores_arr = np.array(
        [continuation_logprob(model, tokenizer, prompt, " " + l) for l in labels],
        dtype=np.float64,
    )
    probs = np.exp(scores_arr - scores_arr.max())
    probs /= probs.sum()
    pred_label = labels[int(probs.argmax())]
    true_label = (
        example["answer"]
        if isinstance(example["answer"], str) and example["answer"] in labels
        else "ABCD"[int(example["answer"])]
    )
    out = {
        "subject": example["subject"],
        "question": example["question"],
        "true_label": true_label,
        "pred_label": pred_label,
        "correct": int(pred_label == true_label),
        "prompt": prompt,
    }
    for label, prob, score in zip(labels, probs, scores_arr):
        out[f"prob_{label}"] = float(prob)
        out[f"score_{label}"] = float(score)
    return out


def evaluate_mmlu_subject(model, tokenizer, subject: str, cfg: Config) -> pd.DataFrame:
    ds = load_mmlu_subject(subject, cfg)
    model.eval()
    rows = [score_mmlu_example(model, tokenizer, ex) for ex in tqdm(ds, desc=f"MMLU {subject}")]
    model.train()
    return pd.DataFrame(rows)


def mmlu_metrics(df: pd.DataFrame) -> Dict[str, float]:
    labels = ["A", "B", "C", "D"]
    y_true, y_pred = df["true_label"].tolist(), df["pred_label"].tolist()
    prec_mac, rec_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    probs = df[[f"prob_{l}" for l in labels]].to_numpy()
    true_idx = np.array([labels.index(y) for y in y_true])
    y_true_bin = label_binarize(true_idx, classes=list(range(4)))
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(prec_mac),
        "recall_macro": float(rec_mac),
        "f1_macro": float(f1_mac),
        "precision_weighted": float(prec_w),
        "recall_weighted": float(rec_w),
        "f1_weighted": float(f1_w),
    }
    for idx, label in enumerate(labels):
        if len(np.unique(y_true_bin[:, idx])) < 2:
            metrics[f"auc_pr_{label}"] = float("nan")
            metrics[f"auc_roc_{label}"] = float("nan")
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], probs[:, idx])
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], probs[:, idx])
        metrics[f"auc_pr_{label}"] = float(auc(recall, precision))
        metrics[f"auc_roc_{label}"] = float(auc(fpr, tpr))
    metrics["auc_pr_macro"] = float(np.nanmean([metrics[f"auc_pr_{l}"] for l in labels]))
    metrics["auc_roc_macro"] = float(np.nanmean([metrics[f"auc_roc_{l}"] for l in labels]))
    return metrics
