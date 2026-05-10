"""
plots.py — All diagnostic visualisations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

sns.set_theme(style="whitegrid", context="talk")


def plot_and_save(fig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(history_df: pd.DataFrame, run_name: str):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    batch_rows = (
        history_df[history_df["record_type"] == "batch"].copy()
        if "record_type" in history_df.columns
        else history_df.copy()
    )
    epoch_rows = (
        history_df[history_df["record_type"] == "epoch"].copy()
        if "record_type" in history_df.columns
        else history_df.copy()
    )
    x = batch_rows["step"] if "step" in batch_rows.columns else batch_rows.index
    axes[0, 0].plot(x, batch_rows["task_loss"], label="task loss")
    axes[0, 0].plot(x, batch_rows["total_loss"], label="total loss")
    axes[0, 0].set_title(f"{run_name} loss")
    axes[0, 0].legend()

    if "carbon_loss" in history_df.columns:
        axes[0, 1].plot(x, batch_rows["carbon_loss"], label="carbon loss", color="tab:orange")
        axes[0, 1].plot(x, batch_rows["reg_loss"], label="reg loss", color="tab:green")
        axes[0, 1].set_title(f"{run_name} joint breakdown")
        axes[0, 1].legend()

    if "val_loss" in history_df.columns and history_df["val_loss"].notna().any():
        val_rows = history_df.dropna(subset=["val_loss"])
        axes[1, 0].plot(val_rows["epoch"], val_rows["val_loss"], marker="o")
        axes[1, 0].set_title(f"{run_name} validation loss")
        axes[1, 0].set_xlabel("epoch")

    axes[1, 1].plot(epoch_rows["epoch"], epoch_rows["task_loss"], marker=".")
    axes[1, 1].set_title(f"{run_name} task loss by epoch")
    axes[1, 1].set_xlabel("epoch")
    return fig


def plot_threshold_sweep(sweep_df: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    ax.plot(sweep_df["threshold"], sweep_df["exact_match"], label="EM")
    ax.set_xlabel("threshold")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_confidence_distribution(rows: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    possible = rows[~rows["is_impossible"]]
    impossible = rows[rows["is_impossible"]]
    sns.kdeplot(possible["confidence"], ax=ax, label="answerable", fill=True, alpha=0.3)
    sns.kdeplot(impossible["confidence"], ax=ax, label="unanswerable", fill=True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("confidence = answer score - no answer score")
    ax.legend()
    return fig


def plot_tradeoff(summary_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(summary_df["emissions_kgco2eq"], summary_df["squad_f1"], s=120)
    for _, row in summary_df.iterrows():
        ax.annotate(
            row["run_name"],
            (row["emissions_kgco2eq"], row["squad_f1"]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_xlabel("kg CO2eq")
    ax.set_ylabel("SQuAD v2 F1")
    ax.set_title("Carbon vs performance tradeoff")
    return fig


def plot_subject_bars(mmlu_summary: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(12, 7))
    plot_df = mmlu_summary.melt(
        id_vars="subject",
        value_vars=["accuracy", "f1_macro", "precision_macro", "recall_macro"],
        var_name="metric",
        value_name="value",
    )
    sns.barplot(data=plot_df, x="subject", y="value", hue="metric", ax=ax)
    ax.set_title("MMLU subject metrics")
    ax.set_ylabel("score")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_pr_roc(df: pd.DataFrame, subject: str):
    labels = ["A", "B", "C", "D"]
    y_true = df["true_label"].tolist()
    y_score = df[[f"prob_{l}" for l in labels]].to_numpy()
    true_idx = np.array([labels.index(y) for y in y_true])
    y_true_bin = label_binarize(true_idx, classes=list(range(4)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for idx, label in enumerate(labels):
        if len(np.unique(y_true_bin[:, idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true_bin[:, idx], y_score[:, idx])
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        axes[0].plot(recall, precision, label=label)
        axes[1].plot(fpr, tpr, label=label)
    axes[0].set_title(f"{subject} PR curves")
    axes[1].set_title(f"{subject} ROC curves")
    for ax, xlabel, ylabel in [
        (axes[0], "recall", "precision"),
        (axes[1], "false positive rate", "true positive rate"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
    return fig


def plot_confusion(df: pd.DataFrame, subject: str):
    from sklearn.metrics import confusion_matrix
    labels = ["A", "B", "C", "D"]
    cm = confusion_matrix(df["true_label"], df["pred_label"], labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"{subject} confusion matrix")
    return fig


def plot_emissions_proxy(run_summaries: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=run_summaries, x="run_name", y="emissions_kgco2eq", ax=ax)
    ax.set_title("Stage emissions")
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_emissions_over_time(run_summaries: pd.DataFrame):
    df = run_summaries.copy().reset_index(drop=True)
    df["elapsed_hours"] = df["duration_s"].fillna(0).cumsum() / 3600.0
    df["cumulative_emissions"] = df["emissions_kgco2eq"].fillna(0).cumsum()
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.step(df["elapsed_hours"], df["cumulative_emissions"], where="post", linewidth=2)
    ax.scatter(df["elapsed_hours"], df["cumulative_emissions"], s=60)
    for _, row in df.iterrows():
        ax.annotate(
            row["run_name"],
            (row["elapsed_hours"], row["cumulative_emissions"]),
            textcoords="offset points",
            xytext=(5, 5),
        )
    ax.set_xlabel("elapsed hours")
    ax.set_ylabel("cumulative kg CO2eq")
    ax.set_title("Cumulative CodeCarbon emissions over time")
    return fig


def plot_summary_grid(summary_df: pd.DataFrame, mmlu_summary: pd.DataFrame, sweep_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes[0, 0].plot(summary_df["run_name"], summary_df["squad_f1"], marker="o")
    axes[0, 0].set_title("SQuAD F1 by run")
    axes[0, 0].tick_params(axis="x", rotation=20)

    axes[0, 1].scatter(summary_df["emissions_kgco2eq"], summary_df["squad_f1"], s=100)
    axes[0, 1].set_title("Carbon tradeoff")
    axes[0, 1].set_xlabel("kg CO2eq")
    axes[0, 1].set_ylabel("SQuAD F1")

    axes[1, 0].plot(sweep_df["threshold"], sweep_df["f1"], label="F1")
    axes[1, 0].plot(sweep_df["threshold"], sweep_df["exact_match"], label="EM")
    axes[1, 0].set_title("Threshold sweep")
    axes[1, 0].legend()

    sns.barplot(data=mmlu_summary, x="subject", y="accuracy", ax=axes[1, 1])
    axes[1, 1].set_title("MMLU accuracy")
    axes[1, 1].tick_params(axis="x", rotation=20)
    return fig
