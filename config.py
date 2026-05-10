"""
config.py
---------
Central configuration dataclass.  Every hyper-parameter is CLI-tunable.

model_zoo.json format
---------------------
    "llama_1b": {
        "model_name": "meta-llama/Llama-3.2-1B",
        "flag": true
    }

Output convention
-----------------
    All results are written to ``<model_alias>_outputs/``  (e.g. ``llama_1b_outputs/``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ---------------------------------------------------------------------------
# Locate model_zoo.json (sits next to this file)
# ---------------------------------------------------------------------------
_PKG_DIR = Path(__file__).resolve().parent
MODEL_ZOO_PATH = _PKG_DIR / "model_zoo.json"


def load_model_zoo(path: Path = MODEL_ZOO_PATH) -> dict:
    """Return the full model-zoo registry as a dict."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_model(alias: str, zoo_path: Path = MODEL_ZOO_PATH) -> dict:
    """
    Look up *alias* in model_zoo.json.

    Returns
    -------
    dict  with keys ``model_name`` (HF id) and ``flag`` (bool).

    Raises
    ------
    ValueError  if the alias is unknown or its ``flag`` is ``false``.
    """
    zoo = load_model_zoo(zoo_path)
    if alias not in zoo:
        available = ", ".join(sorted(zoo.keys()))
        raise ValueError(
            f"Unknown model alias '{alias}'. "
            f"Available aliases:\n  {available}"
        )
    entry = zoo[alias]
    if not entry.get("flag", False):
        raise ValueError(
            f"Model '{alias}' exists in model_zoo.json but its flag is set to false. "
            f"Set flag to true to enable it."
        )
    return entry


# ---------------------------------------------------------------------------
# LoRA target modules — sensible defaults per known family
# ---------------------------------------------------------------------------
_DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ---------------------------------------------------------------------------
# Main config dataclass
# ---------------------------------------------------------------------------
@dataclass
class Config:
    # ── Model ────────────────────────────────────────────────────────────────
    model_alias: str = "qwen_1b"
    """Short alias from model_zoo.json (used as the primary CLI argument)."""

    # Resolved at runtime; do not set manually.
    base_model: str = ""
    lora_target_modules: List[str] = field(default_factory=lambda: list(_DEFAULT_LORA_TARGETS))
    trust_remote_code: bool = True
    use_fast_tokenizer: bool = True

    # ── Datasets ─────────────────────────────────────────────────────────────
    squad_dataset: str = "rajpurkar/squad_v2"
    mmlu_dataset: str = "cais/mmlu"
    mmlu_subjects: List[str] = field(
        default_factory=lambda: ["abstract_algebra", "philosophy", "formal_logic"]
    )

    # ── Tokenisation / sequence lengths ──────────────────────────────────────
    max_source_len: int = 512
    max_context_tokens: int = 384
    max_new_tokens: int = 32
    use_kv_cache: bool = True
    kv_cache_energy_reduction: float = 0.1636

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # ── Training ─────────────────────────────────────────────────────────────
    train_batch_size: int = 1
    eval_batch_size: int = 2
    grad_accum: int = 16
    epochs: int = 1
    lr: float = 2e-4
    weight_decay: float = 0.0

    # ── Green-loss hyper-parameters ──────────────────────────────────────────
    lambda_weight: float = 0.01
    mu: float = 1e-4
    lambda_grid: List[float] = field(default_factory=lambda: [0.01, 0.03, 0.1])

    # ── Evaluation ───────────────────────────────────────────────────────────
    val_threshold_steps: int = 51
    no_answer_text: str = "no answer"
    f1_tolerance: float = 2.0

    # ── Fast-dev / smoke-test ────────────────────────────────────────────────
    fast_dev_run: bool = False
    fast_dev_squad_train: int = 256
    fast_dev_squad_val: int = 128
    fast_dev_mmlu: int = 20

    # ── Misc ─────────────────────────────────────────────────────────────────
    seed: int = 42
    country_iso_code: str = "USA"

    # ── Derived paths (read-only after __post_init__) ────────────────────────
    out_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    carbon_dir: Path = field(init=False)
    fig_dir: Path = field(init=False)
    metrics_dir: Path = field(init=False)
    log_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        # Resolve model from zoo
        if self.model_alias:
            entry = resolve_model(self.model_alias)
            if not self.base_model:
                self.base_model = entry["model_name"]
            # trust_remote_code: Qwen needs it, most others don't
            if "qwen" in self.base_model.lower():
                self.trust_remote_code = True
            else:
                self.trust_remote_code = False

        # Output directory: <model_alias>_outputs/
        self.out_dir = Path(f"{self.model_alias}_outputs")
        self.checkpoint_dir = self.out_dir / "checkpoints"
        self.carbon_dir = self.out_dir / "carbon"
        self.fig_dir = self.out_dir / "figures"
        self.metrics_dir = self.out_dir / "metrics"
        self.log_dir = self.out_dir / "logs"

        for d in [
            self.out_dir,
            self.checkpoint_dir,
            self.carbon_dir,
            self.fig_dir,
            self.metrics_dir,
            self.log_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    @classmethod
    def from_cli(cls, args) -> "Config":
        """Build a Config from an argparse Namespace."""
        kwargs: dict = {}
        if hasattr(args, "model") and args.model:
            kwargs["model_alias"] = args.model
        # Map every CLI flag that exists on the namespace
        for key in [
            "epochs", "lr", "lora_r", "lora_alpha", "lora_dropout",
            "train_batch_size", "eval_batch_size", "grad_accum",
            "mu", "seed", "fast_dev_run", "weight_decay",
            "max_source_len", "max_context_tokens", "max_new_tokens",
            "use_kv_cache", "kv_cache_energy_reduction", "lambda_weight",
            "country_iso_code", "val_threshold_steps", "f1_tolerance",
            "no_answer_text", "fast_dev_squad_train", "fast_dev_squad_val",
            "fast_dev_mmlu",
        ]:
            if hasattr(args, key) and getattr(args, key) is not None:
                kwargs[key] = getattr(args, key)
        return cls(**kwargs)
