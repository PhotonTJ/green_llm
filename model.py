"""
model.py - Model construction and LoRA-related helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

from green_llm.config import Config
from green_llm.utils import get_device, get_torch_dtype


def _model_config(model) -> Any:
    return getattr(model, "config", getattr(getattr(model, "base_model", None), "config", None))


def _config_value(config: Any, names: List[str], default: float) -> float:
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return float(value)
    return float(default)


def transformer_dimension_summary(model) -> Dict[str, float]:
    """Return transformer dimensions used by the carbon surrogate."""
    config = _model_config(model)
    hidden = _config_value(config, ["hidden_size", "d_model", "n_embd"], 2048.0)
    layers = _config_value(config, ["num_hidden_layers", "n_layer", "num_layers"], 24.0)
    heads = max(1.0, _config_value(config, ["num_attention_heads", "n_head"], 16.0))
    kv_heads = _config_value(config, ["num_key_value_heads", "num_kv_heads"], heads)
    head_dim = _config_value(config, ["head_dim"], hidden / heads)
    intermediate = _config_value(config, ["intermediate_size", "ffn_dim"], 4.0 * hidden)
    return {
        "hidden_size": hidden,
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate,
    }


def set_kv_cache(model, enabled: bool) -> Dict[int, tuple[Any, bool]]:
    """Set use_cache on visible model configs and return previous values."""
    previous: Dict[int, tuple[Any, bool]] = {}
    candidates = [model, getattr(model, "base_model", None), getattr(model, "model", None)]
    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is None or not hasattr(config, "use_cache"):
            continue
        key = id(config)
        if key in previous:
            continue
        previous[key] = (config, bool(config.use_cache))
        config.use_cache = bool(enabled)
    return previous


@contextmanager
def kv_cache_context(model, enabled: bool) -> Iterator[None]:
    previous = set_kv_cache(model, enabled)
    try:
        yield
    finally:
        for config, old_value in previous.values():
            config.use_cache = old_value


def build_lora_model(
    cfg: Config,
    device: Optional[torch.device] = None,
    for_training: bool = True,
):
    if device is None:
        device = get_device()
    dtype = get_torch_dtype()

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
        trust_remote_code=cfg.trust_remote_code,
    )
    model.config.use_cache = bool(getattr(cfg, "use_kv_cache", True)) and not for_training
    if for_training:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    return model


def count_trainable_parameters(model) -> Dict[str, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total}


def trainable_frobenius_norm(model) -> torch.Tensor:
    device = next(model.parameters()).device
    terms = [p.float().pow(2).sum() for p in model.parameters() if p.requires_grad]
    return torch.stack(terms).sum() if terms else torch.tensor(0.0, device=device)


def lora_reg_loss(model) -> torch.Tensor:
    return trainable_frobenius_norm(model)


def _sequence_stats(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    batch_size, seq_len = input_ids.shape
    if attention_mask is None:
        lengths = torch.full(
            (batch_size,), float(seq_len), device=input_ids.device, dtype=torch.float32
        )
    else:
        lengths = attention_mask.float().sum(dim=1).clamp_min(1.0)
    return {
        "batch_size": torch.tensor(float(batch_size), device=input_ids.device),
        "avg_seq_len": lengths.mean().clamp_min(1.0),
        "token_count": lengths.sum().clamp_min(1.0),
    }


def carbon_proxy_terms(
    model,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    dims = transformer_dimension_summary(model)
    stats = _sequence_stats(input_ids, attention_mask)
    device = input_ids.device

    bsz = stats["batch_size"]
    seq_len = stats["avg_seq_len"]
    token_count = stats["token_count"]
    layers = torch.tensor(dims["num_layers"], device=device)
    heads = torch.tensor(dims["num_attention_heads"], device=device)
    kv_heads = torch.tensor(dims["num_key_value_heads"], device=device)
    head_dim = torch.tensor(dims["head_dim"], device=device)
    hidden = torch.tensor(dims["hidden_size"], device=device)
    intermediate = torch.tensor(dims["intermediate_size"], device=device)

    param_norm = trainable_frobenius_norm(model)

    # Attention follows k1*T^2*H*d_qkv + k2*T*H*d_qkv from the theory notes.
    attention_quadratic_proxy = bsz * layers * seq_len.pow(2) * heads * head_dim / 1e8
    attention_linear_proxy = bsz * layers * seq_len * heads * head_dim / 1e8

    # SwiGLU-style decoder blocks typically use gate/up/down projections.
    ffn_proxy = bsz * layers * seq_len * 3.0 * hidden * intermediate / 1e8
    normalization_proxy = bsz * layers * seq_len * hidden / 1e8
    flops_proxy = (
        attention_quadratic_proxy
        + attention_linear_proxy
        + ffn_proxy
        + normalization_proxy
    )

    activation_memory_proxy = token_count * layers * hidden / 1e8
    kv_cache_proxy = token_count * layers * 2.0 * kv_heads * head_dim / 1e8
    memory_proxy = activation_memory_proxy + kv_cache_proxy
    network_proxy = token_count * hidden / 1e8

    return {
        "param_norm": param_norm,
        "flops_proxy": flops_proxy,
        "memory_proxy": memory_proxy,
        "attention_quadratic_proxy": attention_quadratic_proxy,
        "attention_linear_proxy": attention_linear_proxy,
        "ffn_proxy": ffn_proxy,
        "normalization_proxy": normalization_proxy,
        "activation_memory_proxy": activation_memory_proxy,
        "kv_cache_proxy": kv_cache_proxy,
        "network_proxy": network_proxy,
        "avg_seq_len": stats["avg_seq_len"],
        "token_count": stats["token_count"],
        "num_layers": layers,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "hidden_size": hidden,
    }


def surrogate_carbon_loss(
    model,
    input_ids: torch.Tensor,
    weights: List[float],
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    terms = carbon_proxy_terms(model, input_ids, attention_mask)
    feature_names = ["param_norm", "flops_proxy", "memory_proxy", "network_proxy"]
    if len(weights) > len(feature_names):
        raise ValueError(f"Expected at most {len(feature_names)} carbon weights, got {len(weights)}")
    loss = torch.tensor(0.0, device=input_ids.device, dtype=torch.float32)
    for name, weight in zip(feature_names, weights):
        loss = loss + torch.tensor(float(weight), device=input_ids.device) * terms[name]
    return loss


def compute_task_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss


def batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    if device is None:
        device = get_device()
    return {k: v.to(device) for k, v in batch.items()}
