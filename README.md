# Green-LLM: Carbon-Aware LoRA Fine-Tuning

A modular PyTorch pipeline for **carbon-aware LoRA fine-tuning** of causal language models on SQuAD v2 with zero-shot MMLU evaluation. Emissions are tracked end-to-end via [CodeCarbon](https://codecarbon.io/).

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r green_llm/requirements.txt
```

### 2. Set up HuggingFace access (for gated models)

Copy `green_llm/setup.env.example` to `green_llm/setup.env`, then paste your HuggingFace token:

```bash
# Linux / macOS
cp green_llm/setup.env.example green_llm/setup.env
source green_llm/setup.env

# Windows PowerShell
Copy-Item green_llm\setup.env.example green_llm\setup.env
Get-Content green_llm\setup.env | ForEach-Object {
  if ($_ -match '^export\s+(\w+)=(.*)') {
    [System.Environment]::SetEnvironmentVariable($Matches[1], $Matches[2].Trim('"'))
  }
}
```

### 3. List available models

```bash
python -m green_llm --list_models
```

```
Alias                     HuggingFace Model ID                               Enabled
-------------------------------------------------------------------------------------
deepseek_r1_llama_8b      deepseek-ai/DeepSeek-R1-Distill-Llama-8B           ✓
gemma_2b                  google/gemma-3-2b                                  ✓
gemma_4b                  google/gemma-3-4b                                  ✓
llama_1b                  meta-llama/Llama-3.2-1B                            ✓
llama_8b                  meta-llama/Meta-Llama-3.1-8B                       ✓
mistral_3b                mistralai/Mistral-3B-v0.2                          ✓
mistral_7b                mistralai/Mistral-7B-v0.3                          ✓
qwen_1b                   Qwen/Qwen2.5-1.5B                                 ✓
qwen_7b                   Qwen/Qwen2.5-7B                                   ✓
qwen_14b                  Qwen/Qwen2.5-14B                                  ✗
```

---

## Commands

### Training only

```bash
python -m green_llm train --model llama_1b
python -m green_llm train --model mistral_7b --loss_mode joint --epochs 2
```

### Inference only

```bash
python -m green_llm infer --model llama_1b
python -m green_llm infer --model llama_1b --skip_mmlu
```

### Full pipeline (calibration → train → infer → plots)

```bash
python -m green_llm pipeline --model qwen_1b --loss_mode joint
python -m green_llm pipeline --model llama_8b --loss_mode joint --epochs 3
```

### Smoke test (fast dev run)

```bash
python -m green_llm pipeline --model qwen_1b --fast_dev_run
```

---

## Output Directory

All outputs are saved to **`<model_alias>_outputs/`**, e.g.:

```
llama_1b_outputs/
├── checkpoints/          # LoRA adapter weights
├── carbon/               # CodeCarbon emissions.csv per run
├── figures/              # PNG plots
├── metrics/              # CSV / JSON summaries
└── logs/                 # Carbon summary JSON per stage
```

---

## Model Zoo (`model_zoo.json`)

```json
{
  "llama_1b": {
    "model_name": "meta-llama/Llama-3.2-1B",
    "flag": true
  }
}
```

- **`model_name`** — HuggingFace model ID
- **`flag`** — `true` to enable, `false` to disable

Add a new model by adding an entry and using its alias:

```bash
python -m green_llm pipeline --model my_custom_model
```

---

## All CLI Arguments

Every config field is tunable from the command line:

| Argument | Type | Default | Description |
|---|---|---|---|
| `--model` | str | `qwen_1b` | Model alias from `model_zoo.json` |
| `--loss_mode` | str | `ce` | `ce` (cross-entropy) or `joint` (carbon-aware) |
| `--epochs` | int | `1` | Number of training epochs |
| `--lr` | float | `2e-4` | Learning rate |
| `--weight_decay` | float | `0.0` | Weight decay |
| `--train_batch_size` | int | `1` | Training batch size |
| `--eval_batch_size` | int | `2` | Evaluation batch size |
| `--grad_accum` | int | `16` | Gradient accumulation steps |
| `--lora_r` | int | `16` | LoRA rank |
| `--lora_alpha` | int | `32` | LoRA alpha |
| `--lora_dropout` | float | `0.05` | LoRA dropout |
| `--max_source_len` | int | `512` | Max input sequence length |
| `--max_context_tokens` | int | `384` | Max context window tokens |
| `--max_new_tokens` | int | `32` | Max tokens to generate |
| `--use_kv_cache` / `--no-use_kv_cache` | flag | on | Enable/disable KV cache during generation |
| `--kv_cache_energy_reduction` | float | `0.1636` | Expected fractional inference-energy reduction from KV cache |
| `--lambda_weight` | float | `0.01` | Carbon surrogate weight for direct `train --loss_mode joint` runs |
| `--mu` | float | `1e-4` | LoRA regularisation weight (μ) |
| `--f1_tolerance` | float | `2.0` | F1 tolerance for λ selection |
| `--val_threshold_steps` | int | `51` | No-answer threshold sweep steps |
| `--no_answer_text` | str | `no answer` | Unanswerable text label |
| `--seed` | int | `42` | Random seed |
| `--country_iso_code` | str | `USA` | ISO code for CodeCarbon |
| `--fast_dev_run` | flag | off | Enable smoke-test mode |
| `--fast_dev_squad_train` | int | `256` | Train samples in fast-dev |
| `--fast_dev_squad_val` | int | `128` | Val samples in fast-dev |
| `--fast_dev_mmlu` | int | `20` | MMLU samples in fast-dev |
| `--skip_calibration` | flag | off | Skip surrogate-weight calibration |
| `--skip_mmlu` | flag | off | Skip MMLU evaluation |

---

## Repository Structure

```
green_llm/
├── __init__.py
├── __main__.py          # python -m green_llm entry
├── train.py             # CLI: train / infer / pipeline subcommands
├── config.py            # Config dataclass — resolves model aliases
├── model_zoo.json       # Model alias registry
├── setup.env            # HuggingFace auto-login
├── requirements.txt
├── README.md            # This file
│
├── utils.py             # Seeding, prompts, SQuAD scoring, I/O
├── data.py              # Dataset loading, encoding, DataLoader
├── model.py             # LoRA construction, carbon proxy terms
├── carbon.py            # CodeCarbon wrappers, calibration
├── trainer.py           # CE & joint training loops
├── evaluate.py          # SQuAD generation, MMLU scoring
└── plots.py             # All diagnostic visualisations
```

---

## Mathematical Background

### Joint Loss

$$L_{\text{joint}}(\theta) = L_{\text{task}}(\theta) + \lambda \cdot \tilde{C}(\theta, x) + \mu \cdot \|\theta\|_F^2$$

### Differentiable Carbon Surrogate

$$\tilde{C}(\theta, x) = \omega_1 \|\theta\|_F^2 + \omega_2 \cdot \text{FLOPs}(x) + \omega_3 \cdot \text{memory}(x)$$

Weights $\omega_1, \omega_2, \omega_3$ are fitted via non-negative linear regression from live CodeCarbon calibration runs at context lengths 128 / 256 / 384.

### Transformer Proxy Details

The implementation computes FLOPs and memory proxies from the model architecture:

$$\text{Attention}(T) = \kappa_1 T^2 H d_{qkv} + \kappa_2 T H d_{qkv}$$

The total FLOPs proxy also includes FFN and normalisation terms, while the memory proxy includes activation memory plus KV-cache memory using the number of KV heads. The parameter term is the true trainable Frobenius norm, so its gradient is proportional to $2\theta$ as required by the theory.

Calibration also fits the input-length carbon curve:

$$C(T) = c_0 + c_1T + c_2T^2$$

The fitted coefficients are saved under `input_length_carbon` in `surrogate_weights.json`.

### KV Cache and Threshold Scope

KV cache is enabled for generation by default and disabled during training to keep gradient checkpointing stable. The existing threshold sweep is a SQuAD no-answer confidence threshold; true sparse QKV-threshold attention would require model-specific attention-kernel changes and is not treated as implemented by the generic HuggingFace wrapper.

### λ Selection

Grid search over `[0.01, 0.03, 0.1]`. Only λ values whose validation F1 is within `--f1_tolerance` of the CE baseline are eligible; the one with lowest emissions is selected.

---

## Citation

If you use this codebase, please cite the associated paper.
