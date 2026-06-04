# TASD: Training-free Structure-Aware Speculative Decoding

Code completion with speculative decoding that preserves structural integrity without training.

## Algorithm

TASD accelerates autoregressive code generation by drafting multi-token blocks with a lightweight draft model and verifying them in a single forward pass with the target model. A structural guard ensures generated code remains well-formed.

```
TASD(target_model, draft_model, tokenizer, prompt):
    1. Pre-fill KV cache for prompt on both models
    2. While tokens < max_new_tokens:
       a. Draft: generate draft_blocks x draft_len tokens via draft model
          - Incremental forward with KV cache
          - Per-token early stop (EOS, off-structure)
       b. Verify: single target forward on all draft tokens
       c. Accept: prefix match + top_k relaxation + prefix_budget
          - Strict prefix (token-by-token argmax match)
          - Window acceptance (loose matching in sliding windows)
          - Prefix budget (risk-controlled probabilistic acceptance)
       d. Guard: structural check → trim to safe prefix
          - Off-structure: def/class/import detection
          - Repetition: consecutive identical token detection
          - Unbalanced brackets: {[( mismatch check
          - Bad tail: trailing commas, unclosed delimiters
       e. Apply: extend generated tokens, trim KV caches
       f. Repair: if nothing accepted, fall back to AR for 1 token
```

### Key Differences from Standard Speculative Decoding

| Feature | Standard SD | TASD |
|---------|------------|------|
| Draft blocks | 1 | `draft_blocks=2` (multi-block) |
| Acceptance | Strict argmax only | Top-k relaxation + prefix budget |
| Structural safety | None | Token-level structural guard |
| KV cache | Discard/recompute | Trim to accepted length |
| Repair | Always 1 AR token | Degrades to full AR after 5 consecutive repairs |

## Installation

```bash
pip install torch transformers
```

## Usage

```python
from src.tasd_decode import tasd_decode

result = tasd_decode(
    target_model=target_model,      # e.g., Qwen2.5-14B-Instruct-AWQ
    draft_model=draft_model,         # e.g., Qwen2.5-3B-Instruct
    tokenizer=tokenizer,
    prompt="train_pipeline = [",
    structure_type="pipeline_stage_config",
    max_new_tokens=128,
    temperature=0.0,
    draft_len=8,
    draft_blocks=2,
    top_k_accept=3,
    min_token_prob=1e-4,
    prefix_budget=0.2,
    window_len=2,
)
# result["generated_text"], result["tokens_per_second"]
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Maximum tokens to generate |
| `temperature` | 0.0 | Sampling temperature (greedy when 0) |
| `draft_len` | 8 | Tokens per draft block |
| `draft_blocks` | 2 | Number of draft blocks per round |
| `top_k_accept` | 3 | Relaxed acceptance: accept draft token if in top-k |
| `min_token_prob` | 1e-4 | Accept draft token if probability above threshold |
| `prefix_budget` | 0.2 | Risk budget for probabilistic prefix acceptance |
| `window_len` | 2 | Sliding window size for loose matching |
| `structure_type` | "argparse" | One of: argparse, dict_config, openmmlab_config, pipeline_stage_config, complex_nested_config, rich_cli_option_groups |

## Structure Types

TASD is designed for structured code completion with repeated skeletons:

| Structure Type | Example | Guard Rules |
|---------------|---------|-------------|
| `argparse` | `parser.add_argument("--lr", type=float, ...)` | Duplicate --option, off-structure |
| `dict_config` | `config = {"lr": 0.01, "batch_size": 32, ...}` | Bracket balance, off-structure |
| `openmmlab_config` | `model = dict(type="FasterRCNN", ...)` | Config pattern preservation |
| `pipeline_stage_config` | `train_pipeline = [dict(type="LoadImage"), ...]` | Stage type/name patterns |
| `complex_nested_config` | Nested dicts with 2+ levels | Deep bracket balance |
| `rich_cli_option_groups` | Multi-option CLI with choices/default/help | Duplicate option, rich fields |

## Evaluation

Benchmark scripts:

```bash
# Full experiment (AR + Greedy SD + TASD on 80 samples)
python run_kv_cache_exp.py --benchmarks argparse dict_config openmmlab --sample-limit 80

# Generate benchmark samples (protocol-compliant)
python generate_benchmarks.py

# GSD-only (supplementary baseline)
python run_gsd_only.py

# Structure coverage scan
python final_structure_coverage_scan.py
```

## Baselines

- **AR**: Standard autoregressive greedy decoding
- **Greedy SD**: Standard speculative decoding with strict argmax matching (no relaxation, no guard)

## Results Summary

Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-3B-Instruct (draft), max_new_tokens=128, n=80:

| Benchmark | AR TPS | Greedy SD | TASD | TASD Speedup |
|-----------|--------|-----------|------|-------------|
| Real-Python-Argparse | 32.98 | 0.82x | 42.92 | **1.30x** |
| Real-Python-DictConfig | 32.67 | 0.87x | 42.62 | **1.30x** |
| OpenMMLab-Config | 32.91 | 0.80x | 47.34 | **1.44x** |
| Rich-CLI-Option-Groups | 33.14 | 0.83x | 49.12 | **1.48x** |
| Complex-Nested-Config | 32.71 | 0.85x | 48.23 | **1.47x** |
| Pipeline-Stage-Config | 32.24 | 0.80x | 49.36 | **1.53x** |

TASD achieves 1.30x-1.53x speedup over AR across 6 benchmark categories while preserving structural quality on par with Greedy SD. See [benchmark coverage report](results/benchmark_coverage_section_draft.md) and [selection protocol](results/benchmark_selection_protocol.md) for details.

## Citation

```bibtex
@misc{tasd2025,
  title={TASD: Training-free Structure-Aware Speculative Decoding},
  author={},
  year={2025},
}
```

## License

MIT
