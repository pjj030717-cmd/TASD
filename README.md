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
    draft_model=draft_model,         # e.g., Qwen2.5-1.5B-Instruct
    tokenizer=tokenizer,
    prompt="train_pipeline = [",
    structure_type="pipeline_stage_config",
    max_new_tokens=128,
    temperature=0.0,
    draft_len=16,
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
| `draft_len` | 16 | Tokens per draft block |
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
# Main 4-method comparison (6 benchmarks × 80 samples)
python run_4method_comparison.py   # AR + Greedy SD + FLY + TASD

# N-gram SpecDec (6 benchmarks × 80 samples)
python run_ngram_480.py

# N-gram pilot (3 benchmarks × 20 samples)
python run_ngram_pilot.py

# FLY pilot
python run_fly_pilot.py

# Ablation studies
python run_ablation.py

# Draft model comparison (1.5B vs 3B)
python run_draft_1_5b_eval.py
```

## Baselines

- **AR**: Standard autoregressive greedy decoding
- **Greedy SD**: Deterministic argmax-matching speculative decoding baseline (k=16, no relaxation, no guard)
- **FLY**: N-gram draft + draft model + window acceptance (k=15, use_ngram=True, max_ngram_size=4, num_ngram_pred_tokens=6, win_len=6, entropy_thre=0.3)
- **N-gram SD**: Training-free speculative decoding using prompt n-gram matching (n=3-8, draft up to 16 tokens, zero model overhead) — included as 3×20 diagnostic pilot

## Results Summary

### Main 4-Method Result (6 benchmarks × 80 samples)

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Settings**: temperature=0.0, max_new_tokens=128, TASD: draft_blocks=2, draft_len=16, top_k_accept=3

Full report: [results/comparison_4method_80.md](results/comparison_4method_80.md)

| Method | TPS | Speedup | SQ |
|--------|-----|---------|----|
| AR | 33.2 | 1.00x | 0.8910 |
| Greedy SD | 22.0 | 0.66x | 0.8612 |
| FLY | 54.5 | 1.64x | 0.8895 |
| TASD | 64.2 | 1.93x | 0.8825 |

TASD achieves 1.93x average speedup over AR, with per-benchmark range 1.80x–2.01x. Structural quality (SQ) is comparable to AR (0.8825 vs 0.8910).

### N-gram Diagnostic Pilot (3 benchmarks × 20 samples)

Full report: [results/comparison_5method_ngram_pilot.md](results/comparison_5method_ngram_pilot.md)

| Method | TPS | Speedup | SQ |
|--------|-----|---------|----|
| AR | 33.5 | 1.00x | 0.8915 |
| Greedy SD | 19.8 | 0.59x | 0.8370 |
| N-gram SD | 49.0 | 1.46x | 0.8512 |
| FLY | 43.1 | 1.29x | 0.8823 |
| TASD | 61.4 | 1.83x | 0.8999 |

N-gram SD (zero model overhead) achieves 1.46x but with lower SQ. TASD maintains 1.83x with SQ on par with AR.

### Per-Benchmark Breakdown (Main 4-Method, 6×80)

| Benchmark | AR | Greedy SD | FLY | TASD |
|-----------|-----|-----------|-----|------|
| Real-Python-Argparse | 33.2 | 31.4 | 63.9 | 61.9 |
| Real-Python-DictConfig | 33.3 | 21.8 | 58.7 | 60.0 |
| OpenMMLab-Config | 33.2 | 22.1 | 35.1 | 64.0 |
| Rich-CLI-Option-Groups | 32.9 | 20.5 | 69.5 | 66.1 |
| Complex-Nested-Config | 33.3 | 18.8 | 57.9 | 66.4 |
| Pipeline-Stage-Config | 33.4 | 17.5 | 41.7 | 66.7 |

### Archive: 3B Draft Model Results (Deprecated)

The following results used Qwen2.5-3B-Instruct as draft model with draft_len=8. They are retained for historical reference but superseded by the 1.5B draft results above:

| Benchmark | AR TPS | Greedy SD | TASD (3B) | TASD Speedup |
|-----------|--------|-----------|-----------|-------------|
| Real-Python-Argparse | 32.98 | 0.82x | 42.92 | 1.30x |
| Real-Python-DictConfig | 32.67 | 0.87x | 42.62 | 1.30x |
| OpenMMLab-Config | 32.91 | 0.80x | 47.34 | 1.44x |
| Rich-CLI-Option-Groups | 33.14 | 0.83x | 49.12 | 1.48x |
| Complex-Nested-Config | 32.71 | 0.85x | 48.23 | 1.47x |
| Pipeline-Stage-Config | 32.24 | 0.80x | 49.36 | 1.53x |

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
