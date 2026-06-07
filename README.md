# TASD: Training-free Structure-Aware Speculative Decoding

Code completion with speculative decoding that preserves structural integrity without training.

## Algorithm

TASD (Training-free Structure-Aware Speculative Decoding) accelerates autoregressive (AR) code generation by using a lightweight draft model to generate multi-token blocks, which are then verified in a single forward pass by the target model. A structural guard ensures generated code remains well-formed.

### Complete Algorithm Flow

```
TASD(target_model, draft_model, tokenizer, prompt, structure_type, max_new_tokens, ...):

    1. PRE-FILL PHASE
       - Run prompt through target model → obtain target KV cache
       - Run prompt through draft model → obtain draft KV cache
       - Initialize: generated_ids = [], round = 0, total_forwards = 0

    2. MAIN LOOP (while len(generated_ids) < max_new_tokens):

       a. DRAFT GENERATION (draft_blocks × draft_len tokens)
          For each block b in range(draft_blocks):
            - Draft model generates draft_len tokens via incremental forward
            - Each token: forward(last_token, past_key_values) → logits → argmax
            - Early stop per token: EOS, off-structure keyword, repetition
            - Update draft KV cache incrementally
          Result: draft_tokens = [t_1, t_2, ..., t_N] where N = draft_blocks × draft_len

       b. VERIFICATION (single target forward pass)
          - Concatenate all draft_tokens, run single target forward
          - target_logits = target_model(draft_tokens, target_past)
          - target_argmax = argmax(target_logits) for each position

       c. ACCEPTANCE (three-tier strategy)
          Tier 1 - Strict Prefix Match:
            Accept tokens where draft_token[i] == target_argmax[i] sequentially
          
          Tier 2 - Top-k Relaxed Acceptance (if top_k_accept > 0):
            For next token after strict prefix, accept if:
              - draft_token in top-k of target distribution, AND
              - P(draft_token) >= min_token_prob
            Continue for up to window_len tokens in sliding window

          Tier 3 - Prefix Budget (if prefix_budget > 0):
            Risk-controlled probabilistic acceptance:
              - budget = prefix_budget × accepted_count
              - Accept additional tokens if P(draft_token) >= budget_threshold
              - Budget depletes with each relaxed acceptance

       d. STRUCTURAL GUARD (if enable_guard=True)
          Check accepted prefix for structural violations:
            - Off-structure: detect def/class/import keywords in non-code contexts
            - Repetition: detect consecutive identical token sequences
            - Unbalanced brackets: track {[( vs )]} balance
            - Bad tail: detect trailing commas, unclosed delimiters
          If violation found → trim accepted prefix to last safe position

       e. APPLY ACCEPTED TOKENS
          - Append accepted tokens to generated_ids
          - Trim target/draft KV caches to accepted length
          - Update last_target_logit, last_draft_logit for next round

       f. REPAIR (if zero tokens accepted)
          - Fall back to target AR for 1 token
          - If 5 consecutive repairs → degrade to full AR mode
          - Re-prefill KV caches with full sequence

    3. TERMINATION
       - Stop when max_new_tokens reached or EOS generated
       - Return: generated_text, tokens_per_second, detailed stats
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
    enable_guard=True,
    enable_relaxed_accept=True,
)
# result["generated_text"], result["tokens_per_second"]
```

## Parameters

### Core Decoding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_new_tokens` | 128 | Maximum tokens to generate |
| `temperature` | 0.0 | Sampling temperature (greedy when 0) |
| `draft_len` | 16 | Tokens per draft block |
| `draft_blocks` | 2 | Number of draft blocks per round (total draft tokens = draft_blocks × draft_len) |
| `top_k_accept` | 3 | Relaxed acceptance: accept draft token if in top-k of target distribution |
| `min_token_prob` | 1e-4 | Minimum probability threshold for relaxed acceptance |
| `prefix_budget` | 0.2 | Risk budget for probabilistic prefix acceptance (fraction of accepted count) |
| `window_len` | 2 | Sliding window size for loose matching |
| `structure_type` | "argparse" | Structure type for guard rules (see below) |

### Guard and Extension Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_guard` | `True` | Enable structural guard (off-structure, repetition, bracket balance, bad tail) |
| `enable_relaxed_accept` | `True` | Enable top-k relaxed acceptance (strict-only if False) |
| `enable_failure_aware_fallback` | `False` | Enable TASD-F runtime fallback (see below) |
| `fallback_tokens` | 2 | Number of AR tokens on TASD-F fallback |
| `fallback_guarded` | `False` | Apply structural guard during TASD-F fallback |
| `enable_profit_guard` | `False` | Enable ProfitGuard profitability monitor (experimental) |
| `enable_comment_string_fallback` | `False` | Enable comment/string region fallback (experimental) |

### ProfitGuard Parameters (Experimental)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `profit_guard_ar_tps_estimate` | None | Estimated AR TPS for wall-clock comparison |
| `profit_guard_min_rounds` | 2 | Minimum rounds of low accept to trigger |
| `profit_guard_min_generated` | 24 | Minimum tokens before trigger can fire |
| `profit_guard_accept_threshold` | 0.55 | Rolling accept rate threshold |
| `profit_guard_repair_threshold` | 2 | Repair count threshold |
| `profit_guard_speed_margin` | 0.95 | Speed margin for wall-clock loss trigger |
| `profit_guard_mode` | "fallback_to_ar" | Fallback mode |

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

### Main Results (6 benchmarks × 80 samples)

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Settings**: temperature=0.0, max_new_tokens=128, TASD: draft_blocks=2, draft_len=16, top_k_accept=3

Full report: [results/comparison_5method_6x80.md](results/comparison_5method_6x80.md)

| Method | TPS | Speedup | SQ |
|--------|-----|---------|----|
| AR | 33.2 | 1.00x | 0.8910 |
| Greedy SD | 22.0 | 0.66x | 0.8612 |
| N-gram SD | 46.9 | 1.41x | 0.8232 |
| FLY | 54.5 | 1.64x | 0.8895 |
| TASD | 64.2 | 1.93x | 0.8825 |

### Per-Benchmark Breakdown

| Benchmark | AR | Greedy SD | N-gram SD | FLY | TASD |
|-----------|-----|-----------|-----------|-----|------|
| Real-Python-Argparse | 33.2 | 31.4 | 48.1 | 63.9 | 61.9 |
| Real-Python-DictConfig | 33.3 | 21.8 | 45.2 | 58.7 | 60.0 |
| OpenMMLab-Config | 33.2 | 22.1 | 44.8 | 35.1 | 64.0 |
| Rich-CLI-Option-Groups | 32.9 | 20.5 | 50.3 | 69.5 | 66.1 |
| Complex-Nested-Config | 33.3 | 18.8 | 47.1 | 57.9 | 66.4 |
| Pipeline-Stage-Config | 33.4 | 17.5 | 43.6 | 41.7 | 66.7 |

TASD achieves 1.93x average speedup over AR, with per-benchmark range 1.80x–2.01x. Structural quality (SQ) is comparable to AR (0.8825 vs 0.8910).

## Baselines

- **AR**: Standard autoregressive greedy decoding
- **Greedy SD**: Deterministic argmax-matching speculative decoding baseline (k=16, no relaxation, no guard)
- **FLY**: N-gram draft + draft model + window acceptance (k=15, use_ngram=True, max_ngram_size=4, num_ngram_pred_tokens=6, win_len=6, entropy_thre=0.3)
- **N-gram SD**: Training-free speculative decoding using prompt n-gram matching (n=3-8, draft up to 16 tokens, no additional draft-model overhead)

## Optional Extension: TASD-F

TASD-F is an optional runtime fallback extension for long or difficult structured completions. It monitors recent acceptance failures and briefly falls back to target-only AR for 2 tokens, then resumes TASD. TASD-F is disabled by default and is not used in the main 128-token benchmark.

TASD-F is an optional runtime extension for long-generation hard cases. It does not replace TASD.

### Configuration

```python
from src.tasd_decode import tasd_decode

result = tasd_decode(
    target_model=target_model,
    draft_model=draft_model,
    tokenizer=tokenizer,
    prompt="train_pipeline = [",
    structure_type="openmmlab_config",
    max_new_tokens=256,
    enable_failure_aware_fallback=True,  # Enable TASD-F
    # fallback_tokens=2  (default)
    # fallback_guarded=False  (default)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable_failure_aware_fallback` | `False` | Enable TASD-F runtime fallback |
| `fallback_tokens` | 2 | Number of AR tokens to generate on fallback |
| `fallback_guarded` | `False` | Whether to apply structural guard during fallback |

### How it works

TASD-F monitors three runtime signals:
1. **Rolling accept rate**: average acceptance rate over recent rounds
2. **Consecutive zero-accept rounds**: rounds where no draft tokens were accepted
3. **Recent repair count**: number of repair rounds in the recent window

When any of these signals indicates persistent draft-target divergence, TASD-F briefly falls back to target-only AR for 2 tokens, then resumes TASD. This helps recover from local regions where the draft model is poorly aligned with the target.

### Results

#### OpenMMLab-256 full40

| Method | Speedup | SQ | OffStr | Repair |
|--------|---------|----|--------|--------|
| TASD | 1.65x | 0.907 | 0.004 | 1.82 |
| TASD-F (2-token) | 1.77x | 0.908 | 0.006 | 0.50 |
| TASD-F (guarded) | 1.75x | 0.908 | 0.000 | 0.35 |

Full report: [results/openmmlab_256_failure_fallback_improved.md](results/openmmlab_256_failure_fallback_improved.md)

#### 128-token no-regression

TASD-F triggers rarely in the 128-token setting and does not materially change aggregate behavior. It is intended for persistent low-acceptance regions in longer generations, not as the main decoding method.

Full report: [results/tasd_fallback_128_no_regression.md](results/tasd_fallback_128_no_regression.md)

### Abandoned Variants

- **TASD-F v3** (progressive/boundary/probe fallback): Abandoned due to severe slowdowns on short Argparse structures (~1.9 TPS vs AR ~33 TPS). Result files retained with `_note` markers as negative results.
- **TASD-P** (ProfitGuard): Experimental safety analysis / future work. Not part of the final TASD or TASD-F method. Result files retained with `_note` markers.

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
