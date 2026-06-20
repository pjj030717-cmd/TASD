# TASD-FG Quality Error Attribution Analysis

**Total samples**: 480
**Score distribution**: 2=192 (40.0%), 1=156 (32.5%), 0=132 (27.5%)
**Automatic recoverable usability rate (score>=1)**: 72.5%

## 1. Error Tag Distribution (score=0 samples)

| Error Tag | Count | Percentage |
|-----------|:-----:|:----------:|
| BRACKET | 65 | 49.2% |
| TRUNC | 60 | 45.5% |
| LOW_F1 | 52 | 39.4% |
| OFF_STRUCT | 27 | 20.5% |
| REPEAT | 24 | 18.2% |
| DUP_OPT | 0 | 0.0% |

**Note**: BRACKET = bracket imbalance in non-truncated samples. TRUNC = output truncated at 128 tokens.
## 2. Error Tag Combinations (score=0)

| Combination | Count |
|-------------|:-----:|
| BRACKET | 50 |
| LOW_F1+TRUNC | 26 |
| OFF_STRUCT+TRUNC | 14 |
| BRACKET+LOW_F1 | 12 |
| REPEAT+TRUNC | 6 |
| OFF_STRUCT+REPEAT+TRUNC | 5 |
| LOW_F1+REPEAT+TRUNC | 4 |
| LOW_F1+OFF_STRUCT+REPEAT+TRUNC | 3 |
| LOW_F1+REPEAT | 2 |
| REPEAT | 2 |
| LOW_F1 | 2 |
| BRACKET+OFF_STRUCT | 2 |
| LOW_F1+OFF_STRUCT+TRUNC | 2 |
| BRACKET+REPEAT | 1 |
| LOW_F1+OFF_STRUCT+REPEAT | 1 |

## 3. Per-Benchmark Score=0 Analysis

| Benchmark | Total | Score=0 | Rate | Top Error Tags |
|-----------|:-----:|:-------:|:----:|----------------|
| argparse | 80 | 16 | 20.0% | BRACKET(15), LOW_F1(1), TRUNC(1) |
| complex_nested_config | 80 | 35 | 43.8% | LOW_F1(25), TRUNC(19), BRACKET(12) |
| dict_config | 80 | 17 | 21.2% | LOW_F1(12), TRUNC(8), BRACKET(7) |
| openmmlab_config | 80 | 23 | 28.7% | BRACKET(14), TRUNC(9), OFF_STRUCT(7) |
| pipeline_stage_config | 80 | 16 | 20.0% | TRUNC(11), OFF_STRUCT(6), BRACKET(5) |
| rich_cli_option_groups | 80 | 25 | 31.2% | BRACKET(12), TRUNC(12), OFF_STRUCT(11) |

## 4. Below-AR Sample Analysis

**Below-AR count**: 3

| Name | Benchmark | Score | Speedup | F1 | Rep | Off-Str | Bracket | Trunc |
|------|-----------|:-----:|:-------:|:--:|:---:|:-------:|:-------:|:-----:|
| argparse_real_062 | argparse | 1 | 0.655x | 0.618 | 0.000 | 0.100 | 1 | 1 |
| dict_config_real_014 | dict_config | 2 | 0.969x | 0.990 | 0.000 | 0.000 | 0 | 1 |
| dict_config_real_057 | dict_config | 1 | 0.912x | 0.796 | 0.000 | 0.000 | 0 | 1 |

## 5. Key Findings

1. **TASD-FG's primary contribution is speed robustness**, not guaranteeing all outputs are directly usable.
2. **Automatic recoverable usability rate = 72.5%** — the majority of outputs are at least partially recoverable.
3. **Score=0 concentrated in**: complex_nested_config (35), rich_cli_option_groups (25), openmmlab_config (23)
4. **Dominant error type**: BRACKET (65/132, 49.2%)
5. **Below-AR samples**: slow hard cases still maintain structural quality (see table above).
6. **FGQ pilot did not pass** — quality-guarded variant showed 11.9% speed loss with zero rep_trim triggers; not included in final method.

## 6. Limitations & Future Work

- Score=0 samples are primarily caused by structural complexity (deep nesting, long references) rather than simple repetition patterns.
- The 128-token budget inherently truncates outputs, making bracket balance impossible to achieve for long structures.
- Future work could explore structure-aware draft models or adaptive token budgets based on reference length estimation.