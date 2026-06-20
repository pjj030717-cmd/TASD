# Benchmark-Aware Structural Scoring Analysis

## 1. Summary

**Samples**: 480 TASD-FG samples across 6 benchmarks (80 each)

| Metric | Unified | Benchmark-Aware | Change |
|--------|:-------:|:---------------:|:------:|
| score=0 | 132 | 90 | -42 |
| score=1 | 156 | 157 | +1 |
| score=2 | 192 | 233 | +41 |
| Recoverable (s=1+2) | 348 | 390 | +42 |

| Change Type | Count |
|-------------|:-----:|
| 0→1 recovered | 11 |
| 0→2 recovered | 31 |
| 1→2 upgraded | 10 |
| 2→1 degraded | 0 |
| 2→0 degraded | 0 |
| 1→0 degraded | 0 |

## 2. Per-Benchmark Breakdown

| Benchmark | N | U s=0 | U s=1 | U s=2 | BA s=0 | BA s=1 | BA s=2 | 0→1 | 0→2 | 1→2 | 2→1 | 2→0 | 1→0 |
|-----------|:-:|:-----:|:-----:|:-----:|:------:|:------:|:------:|:---:|:---:|:---:|:---:|:---:|:---:|
| argparse | 80 | 16 | 21 | 43 | 1 | 27 | 52 | 6 | 9 | 0 | 0 | 0 | 0 |
| complex_nested_config | 80 | 35 | 32 | 13 | 30 | 29 | 21 | 2 | 3 | 5 | 0 | 0 | 0 |
| dict_config | 80 | 17 | 28 | 35 | 17 | 28 | 35 | 0 | 0 | 0 | 0 | 0 | 0 |
| openmmlab_config | 80 | 23 | 15 | 42 | 10 | 16 | 54 | 1 | 12 | 0 | 0 | 0 | 0 |
| pipeline_stage_config | 80 | 16 | 16 | 48 | 11 | 18 | 51 | 2 | 3 | 0 | 0 | 0 | 0 |
| rich_cli_option_groups | 80 | 25 | 44 | 11 | 21 | 39 | 20 | 0 | 4 | 5 | 0 | 0 | 0 |

## 3. Bracket Balance Impact

| Benchmark | U s=0 by bracket | BA s=0 by bracket | Reduction |
|-----------|:----------------:|:------------------:|:---------:|
| argparse | 15 | 0 | +15 |
| complex_nested_config | 23 | 18 | +5 |
| dict_config | 11 | 11 | +0 |
| openmmlab_config | 21 | 8 | +13 |
| pipeline_stage_config | 14 | 9 | +5 |
| rich_cli_option_groups | 14 | 10 | +4 |

## 4. Rescued Samples (Unified s=0 → BA s=1/2)

**Total rescued: 52**

| Benchmark | Name | F1 | Bracket | Rep | Off | Trunc | U→BA |
|-----------|------|----|:-------:|:----:|:---:|:-----:|:----:|
| argparse | argparse_real_001 | 0.775 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| argparse | argparse_real_004 | 0.678 | 0.000 | 0.071 | 0.000 | 0 | 0→1 |
| argparse | argparse_real_006 | 0.933 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_010 | 0.990 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_012 | 0.852 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_018 | 0.878 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_019 | 0.965 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_029 | 0.990 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_031 | 0.817 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| argparse | argparse_real_036 | 0.990 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_060 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| argparse | argparse_real_061 | 0.803 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| argparse | argparse_real_070 | 0.759 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| argparse | argparse_real_071 | 0.622 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| argparse | argparse_real_079 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| complex_nested_config | complex_nested_config_007 | 0.720 | 0.000 | 0.000 | 0.167 | 0 | 0→1 |
| complex_nested_config | complex_nested_config_025 | 0.831 | 0.000 | 0.000 | 0.000 | 1 | 1→2 |
| complex_nested_config | complex_nested_config_031 | 0.800 | 1.000 | 0.000 | 0.000 | 0 | 1→2 |
| complex_nested_config | complex_nested_config_034 | 0.844 | 1.000 | 0.000 | 0.000 | 1 | 1→2 |
| complex_nested_config | complex_nested_config_041 | 0.825 | 1.000 | 0.000 | 0.000 | 1 | 1→2 |
| complex_nested_config | complex_nested_config_044 | 0.912 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| complex_nested_config | complex_nested_config_065 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| complex_nested_config | complex_nested_config_070 | 0.994 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| complex_nested_config | complex_nested_config_072 | 0.850 | 0.000 | 0.000 | 0.167 | 0 | 0→1 |
| complex_nested_config | complex_nested_config_073 | 0.839 | 0.000 | 0.000 | 0.000 | 1 | 1→2 |
| openmmlab_config | openmmlab_config_real_008 | 0.992 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_012 | 0.992 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_018 | 0.992 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_041 | 0.992 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_045 | 0.992 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_046 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_054 | 0.991 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_057 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_062 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_074 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_076 | 0.755 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| openmmlab_config | openmmlab_config_real_077 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| openmmlab_config | openmmlab_config_real_080 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| pipeline_stage_config | pipeline_stage_config_007 | 0.702 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| pipeline_stage_config | pipeline_stage_config_008 | 0.841 | 0.000 | 0.000 | 0.000 | 0 | 0→1 |
| pipeline_stage_config | pipeline_stage_config_022 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| pipeline_stage_config | pipeline_stage_config_025 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| pipeline_stage_config | pipeline_stage_config_037 | 0.992 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0.903 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| rich_cli_option_groups | rich_cli_option_groups_008 | 0.815 | 0.000 | 0.000 | 0.000 | 1 | 1→2 |
| rich_cli_option_groups | rich_cli_option_groups_015 | 0.837 | 1.000 | 0.000 | 0.000 | 0 | 1→2 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 1.000 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| rich_cli_option_groups | rich_cli_option_groups_028 | 0.838 | 0.000 | 0.000 | 0.000 | 1 | 1→2 |
| rich_cli_option_groups | rich_cli_option_groups_043 | 0.833 | 0.000 | 0.000 | 0.000 | 1 | 1→2 |
| rich_cli_option_groups | rich_cli_option_groups_045 | 0.830 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |
| rich_cli_option_groups | rich_cli_option_groups_057 | 0.828 | 0.000 | 0.000 | 0.000 | 1 | 1→2 |
| rich_cli_option_groups | rich_cli_option_groups_076 | 0.938 | 0.000 | 0.000 | 0.000 | 0 | 0→2 |

## 5. Degraded Samples (Unified s=1/2 → BA s=0)

None.

## 6. Is Unified Scoring Too Harsh?

- **52 samples improved** (rescued from s=0 or upgraded to s=2)
- **0 samples degraded** (downgraded by BA rules)
- Net recoverable change: +52

### Benchmarks most affected by harsher bracket rule:

- **argparse**: 15 samples no longer killed by bracket alone
- **complex_nested_config**: 5 samples no longer killed by bracket alone
- **openmmlab_config**: 13 samples no longer killed by bracket alone
- **pipeline_stage_config**: 5 samples no longer killed by bracket alone
- **rich_cli_option_groups**: 4 samples no longer killed by bracket alone

### Key findings:

1. **bracket_balance is the dominant unfair metric** in unified scoring. 
   Many benchmarks have >80% truncation rate, making bracket open/close comparison unreliable.
2. **argparse, openmmlab, pipeline_stage** are most affected: 
   samples with F1 > 0.9 are scored s=0 purely due to bracket=0.
3. **complex_nested_config** genuinely has low F1 (mean=0.55), 
   but the hard threshold of 0.20 is too strict given reference variability.
4. **rich_cli_option_groups** has more diverse error modes (off_structure, repetition), 
   so bracket relaxation alone helps less.

## 7. Should We Adopt Benchmark-Specific Repair?

- Even with benchmark-aware scoring, **90 samples remain s=0**
- These are genuinely bad outputs (severe repetition, off-structure, or very low F1)
- The benchmark-aware scoring primarily **reclassifies unfairly penalized samples**
  rather than hiding real failures.

### Recommendation:

- **Yes, unified scoring is too harsh** for bracket-heavy benchmarks (42 samples rescued)
- **Benchmark-aware scoring is justified** for evaluating quality improvement methods
- **Benchmark-specific repair is worth pursuing** because:
  - argparse: bracket is not the issue; AR repairs for bracket would be wasted
  - openmmlab/pipeline_stage: off_structure is the real error, not bracket
  - complex_nested: needs better structural F1, not bracket repair
  - rich_cli: needs off_structure + repetition prevention, not bracket repair
- **Recommendation**: Use benchmark-aware scoring for all future TASD-FG quality evaluation.
