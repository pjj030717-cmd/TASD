# TASD-FG-Safe Parameter Ablation Study

## 配置

| Config | top_k | prefix_budget | window_len | relaxed_accept |
|--------|:-----:|:-------------:|:----------:|:--------------:|
| TASD-FG | 3 | 0.2 | 2 | True |
| TASD-FG-Safe-k2 | 2 | 0.1 | 1 | True |
| TASD-FG-Safe-k1 | 1 | 0.05 | 1 | True |
| TASD-FG-strict | 1 | 0.0 | 1 | False |

## Hard Subset (score=0 样本)

| 方法 | N | TPS | speedup | score2 | score1 | score0 | recoverable | F1 | BB |
|------|--:|-----:|-------:|------:|------:|------:|-----------:|---:|---:|
| TASD-FG | 60 | 59.0 | 0.000x | 13 | 4 | 43 | 17/60 (28.3%) | 0.636 | 0.317 |
| TASD-FG-Safe-k2 | 60 | 55.9 | 0.000x | 13 | 7 | 40 | 20/60 (33.3%) | 0.635 | 0.283 |
| TASD-FG-Safe-k1 | 60 | 55.5 | 0.000x | 13 | 7 | 40 | 20/60 (33.3%) | 0.635 | 0.283 |
| TASD-FG-strict | 60 | 52.7 | 0.000x | 14 | 5 | 41 | 19/60 (31.7%) | 0.635 | 0.317 |

### Speed Loss (vs TASD-FG)

| 方法 | TPS | ΔTPS | Speed Loss |
|------|----:|-----:|----------:|
| TASD-FG | 59.0 | +0.0 | +0.0% |
| TASD-FG-Safe-k2 | 55.9 | -3.1 | +5.3% |
| TASD-FG-Safe-k1 | 55.5 | -3.5 | +5.9% |
| TASD-FG-strict | 52.7 | -6.3 | +10.6% |

## Clean Subset (score=2 样本, no-regression)

| 方法 | N | TPS | speedup | score2 | score1 | score0 | score2保持率 | F1 | BB |
|------|--:|-----:|-------:|------:|------:|------:|-----------:|---:|---:|
| TASD-FG | 60 | 60.6 | 0.000x | 50 | 6 | 4 | 83.3% | 0.965 | 0.250 |
| TASD-FG-Safe-k2 | 60 | 55.6 | 0.000x | 49 | 6 | 5 | 81.7% | 0.957 | 0.233 |
| TASD-FG-Safe-k1 | 60 | 56.0 | 0.000x | 49 | 6 | 5 | 81.7% | 0.957 | 0.233 |
| TASD-FG-strict | 60 | 53.4 | 0.000x | 50 | 6 | 4 | 83.3% | 0.965 | 0.250 |

## 决策分析

| 方法 | Hard score0↓ | Clean score2保持 | Speed Loss | 判定 |
|------|:-----------:|:---------------:|:----------:|------|
| TASD-FG-Safe-k2 | 40 (+7%) | 81.7% | +5.3% | 情况 C: 不扩大 |
| TASD-FG-Safe-k1 | 40 (+7%) | 81.7% | +5.9% | 情况 C: 不扩大 |
| TASD-FG-strict | 41 (+5%) | 83.3% | +10.6% | 情况 C: 不扩大 |

**推荐**: 无变体满足条件，保留 TASD-FG 作为唯一版本

## 样本详情

### TASD-FG - Hard Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_001 | 0 | 0 | BRACKET | 45.0 | 0.775 |
| argparse | argparse_real_004 | 0 | 0 | BRACKET,DUP_OPT | 62.2 | 0.678 |
| argparse | argparse_real_006 | 0 | 2 | TRUNC | 61.4 | 0.913 |
| argparse | argparse_real_010 | 0 | 2 | TRUNC | 61.3 | 0.987 |
| argparse | argparse_real_012 | 0 | 0 | BRACKET | 60.9 | 0.852 |
| argparse | argparse_real_018 | 0 | 1 | TRUNC | 61.6 | 0.840 |
| argparse | argparse_real_019 | 0 | 0 | BRACKET | 47.1 | 0.983 |
| argparse | argparse_real_029 | 0 | 2 | TRUNC | 60.3 | 0.967 |
| argparse | argparse_real_030 | 0 | 0 | REPEAT,TRUNC | 62.8 | 1.000 |
| argparse | argparse_real_031 | 0 | 2 | TRUNC | 22.9 | 1.000 |
| complex_nested_config | complex_nested_config_002 | 0 | 0 | LOW_F1,TRUNC | 61.9 | 0.048 |
| complex_nested_config | complex_nested_config_004 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 60.0 | 0.420 |
| complex_nested_config | complex_nested_config_007 | 0 | 0 | OFF_STRUCT,BRACKET | 60.6 | 0.720 |
| complex_nested_config | complex_nested_config_008 | 0 | 0 | LOW_F1,TRUNC | 58.3 | 0.011 |
| complex_nested_config | complex_nested_config_014 | 0 | 0 | LOW_F1,TRUNC | 61.1 | 0.001 |
| complex_nested_config | complex_nested_config_016 | 0 | 0 | LOW_F1,TRUNC | 61.5 | 0.147 |
| complex_nested_config | complex_nested_config_017 | 0 | 0 | LOW_F1,DUP_OPT,TRUNC | 62.4 | 0.118 |
| complex_nested_config | complex_nested_config_018 | 0 | 0 | LOW_F1,BRACKET | 61.5 | 0.352 |
| complex_nested_config | complex_nested_config_020 | 0 | 0 | LOW_F1,BRACKET | 62.3 | 0.168 |
| complex_nested_config | complex_nested_config_021 | 0 | 0 | REPEAT,LOW_F1,BRACKET,DUP_OPT | 62.2 | 0.286 |
| dict_config | dict_config_real_002 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 60.0 | 0.376 |
| dict_config | dict_config_real_005 | 0 | 0 | BRACKET | 60.6 | 0.789 |
| dict_config | dict_config_real_008 | 0 | 0 | BRACKET | 41.9 | 0.857 |
| dict_config | dict_config_real_011 | 0 | 0 | LOW_F1,TRUNC | 58.9 | 0.107 |
| dict_config | dict_config_real_015 | 0 | 2 | TRUNC | 61.3 | 1.000 |
| dict_config | dict_config_real_016 | 0 | 1 | DUP_OPT | 60.0 | 1.000 |
| dict_config | dict_config_real_022 | 0 | 0 | LOW_F1,TRUNC | 60.0 | 0.193 |
| dict_config | dict_config_real_041 | 0 | 2 | TRUNC | 47.2 | 0.979 |
| dict_config | dict_config_real_045 | 0 | 0 | LOW_F1,TRUNC | 60.5 | 0.128 |
| dict_config | dict_config_real_049 | 0 | 0 | LOW_F1,TRUNC | 59.9 | 0.070 |
| openmmlab_config | openmmlab_config_real_001 | 0 | 0 | OFF_STRUCT,TRUNC | 61.1 | 0.988 |
| openmmlab_config | openmmlab_config_real_003 | 0 | 0 | OFF_STRUCT,TRUNC | 60.8 | 0.988 |
| openmmlab_config | openmmlab_config_real_008 | 0 | 2 | TRUNC | 59.0 | 1.000 |
| openmmlab_config | openmmlab_config_real_012 | 0 | 2 | TRUNC | 58.5 | 1.000 |
| openmmlab_config | openmmlab_config_real_017 | 0 | 2 | TRUNC | 60.3 | 0.991 |
| openmmlab_config | openmmlab_config_real_018 | 0 | 0 | BRACKET | 60.8 | 0.992 |
| openmmlab_config | openmmlab_config_real_021 | 0 | 2 | TRUNC | 58.6 | 0.991 |
| openmmlab_config | openmmlab_config_real_025 | 0 | 0 | LOW_F1,TRUNC | 60.3 | 0.000 |
| openmmlab_config | openmmlab_config_real_029 | 0 | 0 | LOW_F1,TRUNC | 62.1 | 0.000 |
| openmmlab_config | openmmlab_config_real_041 | 0 | 2 | TRUNC | 59.0 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_007 | 0 | 0 | BRACKET | 62.7 | 0.702 |
| pipeline_stage_config | pipeline_stage_config_008 | 0 | 0 | BRACKET | 61.9 | 0.841 |
| pipeline_stage_config | pipeline_stage_config_010 | 0 | 0 | OFF_STRUCT,TRUNC | 62.8 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_019 | 0 | 0 | OFF_STRUCT,TRUNC | 44.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_022 | 0 | 0 | BRACKET | 60.8 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_025 | 0 | 2 | TRUNC | 61.0 | 0.984 |
| pipeline_stage_config | pipeline_stage_config_034 | 0 | 2 | TRUNC | 59.6 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_037 | 0 | 0 | BRACKET | 60.5 | 0.992 |
| pipeline_stage_config | pipeline_stage_config_041 | 0 | 0 | REPEAT,DUP_OPT,TRUNC | 61.1 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_043 | 0 | 0 | LOW_F1,TRUNC | 62.6 | 0.000 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0 | 0 | BRACKET | 61.4 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_013 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1,TRUNC | 61.3 | 0.164 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 0 | 1 | DUP_OPT,TRUNC | 62.3 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_017 | 0 | 0 | LOW_F1,BRACKET,DUP_OPT | 61.7 | 0.444 |
| rich_cli_option_groups | rich_cli_option_groups_022 | 0 | 0 | OFF_STRUCT,LOW_F1,TRUNC | 61.9 | 0.325 |
| rich_cli_option_groups | rich_cli_option_groups_025 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1 | 59.3 | 0.237 |
| rich_cli_option_groups | rich_cli_option_groups_027 | 0 | 0 | LOW_F1,TRUNC | 60.9 | 0.025 |
| rich_cli_option_groups | rich_cli_option_groups_029 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 61.2 | 0.684 |
| rich_cli_option_groups | rich_cli_option_groups_031 | 0 | 1 | LOW_F1,TRUNC | 61.9 | 0.243 |
| rich_cli_option_groups | rich_cli_option_groups_032 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 61.4 | 0.785 |

### TASD-FG-Safe-k2 - Hard Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_001 | 0 | 0 | BRACKET | 61.3 | 0.775 |
| argparse | argparse_real_004 | 0 | 1 | TRUNC | 40.5 | 0.678 |
| argparse | argparse_real_006 | 0 | 2 | TRUNC | 62.1 | 0.913 |
| argparse | argparse_real_010 | 0 | 2 | TRUNC | 63.2 | 0.987 |
| argparse | argparse_real_012 | 0 | 0 | BRACKET | 61.6 | 0.852 |
| argparse | argparse_real_018 | 0 | 0 | BRACKET,DUP_OPT | 49.0 | 0.860 |
| argparse | argparse_real_019 | 0 | 2 | TRUNC | 41.0 | 0.982 |
| argparse | argparse_real_029 | 0 | 2 | TRUNC | 62.2 | 0.967 |
| argparse | argparse_real_030 | 0 | 0 | REPEAT,TRUNC | 62.3 | 1.000 |
| argparse | argparse_real_031 | 0 | 0 | BRACKET,DUP_OPT | 12.8 | 1.000 |
| complex_nested_config | complex_nested_config_002 | 0 | 0 | LOW_F1,TRUNC | 59.0 | 0.048 |
| complex_nested_config | complex_nested_config_004 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 59.3 | 0.420 |
| complex_nested_config | complex_nested_config_007 | 0 | 0 | OFF_STRUCT,BRACKET | 61.4 | 0.720 |
| complex_nested_config | complex_nested_config_008 | 0 | 0 | LOW_F1,TRUNC | 61.2 | 0.011 |
| complex_nested_config | complex_nested_config_014 | 0 | 0 | LOW_F1,TRUNC | 61.9 | 0.001 |
| complex_nested_config | complex_nested_config_016 | 0 | 0 | LOW_F1,TRUNC | 61.6 | 0.147 |
| complex_nested_config | complex_nested_config_017 | 0 | 0 | LOW_F1,DUP_OPT,TRUNC | 62.1 | 0.118 |
| complex_nested_config | complex_nested_config_018 | 0 | 0 | LOW_F1,BRACKET | 61.0 | 0.352 |
| complex_nested_config | complex_nested_config_020 | 0 | 0 | LOW_F1,BRACKET | 59.8 | 0.168 |
| complex_nested_config | complex_nested_config_021 | 0 | 1 | LOW_F1,TRUNC | 42.6 | 0.286 |
| dict_config | dict_config_real_002 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 34.7 | 0.389 |
| dict_config | dict_config_real_005 | 0 | 0 | BRACKET | 60.0 | 0.789 |
| dict_config | dict_config_real_008 | 0 | 1 | TRUNC | 32.0 | 0.819 |
| dict_config | dict_config_real_011 | 0 | 0 | LOW_F1,TRUNC | 62.7 | 0.107 |
| dict_config | dict_config_real_015 | 0 | 2 | TRUNC | 63.3 | 1.000 |
| dict_config | dict_config_real_016 | 0 | 1 | DUP_OPT | 62.5 | 1.000 |
| dict_config | dict_config_real_022 | 0 | 0 | LOW_F1,TRUNC | 49.8 | 0.188 |
| dict_config | dict_config_real_041 | 0 | 2 | TRUNC | 29.4 | 0.981 |
| dict_config | dict_config_real_045 | 0 | 0 | LOW_F1,TRUNC | 60.9 | 0.128 |
| dict_config | dict_config_real_049 | 0 | 0 | LOW_F1,TRUNC | 60.8 | 0.070 |
| openmmlab_config | openmmlab_config_real_001 | 0 | 0 | OFF_STRUCT,TRUNC | 60.7 | 0.988 |
| openmmlab_config | openmmlab_config_real_003 | 0 | 0 | OFF_STRUCT,TRUNC | 60.5 | 0.988 |
| openmmlab_config | openmmlab_config_real_008 | 0 | 2 | TRUNC | 31.0 | 0.982 |
| openmmlab_config | openmmlab_config_real_012 | 0 | 2 | TRUNC | 31.6 | 0.982 |
| openmmlab_config | openmmlab_config_real_017 | 0 | 2 | TRUNC | 62.2 | 0.991 |
| openmmlab_config | openmmlab_config_real_018 | 0 | 0 | BRACKET | 61.5 | 0.992 |
| openmmlab_config | openmmlab_config_real_021 | 0 | 2 | TRUNC | 61.0 | 0.991 |
| openmmlab_config | openmmlab_config_real_025 | 0 | 0 | LOW_F1,TRUNC | 61.7 | 0.000 |
| openmmlab_config | openmmlab_config_real_029 | 0 | 0 | LOW_F1,TRUNC | 62.9 | 0.000 |
| openmmlab_config | openmmlab_config_real_041 | 0 | 2 | TRUNC | 62.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_007 | 0 | 0 | BRACKET | 62.5 | 0.702 |
| pipeline_stage_config | pipeline_stage_config_008 | 0 | 0 | BRACKET | 61.8 | 0.841 |
| pipeline_stage_config | pipeline_stage_config_010 | 0 | 1 | OFF_STRUCT,TRUNC | 45.8 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_019 | 0 | 0 | OFF_STRUCT,TRUNC | 32.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_022 | 0 | 0 | BRACKET | 61.9 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_025 | 0 | 2 | TRUNC | 38.5 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_034 | 0 | 2 | TRUNC | 61.8 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_037 | 0 | 0 | BRACKET | 61.0 | 0.992 |
| pipeline_stage_config | pipeline_stage_config_041 | 0 | 0 | REPEAT,DUP_OPT,TRUNC | 61.6 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_043 | 0 | 0 | LOW_F1,TRUNC | 63.3 | 0.000 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0 | 0 | BRACKET | 62.7 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_013 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1,TRUNC | 62.8 | 0.164 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 0 | 1 | DUP_OPT,TRUNC | 62.4 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_017 | 0 | 0 | LOW_F1,BRACKET,DUP_OPT | 62.5 | 0.444 |
| rich_cli_option_groups | rich_cli_option_groups_022 | 0 | 0 | OFF_STRUCT,LOW_F1,TRUNC | 62.0 | 0.325 |
| rich_cli_option_groups | rich_cli_option_groups_025 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1 | 62.0 | 0.237 |
| rich_cli_option_groups | rich_cli_option_groups_027 | 0 | 0 | LOW_F1,TRUNC | 62.2 | 0.025 |
| rich_cli_option_groups | rich_cli_option_groups_029 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 62.8 | 0.684 |
| rich_cli_option_groups | rich_cli_option_groups_031 | 0 | 1 | LOW_F1,TRUNC | 62.1 | 0.243 |
| rich_cli_option_groups | rich_cli_option_groups_032 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 62.7 | 0.785 |

### TASD-FG-Safe-k1 - Hard Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_001 | 0 | 0 | BRACKET | 62.4 | 0.775 |
| argparse | argparse_real_004 | 0 | 1 | TRUNC | 37.5 | 0.678 |
| argparse | argparse_real_006 | 0 | 2 | TRUNC | 62.3 | 0.913 |
| argparse | argparse_real_010 | 0 | 2 | TRUNC | 62.5 | 0.987 |
| argparse | argparse_real_012 | 0 | 0 | BRACKET | 61.0 | 0.852 |
| argparse | argparse_real_018 | 0 | 0 | BRACKET,DUP_OPT | 47.0 | 0.860 |
| argparse | argparse_real_019 | 0 | 2 | TRUNC | 38.5 | 0.982 |
| argparse | argparse_real_029 | 0 | 2 | TRUNC | 61.9 | 0.967 |
| argparse | argparse_real_030 | 0 | 0 | REPEAT,TRUNC | 63.2 | 1.000 |
| argparse | argparse_real_031 | 0 | 0 | BRACKET,DUP_OPT | 12.7 | 1.000 |
| complex_nested_config | complex_nested_config_002 | 0 | 0 | LOW_F1,TRUNC | 60.7 | 0.048 |
| complex_nested_config | complex_nested_config_004 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 60.3 | 0.420 |
| complex_nested_config | complex_nested_config_007 | 0 | 0 | OFF_STRUCT,BRACKET | 59.8 | 0.720 |
| complex_nested_config | complex_nested_config_008 | 0 | 0 | LOW_F1,TRUNC | 60.7 | 0.011 |
| complex_nested_config | complex_nested_config_014 | 0 | 0 | LOW_F1,TRUNC | 60.4 | 0.001 |
| complex_nested_config | complex_nested_config_016 | 0 | 0 | LOW_F1,TRUNC | 60.8 | 0.147 |
| complex_nested_config | complex_nested_config_017 | 0 | 0 | LOW_F1,DUP_OPT,TRUNC | 61.8 | 0.118 |
| complex_nested_config | complex_nested_config_018 | 0 | 0 | LOW_F1,BRACKET | 59.9 | 0.352 |
| complex_nested_config | complex_nested_config_020 | 0 | 0 | LOW_F1,BRACKET | 60.0 | 0.168 |
| complex_nested_config | complex_nested_config_021 | 0 | 1 | LOW_F1,TRUNC | 42.6 | 0.286 |
| dict_config | dict_config_real_002 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 34.2 | 0.389 |
| dict_config | dict_config_real_005 | 0 | 0 | BRACKET | 59.4 | 0.789 |
| dict_config | dict_config_real_008 | 0 | 1 | TRUNC | 33.2 | 0.819 |
| dict_config | dict_config_real_011 | 0 | 0 | LOW_F1,TRUNC | 62.2 | 0.107 |
| dict_config | dict_config_real_015 | 0 | 2 | TRUNC | 61.6 | 1.000 |
| dict_config | dict_config_real_016 | 0 | 1 | DUP_OPT | 62.4 | 1.000 |
| dict_config | dict_config_real_022 | 0 | 0 | LOW_F1,TRUNC | 50.2 | 0.188 |
| dict_config | dict_config_real_041 | 0 | 2 | TRUNC | 29.2 | 0.981 |
| dict_config | dict_config_real_045 | 0 | 0 | LOW_F1,TRUNC | 60.5 | 0.128 |
| dict_config | dict_config_real_049 | 0 | 0 | LOW_F1,TRUNC | 61.5 | 0.070 |
| openmmlab_config | openmmlab_config_real_001 | 0 | 0 | OFF_STRUCT,TRUNC | 60.9 | 0.988 |
| openmmlab_config | openmmlab_config_real_003 | 0 | 0 | OFF_STRUCT,TRUNC | 62.7 | 0.988 |
| openmmlab_config | openmmlab_config_real_008 | 0 | 2 | TRUNC | 31.0 | 0.982 |
| openmmlab_config | openmmlab_config_real_012 | 0 | 2 | TRUNC | 30.8 | 0.982 |
| openmmlab_config | openmmlab_config_real_017 | 0 | 2 | TRUNC | 62.0 | 0.991 |
| openmmlab_config | openmmlab_config_real_018 | 0 | 0 | BRACKET | 62.8 | 0.992 |
| openmmlab_config | openmmlab_config_real_021 | 0 | 2 | TRUNC | 62.3 | 0.991 |
| openmmlab_config | openmmlab_config_real_025 | 0 | 0 | LOW_F1,TRUNC | 63.2 | 0.000 |
| openmmlab_config | openmmlab_config_real_029 | 0 | 0 | LOW_F1,TRUNC | 62.7 | 0.000 |
| openmmlab_config | openmmlab_config_real_041 | 0 | 2 | TRUNC | 62.4 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_007 | 0 | 0 | BRACKET | 60.7 | 0.702 |
| pipeline_stage_config | pipeline_stage_config_008 | 0 | 0 | BRACKET | 61.4 | 0.841 |
| pipeline_stage_config | pipeline_stage_config_010 | 0 | 1 | OFF_STRUCT,TRUNC | 44.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_019 | 0 | 0 | OFF_STRUCT,TRUNC | 32.9 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_022 | 0 | 0 | BRACKET | 61.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_025 | 0 | 2 | TRUNC | 38.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_034 | 0 | 2 | TRUNC | 62.4 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_037 | 0 | 0 | BRACKET | 62.7 | 0.992 |
| pipeline_stage_config | pipeline_stage_config_041 | 0 | 0 | REPEAT,DUP_OPT,TRUNC | 62.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_043 | 0 | 0 | LOW_F1,TRUNC | 62.4 | 0.000 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0 | 0 | BRACKET | 60.8 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_013 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1,TRUNC | 60.8 | 0.164 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 0 | 1 | DUP_OPT,TRUNC | 62.4 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_017 | 0 | 0 | LOW_F1,BRACKET,DUP_OPT | 62.1 | 0.444 |
| rich_cli_option_groups | rich_cli_option_groups_022 | 0 | 0 | OFF_STRUCT,LOW_F1,TRUNC | 61.6 | 0.325 |
| rich_cli_option_groups | rich_cli_option_groups_025 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1 | 61.5 | 0.237 |
| rich_cli_option_groups | rich_cli_option_groups_027 | 0 | 0 | LOW_F1,TRUNC | 60.2 | 0.025 |
| rich_cli_option_groups | rich_cli_option_groups_029 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 60.1 | 0.684 |
| rich_cli_option_groups | rich_cli_option_groups_031 | 0 | 1 | LOW_F1,TRUNC | 60.4 | 0.243 |
| rich_cli_option_groups | rich_cli_option_groups_032 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 61.9 | 0.785 |

### TASD-FG-strict - Hard Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_001 | 0 | 1 | TRUNC | 55.4 | 0.759 |
| argparse | argparse_real_004 | 0 | 0 | BRACKET,DUP_OPT | 49.0 | 0.678 |
| argparse | argparse_real_006 | 0 | 2 | TRUNC | 54.1 | 0.913 |
| argparse | argparse_real_010 | 0 | 2 | TRUNC | 53.2 | 0.987 |
| argparse | argparse_real_012 | 0 | 0 | BRACKET | 58.6 | 0.852 |
| argparse | argparse_real_018 | 0 | 1 | TRUNC | 49.2 | 0.840 |
| argparse | argparse_real_019 | 0 | 0 | BRACKET | 43.4 | 0.983 |
| argparse | argparse_real_029 | 0 | 2 | TRUNC | 52.0 | 0.967 |
| argparse | argparse_real_030 | 0 | 0 | REPEAT,TRUNC | 56.0 | 1.000 |
| argparse | argparse_real_031 | 0 | 2 | TRUNC | 22.4 | 1.000 |
| complex_nested_config | complex_nested_config_002 | 0 | 0 | LOW_F1,TRUNC | 36.8 | 0.048 |
| complex_nested_config | complex_nested_config_004 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 56.6 | 0.420 |
| complex_nested_config | complex_nested_config_007 | 0 | 0 | OFF_STRUCT,BRACKET | 51.7 | 0.720 |
| complex_nested_config | complex_nested_config_008 | 0 | 0 | LOW_F1,TRUNC | 56.2 | 0.011 |
| complex_nested_config | complex_nested_config_014 | 0 | 0 | LOW_F1,TRUNC | 54.8 | 0.001 |
| complex_nested_config | complex_nested_config_016 | 0 | 0 | LOW_F1,TRUNC | 52.7 | 0.147 |
| complex_nested_config | complex_nested_config_017 | 0 | 0 | LOW_F1,DUP_OPT,TRUNC | 53.0 | 0.118 |
| complex_nested_config | complex_nested_config_018 | 0 | 0 | LOW_F1,BRACKET | 58.7 | 0.352 |
| complex_nested_config | complex_nested_config_020 | 0 | 0 | LOW_F1,BRACKET | 57.9 | 0.168 |
| complex_nested_config | complex_nested_config_021 | 0 | 0 | REPEAT,LOW_F1,BRACKET,DUP_OPT | 53.7 | 0.286 |
| dict_config | dict_config_real_002 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 56.0 | 0.376 |
| dict_config | dict_config_real_005 | 0 | 0 | BRACKET | 60.4 | 0.789 |
| dict_config | dict_config_real_008 | 0 | 2 | TRUNC | 36.1 | 0.850 |
| dict_config | dict_config_real_011 | 0 | 0 | LOW_F1,TRUNC | 58.3 | 0.107 |
| dict_config | dict_config_real_015 | 0 | 2 | TRUNC | 58.0 | 1.000 |
| dict_config | dict_config_real_016 | 0 | 1 | DUP_OPT | 56.3 | 1.000 |
| dict_config | dict_config_real_022 | 0 | 0 | LOW_F1,TRUNC | 55.3 | 0.193 |
| dict_config | dict_config_real_041 | 0 | 0 | BRACKET | 38.2 | 0.963 |
| dict_config | dict_config_real_045 | 0 | 0 | LOW_F1,TRUNC | 59.2 | 0.128 |
| dict_config | dict_config_real_049 | 0 | 0 | LOW_F1,TRUNC | 58.2 | 0.070 |
| openmmlab_config | openmmlab_config_real_001 | 0 | 0 | OFF_STRUCT,TRUNC | 49.7 | 0.988 |
| openmmlab_config | openmmlab_config_real_003 | 0 | 0 | OFF_STRUCT,TRUNC | 49.5 | 0.988 |
| openmmlab_config | openmmlab_config_real_008 | 0 | 2 | TRUNC | 53.5 | 1.000 |
| openmmlab_config | openmmlab_config_real_012 | 0 | 2 | TRUNC | 53.4 | 1.000 |
| openmmlab_config | openmmlab_config_real_017 | 0 | 2 | TRUNC | 43.1 | 0.991 |
| openmmlab_config | openmmlab_config_real_018 | 0 | 0 | BRACKET | 53.1 | 0.992 |
| openmmlab_config | openmmlab_config_real_021 | 0 | 2 | TRUNC | 47.0 | 0.991 |
| openmmlab_config | openmmlab_config_real_025 | 0 | 0 | LOW_F1,TRUNC | 59.3 | 0.000 |
| openmmlab_config | openmmlab_config_real_029 | 0 | 0 | LOW_F1,TRUNC | 59.6 | 0.000 |
| openmmlab_config | openmmlab_config_real_041 | 0 | 2 | TRUNC | 54.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_007 | 0 | 0 | BRACKET | 53.7 | 0.702 |
| pipeline_stage_config | pipeline_stage_config_008 | 0 | 0 | BRACKET | 53.7 | 0.841 |
| pipeline_stage_config | pipeline_stage_config_010 | 0 | 0 | OFF_STRUCT,TRUNC | 53.5 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_019 | 0 | 2 | TRUNC | 47.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_022 | 0 | 0 | BRACKET | 58.9 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_025 | 0 | 2 | TRUNC | 50.4 | 0.984 |
| pipeline_stage_config | pipeline_stage_config_034 | 0 | 2 | TRUNC | 51.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_037 | 0 | 0 | BRACKET | 57.3 | 0.992 |
| pipeline_stage_config | pipeline_stage_config_041 | 0 | 0 | REPEAT,DUP_OPT,TRUNC | 57.2 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_043 | 0 | 0 | LOW_F1,TRUNC | 58.0 | 0.000 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0 | 0 | BRACKET | 52.3 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_013 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1,TRUNC | 52.7 | 0.164 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 0 | 1 | DUP_OPT,TRUNC | 55.3 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_017 | 0 | 0 | LOW_F1,BRACKET,DUP_OPT | 56.8 | 0.444 |
| rich_cli_option_groups | rich_cli_option_groups_022 | 0 | 0 | OFF_STRUCT,LOW_F1,TRUNC | 53.0 | 0.325 |
| rich_cli_option_groups | rich_cli_option_groups_025 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1 | 52.8 | 0.237 |
| rich_cli_option_groups | rich_cli_option_groups_027 | 0 | 0 | LOW_F1,TRUNC | 51.7 | 0.025 |
| rich_cli_option_groups | rich_cli_option_groups_029 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 54.8 | 0.684 |
| rich_cli_option_groups | rich_cli_option_groups_031 | 0 | 1 | LOW_F1,TRUNC | 56.9 | 0.243 |
| rich_cli_option_groups | rich_cli_option_groups_032 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 50.3 | 0.785 |

### TASD-FG - Clean Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_002 | 2 | 2 | TRUNC | 57.7 | 1.000 |
| argparse | argparse_real_003 | 2 | 2 | TRUNC | 60.7 | 0.933 |
| argparse | argparse_real_007 | 2 | 2 | TRUNC | 62.3 | 1.000 |
| argparse | argparse_real_008 | 2 | 0 | DUP_OPT,TRUNC | 61.9 | 1.000 |
| argparse | argparse_real_009 | 2 | 2 | TRUNC | 60.2 | 0.901 |
| argparse | argparse_real_013 | 2 | 2 | TRUNC | 59.1 | 0.990 |
| argparse | argparse_real_014 | 2 | 2 | TRUNC | 58.7 | 0.981 |
| argparse | argparse_real_015 | 2 | 0 | BRACKET | 58.8 | 0.967 |
| argparse | argparse_real_016 | 2 | 2 | TRUNC | 59.9 | 0.865 |
| argparse | argparse_real_017 | 2 | 2 | TRUNC | 60.3 | 1.000 |
| complex_nested_config | complex_nested_config_006 | 2 | 2 | TRUNC | 59.3 | 1.000 |
| complex_nested_config | complex_nested_config_022 | 2 | 2 | TRUNC | 59.7 | 1.000 |
| complex_nested_config | complex_nested_config_026 | 2 | 1 | TRUNC | 59.8 | 0.833 |
| complex_nested_config | complex_nested_config_030 | 2 | 2 | TRUNC | 61.7 | 1.000 |
| complex_nested_config | complex_nested_config_035 | 2 | 1 | TRUNC | 61.0 | 0.947 |
| complex_nested_config | complex_nested_config_042 | 2 | 1 | TRUNC | 62.1 | 0.960 |
| complex_nested_config | complex_nested_config_043 | 2 | 2 | TRUNC | 59.4 | 0.943 |
| complex_nested_config | complex_nested_config_055 | 2 | 2 | TRUNC | 59.6 | 0.959 |
| complex_nested_config | complex_nested_config_064 | 2 | 1 | TRUNC | 61.4 | 0.844 |
| complex_nested_config | complex_nested_config_068 | 2 | 2 | clean | 60.1 | 0.978 |
| dict_config | dict_config_real_001 | 2 | 2 | TRUNC | 59.7 | 1.000 |
| dict_config | dict_config_real_013 | 2 | 2 | TRUNC | 61.2 | 0.898 |
| dict_config | dict_config_real_014 | 2 | 0 | BRACKET | 47.7 | 0.981 |
| dict_config | dict_config_real_018 | 2 | 2 | TRUNC | 45.3 | 0.993 |
| dict_config | dict_config_real_020 | 2 | 2 | TRUNC | 61.8 | 1.000 |
| dict_config | dict_config_real_027 | 2 | 2 | TRUNC | 61.1 | 0.986 |
| dict_config | dict_config_real_028 | 2 | 2 | TRUNC | 59.6 | 0.986 |
| dict_config | dict_config_real_029 | 2 | 2 | TRUNC | 62.0 | 0.986 |
| dict_config | dict_config_real_030 | 2 | 2 | clean | 61.5 | 0.892 |
| dict_config | dict_config_real_032 | 2 | 2 | TRUNC | 62.7 | 0.983 |
| openmmlab_config | openmmlab_config_real_002 | 2 | 2 | TRUNC | 62.2 | 0.991 |
| openmmlab_config | openmmlab_config_real_004 | 2 | 0 | OFF_STRUCT,TRUNC | 61.6 | 0.987 |
| openmmlab_config | openmmlab_config_real_005 | 2 | 2 | TRUNC | 62.3 | 0.985 |
| openmmlab_config | openmmlab_config_real_006 | 2 | 2 | TRUNC | 62.1 | 0.991 |
| openmmlab_config | openmmlab_config_real_007 | 2 | 2 | TRUNC | 62.8 | 1.000 |
| openmmlab_config | openmmlab_config_real_009 | 2 | 2 | TRUNC | 62.0 | 0.982 |
| openmmlab_config | openmmlab_config_real_011 | 2 | 2 | TRUNC | 62.1 | 1.000 |
| openmmlab_config | openmmlab_config_real_013 | 2 | 2 | TRUNC | 62.1 | 0.982 |
| openmmlab_config | openmmlab_config_real_015 | 2 | 2 | TRUNC | 61.8 | 1.000 |
| openmmlab_config | openmmlab_config_real_016 | 2 | 2 | TRUNC | 62.9 | 0.991 |
| pipeline_stage_config | pipeline_stage_config_001 | 2 | 2 | TRUNC | 61.9 | 0.932 |
| pipeline_stage_config | pipeline_stage_config_002 | 2 | 2 | TRUNC | 62.2 | 0.980 |
| pipeline_stage_config | pipeline_stage_config_003 | 2 | 2 | TRUNC | 62.1 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_004 | 2 | 2 | TRUNC | 61.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_005 | 2 | 2 | TRUNC | 62.0 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_006 | 2 | 2 | TRUNC | 61.4 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_009 | 2 | 2 | TRUNC | 62.3 | 0.978 |
| pipeline_stage_config | pipeline_stage_config_011 | 2 | 2 | TRUNC | 62.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_012 | 2 | 2 | TRUNC | 62.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_014 | 2 | 2 | TRUNC | 62.4 | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_005 | 2 | 2 | TRUNC | 62.3 | 0.939 |
| rich_cli_option_groups | rich_cli_option_groups_010 | 2 | 2 | clean | 61.5 | 0.983 |
| rich_cli_option_groups | rich_cli_option_groups_012 | 2 | 2 | TRUNC | 61.3 | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_019 | 2 | 2 | TRUNC | 62.8 | 0.936 |
| rich_cli_option_groups | rich_cli_option_groups_033 | 2 | 2 | clean | 57.8 | 0.971 |
| rich_cli_option_groups | rich_cli_option_groups_035 | 2 | 1 | TRUNC | 59.9 | 0.743 |
| rich_cli_option_groups | rich_cli_option_groups_051 | 2 | 2 | TRUNC | 58.5 | 0.938 |
| rich_cli_option_groups | rich_cli_option_groups_052 | 2 | 2 | clean | 61.5 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_059 | 2 | 2 | clean | 60.3 | 0.967 |
| rich_cli_option_groups | rich_cli_option_groups_068 | 2 | 1 | TRUNC | 60.8 | 0.826 |

### TASD-FG-Safe-k2 - Clean Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_002 | 2 | 2 | TRUNC | 60.8 | 1.000 |
| argparse | argparse_real_003 | 2 | 2 | TRUNC | 58.2 | 0.933 |
| argparse | argparse_real_007 | 2 | 2 | TRUNC | 58.7 | 1.000 |
| argparse | argparse_real_008 | 2 | 0 | DUP_OPT,TRUNC | 60.7 | 1.000 |
| argparse | argparse_real_009 | 2 | 2 | TRUNC | 60.4 | 0.901 |
| argparse | argparse_real_013 | 2 | 2 | TRUNC | 60.6 | 0.990 |
| argparse | argparse_real_014 | 2 | 2 | TRUNC | 60.9 | 0.981 |
| argparse | argparse_real_015 | 2 | 0 | BRACKET | 57.4 | 0.967 |
| argparse | argparse_real_016 | 2 | 2 | TRUNC | 58.0 | 0.865 |
| argparse | argparse_real_017 | 2 | 2 | TRUNC | 58.8 | 1.000 |
| complex_nested_config | complex_nested_config_006 | 2 | 2 | TRUNC | 60.0 | 1.000 |
| complex_nested_config | complex_nested_config_022 | 2 | 2 | TRUNC | 60.6 | 1.000 |
| complex_nested_config | complex_nested_config_026 | 2 | 1 | TRUNC | 43.4 | 0.829 |
| complex_nested_config | complex_nested_config_030 | 2 | 2 | TRUNC | 61.3 | 1.000 |
| complex_nested_config | complex_nested_config_035 | 2 | 1 | TRUNC | 61.8 | 0.947 |
| complex_nested_config | complex_nested_config_042 | 2 | 1 | TRUNC | 61.0 | 0.960 |
| complex_nested_config | complex_nested_config_043 | 2 | 2 | TRUNC | 61.4 | 0.943 |
| complex_nested_config | complex_nested_config_055 | 2 | 0 | BRACKET,DUP_OPT | 19.1 | 0.884 |
| complex_nested_config | complex_nested_config_064 | 2 | 1 | TRUNC | 42.5 | 0.767 |
| complex_nested_config | complex_nested_config_068 | 2 | 2 | clean | 59.3 | 0.978 |
| dict_config | dict_config_real_001 | 2 | 2 | TRUNC | 60.7 | 1.000 |
| dict_config | dict_config_real_013 | 2 | 2 | TRUNC | 62.1 | 0.898 |
| dict_config | dict_config_real_014 | 2 | 2 | TRUNC | 29.7 | 0.973 |
| dict_config | dict_config_real_018 | 2 | 2 | TRUNC | 44.3 | 0.993 |
| dict_config | dict_config_real_020 | 2 | 2 | TRUNC | 59.6 | 1.000 |
| dict_config | dict_config_real_027 | 2 | 2 | TRUNC | 30.4 | 0.960 |
| dict_config | dict_config_real_028 | 2 | 2 | TRUNC | 31.1 | 0.960 |
| dict_config | dict_config_real_029 | 2 | 2 | TRUNC | 31.3 | 0.960 |
| dict_config | dict_config_real_030 | 2 | 2 | clean | 46.1 | 0.941 |
| dict_config | dict_config_real_032 | 2 | 2 | TRUNC | 62.2 | 0.983 |
| openmmlab_config | openmmlab_config_real_002 | 2 | 2 | TRUNC | 59.4 | 0.991 |
| openmmlab_config | openmmlab_config_real_004 | 2 | 0 | OFF_STRUCT,TRUNC | 62.2 | 0.987 |
| openmmlab_config | openmmlab_config_real_005 | 2 | 2 | TRUNC | 61.2 | 0.985 |
| openmmlab_config | openmmlab_config_real_006 | 2 | 2 | TRUNC | 62.4 | 0.991 |
| openmmlab_config | openmmlab_config_real_007 | 2 | 2 | TRUNC | 62.8 | 1.000 |
| openmmlab_config | openmmlab_config_real_009 | 2 | 2 | TRUNC | 62.7 | 0.982 |
| openmmlab_config | openmmlab_config_real_011 | 2 | 2 | TRUNC | 62.6 | 1.000 |
| openmmlab_config | openmmlab_config_real_013 | 2 | 2 | TRUNC | 62.2 | 0.982 |
| openmmlab_config | openmmlab_config_real_015 | 2 | 2 | TRUNC | 62.3 | 1.000 |
| openmmlab_config | openmmlab_config_real_016 | 2 | 2 | TRUNC | 61.1 | 0.991 |
| pipeline_stage_config | pipeline_stage_config_001 | 2 | 0 | OFF_STRUCT,TRUNC | 34.3 | 0.721 |
| pipeline_stage_config | pipeline_stage_config_002 | 2 | 2 | TRUNC | 61.9 | 0.980 |
| pipeline_stage_config | pipeline_stage_config_003 | 2 | 2 | TRUNC | 61.5 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_004 | 2 | 2 | TRUNC | 62.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_005 | 2 | 2 | TRUNC | 54.0 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_006 | 2 | 2 | TRUNC | 59.5 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_009 | 2 | 2 | TRUNC | 59.6 | 0.978 |
| pipeline_stage_config | pipeline_stage_config_011 | 2 | 2 | TRUNC | 62.1 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_012 | 2 | 2 | TRUNC | 61.5 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_014 | 2 | 2 | TRUNC | 35.6 | 0.947 |
| rich_cli_option_groups | rich_cli_option_groups_005 | 2 | 2 | TRUNC | 62.0 | 0.939 |
| rich_cli_option_groups | rich_cli_option_groups_010 | 2 | 2 | clean | 61.5 | 0.983 |
| rich_cli_option_groups | rich_cli_option_groups_012 | 2 | 2 | TRUNC | 62.1 | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_019 | 2 | 2 | TRUNC | 61.7 | 0.936 |
| rich_cli_option_groups | rich_cli_option_groups_033 | 2 | 2 | clean | 61.6 | 0.971 |
| rich_cli_option_groups | rich_cli_option_groups_035 | 2 | 1 | TRUNC | 59.7 | 0.743 |
| rich_cli_option_groups | rich_cli_option_groups_051 | 2 | 2 | TRUNC | 58.7 | 0.938 |
| rich_cli_option_groups | rich_cli_option_groups_052 | 2 | 2 | TRUNC | 35.4 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_059 | 2 | 2 | clean | 58.8 | 0.967 |
| rich_cli_option_groups | rich_cli_option_groups_068 | 2 | 1 | TRUNC | 58.2 | 0.826 |

### TASD-FG-Safe-k1 - Clean Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_002 | 2 | 2 | TRUNC | 59.9 | 1.000 |
| argparse | argparse_real_003 | 2 | 2 | TRUNC | 60.0 | 0.933 |
| argparse | argparse_real_007 | 2 | 2 | TRUNC | 58.8 | 1.000 |
| argparse | argparse_real_008 | 2 | 0 | DUP_OPT,TRUNC | 58.3 | 1.000 |
| argparse | argparse_real_009 | 2 | 2 | TRUNC | 58.6 | 0.901 |
| argparse | argparse_real_013 | 2 | 2 | TRUNC | 59.1 | 0.990 |
| argparse | argparse_real_014 | 2 | 2 | TRUNC | 60.1 | 0.981 |
| argparse | argparse_real_015 | 2 | 0 | BRACKET | 60.4 | 0.967 |
| argparse | argparse_real_016 | 2 | 2 | TRUNC | 58.4 | 0.865 |
| argparse | argparse_real_017 | 2 | 2 | TRUNC | 59.7 | 1.000 |
| complex_nested_config | complex_nested_config_006 | 2 | 2 | TRUNC | 60.5 | 1.000 |
| complex_nested_config | complex_nested_config_022 | 2 | 2 | TRUNC | 61.7 | 1.000 |
| complex_nested_config | complex_nested_config_026 | 2 | 1 | TRUNC | 44.2 | 0.829 |
| complex_nested_config | complex_nested_config_030 | 2 | 2 | TRUNC | 59.8 | 1.000 |
| complex_nested_config | complex_nested_config_035 | 2 | 1 | TRUNC | 57.4 | 0.947 |
| complex_nested_config | complex_nested_config_042 | 2 | 1 | TRUNC | 58.5 | 0.960 |
| complex_nested_config | complex_nested_config_043 | 2 | 2 | TRUNC | 62.1 | 0.943 |
| complex_nested_config | complex_nested_config_055 | 2 | 0 | BRACKET,DUP_OPT | 19.6 | 0.884 |
| complex_nested_config | complex_nested_config_064 | 2 | 1 | TRUNC | 44.3 | 0.767 |
| complex_nested_config | complex_nested_config_068 | 2 | 2 | clean | 61.6 | 0.978 |
| dict_config | dict_config_real_001 | 2 | 2 | TRUNC | 61.8 | 1.000 |
| dict_config | dict_config_real_013 | 2 | 2 | TRUNC | 62.4 | 0.898 |
| dict_config | dict_config_real_014 | 2 | 2 | TRUNC | 30.3 | 0.973 |
| dict_config | dict_config_real_018 | 2 | 2 | TRUNC | 45.2 | 0.993 |
| dict_config | dict_config_real_020 | 2 | 2 | TRUNC | 63.2 | 1.000 |
| dict_config | dict_config_real_027 | 2 | 2 | TRUNC | 31.3 | 0.960 |
| dict_config | dict_config_real_028 | 2 | 2 | TRUNC | 31.0 | 0.960 |
| dict_config | dict_config_real_029 | 2 | 2 | TRUNC | 31.3 | 0.960 |
| dict_config | dict_config_real_030 | 2 | 2 | clean | 47.1 | 0.941 |
| dict_config | dict_config_real_032 | 2 | 2 | TRUNC | 62.8 | 0.983 |
| openmmlab_config | openmmlab_config_real_002 | 2 | 2 | TRUNC | 62.3 | 0.991 |
| openmmlab_config | openmmlab_config_real_004 | 2 | 0 | OFF_STRUCT,TRUNC | 62.6 | 0.987 |
| openmmlab_config | openmmlab_config_real_005 | 2 | 2 | TRUNC | 61.5 | 0.985 |
| openmmlab_config | openmmlab_config_real_006 | 2 | 2 | TRUNC | 61.6 | 0.991 |
| openmmlab_config | openmmlab_config_real_007 | 2 | 2 | TRUNC | 61.5 | 1.000 |
| openmmlab_config | openmmlab_config_real_009 | 2 | 2 | TRUNC | 60.9 | 0.982 |
| openmmlab_config | openmmlab_config_real_011 | 2 | 2 | TRUNC | 62.2 | 1.000 |
| openmmlab_config | openmmlab_config_real_013 | 2 | 2 | TRUNC | 61.5 | 0.982 |
| openmmlab_config | openmmlab_config_real_015 | 2 | 2 | TRUNC | 61.7 | 1.000 |
| openmmlab_config | openmmlab_config_real_016 | 2 | 2 | TRUNC | 61.5 | 0.991 |
| pipeline_stage_config | pipeline_stage_config_001 | 2 | 0 | OFF_STRUCT,TRUNC | 35.0 | 0.721 |
| pipeline_stage_config | pipeline_stage_config_002 | 2 | 2 | TRUNC | 63.0 | 0.980 |
| pipeline_stage_config | pipeline_stage_config_003 | 2 | 2 | TRUNC | 62.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_004 | 2 | 2 | TRUNC | 62.6 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_005 | 2 | 2 | TRUNC | 62.2 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_006 | 2 | 2 | TRUNC | 62.8 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_009 | 2 | 2 | TRUNC | 62.3 | 0.978 |
| pipeline_stage_config | pipeline_stage_config_011 | 2 | 2 | TRUNC | 63.0 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_012 | 2 | 2 | TRUNC | 62.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_014 | 2 | 2 | TRUNC | 35.9 | 0.947 |
| rich_cli_option_groups | rich_cli_option_groups_005 | 2 | 2 | TRUNC | 60.6 | 0.939 |
| rich_cli_option_groups | rich_cli_option_groups_010 | 2 | 2 | clean | 60.8 | 0.983 |
| rich_cli_option_groups | rich_cli_option_groups_012 | 2 | 2 | TRUNC | 60.6 | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_019 | 2 | 2 | TRUNC | 59.9 | 0.936 |
| rich_cli_option_groups | rich_cli_option_groups_033 | 2 | 2 | clean | 61.0 | 0.971 |
| rich_cli_option_groups | rich_cli_option_groups_035 | 2 | 1 | TRUNC | 59.8 | 0.743 |
| rich_cli_option_groups | rich_cli_option_groups_051 | 2 | 2 | TRUNC | 59.4 | 0.938 |
| rich_cli_option_groups | rich_cli_option_groups_052 | 2 | 2 | TRUNC | 36.6 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_059 | 2 | 2 | clean | 60.6 | 0.967 |
| rich_cli_option_groups | rich_cli_option_groups_068 | 2 | 1 | TRUNC | 59.7 | 0.826 |

### TASD-FG-strict - Clean Subset

| Benchmark | Name | Original | New | Error Tags | TPS | F1 |
|-----------|------|:--------:|:---:|------------|----:|---:|
| argparse | argparse_real_002 | 2 | 0 | BRACKET | 52.1 | 1.000 |
| argparse | argparse_real_003 | 2 | 2 | TRUNC | 49.4 | 0.933 |
| argparse | argparse_real_007 | 2 | 2 | TRUNC | 53.2 | 1.000 |
| argparse | argparse_real_008 | 2 | 0 | DUP_OPT,TRUNC | 51.5 | 1.000 |
| argparse | argparse_real_009 | 2 | 2 | TRUNC | 52.5 | 0.901 |
| argparse | argparse_real_013 | 2 | 2 | TRUNC | 56.6 | 0.990 |
| argparse | argparse_real_014 | 2 | 2 | TRUNC | 54.7 | 0.981 |
| argparse | argparse_real_015 | 2 | 0 | BRACKET | 49.8 | 0.967 |
| argparse | argparse_real_016 | 2 | 2 | TRUNC | 52.4 | 0.865 |
| argparse | argparse_real_017 | 2 | 2 | TRUNC | 59.0 | 1.000 |
| complex_nested_config | complex_nested_config_006 | 2 | 2 | TRUNC | 52.5 | 1.000 |
| complex_nested_config | complex_nested_config_022 | 2 | 2 | TRUNC | 55.5 | 1.000 |
| complex_nested_config | complex_nested_config_026 | 2 | 1 | TRUNC | 43.5 | 0.833 |
| complex_nested_config | complex_nested_config_030 | 2 | 2 | TRUNC | 55.4 | 1.000 |
| complex_nested_config | complex_nested_config_035 | 2 | 1 | TRUNC | 49.3 | 0.947 |
| complex_nested_config | complex_nested_config_042 | 2 | 1 | TRUNC | 54.5 | 0.960 |
| complex_nested_config | complex_nested_config_043 | 2 | 2 | TRUNC | 58.7 | 0.943 |
| complex_nested_config | complex_nested_config_055 | 2 | 2 | TRUNC | 57.4 | 0.959 |
| complex_nested_config | complex_nested_config_064 | 2 | 1 | TRUNC | 53.2 | 0.844 |
| complex_nested_config | complex_nested_config_068 | 2 | 2 | clean | 55.5 | 0.978 |
| dict_config | dict_config_real_001 | 2 | 2 | clean | 56.2 | 1.000 |
| dict_config | dict_config_real_013 | 2 | 2 | TRUNC | 49.4 | 0.898 |
| dict_config | dict_config_real_014 | 2 | 2 | TRUNC | 33.9 | 1.000 |
| dict_config | dict_config_real_018 | 2 | 2 | TRUNC | 43.8 | 0.993 |
| dict_config | dict_config_real_020 | 2 | 2 | TRUNC | 55.3 | 1.000 |
| dict_config | dict_config_real_027 | 2 | 2 | TRUNC | 57.4 | 0.986 |
| dict_config | dict_config_real_028 | 2 | 2 | TRUNC | 58.3 | 0.986 |
| dict_config | dict_config_real_029 | 2 | 2 | TRUNC | 58.0 | 0.986 |
| dict_config | dict_config_real_030 | 2 | 2 | clean | 55.9 | 0.892 |
| dict_config | dict_config_real_032 | 2 | 2 | TRUNC | 53.0 | 0.983 |
| openmmlab_config | openmmlab_config_real_002 | 2 | 2 | TRUNC | 50.5 | 0.991 |
| openmmlab_config | openmmlab_config_real_004 | 2 | 0 | OFF_STRUCT,TRUNC | 51.1 | 0.987 |
| openmmlab_config | openmmlab_config_real_005 | 2 | 2 | TRUNC | 52.9 | 0.985 |
| openmmlab_config | openmmlab_config_real_006 | 2 | 2 | TRUNC | 49.3 | 0.991 |
| openmmlab_config | openmmlab_config_real_007 | 2 | 2 | TRUNC | 56.4 | 1.000 |
| openmmlab_config | openmmlab_config_real_009 | 2 | 2 | TRUNC | 56.3 | 0.982 |
| openmmlab_config | openmmlab_config_real_011 | 2 | 2 | TRUNC | 59.3 | 1.000 |
| openmmlab_config | openmmlab_config_real_013 | 2 | 2 | TRUNC | 56.1 | 0.982 |
| openmmlab_config | openmmlab_config_real_015 | 2 | 2 | TRUNC | 51.1 | 1.000 |
| openmmlab_config | openmmlab_config_real_016 | 2 | 2 | TRUNC | 45.0 | 0.991 |
| pipeline_stage_config | pipeline_stage_config_001 | 2 | 2 | TRUNC | 55.5 | 0.932 |
| pipeline_stage_config | pipeline_stage_config_002 | 2 | 2 | TRUNC | 58.6 | 0.980 |
| pipeline_stage_config | pipeline_stage_config_003 | 2 | 2 | TRUNC | 57.7 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_004 | 2 | 2 | TRUNC | 51.9 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_005 | 2 | 2 | TRUNC | 56.3 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_006 | 2 | 2 | TRUNC | 55.9 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_009 | 2 | 2 | TRUNC | 54.6 | 0.978 |
| pipeline_stage_config | pipeline_stage_config_011 | 2 | 2 | TRUNC | 52.2 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_012 | 2 | 2 | TRUNC | 53.1 | 1.000 |
| pipeline_stage_config | pipeline_stage_config_014 | 2 | 2 | TRUNC | 46.8 | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_005 | 2 | 2 | TRUNC | 50.1 | 0.939 |
| rich_cli_option_groups | rich_cli_option_groups_010 | 2 | 2 | clean | 53.6 | 0.983 |
| rich_cli_option_groups | rich_cli_option_groups_012 | 2 | 2 | TRUNC | 54.6 | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_019 | 2 | 2 | TRUNC | 57.3 | 0.936 |
| rich_cli_option_groups | rich_cli_option_groups_033 | 2 | 2 | clean | 55.9 | 0.971 |
| rich_cli_option_groups | rich_cli_option_groups_035 | 2 | 1 | TRUNC | 54.3 | 0.743 |
| rich_cli_option_groups | rich_cli_option_groups_051 | 2 | 2 | TRUNC | 57.8 | 0.938 |
| rich_cli_option_groups | rich_cli_option_groups_052 | 2 | 2 | clean | 58.3 | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_059 | 2 | 2 | clean | 56.2 | 0.967 |
| rich_cli_option_groups | rich_cli_option_groups_068 | 2 | 1 | TRUNC | 47.9 | 0.826 |
