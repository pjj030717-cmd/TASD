# TASD-FG-ES: Structure-Aware Early Stopping Pilot

## 配置

| Config | top_k | prefix_budget | window_len | early_stopping |
|--------|:-----:|:-------------:|:----------:|:--------------:|
| TASD-FG | 3 | 0.2 | 2 | OFF |
| TASD-FG-ES | 3 | 0.2 | 2 | ON |

## Hard Subset (score=0 样本)

| 方法 | N | TPS | speedup | score2 | score1 | score0 | recoverable | 平均长度 | F1 | BB |
|------|--:|-----:|-------:|------:|------:|------:|-----------:|-------:|---:|---:|
| TASD-FG | 60 | 59.2 | — | 1 | 2 | 57 | 3/60 (5.0%) | 128 | 0.635 | 0.300 |
| TASD-FG-ES | 60 | 59.6 | — | 1 | 1 | 58 | 2/60 (3.3%) | 124 | 0.629 | 0.317 |

### Hard Subset 对比

| 指标 | TASD-FG | TASD-FG-ES | 变化 |
|------|--------:|----------:|------|
| score=0 | 57 | 58 | +1 |
| recoverable | 3 (5.0%) | 2 (3.3%) | -1 |
| 平均长度 | 128 | 124 | -4 |
| TPS | 59.2 | 59.6 | +0.4 |
| F1 | 0.635 | 0.629 | -0.006 |

### TASD-FG-ES Stop Reason Distribution (Hard)

| Stop Reason | Count |
|-------------|------:|
| max_tokens | 55 |
| structure_complete | 5 |

## Clean Subset (score=2 样本, no-regression)

| 方法 | N | TPS | speedup | score2 | score1 | score0 | score2保持率 | 平均长度 | F1 | BB |
|------|--:|-----:|-------:|------:|------:|------:|-----------:|-------:|---:|---:|
| TASD-FG | 60 | 61.7 | — | 56 | 1 | 3 | 93.3% | 128 | 0.971 | 0.233 |
| TASD-FG-ES | 60 | 62.2 | — | 55 | 1 | 4 | 91.7% | 125 | 0.967 | 0.233 |

### Clean Subset 对比

| 指标 | TASD-FG | TASD-FG-ES | 变化 |
|------|--------:|----------:|------|
| score2保持率 | 93.3% | 91.7% | -1.7pp |
| score=0退化 | 3 | 4 | +1 |
| 平均长度 | 128 | 125 | -3 |
| TPS | 61.7 | 62.2 | +0.5 |
| F1 | 0.971 | 0.967 | -0.004 |

### TASD-FG-ES Stop Reason Distribution (Clean)

| Stop Reason | Count |
|-------------|------:|
| max_tokens | 57 |
| structure_complete | 3 |

## 决策分析

| 指标 | TASD-FG | TASD-FG-ES | 变化 |
|------|--------:|----------:|------|
| Hard score=0 | 57 | 58 | -1 (-2%) |
| Recoverable | 5.0% | 3.3% | -1.7pp |
| Clean retention | 93.3% | 91.7% | -1.7pp |
| TPS (hard) | 59.2 | 59.6 | +0.7% |
| 平均长度 (hard) | 128 | 124 | -2.7% |

**判定**: 情况 C: 不扩大

**建议**: 保留 TASD-FG 作为唯一版本，early stopping 未带来足够提升

## 样本详情

### TASD-FG - Hard Subset

| Benchmark | Name | Original | New | Error Tags | TPS | Len | Stop | F1 |
|-----------|------|:--------:|:---:|------------|----:|----:|------|----:|
| argparse | argparse_real_001 | 0 | 0 | BRACKET | 45.7 | 128 | max_tokens | 0.775 |
| argparse | argparse_real_004 | 0 | 0 | BRACKET,DUP_OPT | 61.0 | 128 | max_tokens | 0.678 |
| argparse | argparse_real_006 | 0 | 0 | BRACKET | 63.1 | 128 | max_tokens | 0.933 |
| argparse | argparse_real_010 | 0 | 0 | BRACKET | 60.9 | 128 | max_tokens | 0.990 |
| argparse | argparse_real_012 | 0 | 0 | BRACKET | 61.8 | 128 | max_tokens | 0.852 |
| argparse | argparse_real_018 | 0 | 0 | BRACKET | 59.9 | 128 | max_tokens | 0.878 |
| argparse | argparse_real_019 | 0 | 2 | TRUNC | 56.1 | 128 | max_tokens | 0.977 |
| argparse | argparse_real_029 | 0 | 0 | BRACKET | 62.4 | 128 | max_tokens | 0.990 |
| argparse | argparse_real_030 | 0 | 1 | REPEAT,TRUNC | 7.1 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_031 | 0 | 0 | REPEAT,TRUNC | 3.5 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_002 | 0 | 0 | LOW_F1,BRACKET | 58.8 | 128 | max_tokens | 0.042 |
| complex_nested_config | complex_nested_config_004 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 62.0 | 128 | max_tokens | 0.420 |
| complex_nested_config | complex_nested_config_007 | 0 | 1 | OFF_STRUCT,TRUNC | 60.7 | 128 | max_tokens | 0.740 |
| complex_nested_config | complex_nested_config_008 | 0 | 0 | LOW_F1,TRUNC | 62.7 | 128 | max_tokens | 0.011 |
| complex_nested_config | complex_nested_config_014 | 0 | 0 | LOW_F1,TRUNC | 63.2 | 128 | max_tokens | 0.001 |
| complex_nested_config | complex_nested_config_016 | 0 | 0 | LOW_F1,BRACKET | 63.4 | 128 | max_tokens | 0.147 |
| complex_nested_config | complex_nested_config_017 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 62.4 | 128 | max_tokens | 0.136 |
| complex_nested_config | complex_nested_config_018 | 0 | 0 | LOW_F1,BRACKET | 61.8 | 128 | max_tokens | 0.352 |
| complex_nested_config | complex_nested_config_020 | 0 | 0 | LOW_F1,BRACKET | 62.2 | 128 | max_tokens | 0.168 |
| complex_nested_config | complex_nested_config_021 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 62.6 | 128 | max_tokens | 0.346 |
| dict_config | dict_config_real_002 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 63.0 | 128 | max_tokens | 0.376 |
| dict_config | dict_config_real_005 | 0 | 0 | BRACKET | 63.0 | 128 | max_tokens | 0.789 |
| dict_config | dict_config_real_008 | 0 | 0 | BRACKET | 37.9 | 128 | max_tokens | 0.871 |
| dict_config | dict_config_real_011 | 0 | 0 | LOW_F1,TRUNC | 63.6 | 128 | max_tokens | 0.100 |
| dict_config | dict_config_real_015 | 0 | 0 | REPEAT,BRACKET,DUP_OPT | 63.9 | 128 | max_tokens | 0.901 |
| dict_config | dict_config_real_016 | 0 | 0 | REPEAT,DUP_OPT | 63.4 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_022 | 0 | 0 | LOW_F1,BRACKET | 63.1 | 128 | max_tokens | 0.175 |
| dict_config | dict_config_real_041 | 0 | 0 | BRACKET | 62.9 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_045 | 0 | 0 | LOW_F1,TRUNC | 62.4 | 128 | max_tokens | 0.128 |
| dict_config | dict_config_real_049 | 0 | 0 | LOW_F1,TRUNC | 62.0 | 128 | max_tokens | 0.070 |
| openmmlab_config | openmmlab_config_real_001 | 0 | 0 | OFF_STRUCT,TRUNC | 61.6 | 128 | max_tokens | 0.991 |
| openmmlab_config | openmmlab_config_real_003 | 0 | 0 | OFF_STRUCT,TRUNC | 61.7 | 128 | max_tokens | 0.991 |
| openmmlab_config | openmmlab_config_real_008 | 0 | 0 | BRACKET | 61.1 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_012 | 0 | 0 | BRACKET | 60.8 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_017 | 0 | 0 | OFF_STRUCT,TRUNC | 63.1 | 128 | max_tokens | 0.989 |
| openmmlab_config | openmmlab_config_real_018 | 0 | 0 | BRACKET | 61.0 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_021 | 0 | 0 | OFF_STRUCT,TRUNC | 61.0 | 128 | max_tokens | 0.989 |
| openmmlab_config | openmmlab_config_real_025 | 0 | 0 | LOW_F1,TRUNC | 62.6 | 128 | max_tokens | 0.000 |
| openmmlab_config | openmmlab_config_real_029 | 0 | 0 | LOW_F1,TRUNC | 62.6 | 128 | max_tokens | 0.000 |
| openmmlab_config | openmmlab_config_real_041 | 0 | 0 | BRACKET | 62.7 | 128 | max_tokens | 0.992 |
| pipeline_stage_config | pipeline_stage_config_007 | 0 | 0 | BRACKET | 62.0 | 128 | max_tokens | 0.702 |
| pipeline_stage_config | pipeline_stage_config_008 | 0 | 0 | BRACKET | 62.2 | 128 | max_tokens | 0.841 |
| pipeline_stage_config | pipeline_stage_config_010 | 0 | 0 | OFF_STRUCT,TRUNC | 62.0 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_019 | 0 | 0 | OFF_STRUCT,TRUNC | 47.1 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_022 | 0 | 0 | BRACKET | 62.4 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_025 | 0 | 0 | BRACKET | 62.9 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_034 | 0 | 0 | OFF_STRUCT,TRUNC | 62.4 | 128 | max_tokens | 0.988 |
| pipeline_stage_config | pipeline_stage_config_037 | 0 | 0 | BRACKET | 60.9 | 128 | max_tokens | 0.992 |
| pipeline_stage_config | pipeline_stage_config_041 | 0 | 0 | REPEAT,DUP_OPT,TRUNC | 63.2 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_043 | 0 | 0 | LOW_F1,TRUNC | 63.1 | 128 | max_tokens | 0.000 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0 | 0 | BRACKET | 61.7 | 128 | max_tokens | 0.903 |
| rich_cli_option_groups | rich_cli_option_groups_013 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1,TRUNC | 62.0 | 128 | max_tokens | 0.174 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 0 | 0 | BRACKET,DUP_OPT | 62.7 | 128 | max_tokens | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_017 | 0 | 0 | LOW_F1,BRACKET,DUP_OPT | 61.8 | 128 | max_tokens | 0.444 |
| rich_cli_option_groups | rich_cli_option_groups_022 | 0 | 0 | OFF_STRUCT,LOW_F1,TRUNC | 61.3 | 128 | max_tokens | 0.325 |
| rich_cli_option_groups | rich_cli_option_groups_025 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1 | 62.2 | 128 | max_tokens | 0.237 |
| rich_cli_option_groups | rich_cli_option_groups_027 | 0 | 0 | LOW_F1,TRUNC | 62.3 | 128 | max_tokens | 0.026 |
| rich_cli_option_groups | rich_cli_option_groups_029 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 62.2 | 128 | max_tokens | 0.718 |
| rich_cli_option_groups | rich_cli_option_groups_031 | 0 | 0 | LOW_F1,TRUNC | 61.2 | 128 | max_tokens | 0.196 |
| rich_cli_option_groups | rich_cli_option_groups_032 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 60.5 | 128 | max_tokens | 0.785 |

### TASD-FG-ES - Hard Subset

| Benchmark | Name | Original | New | Error Tags | TPS | Len | Stop | F1 |
|-----------|------|:--------:|:---:|------------|----:|----:|------|----:|
| argparse | argparse_real_001 | 0 | 0 | BRACKET | 58.5 | 128 | max_tokens | 0.775 |
| argparse | argparse_real_004 | 0 | 0 | BRACKET,DUP_OPT | 59.7 | 128 | max_tokens | 0.678 |
| argparse | argparse_real_006 | 0 | 0 | BRACKET | 59.3 | 128 | max_tokens | 0.933 |
| argparse | argparse_real_010 | 0 | 0 | BRACKET | 61.0 | 128 | max_tokens | 0.990 |
| argparse | argparse_real_012 | 0 | 0 | BRACKET | 60.7 | 128 | max_tokens | 0.852 |
| argparse | argparse_real_018 | 0 | 0 | BRACKET | 61.1 | 128 | max_tokens | 0.878 |
| argparse | argparse_real_019 | 0 | 2 | TRUNC | 56.8 | 128 | max_tokens | 0.977 |
| argparse | argparse_real_029 | 0 | 0 | BRACKET | 61.5 | 128 | max_tokens | 0.990 |
| argparse | argparse_real_030 | 0 | 1 | REPEAT,TRUNC | 7.1 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_031 | 0 | 0 | REPEAT,TRUNC | 3.6 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_002 | 0 | 0 | LOW_F1 | 62.0 | 118 | structure_complete | 0.036 |
| complex_nested_config | complex_nested_config_004 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 63.4 | 128 | max_tokens | 0.420 |
| complex_nested_config | complex_nested_config_007 | 0 | 0 | LOW_F1,BRACKET | 46.6 | 31 | structure_complete | 0.400 |
| complex_nested_config | complex_nested_config_008 | 0 | 0 | LOW_F1,TRUNC | 62.6 | 128 | max_tokens | 0.011 |
| complex_nested_config | complex_nested_config_014 | 0 | 0 | LOW_F1,TRUNC | 63.1 | 128 | max_tokens | 0.001 |
| complex_nested_config | complex_nested_config_016 | 0 | 0 | LOW_F1,BRACKET | 63.0 | 128 | max_tokens | 0.147 |
| complex_nested_config | complex_nested_config_017 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT,TRUNC | 62.9 | 128 | max_tokens | 0.136 |
| complex_nested_config | complex_nested_config_018 | 0 | 0 | LOW_F1,BRACKET | 62.7 | 128 | max_tokens | 0.352 |
| complex_nested_config | complex_nested_config_020 | 0 | 0 | LOW_F1,BRACKET | 62.6 | 128 | max_tokens | 0.168 |
| complex_nested_config | complex_nested_config_021 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 63.3 | 128 | max_tokens | 0.346 |
| dict_config | dict_config_real_002 | 0 | 0 | REPEAT,LOW_F1,DUP_OPT | 63.1 | 128 | max_tokens | 0.376 |
| dict_config | dict_config_real_005 | 0 | 0 | BRACKET | 63.0 | 128 | max_tokens | 0.789 |
| dict_config | dict_config_real_008 | 0 | 0 | BRACKET | 56.6 | 66 | structure_complete | 0.830 |
| dict_config | dict_config_real_011 | 0 | 0 | LOW_F1,TRUNC | 63.7 | 128 | max_tokens | 0.100 |
| dict_config | dict_config_real_015 | 0 | 0 | REPEAT,BRACKET,DUP_OPT | 63.8 | 128 | max_tokens | 0.901 |
| dict_config | dict_config_real_016 | 0 | 0 | REPEAT,DUP_OPT | 63.9 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_022 | 0 | 0 | LOW_F1,BRACKET | 56.8 | 128 | max_tokens | 0.175 |
| dict_config | dict_config_real_041 | 0 | 0 | BRACKET | 63.6 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_045 | 0 | 0 | LOW_F1,TRUNC | 62.4 | 128 | max_tokens | 0.128 |
| dict_config | dict_config_real_049 | 0 | 0 | LOW_F1,TRUNC | 63.3 | 128 | max_tokens | 0.070 |
| openmmlab_config | openmmlab_config_real_001 | 0 | 0 | OFF_STRUCT,TRUNC | 63.8 | 128 | max_tokens | 0.991 |
| openmmlab_config | openmmlab_config_real_003 | 0 | 0 | OFF_STRUCT,TRUNC | 63.2 | 128 | max_tokens | 0.991 |
| openmmlab_config | openmmlab_config_real_008 | 0 | 0 | BRACKET | 61.8 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_012 | 0 | 0 | BRACKET | 61.4 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_017 | 0 | 0 | OFF_STRUCT,TRUNC | 63.6 | 128 | max_tokens | 0.989 |
| openmmlab_config | openmmlab_config_real_018 | 0 | 0 | BRACKET | 63.5 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_021 | 0 | 0 | OFF_STRUCT,TRUNC | 63.3 | 128 | max_tokens | 0.989 |
| openmmlab_config | openmmlab_config_real_025 | 0 | 0 | LOW_F1,TRUNC | 63.9 | 128 | max_tokens | 0.000 |
| openmmlab_config | openmmlab_config_real_029 | 0 | 0 | LOW_F1,TRUNC | 63.4 | 128 | max_tokens | 0.000 |
| openmmlab_config | openmmlab_config_real_041 | 0 | 0 | BRACKET | 63.3 | 128 | max_tokens | 0.992 |
| pipeline_stage_config | pipeline_stage_config_007 | 0 | 0 | BRACKET | 63.4 | 128 | max_tokens | 0.702 |
| pipeline_stage_config | pipeline_stage_config_008 | 0 | 0 | BRACKET | 62.9 | 128 | max_tokens | 0.841 |
| pipeline_stage_config | pipeline_stage_config_010 | 0 | 0 | BRACKET | 61.7 | 110 | structure_complete | 1.000 |
| pipeline_stage_config | pipeline_stage_config_019 | 0 | 0 | BRACKET | 42.4 | 104 | structure_complete | 1.000 |
| pipeline_stage_config | pipeline_stage_config_022 | 0 | 0 | BRACKET | 60.3 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_025 | 0 | 0 | BRACKET | 59.7 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_034 | 0 | 0 | OFF_STRUCT,TRUNC | 59.2 | 128 | max_tokens | 0.988 |
| pipeline_stage_config | pipeline_stage_config_037 | 0 | 0 | BRACKET | 59.5 | 128 | max_tokens | 0.992 |
| pipeline_stage_config | pipeline_stage_config_041 | 0 | 0 | REPEAT,DUP_OPT,TRUNC | 60.5 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_043 | 0 | 0 | LOW_F1,TRUNC | 63.3 | 128 | max_tokens | 0.000 |
| rich_cli_option_groups | rich_cli_option_groups_007 | 0 | 0 | BRACKET | 63.4 | 128 | max_tokens | 0.903 |
| rich_cli_option_groups | rich_cli_option_groups_013 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1,TRUNC | 61.3 | 128 | max_tokens | 0.174 |
| rich_cli_option_groups | rich_cli_option_groups_016 | 0 | 0 | BRACKET,DUP_OPT | 63.2 | 128 | max_tokens | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_017 | 0 | 0 | LOW_F1,BRACKET,DUP_OPT | 62.6 | 128 | max_tokens | 0.444 |
| rich_cli_option_groups | rich_cli_option_groups_022 | 0 | 0 | OFF_STRUCT,LOW_F1,TRUNC | 62.6 | 128 | max_tokens | 0.325 |
| rich_cli_option_groups | rich_cli_option_groups_025 | 0 | 0 | REPEAT,OFF_STRUCT,LOW_F1 | 62.8 | 128 | max_tokens | 0.237 |
| rich_cli_option_groups | rich_cli_option_groups_027 | 0 | 0 | LOW_F1,TRUNC | 63.1 | 128 | max_tokens | 0.026 |
| rich_cli_option_groups | rich_cli_option_groups_029 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 63.5 | 128 | max_tokens | 0.718 |
| rich_cli_option_groups | rich_cli_option_groups_031 | 0 | 0 | LOW_F1,TRUNC | 62.9 | 128 | max_tokens | 0.196 |
| rich_cli_option_groups | rich_cli_option_groups_032 | 0 | 0 | REPEAT,OFF_STRUCT,TRUNC | 63.2 | 128 | max_tokens | 0.785 |

### TASD-FG - Clean Subset

| Benchmark | Name | Original | New | Error Tags | TPS | Len | Stop | F1 |
|-----------|------|:--------:|:---:|------------|----:|----:|------|----:|
| argparse | argparse_real_002 | 2 | 2 | TRUNC | 63.1 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_003 | 2 | 2 | TRUNC | 62.8 | 128 | max_tokens | 0.933 |
| argparse | argparse_real_007 | 2 | 2 | TRUNC | 63.2 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_008 | 2 | 0 | DUP_OPT,TRUNC | 62.9 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_009 | 2 | 2 | TRUNC | 62.8 | 128 | max_tokens | 0.901 |
| argparse | argparse_real_013 | 2 | 2 | TRUNC | 63.0 | 128 | max_tokens | 0.992 |
| argparse | argparse_real_014 | 2 | 2 | TRUNC | 62.8 | 128 | max_tokens | 0.981 |
| argparse | argparse_real_015 | 2 | 2 | TRUNC | 63.0 | 128 | max_tokens | 0.949 |
| argparse | argparse_real_016 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.855 |
| argparse | argparse_real_017 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_006 | 2 | 2 | TRUNC | 63.5 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_022 | 2 | 2 | TRUNC | 63.4 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_026 | 2 | 2 | TRUNC | 63.2 | 128 | max_tokens | 0.989 |
| complex_nested_config | complex_nested_config_030 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_035 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.969 |
| complex_nested_config | complex_nested_config_042 | 2 | 1 | TRUNC | 62.9 | 128 | max_tokens | 0.960 |
| complex_nested_config | complex_nested_config_043 | 2 | 2 | TRUNC | 63.1 | 128 | max_tokens | 0.943 |
| complex_nested_config | complex_nested_config_055 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 0.988 |
| complex_nested_config | complex_nested_config_064 | 2 | 2 | TRUNC | 63.4 | 128 | max_tokens | 0.986 |
| complex_nested_config | complex_nested_config_068 | 2 | 2 | clean | 63.7 | 128 | max_tokens | 0.978 |
| dict_config | dict_config_real_001 | 2 | 2 | TRUNC | 63.6 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_013 | 2 | 0 | BRACKET | 62.7 | 128 | max_tokens | 0.900 |
| dict_config | dict_config_real_014 | 2 | 2 | TRUNC | 30.1 | 128 | max_tokens | 0.991 |
| dict_config | dict_config_real_018 | 2 | 0 | BRACKET | 40.9 | 128 | max_tokens | 0.986 |
| dict_config | dict_config_real_020 | 2 | 2 | clean | 64.0 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_027 | 2 | 2 | TRUNC | 63.7 | 128 | max_tokens | 0.912 |
| dict_config | dict_config_real_028 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 0.912 |
| dict_config | dict_config_real_029 | 2 | 2 | TRUNC | 63.7 | 128 | max_tokens | 0.912 |
| dict_config | dict_config_real_030 | 2 | 2 | clean | 63.3 | 128 | max_tokens | 0.892 |
| dict_config | dict_config_real_032 | 2 | 2 | clean | 63.7 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_002 | 2 | 2 | TRUNC | 61.7 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_004 | 2 | 2 | TRUNC | 62.6 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_005 | 2 | 2 | TRUNC | 61.3 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_006 | 2 | 2 | TRUNC | 63.5 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_007 | 2 | 2 | TRUNC | 64.0 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_009 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 0.981 |
| openmmlab_config | openmmlab_config_real_011 | 2 | 2 | TRUNC | 64.2 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_013 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 0.981 |
| openmmlab_config | openmmlab_config_real_015 | 2 | 2 | TRUNC | 64.6 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_016 | 2 | 2 | TRUNC | 61.7 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_001 | 2 | 2 | TRUNC | 61.6 | 128 | max_tokens | 0.932 |
| pipeline_stage_config | pipeline_stage_config_002 | 2 | 2 | TRUNC | 62.9 | 128 | max_tokens | 0.980 |
| pipeline_stage_config | pipeline_stage_config_003 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_004 | 2 | 2 | TRUNC | 60.3 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_005 | 2 | 2 | TRUNC | 59.2 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_006 | 2 | 2 | TRUNC | 55.8 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_009 | 2 | 2 | TRUNC | 60.0 | 128 | max_tokens | 0.978 |
| pipeline_stage_config | pipeline_stage_config_011 | 2 | 2 | TRUNC | 60.6 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_012 | 2 | 2 | TRUNC | 62.6 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_014 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_005 | 2 | 2 | TRUNC | 61.4 | 128 | max_tokens | 0.939 |
| rich_cli_option_groups | rich_cli_option_groups_010 | 2 | 2 | clean | 61.9 | 128 | max_tokens | 0.983 |
| rich_cli_option_groups | rich_cli_option_groups_012 | 2 | 2 | TRUNC | 61.7 | 128 | max_tokens | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_019 | 2 | 2 | TRUNC | 62.6 | 128 | max_tokens | 0.936 |
| rich_cli_option_groups | rich_cli_option_groups_033 | 2 | 2 | clean | 61.8 | 128 | max_tokens | 0.971 |
| rich_cli_option_groups | rich_cli_option_groups_035 | 2 | 2 | TRUNC | 62.2 | 128 | max_tokens | 0.885 |
| rich_cli_option_groups | rich_cli_option_groups_051 | 2 | 2 | TRUNC | 61.7 | 128 | max_tokens | 0.938 |
| rich_cli_option_groups | rich_cli_option_groups_052 | 2 | 2 | clean | 62.4 | 128 | max_tokens | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_059 | 2 | 2 | clean | 62.0 | 128 | max_tokens | 0.967 |
| rich_cli_option_groups | rich_cli_option_groups_068 | 2 | 2 | TRUNC | 62.7 | 128 | max_tokens | 0.906 |

### TASD-FG-ES - Clean Subset

| Benchmark | Name | Original | New | Error Tags | TPS | Len | Stop | F1 |
|-----------|------|:--------:|:---:|------------|----:|----:|------|----:|
| argparse | argparse_real_002 | 2 | 2 | TRUNC | 62.2 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_003 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.933 |
| argparse | argparse_real_007 | 2 | 2 | TRUNC | 63.2 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_008 | 2 | 0 | DUP_OPT,TRUNC | 62.9 | 128 | max_tokens | 1.000 |
| argparse | argparse_real_009 | 2 | 2 | TRUNC | 62.4 | 128 | max_tokens | 0.901 |
| argparse | argparse_real_013 | 2 | 2 | TRUNC | 63.4 | 128 | max_tokens | 0.992 |
| argparse | argparse_real_014 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.981 |
| argparse | argparse_real_015 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.949 |
| argparse | argparse_real_016 | 2 | 2 | TRUNC | 62.9 | 128 | max_tokens | 0.855 |
| argparse | argparse_real_017 | 2 | 2 | TRUNC | 62.7 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_006 | 2 | 2 | TRUNC | 63.0 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_022 | 2 | 2 | TRUNC | 63.0 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_026 | 2 | 2 | TRUNC | 62.9 | 128 | max_tokens | 0.989 |
| complex_nested_config | complex_nested_config_030 | 2 | 2 | TRUNC | 63.3 | 128 | max_tokens | 1.000 |
| complex_nested_config | complex_nested_config_035 | 2 | 2 | TRUNC | 62.2 | 128 | max_tokens | 0.969 |
| complex_nested_config | complex_nested_config_042 | 2 | 1 | TRUNC | 63.7 | 128 | max_tokens | 0.960 |
| complex_nested_config | complex_nested_config_043 | 2 | 2 | TRUNC | 62.5 | 128 | max_tokens | 0.943 |
| complex_nested_config | complex_nested_config_055 | 2 | 2 | TRUNC | 62.9 | 128 | max_tokens | 0.988 |
| complex_nested_config | complex_nested_config_064 | 2 | 2 | TRUNC | 62.1 | 128 | max_tokens | 0.986 |
| complex_nested_config | complex_nested_config_068 | 2 | 2 | clean | 62.2 | 128 | max_tokens | 0.978 |
| dict_config | dict_config_real_001 | 2 | 2 | TRUNC | 62.8 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_013 | 2 | 0 | BRACKET | 61.8 | 109 | structure_complete | 1.000 |
| dict_config | dict_config_real_014 | 2 | 0 | BRACKET | 38.0 | 18 | structure_complete | 0.667 |
| dict_config | dict_config_real_018 | 2 | 0 | BRACKET | 56.2 | 67 | structure_complete | 0.972 |
| dict_config | dict_config_real_020 | 2 | 2 | clean | 64.0 | 128 | max_tokens | 1.000 |
| dict_config | dict_config_real_027 | 2 | 2 | TRUNC | 64.4 | 128 | max_tokens | 0.912 |
| dict_config | dict_config_real_028 | 2 | 2 | TRUNC | 63.9 | 128 | max_tokens | 0.912 |
| dict_config | dict_config_real_029 | 2 | 2 | TRUNC | 62.8 | 128 | max_tokens | 0.912 |
| dict_config | dict_config_real_030 | 2 | 2 | clean | 63.4 | 128 | max_tokens | 0.892 |
| dict_config | dict_config_real_032 | 2 | 2 | clean | 64.3 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_002 | 2 | 2 | TRUNC | 62.6 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_004 | 2 | 2 | TRUNC | 63.6 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_005 | 2 | 2 | TRUNC | 61.4 | 128 | max_tokens | 0.992 |
| openmmlab_config | openmmlab_config_real_006 | 2 | 2 | TRUNC | 63.5 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_007 | 2 | 2 | TRUNC | 63.5 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_009 | 2 | 2 | TRUNC | 63.5 | 128 | max_tokens | 0.981 |
| openmmlab_config | openmmlab_config_real_011 | 2 | 2 | TRUNC | 63.2 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_013 | 2 | 2 | TRUNC | 61.2 | 128 | max_tokens | 0.981 |
| openmmlab_config | openmmlab_config_real_015 | 2 | 2 | TRUNC | 61.2 | 128 | max_tokens | 1.000 |
| openmmlab_config | openmmlab_config_real_016 | 2 | 2 | TRUNC | 61.2 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_001 | 2 | 2 | TRUNC | 62.9 | 128 | max_tokens | 0.932 |
| pipeline_stage_config | pipeline_stage_config_002 | 2 | 2 | TRUNC | 63.5 | 128 | max_tokens | 0.980 |
| pipeline_stage_config | pipeline_stage_config_003 | 2 | 2 | TRUNC | 63.4 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_004 | 2 | 2 | TRUNC | 64.1 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_005 | 2 | 2 | TRUNC | 64.0 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_006 | 2 | 2 | TRUNC | 64.2 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_009 | 2 | 2 | TRUNC | 63.9 | 128 | max_tokens | 0.978 |
| pipeline_stage_config | pipeline_stage_config_011 | 2 | 2 | TRUNC | 63.4 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_012 | 2 | 2 | TRUNC | 62.9 | 128 | max_tokens | 1.000 |
| pipeline_stage_config | pipeline_stage_config_014 | 2 | 2 | TRUNC | 61.7 | 128 | max_tokens | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_005 | 2 | 2 | TRUNC | 62.8 | 128 | max_tokens | 0.939 |
| rich_cli_option_groups | rich_cli_option_groups_010 | 2 | 2 | clean | 62.3 | 128 | max_tokens | 0.983 |
| rich_cli_option_groups | rich_cli_option_groups_012 | 2 | 2 | TRUNC | 62.3 | 128 | max_tokens | 0.990 |
| rich_cli_option_groups | rich_cli_option_groups_019 | 2 | 2 | TRUNC | 63.1 | 128 | max_tokens | 0.936 |
| rich_cli_option_groups | rich_cli_option_groups_033 | 2 | 2 | clean | 61.4 | 128 | max_tokens | 0.971 |
| rich_cli_option_groups | rich_cli_option_groups_035 | 2 | 2 | TRUNC | 60.9 | 128 | max_tokens | 0.885 |
| rich_cli_option_groups | rich_cli_option_groups_051 | 2 | 2 | TRUNC | 60.2 | 128 | max_tokens | 0.938 |
| rich_cli_option_groups | rich_cli_option_groups_052 | 2 | 2 | clean | 61.2 | 128 | max_tokens | 1.000 |
| rich_cli_option_groups | rich_cli_option_groups_059 | 2 | 2 | clean | 61.9 | 128 | max_tokens | 0.967 |
| rich_cli_option_groups | rich_cli_option_groups_068 | 2 | 2 | TRUNC | 60.4 | 128 | max_tokens | 0.906 |
