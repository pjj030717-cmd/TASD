# TASD-FG Risk-Aware AR Fallback: Oracle Analysis

**Total samples**: 480
**TASD-FG score=0**: 132
**TASD-FG sp_mean**: 2.004
**AR sp_mean**: 1.0

## 1. Route Score=0 Samples Only (按 risk 排序)

| Configuration | Route % | score2 | score1 | score0 | score2% | s0→s2 | s0→s1 | s0→s0 | Est TPS | Est Speedup |
|---------------|:-------:|:------:|:------:|:------:|:-------:|:-----:|:-----:|:-----:|:-------:|:-----------:|
| TASD-FG (baseline) | 0.0% | 192 | 156 | 132 | 40.0% | 0 | 0 | 0 | 66.4 | 2.000 |
| Worst 10% s0 → AR | 2.7% | 198 | 161 | 121 | 41.2% | 6 | 5 | 2 | 65.5 | 1.972 |
| Worst 20% s0 → AR | 5.4% | 203 | 165 | 112 | 42.3% | 11 | 9 | 6 | 64.5 | 1.944 |
| Worst 30% s0 → AR | 8.1% | 207 | 168 | 105 | 43.1% | 15 | 12 | 12 | 63.6 | 1.916 |
| All s0 → AR | 27.5% | 257 | 198 | 25 | 53.5% | 65 | 42 | 25 | 57.3 | 1.727 |
| AR (pure) | 100.0% | 251 | 155 | 74 | 52.3% | — | — | — | 33.2 | 1.000 |

## 2. Route Worst K% of ALL Samples (risk-based, 包含非 s0)

| Configuration | Route % | score2 | score1 | score0 | AR Degraded | Est TPS | Est Speedup |
|---------------|:-------:|:------:|:------:|:------:|:-----------:|:-------:|:-----------:|
| Worst 10% all → AR | 10.0% | 210 | 171 | 99 | 0 | 63.0 | 1.897 |
| Worst 20% all → AR | 20.0% | 235 | 187 | 58 | 0 | 59.8 | 1.802 |
| Worst 30% all → AR | 30.0% | 261 | 188 | 31 | 6 | 56.5 | 1.701 |

## 3. TASD-FG score=0 → AR 救援分析

| 指标 | 值 |
|------|----|
| TASD-FG score=0 总数 | 132 |
| AR 救回 score=2 | 65 (49.2%) |
| AR 部分救回 score=1 | 42 (31.8%) |
| AR 也救不回 score=0 | 25 (18.9%) |

## 4. Route All s0 前后对比

| 指标 | TASD-FG | Route All s0 | 变化 |
|------|--------:|:-----------:|------|
| score=2 | 192 | 257 | +65 |
| score=1 | 156 | 198 | +42 |
| score=0 | 132 | 25 | -107 |
| score2% | 40.0% | 53.5% | +13.5pp |
| Est TPS | 66.4 | 57.3 | -9.1 |
| Est Speedup | 2.000 | 1.727 | -0.273 |

## 5. Per-Benchmark Breakdown (Route All s0)

| Benchmark | N | TASD-FG s0 | AR s0 | Routed→s0 | Rescued→s2 | Still s0 |
|-----------|--:|:----------:|:-----:|:---------:|:----------:|:--------:|
| argparse | 80 | 16 | 7 | 1 | 10 | 1 |
| complex_nested_config | 80 | 35 | 29 | 14 | 2 | 14 |
| dict_config | 80 | 17 | 29 | 6 | 3 | 6 |
| openmmlab_config | 80 | 23 | 6 | 2 | 17 | 2 |
| pipeline_stage_config | 80 | 16 | 0 | 0 | 15 | 0 |
| rich_cli_option_groups | 80 | 25 | 3 | 2 | 18 | 2 |

## 6. Risk Score Distribution (Top 30 s0)

| Rank | Benchmark | Name | AR | Risk | F1 | Off-Str | Rep | BB |
|-----:|-----------|------|:--:|-----:|-----:|:-------:|:---:|:--:|
| 1 | rich_cli_option_groups | rich_cli_option_groups_013 | 2 | 177.5 | 0.174 | 1.00 | 0.31 | 1.0 |
| 2 | rich_cli_option_groups | rich_cli_option_groups_025 | 2 | 170.4 | 0.237 | 0.91 | 0.25 | 1.0 |
| 3 | rich_cli_option_groups | rich_cli_option_groups_044 | 2 | 165.9 | 0.483 | 1.00 | 0.50 | 1.0 |
| 4 | rich_cli_option_groups | rich_cli_option_groups_022 | 2 | 163.7 | 0.325 | 1.00 | 0.00 | 1.0 |
| 5 | rich_cli_option_groups | rich_cli_option_groups_061 | 0 | 160.3 | 0.340 | 0.91 | 0.00 | 1.0 |
| 6 | complex_nested_config | complex_nested_config_059 | 2 | 160.3 | 0.095 | 0.00 | 0.00 | 0.0 |
| 7 | rich_cli_option_groups | rich_cli_option_groups_029 | 2 | 158.8 | 0.718 | 0.98 | 0.77 | 1.0 |
| 8 | dict_config | dict_config_real_079 | 0 | 158.5 | 0.130 | 0.00 | 0.00 | 0.0 |
| 9 | complex_nested_config | complex_nested_config_016 | 1 | 157.6 | 0.147 | 0.00 | 0.00 | 0.0 |
| 10 | complex_nested_config | complex_nested_config_024 | 1 | 157.3 | 0.153 | 0.00 | 0.00 | 0.0 |
| 11 | complex_nested_config | complex_nested_config_020 | 1 | 156.6 | 0.168 | 0.00 | 0.00 | 0.0 |
| 12 | rich_cli_option_groups | rich_cli_option_groups_077 | 1 | 156.3 | 0.703 | 0.97 | 0.61 | 1.0 |
| 13 | dict_config | dict_config_real_022 | 1 | 156.3 | 0.175 | 0.00 | 0.00 | 0.0 |
| 14 | complex_nested_config | complex_nested_config_063 | 1 | 155.6 | 0.116 | 0.13 | 0.38 | 0.0 |
| 15 | rich_cli_option_groups | rich_cli_option_groups_032 | 1 | 154.1 | 0.785 | 0.98 | 0.70 | 1.0 |
| 16 | dict_config | dict_config_real_078 | 0 | 153.7 | 0.227 | 0.00 | 0.00 | 0.0 |
| 17 | complex_nested_config | complex_nested_config_017 | 0 | 152.4 | 0.136 | 0.00 | 0.46 | 1.0 |
| 18 | complex_nested_config | complex_nested_config_080 | 0 | 151.3 | 0.108 | 0.00 | 0.33 | 1.0 |
| 19 | rich_cli_option_groups | rich_cli_option_groups_046 | 2 | 150.6 | 0.313 | 0.00 | 0.06 | 0.0 |
| 20 | complex_nested_config | complex_nested_config_056 | 1 | 150.1 | 0.132 | 0.00 | 0.33 | 0.0 |
| 21 | openmmlab_config | openmmlab_config_real_025 | 2 | 150.0 | 0.000 | 0.00 | 0.00 | 1.0 |
| 22 | openmmlab_config | openmmlab_config_real_029 | 2 | 150.0 | 0.000 | 0.00 | 0.00 | 1.0 |
| 23 | pipeline_stage_config | pipeline_stage_config_043 | 2 | 150.0 | 0.000 | 0.00 | 0.00 | 1.0 |
| 24 | pipeline_stage_config | pipeline_stage_config_070 | 2 | 150.0 | 0.000 | 0.00 | 0.00 | 1.0 |
| 25 | complex_nested_config | complex_nested_config_046 | 0 | 150.0 | 0.000 | 0.00 | 0.00 | 1.0 |
| 26 | complex_nested_config | complex_nested_config_014 | 1 | 149.9 | 0.001 | 0.00 | 0.00 | 1.0 |
| 27 | rich_cli_option_groups | rich_cli_option_groups_055 | 2 | 149.9 | 0.778 | 0.98 | 0.47 | 1.0 |
| 28 | dict_config | dict_config_real_050 | 0 | 149.7 | 0.007 | 0.00 | 0.00 | 1.0 |
| 29 | dict_config | dict_config_real_069 | 0 | 149.7 | 0.007 | 0.00 | 0.00 | 1.0 |
| 30 | complex_nested_config | complex_nested_config_033 | 0 | 149.7 | 0.007 | 0.00 | 0.00 | 1.0 |

## 7. 决策分析

### 关键数字

- Route all s0 (132/480 = 27.5%): score=0 132 → 25 (-107)

- AR 救回 65/132 (49.2%) 个 score=0 → score=2

- AR 也救不回 25/132 (18.9%) 个

- Speedup: 2.004x → 1.727x

### 判定

| 条件 | 阈值 | 实际 | 满足? |
|------|------|------|:-----:|
| AR rescue rate | >= 30% s0→s2 | 49.2% | ✅ |
| Speedup | >= 1.5x | 1.727x | ✅ |

**结论: 可考虑实现真实 risk detector。** Oracle AR fallback 能救回足够多的样本，速度损失可接受。
