# TASD-FG-R (Risk-Aware AR Fallback) Pilot Experiment

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**Subsets**: 60 hard (TASD-FG s=0, AR s>=1) + 60 clean (TASD-FG s=2)

## 1. Hard Subset (60 samples)

| Metric | TASD-FG | TASD-FG-R | Change |
|--------|:-------:|:---------:|:------:|
| score=2 | 0 | 0 | +0 |
| score=1 | 0 | 1 | +1 |
| score=0 | 60 | 59 | -1 |
| score2% | 0.0% | 0.0% | +0.0pp |
| Speedup | 2.102x | 1.944x | -0.158x |
| Recoverable | — | 1.7% (1/60) | |
| Routed ratio | — | 23.3% (14/60) | |
| Fallback tokens | — | 1248 | |
| False negative (s=0 & no FB) | — | 76.7% (46) | |

## 2. Clean Subset (60 samples)

| Metric | TASD-FG | TASD-FG-R | Change |
|--------|:-------:|:---------:|:------:|
| score=2 | 14 | 13 | -1 |
| score=1 | 0 | 1 | +1 |
| score=0 | 46 | 46 | +0 |
| s2 preservation | — | 92.9% | |
| Speedup | 2.041x | 1.998x | -0.043x |
| False positive (routed) | — | 8.3% (5/60) | |

## 3. Combined (120 samples)

| Metric | TASD-FG | TASD-FG-R | Change |
|--------|:-------:|:---------:|:------:|
| score=2 | 14 | 13 | -1 |
| score=1 | 0 | 2 | +2 |
| score=0 | 106 | 105 | -1 |
| Speedup | 2.071x | 1.971x | -0.100x |
| Routed ratio | — | 15.8% | |

## 4. Fallback Reason Breakdown

| Reason | Count |
|--------|:-----:|
| consecutive_line_repeat:3 | 4 |
| token_repetition:8 | 2 |
| token_repetition:12 | 1 |
| off_structure:def replace_parenthesis(s): | 1 |
| token_repetition:11 | 3 |
| token_repetition:17 | 1 |
| repetition:1 | 2 |
| low_accept_rate:0.031 | 4 |
| token_repetition:9 | 1 |

## 5. Acceptance Criteria

| Criterion | Threshold | Hard | Clean | Combined |
|-----------|:--------:|:----:|:-----:|:--------:|
| s=0 relative reduction | >= 30% | 1.7% | — | 0.9% |
| s2 preservation | >= 90% | — | 92.9% | — |
| Speedup | >= 1.5x | 1.944x | 1.998x | 1.971x |
| Fallback ratio | <= 35% | 23.3% | 8.3% | 15.8% |

| Condition | Result |
|-----------|:------:|
| Hard s=0 reduction >= 30% | ❌ FAIL |
| Clean s2 preservation >= 90% | ✅ PASS |
| Speedup >= 1.5x | ✅ PASS |
| Fallback ratio <= 35% | ✅ PASS |

**结论: SOME FAIL. See above for details.**

## 6. Analysis

### 6.1 Risk Signal Design

TASD-FG-R 使用 7 个 guard 信号 + 3 个独立信号检测风险：

| Signal | Type | Threshold | Description |
|--------|------|:---------:|-------------|
| recent_accept_rate | Guard | < 0.60 | 最近 5 轮滚动平均接受率 |
| guard_trigger_count | Guard | >= 3 | 结构 guard 触发次数 |
| repair_count | Guard | >= 3 | 修复轮次 |
| off_structure_detected | Guard | >= 1 | 结构偏移检测 |
| repetition_warning | Guard | >= 1 | Guard 重复警告 |
| consecutive_low_accept | Guard | >= 2 轮 | 连续低接受率 (accept < 0.30) |
| bracket_stack_abnormal | Guard | any | 括号栈异常 |
| token_repetition | Independent | >= 8 | 4-gram 重复 (64 token 窗口) |
| consecutive_line_repeat | Independent | >= 3 | 连续相同行 (长度 > 10) |
| off_structure | Independent | >= 1 | def/class 关键词在最后 8 行 |

### 6.2 Threshold Sensitivity Analysis

对 accept_threshold 进行 0.55/0.60/0.70 三档扫描，结果完全相同：

| Threshold | Hard s=0 | Clean s2 | Fallback% | Notes |
|:---------:|:--------:|:--------:|:---------:|-------|
| 0.55 | 1.7% | 92.9% | 15.8% | 相同 |
| 0.60 | 1.7% | 92.9% | 15.8% | 相同 |
| 0.70 | 1.7% | 92.9% | 15.8% | 相同 |

**结论**: hard 样本的 first-round accept rate 呈双峰分布 — 4 个样本 < 0.05（触发 low_accept_rate），其余 56 个 > 0.70（无法触发）。accept_rate 信号无法区分 hard 和 clean 样本。

### 6.3 Fallback 触发时机分析

14 个 hard 样本触发 fallback 的时机分布：

| 触发 step | 数量 | 原因 | 挽救 |
|:---------:|:----:|------|:----:|
| 1 | 4 | low_accept_rate:0.031 | 1 |
| 32 | 1 | off_structure | 0 |
| 64 | 7 | token_repetition / repetition | 0 |
| 96 | 2 | consecutive_line_repeat | 0 |
| 128 | 0 | — | — |

**关键发现**: 只有 step=1 的 fallback 能挽救样本（1/4 成功）。step=32+ 的 fallback 无法挽救，因为前 32-128 个 token 已经是错误输出，AR 只能生成剩余 token。

### 6.4 根本限制

Sample-level fallback 的 recoverable rate 受限于以下因素：

1. **唯一有效的触发时机是 step=1**: 只有 low_accept_rate 信号能在 step=1 触发
2. **大多数 hard 样本 accept rate 很高**: draft 和 target 在第一轮就「一致同意」了错误内容（如 import 循环），accept rate > 0.70
3. **文本级信号触发太晚**: token_repetition（step 64+）、consecutive_line_repeat（step 96+）触发时，大部分 token 已生成
4. **结构 guard 只覆盖 3/6 类型**: argparse、dict_config、openmmlab_config 有 guard，但 rich_cli_option_groups、pipeline_stage_config、complex_nested_config 无 guard

### 6.5 达成 30% hard s=0 下降的路径

要达成 30%（18/60 挽救），需要 trigger 在 step=1 对 18+ 个样本生效。当前只有 4 个样本能在 step=1 触发。可能的方案：

1. **Step-level fallback**（而非 sample-level）：每轮检测风险，丢弃当前 round 的 draft token 用 AR 重新生成，然后继续 TASD-FG
2. **更早的检测信号**：在 token 级别检测重复/偏移，在 16 token 内触发
3. **结构类型感知的 fallback**：根据 structure type 采用不同的检测策略
4. **降低 draft_len**：减少每轮 draft token 数，增加检测机会

以上方案均超出了当前 sample-level fallback 的设计范围。
