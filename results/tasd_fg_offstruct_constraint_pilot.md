# TASD-FG-OS (OffStruct Constraint) Pilot

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**Method**: Block def/class/import/from in accepted tokens at verification time.

**Subsets**: 60 hard (TASD-FG s=0) + 60 clean (TASD-FG s=2), max 10 per benchmark

## 1. Hard Subset (60 samples)

| Metric | TASD-FG | TASD-FG-OS | Change |
|--------|:-------:|:----------:|:------:|
| score=2 | 0 | 0 | +0 |
| score=1 | 0 | 1 | +1 |
| score=0 | 60 | 59 | -1 |
| score=0 reduction | — | 1.7% (1/60) | |
| TPS | 67.16 | 62.0 | +7.67% |
| off_structure | 0.1187 | 0.0472 | -0.0715 (+60.2%) |
| repetition_rate | 0.0858 | 0.0605 | -0.0253 |
| structural_f1 | 0.607 | 0.5985 | -0.0085 |
| OS triggers | — | 41 | |
| OS trimmed tokens | — | 1007 | |
| OS AR repairs | — | 25 | |

### Benchmark Breakdown

| Benchmark | N | FG s=0 | OS s=0 | s=0 Chg | FG s=2 | OS s=2 | s=2 Chg | FG off | OS off |
|-----------|:-:|:------:|:------:|:-------:|:------:|:------:|:-------:|:------:|:------:|
| argparse | 10 | 10 | 10 | +0 | 0 | 0 | +0 | 0.0 | 0.0 |
| complex_nested_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 | 0.028 | 0.0104 |
| dict_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 | 0.0 | 0.0 |
| openmmlab_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 | 0.1699 | 0.1062 |
| pipeline_stage_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 | 0.128 | 0.0629 |
| rich_cli_option_groups | 10 | 10 | 9 | -1 | 0 | 0 | +0 | 0.3861 | 0.104 |

## 2. Clean Subset (60 samples)

| Metric | TASD-FG | TASD-FG-OS | Change |
|--------|:-------:|:----------:|:------:|
| score=2 | 20 | 20 | +0 |
| score=1 | 0 | 0 | +0 |
| score=0 | 40 | 40 | +0 |
| score=0 reduction | — | 0.0% (0/40) | |
| score=2 preservation | — | 100.0% (20/20) | |
| TPS | 66.72 | 66.71 | +0.01% |
| off_structure | 0.0 | 0.0 | +0.0000 (+0.0%) |
| repetition_rate | 0.0023 | 0.0023 | +0.0000 |
| structural_f1 | 0.9773 | 0.9773 | +0.0000 |
| OS triggers | — | 0 | |
| OS trimmed tokens | — | 0 | |
| OS AR repairs | — | 0 | |

### Benchmark Breakdown

| Benchmark | N | FG s=0 | OS s=0 | s=0 Chg | FG s=2 | OS s=2 | s=2 Chg | FG off | OS off |
|-----------|:-:|:------:|:------:|:-------:|:------:|:------:|:-------:|:------:|:------:|
| argparse | 10 | 8 | 8 | +0 | 2 | 2 | +0 | 0.0 | 0.0 |
| complex_nested_config | 10 | 8 | 8 | +0 | 2 | 2 | +0 | 0.0 | 0.0 |
| dict_config | 10 | 1 | 1 | +0 | 9 | 9 | +0 | 0.0 | 0.0 |
| openmmlab_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 | 0.0 | 0.0 |
| pipeline_stage_config | 10 | 9 | 9 | +0 | 1 | 1 | +0 | 0.0 | 0.0 |
| rich_cli_option_groups | 10 | 4 | 4 | +0 | 6 | 6 | +0 | 0.0 | 0.0 |

## 3. Combined (120 samples)

| Metric | TASD-FG | TASD-FG-OS | Change |
|--------|:-------:|:----------:|:------:|
| score=2 | 20 | 20 | +0 |
| score=1 | 0 | 1 | +1 |
| score=0 | 100 | 99 | -1 |
| score=0 reduction | — | 1.0% (1/100) | |
| score=2 preservation | — | 100.0% (20/20) | |
| TPS | 66.94 | 64.36 | +3.86% |
| off_structure | 0.0593 | 0.0236 | -0.0357 (+60.2%) |
| repetition_rate | 0.0441 | 0.0314 | -0.0127 |
| structural_f1 | 0.7921 | 0.7879 | -0.0042 |
| OS triggers | — | 41 | |
| OS trimmed tokens | — | 1007 | |
| OS AR repairs | — | 25 | |

### Benchmark Breakdown

| Benchmark | N | FG s=0 | OS s=0 | s=0 Chg | FG s=2 | OS s=2 | s=2 Chg | FG off | OS off |
|-----------|:-:|:------:|:------:|:-------:|:------:|:------:|:-------:|:------:|:------:|
| argparse | 20 | 18 | 18 | +0 | 2 | 2 | +0 | 0.0 | 0.0 |
| complex_nested_config | 20 | 18 | 18 | +0 | 2 | 2 | +0 | 0.014 | 0.0052 |
| dict_config | 20 | 11 | 11 | +0 | 9 | 9 | +0 | 0.0 | 0.0 |
| openmmlab_config | 20 | 20 | 20 | +0 | 0 | 0 | +0 | 0.0849 | 0.0531 |
| pipeline_stage_config | 20 | 19 | 19 | +0 | 1 | 1 | +0 | 0.064 | 0.0314 |
| rich_cli_option_groups | 20 | 14 | 13 | -1 | 6 | 6 | +0 | 0.193 | 0.052 |

## 4. Acceptance Criteria

| Criterion | Threshold | Hard | Clean | Combined |
|-----------|:--------:|:----:|:-----:|:--------:|
| Hard s=0 reduction | >= 10% | 1.7% | — | — |
| Hard off_structure drop | >= 50% | 60.2% | — | — |
| Clean s2 preservation | >= 90% | — | 100.0% | — |
| Speed loss | <= 5% | — | — | +3.86% |

| Condition | Result |
|-----------|:------:|
| Hard s=0 reduction >= 10% | FAIL |
| Hard off_structure drop >= 50% | PASS |
| Clean s2 preservation >= 90% | PASS |
| Speed loss <= 5% | PASS |

**结论: SOME FAIL.**
