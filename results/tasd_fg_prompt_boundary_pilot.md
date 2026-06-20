# TASD-FG Prompt Boundary Pilot

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**Method**: Append structure-specific boundary suffix to prompt before decoding.

**Subsets**: 60 hard (TASD-FG s=0) + 60 clean (TASD-FG s=2), max 10 per benchmark

## 0. Boundary Suffixes

| Benchmark | Suffix |
|-----------|--------|
| dict_config | `# Complete only this dictionary/list configuration block. Do not define functions or imports.` |
| complex_nested_config | `# Continue this nested configuration block only. Keep keys and nesting style consistent.` |
| openmmlab_config | `# Complete only this OpenMMLab config block. Do not start new Python code.` |
| pipeline_stage_config | `# Continue only this pipeline stage list. Keep dict(type=...) style.` |
| argparse | `# Continue only parser.add_argument(...) entries. Do not add imports or function definitions.` |
| rich_cli_option_groups | `# Continue only CLI option definitions. Do not start new functions/classes/imports.` |

## 1. Hard Subset (60 samples)

| Metric | Original | + Boundary | Change |
|--------|:--------:|:----------:|:------:|
| score=2 | 0 | 2 | +2 |
| score=1 | 0 | 5 | +5 |
| score=0 | 60 | 53 | -7 |
| score=0 reduction | — | 11.7% | |
| TPS | 66.49 | 66.56 | -0.09% |
| off_structure | 0.1232 | 0.0098 | -0.1134 |
| repetition_rate | 0.0817 | 0.0876 | +0.0059 |
| structural_f1 | 0.642 | 0.6521 | +0.0101 |

### 1. Hard Subset Benchmark Breakdown

| Benchmark | N | Orig s=0 | +Boundary s=0 | s=0 Change | Orig s=2 | +Boundary s=2 | s=2 Change |
|-----------|:-:|:--------:|:-------------:|:----------:|:--------:|:-------------:|:----------:|
| argparse | 10 | 10 | 9 | -1 | 0 | 1 | +1 |
| complex_nested_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 |
| dict_config | 10 | 10 | 9 | -1 | 0 | 0 | +0 |
| openmmlab_config | 10 | 10 | 9 | -1 | 0 | 1 | +1 |
| pipeline_stage_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 |
| rich_cli_option_groups | 10 | 10 | 6 | -4 | 0 | 0 | +0 |

## 2. Clean Subset (60 samples)

| Metric | Original | + Boundary | Change |
|--------|:--------:|:----------:|:------:|
| score=2 | 19 | 8 | -11 |
| score=1 | 0 | 7 | +7 |
| score=0 | 41 | 45 | +4 |
| score=0 reduction | — | -9.8% | |
| score=2 preservation | — | 42.1% | |
| TPS | 67.66 | 66.37 | +1.91% |
| off_structure | 0.0 | 0.0279 | +0.0279 |
| repetition_rate | 0.0011 | 0.122 | +0.1209 |
| structural_f1 | 0.9809 | 0.8972 | -0.0837 |

### 2. Clean Subset Benchmark Breakdown

| Benchmark | N | Orig s=0 | +Boundary s=0 | s=0 Change | Orig s=2 | +Boundary s=2 | s=2 Change |
|-----------|:-:|:--------:|:-------------:|:----------:|:--------:|:-------------:|:----------:|
| argparse | 10 | 9 | 8 | -1 | 1 | 1 | +0 |
| complex_nested_config | 10 | 7 | 6 | -1 | 3 | 2 | -1 |
| dict_config | 10 | 1 | 5 | +4 | 9 | 5 | -4 |
| openmmlab_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 |
| pipeline_stage_config | 10 | 10 | 10 | +0 | 0 | 0 | +0 |
| rich_cli_option_groups | 10 | 4 | 6 | +2 | 6 | 0 | -6 |

## 3. Combined (120 samples)

| Metric | Original | + Boundary | Change |
|--------|:--------:|:----------:|:------:|
| score=2 | 19 | 10 | -9 |
| score=1 | 0 | 12 | +12 |
| score=0 | 101 | 98 | -3 |
| score=0 reduction | — | 3.0% | |
| score=2 preservation | — | 52.6% | |
| TPS | 67.08 | 66.46 | +0.92% |
| off_structure | 0.0616 | 0.0188 | -0.0428 |
| repetition_rate | 0.0414 | 0.1048 | +0.0634 |
| structural_f1 | 0.8115 | 0.7747 | -0.0368 |

### 3. Combined Benchmark Breakdown

| Benchmark | N | Orig s=0 | +Boundary s=0 | s=0 Change | Orig s=2 | +Boundary s=2 | s=2 Change |
|-----------|:-:|:--------:|:-------------:|:----------:|:--------:|:-------------:|:----------:|
| argparse | 20 | 19 | 17 | -2 | 1 | 2 | +1 |
| complex_nested_config | 20 | 17 | 16 | -1 | 3 | 2 | -1 |
| dict_config | 20 | 11 | 14 | +3 | 9 | 5 | -4 |
| openmmlab_config | 20 | 20 | 19 | -1 | 0 | 1 | +1 |
| pipeline_stage_config | 20 | 20 | 20 | +0 | 0 | 0 | +0 |
| rich_cli_option_groups | 20 | 14 | 12 | -2 | 6 | 0 | -6 |

## 4. Acceptance Criteria

| Criterion | Threshold | Hard | Clean | Combined |
|-----------|:--------:|:----:|:-----:|:--------:|
| Hard s=0 reduction | >= 20% | 11.7% | — | — |
| Clean s2 preservation | >= 90% | — | 42.1% | — |
| Speed loss | <= 5% | — | — | +0.92% |

| Condition | Result |
|-----------|:------:|
| Hard s=0 reduction >= 20% | FAIL |
| Clean s2 preservation >= 90% | FAIL |
| Speed loss <= 5% | PASS |

**结论: SOME FAIL. Boundary suffix may not be sufficient alone.**
