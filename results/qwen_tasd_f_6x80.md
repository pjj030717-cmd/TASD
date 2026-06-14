# TASD-F Qwen 6×80 Full Experiment

**Target**: Qwen2.5-14B-Instruct-AWQ  |  **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**TASD-F**: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=False, guard_calibrated=True

## TASD vs TASD-F Overall

| Method | Speedup | Below 1.0x | SQ | Fallbacks |
|--------|:-------:|:----------:|:--:|:---------:|
| **TASD** | **1.956x** | 9/480 | - | - |
| **TASD-F** | **1.998x** | 9/480 | 0.5917 | 131 |

## Per-Benchmark Comparison

| Benchmark | TASD sp | TASD-F sp | TASD below | TASD-F below | TASD-F SQ | TASD-F FB |
|-----------|:-------:|:---------:|:----------:|:------------:|:---------:|:---------:|
| argparse | 1.735x | **1.859x** | 8 | 8 | 0.6467 | 117 |
| dict_config | 1.989x | **1.974x** | 1 | 1 | 0.6570 | 13 |
| openmmlab_config | 1.974x | **2.055x** | 0 | 0 | 0.6023 | 1 |
| pipeline_stage_config | 2.018x | **2.042x** | 0 | 0 | 0.5568 | 0 |
| complex_nested_config | 2.004x | **2.030x** | 0 | 0 | 0.4749 | 0 |
| rich_cli_option_groups | 2.015x | **2.027x** | 0 | 0 | 0.6122 | 0 |

## TASD-F Per-Benchmark Details

### argparse (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 1.859x |
| median sp | 2.017x |
| min sp | 0.202x |
| worst-10 sp | 0.533x |
| below-1.0x | 8 |
| mean accept | 0.9237 |
| mean repair | 0.2 |
| mean guard_trig | 1.4 |
| mean trim | 1.4 |
| mean SQ | 0.6467 |
| total fb | 117 |

### dict_config (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 1.974x |
| median sp | 2.016x |
| min sp | 0.637x |
| worst-10 sp | 1.397x |
| below-1.0x | 1 |
| mean accept | 0.9717 |
| mean repair | 0.1 |
| mean guard_trig | 0.8 |
| mean trim | 0.3 |
| mean SQ | 0.6570 |
| total fb | 13 |

### openmmlab_config (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 2.055x |
| median sp | 2.064x |
| min sp | 1.454x |
| worst-10 sp | 1.881x |
| below-1.0x | 0 |
| mean accept | 0.9960 |
| mean repair | 0.0 |
| mean guard_trig | 0.7 |
| mean trim | 0.0 |
| mean SQ | 0.6023 |
| total fb | 1 |

### pipeline_stage_config (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 2.042x |
| median sp | 2.045x |
| min sp | 1.467x |
| worst-10 sp | 1.829x |
| below-1.0x | 0 |
| mean accept | 0.9967 |
| mean repair | 0.0 |
| mean guard_trig | 0.0 |
| mean trim | 0.0 |
| mean SQ | 0.5568 |
| total fb | 0 |

### complex_nested_config (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 2.030x |
| median sp | 2.024x |
| min sp | 1.887x |
| worst-10 sp | 1.925x |
| below-1.0x | 0 |
| mean accept | 1.0000 |
| mean repair | 0.0 |
| mean guard_trig | 0.0 |
| mean trim | 0.0 |
| mean SQ | 0.4749 |
| total fb | 0 |

### rich_cli_option_groups (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 2.027x |
| median sp | 2.036x |
| min sp | 1.740x |
| worst-10 sp | 1.855x |
| below-1.0x | 0 |
| mean accept | 0.9998 |
| mean repair | 0.0 |
| mean guard_trig | 0.0 |
| mean trim | 0.0 |
| mean SQ | 0.6122 |
| total fb | 0 |

## Pass/Fail Criteria

| Criterion | Result | Note |
|-----------|:------:|------|
| mean sp >= TASD×0.97 | PASS | 1.956 → 1.998 |
| below-1.0x reduced | FAIL | 9 → 9 |
| worst-10 sp >= TASD×0.9 | PASS | 1.428 → 1.538 |
| SQ >= TASD-0.02 | PASS | 0.0000 → 0.5916 |
| off_structure <= TASD×1.1 | FAIL | 0.0000 → 0.0484 |

**Overall**: SOME FAIL

## Data Files

- `results/qwen_tasd_f_6x80.json`
- `results/qwen_6x80_checkpoints/`
