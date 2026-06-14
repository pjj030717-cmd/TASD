# TASD-F-G Qwen 6×80 Full Experiment (Guarded Fallback)

**Target**: Qwen2.5-14B-Instruct-AWQ  |  **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**TASD-F-G**: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=True, guard_calibrated=True

## TASD vs TASD-F-G Overall

| Method | Speedup | Below 1.0x | SQ | Fallbacks |
|--------|:-------:|:----------:|:--:|:---------:|
| **TASD** | **1.956x** | 9/480 | - | - |
| **TASD-F-G** | **2.004x** | 3/480 | 0.5908 | 27 |

## Per-Benchmark Comparison

| Benchmark | TASD sp | TASD-F-G sp | TASD b<1 | TASD-F-G b<1 | TASD-F-G SQ | TASD-F-G FB |
|-----------|:-------:|:-----------:|:--------:|:------------:|:-----------:|:-----------:|
| argparse | 1.735x | **1.934x** | 8 | 1 | 0.6420 | 17 |
| dict_config | 1.989x | **1.971x** | 1 | 2 | 0.6567 | 9 |
| openmmlab_config | 1.974x | **2.050x** | 0 | 0 | 0.6023 | 1 |
| pipeline_stage_config | 2.018x | **2.028x** | 0 | 0 | 0.5568 | 0 |
| complex_nested_config | 2.004x | **2.019x** | 0 | 0 | 0.4749 | 0 |
| rich_cli_option_groups | 2.015x | **2.025x** | 0 | 0 | 0.6122 | 0 |

## TASD-F-G Per-Benchmark Details

### argparse (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 1.934x |
| median sp | 2.013x |
| min sp | 0.655x |
| worst-10 sp | 1.272x |
| below-1.0x | 1 |
| mean accept | 0.9655 |
| mean repair | 0.1 |
| mean guard_trig | 0.4 |
| mean trim | 0.3 |
| mean SQ | 0.6420 |
| total fb | 17 |

### dict_config (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 1.971x |
| median sp | 2.030x |
| min sp | 0.912x |
| worst-10 sp | 1.396x |
| below-1.0x | 2 |
| mean accept | 0.9736 |
| mean repair | 0.1 |
| mean guard_trig | 0.9 |
| mean trim | 0.3 |
| mean SQ | 0.6567 |
| total fb | 9 |

### openmmlab_config (80 samples)

| Metric | Value |
|--------|:-----:|
| mean sp | 2.050x |
| median sp | 2.045x |
| min sp | 1.464x |
| worst-10 sp | 1.852x |
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
| mean sp | 2.028x |
| median sp | 2.024x |
| min sp | 1.527x |
| worst-10 sp | 1.827x |
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
| mean sp | 2.019x |
| median sp | 2.029x |
| min sp | 1.802x |
| worst-10 sp | 1.887x |
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
| mean sp | 2.025x |
| median sp | 2.028x |
| min sp | 1.732x |
| worst-10 sp | 1.862x |
| below-1.0x | 0 |
| mean accept | 0.9998 |
| mean repair | 0.0 |
| mean guard_trig | 0.0 |
| mean trim | 0.0 |
| mean SQ | 0.6122 |
| total fb | 0 |

## Pass/Fail

| Criterion | Result | Note |
|-----------|:------:|------|
| mean sp >= TASD×0.97 | PASS | 1.956 → 2.004 |
| below-1.0x reduced | PASS | 9 → 3 |
| worst-10 sp >= TASD×0.9 | PASS | 1.428 → 1.655 |
| SQ >= TASD-0.02 | PASS | 0.0000 → 0.5908 |
| off_structure <= TASD×1.1 | FAIL | 0.0000 → 0.0366 |

**Overall**: SOME FAIL

## Data

- `results/qwen_tasd_fg_6x80.json`
- `results/qwen_6x80_checkpoints/`
