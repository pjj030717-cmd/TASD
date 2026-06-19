# LLaMA 6×80 Full Generalization Experiment

**Target**: meta-llama/Llama-3.1-8B-Instruct  |  **Draft**: meta-llama/Llama-3.2-1B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**TASD-FG**: enable_failure_aware_fallback=True, fallback_guarded=True, guard_calibrated=True

## Overall Results (240 samples)

| Method | Speedup | Below | Worst-10 | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:--------:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0/480 | 1.000x | 0.6228 | 0.8190 | 0.0035 | - |
| Greedy SD | 1.442x | 0/480 | 1.265x | 0.6272 | 0.7988 | 0.0075 | - |
| N-gram SD | 1.224x | 121/480 | 0.685x | 0.4928 | 0.7862 | 0.0029 | - |
| Official FLY | 0.991x | 122/480 | 0.727x | 0.5663 | 0.7872 | 0.0008 | - |
| **TASD** | **1.682x** | 2/480 | 1.495x | 0.6254 | 0.7944 | 0.0028 | - |
| **TASD-FG** | **1.683x** | 0/480 | 1.571x | 0.6279 | 0.7943 | 0.0027 | 3 |

## Per-Benchmark Results

### argparse (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6381 | 0.8032 | 0.0000 | - |
| Greedy SD | 1.422x | 0 | 0.6251 | 0.8259 | 0.0269 | - |
| N-gram SD | 1.258x | 21 | 0.5208 | 0.7425 | 0.0060 | - |
| Official FLY | 0.982x | 22 | 0.6026 | 0.7641 | 0.0020 | - |
| **TASD** | **1.646x** | 2 | 0.6164 | 0.8234 | 0.0011 | - |
| **TASD-FG** | **1.676x** | 0 | 0.6284 | 0.8225 | 0.0000 | 2 |

### dict_config (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6874 | 0.8355 | 0.0083 | - |
| Greedy SD | 1.486x | 0 | 0.6969 | 0.8101 | 0.0019 | - |
| N-gram SD | 1.323x | 14 | 0.5039 | 0.8104 | 0.0047 | - |
| Official FLY | 0.953x | 23 | 0.5895 | 0.8410 | 0.0019 | - |
| **TASD** | **1.688x** | 0 | 0.7025 | 0.8048 | 0.0000 | - |
| **TASD-FG** | **1.690x** | 0 | 0.7025 | 0.8048 | 0.0000 | 0 |

### openmmlab_config (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6933 | 0.7777 | 0.0000 | - |
| Greedy SD | 1.426x | 0 | 0.6951 | 0.7686 | 0.0000 | - |
| N-gram SD | 1.307x | 18 | 0.5969 | 0.7750 | 0.0000 | - |
| Official FLY | 0.980x | 21 | 0.6486 | 0.7437 | 0.0000 | - |
| **TASD** | **1.721x** | 0 | 0.6951 | 0.7686 | 0.0000 | - |
| **TASD-FG** | **1.726x** | 0 | 0.6951 | 0.7686 | 0.0000 | 0 |

### pipeline_stage_config (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6383 | 0.8625 | 0.0000 | - |
| Greedy SD | 1.390x | 0 | 0.6473 | 0.7689 | 0.0000 | - |
| N-gram SD | 0.907x | 33 | 0.5134 | 0.8447 | 0.0000 | - |
| Official FLY | 0.984x | 22 | 0.5672 | 0.8159 | 0.0000 | - |
| **TASD** | **1.663x** | 0 | 0.6398 | 0.7502 | 0.0000 | - |
| **TASD-FG** | **1.635x** | 0 | 0.6428 | 0.7502 | 0.0000 | 1 |

### complex_nested_config (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.4726 | 0.8109 | 0.0008 | - |
| Greedy SD | 1.455x | 0 | 0.4859 | 0.7892 | 0.0118 | - |
| N-gram SD | 1.367x | 17 | 0.4016 | 0.7913 | 0.0065 | - |
| Official FLY | 0.995x | 19 | 0.4068 | 0.7917 | 0.0000 | - |
| **TASD** | **1.691x** | 0 | 0.4859 | 0.7892 | 0.0118 | - |
| **TASD-FG** | **1.661x** | 0 | 0.4859 | 0.7892 | 0.0118 | 0 |

### rich_cli_option_groups (80)

| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |
|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|
| AR | 1.000x | 0 | 0.6070 | 0.8245 | 0.0120 | - |
| Greedy SD | 1.471x | 0 | 0.6129 | 0.8303 | 0.0041 | - |
| N-gram SD | 1.185x | 18 | 0.4201 | 0.7533 | 0.0000 | - |
| Official FLY | 1.050x | 15 | 0.5829 | 0.7670 | 0.0010 | - |
| **TASD** | **1.683x** | 0 | 0.6129 | 0.8303 | 0.0041 | - |
| **TASD-FG** | **1.709x** | 0 | 0.6129 | 0.8303 | 0.0041 | 0 |

## Comparison with Qwen Results

| Model | TASD sp | TASD below | TASD-FG sp | TASD-FG below |
|-------|:-------:|:----------:|:----------:|:-------------:|
| LLaMA-8B | 1.682x | 2/480 | 1.683x | 0/480 |
| Qwen-14B | 1.978x | 9/480 | 2.004x | 3/480 |

## Pass/Fail Criteria

| Criterion | Result | Note |
|-----------|:------:|------|
| TASD-FG sp >= 1.5x | PASS | 1.683x |
| TASD-FG below <= 10 | PASS | 0/480 |
| TASD-FG >= TASD×0.95 | PASS | 1.682 → 1.683 |
| SQ >= TASD-0.03 | PASS | 0.6930 → 0.6945 |

**Overall**: ALL PASS — Generalization confirmed

## Data

- `results/llama_6x80_full.json`
- `results/checkpoints_llama_6x80/`
