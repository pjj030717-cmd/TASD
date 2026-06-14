# Qwen 6×80 Main Experiment Report (Corrected TPS)

**Target**: Qwen2.5-14B-Instruct-AWQ  |  **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, temperature=0.0

> **Note**: All TPS computed as generated_tokens / wall_time, excluding prompt tokens and model loading.

## Methods

| Method | Description |
|--------|-------------|
| AR | Target autoregressive (greedy) |
| Greedy SD | Target-verify greedy draft (draft_len=16, blocks=2, top_k=3, no guard) |
| N-gram SD | Pure n-gram lookup SD (ngram_min=3, max=8, no draft model) |
| **Official FLY** | AMD FLy SPDGenerate (k=15, win_len=6, entropy_thre=0.3, ngram=4/6) |
| **TASD** | Structure-aware SD + Guard-v1.5 calibrated (draft_len=16, blocks=2, top_k=3) |

## Per-Benchmark Results

### argparse (80 samples)

Baseline AR TPS: **33.9** tok/s (generated only)

| Method | Speedup | SQ | Accept/MAT | Below 1.0x |
|--------|:-------:|:--:|:----------:|:----------:|
| AR | **1.000x** | 0.8672 | - | 0 |
| Greedy SD | **1.725x** | 0.8443 | 0.878 | 0 |
| N-gram SD | **1.407x** | 0.6466 | 0.524 | 33 |
| Official FLY | **1.945x** | 0.8790 | 1.37 (MAT) | 6 |
| TASD | **1.735x** | 0.8289 | 0.907 | 8 |

### dict_config (80 samples)

Baseline AR TPS: **33.7** tok/s (generated only)

| Method | Speedup | SQ | Accept/MAT | Below 1.0x |
|--------|:-------:|:--:|:----------:|:----------:|
| AR | **1.000x** | 0.6908 | - | 0 |
| Greedy SD | **1.810x** | 0.6522 | 0.899 | 0 |
| N-gram SD | **1.468x** | 0.5100 | 0.541 | 32 |
| Official FLY | **1.798x** | 0.6917 | 1.39 (MAT) | 7 |
| TASD | **1.989x** | 0.6506 | 0.971 | 1 |

### openmmlab_config (80 samples)

Baseline AR TPS: **33.7** tok/s (generated only)

| Method | Speedup | SQ | Accept/MAT | Below 1.0x |
|--------|:-------:|:--:|:----------:|:----------:|
| AR | **1.000x** | 0.9281 | - | 0 |
| Greedy SD | **1.718x** | 0.8751 | 0.860 | 0 |
| N-gram SD | **1.113x** | 0.7388 | 0.542 | 54 |
| Official FLY | **0.993x** | 0.9358 | 1.83 (MAT) | 42 |
| TASD | **1.974x** | 0.8751 | 0.996 | 0 |

### pipeline_stage_config (80 samples)

Baseline AR TPS: **33.6** tok/s (generated only)

| Method | Speedup | SQ | Accept/MAT | Below 1.0x |
|--------|:-------:|:--:|:----------:|:----------:|
| AR | **1.000x** | 0.9219 | - | 0 |
| Greedy SD | **1.777x** | 0.8918 | 0.890 | 0 |
| N-gram SD | **1.182x** | 0.6777 | 0.535 | 49 |
| Official FLY | **1.127x** | 0.9397 | 1.70 (MAT) | 39 |
| TASD | **2.018x** | 0.8918 | 0.997 | 0 |

### complex_nested_config (80 samples)

Baseline AR TPS: **33.4** tok/s (generated only)

| Method | Speedup | SQ | Accept/MAT | Below 1.0x |
|--------|:-------:|:--:|:----------:|:----------:|
| AR | **1.000x** | 0.4857 | - | 0 |
| Greedy SD | **1.801x** | 0.4744 | 0.902 | 0 |
| N-gram SD | **1.486x** | 0.3425 | 0.553 | 36 |
| Official FLY | **1.728x** | 0.5047 | 1.57 (MAT) | 25 |
| TASD | **2.004x** | 0.4741 | 1.000 | 0 |

### rich_cli_option_groups (80 samples)

Baseline AR TPS: **33.0** tok/s (generated only)

| Method | Speedup | SQ | Accept/MAT | Below 1.0x |
|--------|:-------:|:--:|:----------:|:----------:|
| AR | **1.000x** | 0.5186 | - | 0 |
| Greedy SD | **1.792x** | 0.4931 | 0.894 | 0 |
| N-gram SD | **1.824x** | 0.4665 | 0.685 | 29 |
| Official FLY | **2.143x** | 0.5514 | 1.30 (MAT) | 7 |
| TASD | **2.015x** | 0.4927 | 1.000 | 0 |

## Overall (6 benchmarks × 80 samples = 480 samples)

| Method | Mean Speedup | Mean SQ | Below 1.0x |
|--------|:-----------:|:-------:|:----------:|
| AR | **1.000x** | 0.7354 | 0/0 |
| Greedy SD | **1.770x** | 0.7052 | 0/480 |
| N-gram SD | **1.413x** | 0.5637 | 233/480 |
| Official FLY | **1.622x** | 0.7504 | 126/480 |
| TASD | **1.956x** | 0.7022 | 9/480 |

## Key Findings

- **TASD calibrated**: 1.956x overall
- **Official FLY**: 1.622x overall
- **Greedy SD**: 1.770x overall
- **N-gram SD alone**: 1.413x overall

- TASD vs Greedy SD: 1.956x vs 1.770x (Δ=+0.185x)
- TASD vs Official FLY: 1.956x vs 1.622x (Δ=+0.333x)
- **TASD is the best overall method.**

### Comparison with old d16_b2_k3 experiment

| Benchmark | Old AR TPS | New AR TPS | Old TASD sp | New TASD sp | Change |
|-----------|:----------:|:----------:|:-----------:|:-----------:|:------:|
| argparse | 33 | 34 | 1.437x | **1.735x** | +0.298x |
| dict_config | 33 | 34 | 1.472x | **1.989x** | +0.517x |
| openmmlab_config | 33 | 34 | 0.000x | **1.974x** | +1.974x |
| pipeline_stage_config | 33 | 34 | 1.646x | **2.018x** | +0.372x |
| complex_nested_config | 33 | 33 | 1.588x | **2.004x** | +0.416x |
| rich_cli_option_groups | 33 | 33 | 1.596x | **2.015x** | +0.419x |

## Data Files

- Full data: `results/qwen_5method_6x80.json`
- Checkpoints: `results/qwen_6x80_checkpoints/`
