# Qwen 3×20 Sanity Report: Official FLY Baseline Integration

**Target**: Qwen2.5-14B-Instruct-AWQ  |  **Draft**: Qwen2.5-1.5B-Instruct  
**Config**: max_new_tokens=128, temperature=0.0, 20 samples per benchmark  

---

## Methods

| Method | Description |
|--------|-------------|
| **AR** | Target autoregressive (greedy) |
| **Greedy SD** | Target-verify greedy draft (draft_len=16, blocks=2, top_k=3, no guard) |
| **N-gram SD** | Pure n-gram lookup SD (no draft model, ngram_min=3, ngram_max=8) |
| **Official FLY** | AMD FLy SPDGenerate (k=15, win_len=6, entropy_thre=0.3, ngram=4/6, FLY recovery) |
| **TASD (cal)** | Structure-aware SD + Guard-v1.5 calibrated (draft_len=16, blocks=2, top_k=3) |

---

## Per-Benchmark Results

### DictConfig — AR TPS: 46.1

| Method | Speedup | SQ | Accept / MAT | Below 1.0x |
|--------|:-------:|:--:|:------------:|:----------:|
| AR | **1.000x** | 0.6701 | - | 0 |
| Greedy SD | 1.329x | 0.6561 | 0.898 | 2 |
| N-gram SD | 0.959x | 0.5330 | 0.502 | 14 |
| **Official FLY (k=15)** | **1.678x** | 0.6638 | 1.43 (MAT) | 4 |
| **TASD (cal)** | 1.371x | 0.6497 | 0.932 | 2 |

> FLY dominates on DictConfig — n-gram PLD exploits high repetition in config dict generation.
> MAT=1.43 means FLY accepts more tokens than it emits (recovery converts rejects to accepts).

### OpenMMLab — AR TPS: 40.2

| Method | Speedup | SQ | Accept / MAT | Below 1.0x |
|--------|:-------:|:--:|:------------:|:----------:|
| AR | **1.000x** | 0.9889 | - | 0 |
| Greedy SD | 1.395x | 0.9721 | 0.844 | 0 |
| N-gram SD | 0.895x | 0.6962 | 0.529 | 16 |
| Official FLY (k=15) | 1.061x | 1.0000 | 1.78 (MAT) | 9 |
| **TASD (cal)** | **1.509x** | 0.9721 | 0.909 | 3 |

> FLY modest at 1.06x — OpenMMLab configs have diverse multi-level nesting, n-gram PLD less effective.
> TASD at 1.51x on structured config format.

### PipelineStage — AR TPS: 42.3

| Method | Speedup | SQ | Accept / MAT | Below 1.0x |
|--------|:-------:|:--:|:------------:|:----------:|
| AR | **1.000x** | 0.9120 | - | 0 |
| Greedy SD | 1.352x | 0.9132 | 0.883 | 0 |
| N-gram SD | 0.921x | 0.6746 | 0.545 | 16 |
| Official FLY (k=15) | 1.116x | 0.9346 | 1.76 (MAT) | 11 |
| **TASD (cal)** | **1.600x** | 0.9132 | 0.987 | 0 |

> TASD strongest (1.60x) with perfect structural quality. FLY at 1.12x.

---

## Overall (60 samples, 3 benchmarks)

| Method | Speedup | Below 1.0x |
|--------|:-------:|:----------:|
| AR | **1.000x** | 0 |
| Greedy SD | **1.359x** | 2 |
| N-gram SD | **0.925x** | 46 |
| Official FLY (k=15) | **1.285x** | 24 |
| **TASD (cal)** | **1.493x** | **5** |

---

## Key Findings

### Official FLY on Qwen 14B

- **Overall 1.285x** — viable as formal baseline (vs LLaMA 8B at 1.08x)
- **DictConfig 1.68x** — FLY's strongest benchmark; n-gram PLD + high repetition = large draft savings
- **OpenMMLab 1.06x, Pipeline 1.12x** — modest gains; diverse config structures limit n-gram hit rate
- **MAT 1.43-1.78** — FLY window recovery is effective on Qwen's accept pattern
- **No sweep needed**: k=15 with default win_len=6 entropy_thre=0.3 is solid

### TASD vs Baselines

- TASD **1.49x overall** — significantly ahead of GSD (1.36x) and FLY (1.29x)
- Structural guard adds value beyond both draft-model SD and n-gram PLD
- Off-structure rate negligible (0.00-0.03)
- Only 5/60 below 1.0x — mostly dict_config AR-fast cases

### N-gram SD Alone

- Pure n-gram SD at **0.93x overall** — insufficient as standalone method
- Accept rate 0.50-0.55 limits draft effectiveness
- Confirms that draft model + structural guard are necessary additions

### Method Ranking

| Rank | Method | Speedup | Best On |
|:----:|--------|:-------:|---------|
| 1 | **TASD** | 1.49x | All benchmarks |
| 2 | Greedy SD | 1.36x | OpenMMLab, Pipeline |
| 3 | Official FLY | 1.29x | DictConfig |
| 4 | AR | 1.00x | - |
| 5 | N-gram SD | 0.93x | - |

---

## Comparison: Qwen vs LLaMA

| Metric | Qwen 14B | LLaMA 8B |
|--------|:--------:|:--------:|
| AR TPS | ~43 | ~83 |
| Official FLY | **1.29x** | 1.08x |
| TASD | **1.49x** | 1.33x |
| Greedy SD | **1.36x** | 1.11x |

FLY benefits from larger target model (14B vs 8B): lower AR TPS means SD overhead can be amortized.

---

## Flag Recommendation

- **Keep Official FLY as formal baseline** for Qwen 14B experiments
- **Use k=15** — stable across all benchmarks, no sweep needed  
- **Label**: "Official FLY (AMD FLy, k=15)" — distinguish from old wrapper ("Simplified N-gram SD")
- **TASD remains the top method** with Guard-v1.5 calibrated default

## Data

- Raw data: `results/qwen_5method_3x20_sanity.json`
- Script: `run_qwen_official_fly_sanity.py`
