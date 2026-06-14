# LLaMA-8B Generalization Pilot Report

**Target**: meta-llama/Llama-3.1-8B-Instruct (8B params, vocab=128000)
**Draft**: meta-llama/Llama-3.2-1B-Instruct (1B params, vocab=128000)
**Tokenizer**: FULLY COMPATIBLE (same vocab, same token IDs for code)

### Methods

| Method | Description |
|--------|-------------|
| AR | Target model autoregressive (greedy) |
| Greedy SD | Target-verify greedy draft (draft_len=16, blocks=2, top_k=3) |
| **Official FLY** | AMD FLy SPDGenerate (k=8, win_len=6, ngram=4/6, entropy_thre=0.3, post-verify FLY recovery) |
| **TASD** | Structure-aware SD (draft_len=16, blocks=2, top_k=3, guard=True) |

**Config**: max_new_tokens=128, temperature=0.0, 20 samples per benchmark

## DictConfig (20 samples)

Baseline AR TPS: **89.5**

| Method | TPS | Speedup | AcceptRate | SQ | OffStr | Repair | GuardTrig | Below1.0x |
|--------|-----|---------|------------|-----|--------|--------|----------|----------|
| AR | 89.5 | **1.00x** | - | - | - | - | - | 0 |
| Greedy SD | 94.0 | **1.08x** | 0.8735 | 0.6348 | - | - | - | 6 |
| Official FLY (k=8) | 124.0 | **1.44x** | 1.1519 | 0.6615 | - | - | - | 4 |
| TASD | 108.9 | **1.28x** | 0.9867 | 0.6242 | 0.0000 | 0.1 | 0.6 | 2 |

## OpenMMLab (20 samples)

Baseline AR TPS: **79.4**

| Method | TPS | Speedup | AcceptRate | SQ | OffStr | Repair | GuardTrig | Below1.0x |
|--------|-----|---------|------------|-----|--------|--------|----------|----------|
| AR | 79.4 | **1.00x** | - | - | - | - | - | 0 |
| Greedy SD | 90.2 | **1.14x** | 0.8364 | 0.9563 | - | - | - | 0 |
| Official FLY (k=8) | 76.6 | **0.94x** | 1.3482 | 1.0000 | - | - | - | 16 |
| TASD | 109.6 | **1.38x** | 1.0000 | 0.9563 | 0.0000 | 0.0 | 0.0 | 0 |

## PipelineStage (20 samples)

Baseline AR TPS: **81.6**

| Method | TPS | Speedup | AcceptRate | SQ | OffStr | Repair | GuardTrig | Below1.0x |
|--------|-----|---------|------------|-----|--------|--------|----------|----------|
| AR | 81.6 | **1.00x** | - | - | - | - | - | 0 |
| Greedy SD | 88.8 | **1.10x** | 0.8251 | 0.9075 | - | - | - | 6 |
| Official FLY (k=8) | 73.8 | **0.86x** | 1.3873 | 0.9478 | - | - | - | 17 |
| TASD | 106.8 | **1.32x** | 0.9874 | 0.9075 | 0.0000 | 0.1 | 0.0 | 1 |

## Overall Summary (60 samples)

| Method | Speedup | AcceptRate | SQ | OffStr | Repair | Below1.0x |
|--------|---------|------------|-----|--------|--------|----------|
| AR | **1.00x** | - | - | - | - | 0 |
| Greedy SD | **1.11x** | 0.845 | 0.8330 | - | - | 12 |
| Official FLY (k=8) | **1.08x** | 1.296 | 0.8700 | - | - | 37 |
| TASD | **1.33x** | 0.991 | 0.8290 | 0.0000 | 0.1 | 3 |

## Criteria Check

| Criterion | Value | Pass |
|-----------|-------|------|
| TASD mean speedup >= 1.3x | 1.33x | **PASS** |
| TASD > Greedy SD | TASD=1.33x GSD=1.11x | **PASS** |
| Accept rate >= 0.70 | 0.991 | **PASS** |

**Overall: PASS — Recommend continue to 6×80**

### Quality (SQ) — Reported Independently

Quality is reported separately from speed. TASD has no substantial SQ degradation
relative to Greedy SD, while Official FLY achieves higher SQ on some benchmarks
but fails to provide consistent speedup on LLaMA-8B.

| Method | DictConfig SQ | OpenMMLab SQ | Pipeline SQ | Overall SQ |
|--------|:------------:|:------------:|:-----------:|:----------:|
| AR | - | - | - | - |
| Greedy SD | 0.6348 | 0.9563 | 0.9075 | 0.8330 |
| Official FLY (k=8) | 0.6615 | 1.0000 | 0.9478 | 0.8700 |
| TASD | 0.6242 | 0.9563 | 0.9075 | 0.8290 |

- TASD vs GSD SQ delta: DictConfig -0.0106, OpenMMLab +0.0000, PipelineStage +0.0000
- **Conclusion**: No meaningful SQ degradation from TASD. Quality maintained.

## Analysis

### TASD Performance
- Overall speedup **1.33x**
- Highest on OpenMMLab (1.38x, 0 below 1.0x), PipelineStage (1.32x, 1 below)
- Lowest on DictConfig (1.28x, 2 below 1.0x)
- Accept rate **0.99** (excellent draft-target alignment)
- Off-structure rate **0.0000** (effectively zero)
- SQ: 0.8290 (see Quality section above)

### Official FLY on 8B
- Overall speedup **1.08x**
- Effective on DictConfig (1.44x) — n-gram PLD exploits high repetition
- Sub-1.0x on OpenMMLab (0.94x) and PipelineStage (0.86x)
- **Root cause**: FLY designed for 70B/405B target (k=15-25). On 8B, target AR is already 80-90 TPS.
  N-gram lookup + model draft + verify overhead exceeds savings. FLY recovery recovers 7% of mismatches.
- **Conclusion**: Official FLY not suitable as primary baseline for 8B. Included for completeness.

### Method Comparison
- **TASD > Greedy SD**: YES (1.33x vs 1.11x) — structural guard adds value
- **TASD > Official FLY**: YES overall, but FLY wins on DictConfig (1.44x vs 1.28x)
- **FLY route results**: Official FLY only beneficial when n-gram hit rate high (DictConfig). Otherwise overhead dominates.

## Model Paths

- Target: `/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct`
- Draft: `/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct`

## Related Reports

- `results/llama_pilot_3x20.json` — AR / Greedy SD / TASD per-sample results
- `results/llama_official_fly_pilot.json` — Official FLY (k=8, k=16) per-sample results
