# LLaMA Generalization Pilot Report

**Target**: meta-llama/Llama-3.1-8B-Instruct (vocab=128000, 8B params)
**Draft**: meta-llama/Llama-3.2-1B-Instruct (vocab=128000, 1B params)
**Tokenizer**: FULLY COMPATIBLE (same vocab, same BOS/EOS/PAD, identical encoding)

**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3
**Samples**: 20 per benchmark × 3 benchmarks = 60 total

## Per-Benchmark Results

### dict_config (20 samples)

| Method | TPS | Speedup | Accept | SQ | OffStr | Repair | GuardTrig | Below1.0x |
|--------|-----|---------|--------|-----|--------|--------|----------|----------|
| AR | 89.5 | 1.00x | - | - | - | - | - | - |
| Greedy SD | 94.0 | 1.08x | 0.874 | 0.6348 | - | - | - | 6 |
| Simplified N-gram SD | 59.9 | 0.69x | - | 0.6232 | - | - | - | - |
| **TASD** | 99.2 | **1.14x** | 0.913 | 0.6229 | 0.0000 | 0.2 | 1.6 | 5 |

### openmmlab_config (20 samples)

| Method | TPS | Speedup | Accept | SQ | OffStr | Repair | GuardTrig | Below1.0x |
|--------|-----|---------|--------|-----|--------|--------|----------|----------|
| AR | 79.4 | 1.00x | - | - | - | - | - | - |
| Greedy SD | 90.2 | 1.14x | 0.836 | 0.9563 | - | - | - | 0 |
| Simplified N-gram SD | 62.6 | 0.79x | - | 1.0000 | - | - | - | - |
| **TASD** | 109.6 | **1.38x** | 1.000 | 0.9563 | 0.0000 | 0.0 | 0.0 | 0 |

### pipeline_stage_config (20 samples)

| Method | TPS | Speedup | Accept | SQ | OffStr | Repair | GuardTrig | Below1.0x |
|--------|-----|---------|--------|-----|--------|--------|----------|----------|
| AR | 81.6 | 1.00x | - | - | - | - | - | - |
| Greedy SD | 88.8 | 1.10x | 0.825 | 0.9075 | - | - | - | 6 |
| Simplified N-gram SD | 61.8 | 0.76x | - | 0.9397 | - | - | - | - |
| **TASD** | 106.8 | **1.32x** | 0.987 | 0.9075 | 0.0000 | 0.1 | 0.0 | 1 |

## Overall Summary

| Method | TPS | Speedup | Accept | SQ | OffStr | Repair | Below1.0x |
|--------|-----|---------|--------|-----|--------|--------|----------|
| AR | 83.5 | 1.00x | - | - | - | - | - |
| Greedy SD | 91.0 | 1.10x | 0.845 | 0.8329 | - | - | - |
| Simplified N-gram SD | 61.4 | 0.75x | - | 0.8543 | - | - | - |
| **TASD** | 105.2 | **1.28x** | 0.967 | 0.8289 | 0.0000 | 0.1 | 6 |

## Criteria Check

| Criterion | Value | Pass |
|-----------|-------|------|
| TASD speedup >= 1.3x | 1.28x | FAIL |
| TASD > Greedy SD | TASD=1.28x GSD=1.10x | OK |
| SQ >= best non-TASD - 0.03 | TASD=0.8289 GSD=0.8329 Simplified=0.8543 | OK |
| Accept rate >= 0.70 | 0.967 | OK |

**Overall**: FAIL — Investigate before scaling |

## Tokenizer Compatibility

- Same vocab_size: 128000 ✓
- Same BOS/EOS/PAD: ✓
- Python code encoding test: 43 tokens, identical IDs ✓
- Compatible for speculative decoding: **YES**

## Model Paths

- Target: `/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct`
- Draft: `/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct`
