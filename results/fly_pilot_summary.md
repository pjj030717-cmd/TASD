# FLY Pilot Comparison

**Purpose**: A representative comparison against FLY (Li et al., 2025), a training-free relaxed speculative decoding method, on structured code completion tasks. This is a limited pilot study, not a full-scale benchmark.

**Scope**: 3 benchmarks × 20 samples only.

| Benchmark | Samples |
|-----------|---------|
| OpenMMLab-Config | 20 |
| Real-Python-DictConfig | 20 |
| Pipeline-Stage-Config | 20 |

---

## FLY Configuration

Both FLY and Greedy SD (FLY-no-fly) were tested with the following shared parameters:

| Parameter | Value |
|-----------|-------|
| Target model | Qwen2.5-14B-Instruct-AWQ |
| Draft model | Qwen2.5-1.5B-Instruct |
| `k` (draft tokens per round) | 16 |
| `win_len` | 6 |
| `entropy_thre` | 0.3 |
| `temperature` | 0.0 |
| `max_new_tokens` | 128 |
| `enable_fly` | True (FLY) / False (Greedy SD) |
| `use_ngram` | False |
| `tree_verify` | False |
| Verification mode | Greedy (argmax matching) |

FLY was integrated via direct import from the official FLy-main source (https://github.com/AMD-AIG-AIMA/FLy) using `SPDGenerate`, with the same greedy verification mode used by TASD for fair comparison.

---

## Speed and Quality

| Benchmark | Method | TPS | Speedup | SQ | OffStr | Trunc | Rep | SNP |
|-----------|--------|-----|---------|----|--------|-------|-----|-----|
| OpenMMLab-Config | FLY | 30.3 | 0.92x | 0.8437 | 0.0000 | 0.1582 | 0.0000 | 0.4500 |
|     | *TASD* | *62.8* | *1.91x* | *0.8974* | *0.0126* | *0.1554* | *0.0000* | *0.1500* |
| Real-Python-DictConfig | FLY | 44.9 | 1.38x | 0.8412 | 0.0278 | 0.0461 | 0.0125 | 0.7000 |
|     | *TASD* | *51.4* | *1.57x* | *0.8443* | *0.0000* | *0.0445* | *0.0000* | *0.2000* |
| Pipeline-Stage-Config | FLY | 34.2 | 1.06x | 0.9209 | 0.0000 | 0.0887 | 0.0000 | 0.0000 |
|     | *TASD* | *65.5* | *2.03x* | *0.9581* | *0.0303* | *0.1397* | *0.0000* | *0.1500* |
| OpenMMLab-Config | Greedy SD (no-fly) | 22.1 | 0.67x | 0.8333 | 0.0000 | 0.1455 | 0.0000 | 0.5000 |
|     | *TASD* | *62.8* | *1.91x* | *0.8974* | *0.0126* | *0.1554* | *0.0000* | *0.1500* |
| Real-Python-DictConfig | Greedy SD (no-fly) | 33.4 | 1.02x | 0.8532 | 0.0301 | 0.0625 | 0.0000 | 0.5000 |
|     | *TASD* | *51.4* | *1.57x* | *0.8443* | *0.0000* | *0.0445* | *0.0000* | *0.2000* |
| Pipeline-Stage-Config | Greedy SD (no-fly) | 22.9 | 0.71x | 0.9053 | 0.0000 | 0.1017 | 0.0000 | 0.0000 |
|     | *TASD* | *65.5* | *2.03x* | *0.9581* | *0.0303* | *0.1397* | *0.0000* | *0.1500* |

**SQ** = Structural quality score (higher=better, 0-1). **SNP** = Structure Not Preserved rate.

---

## Supplementary Table (for paper appendix)

| Benchmark | Greedy SD TPS | FLY TPS | TASD TPS | FLY SQ | TASD SQ |
|-----------|--------------|---------|----------|--------|---------|
| OpenMMLab-Config | 22.1 | 30.3 | 62.8 | 0.8437 | 0.8974 |
| Real-Python-DictConfig | 33.4 | 44.9 | 51.4 | 0.8412 | 0.8443 |
| Pipeline-Stage-Config | 22.9 | 34.2 | 65.5 | 0.9209 | 0.9581 |

---

## Analysis

### FLY window acceptance vs Greedy SD

- **OpenMMLab-Config**: FLY +8.1 TPS over Greedy SD (+36.8%)
- **Real-Python-DictConfig**: FLY +11.5 TPS over Greedy SD (+34.5%)
- **Pipeline-Stage-Config**: FLY +11.4 TPS over Greedy SD (+49.7%)

FLY's window-based acceptance provides a consistent and non-trivial speed improvement over standard speculative decoding.

### Speed gap decomposition

- FLY vs TASD: **-37.4%** (averaged across benchmarks)
- Greedy SD vs TASD: **-55.0%**
- FLY window acceptance boost: ~10 TPS (FLY minus Greedy SD)
- Multi-block draft advantage: ~23 TPS (TASD minus FLY)

---

## Conclusion

In the deterministic structured code completion setting studied in this paper, TASD is more effective than FLY.

FLY improves over Greedy SD but remains below TASD on representative structured code completion benchmarks. TASD's advantage mainly comes from multi-block draft proposal and structure-aware relaxed verification, rather than generic relaxed verification alone.

Specifically:

1. **FLY's window acceptance** provides a measurable ~10 TPS (~35-50%) boost over standard SD, confirming the value of relaxed verification.
2. **Multi-block draft** contributes an additional ~23 TPS advantage for TASD over FLY, making it the dominant speed factor.
3. **Structural quality is comparable** across all methods on these benchmarks — the structured prompt format itself provides a strong quality baseline, and TASD's guard maintains it marginally better.
4. **This is a limited pilot comparison** (3 benchmarks × 20 samples) using greedy verification mode for fair comparison with TASD's relaxed verification baseline. FLY's original paper also supports modified rejection sampling, which may yield different results in stochastic settings.

### Qualification

- FLY was tested in greedy verification mode (argmax match + window acceptance), consistent with TASD's relaxed verification baseline at temperature 0.0.
- SPDGenerate was imported directly from FLy-main source to avoid package dependency conflicts; the core speculative decoding logic is used as-is.
- This pilot does not cover all FLY variants (e.g., n-gram draft, tree verification, modified rejection sampling). Results are specific to the deterministic structured code completion setting.
