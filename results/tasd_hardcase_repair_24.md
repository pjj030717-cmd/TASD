# TASD-F Hard-Case Repair Experiment

**Hard cases**: 24 performance failures from 480-sample main experiment
**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Settings**: max_new_tokens=128, temperature=0.0

**TASD-F config**: `enable_failure_aware_fallback=True`, `fallback_tokens=2`, `fallback_guarded=False`

## Table 1: Hard Cases Overall Summary

| Method | Mean TPS | Median TPS | Mean Speedup | Median Speedup | Mean SQ | Mean OffStr | Mean Repair | Mean Accept | Mean Repetition | Mean Truncation |
|--------|----------|------------|--------------|----------------|---------|-------------|-------------|-------------|-----------------|-----------------|
| AR | 31.1 | 31.2 | - | - | 0.8525 | 0.0274 | 0.00 | 1.0000 | 0.4649 | 0.0000 |
| FLY | 34.0 | 34.7 | 1.09x | 1.10x | 0.7861 | 0.0054 | 0.00 | 0.5476 | 0.7004 | 0.0000 |
| TASD | 26.0 | 25.6 | 0.83x | 0.83x | 0.7521 | 0.0219 | 3.83 | 0.4190 | 0.6534 | 0.0000 |
| TASD-F | 28.6 | 29.7 | 0.91x | 0.93x | 0.7636 | 0.2706 | 1.54 | 0.4899 | 0.6432 | 0.0000 |

## Table 2: Below-1.0x Speedup Count

| Method | Below-1.0x Count | Total | Rate |
|--------|-----------------|-------|------|
| FLY | 6 | 24 | 25.0% |
| TASD | 16 | 24 | 66.7% |
| TASD-F | 13 | 24 | 54.2% |

## Table 3: Per-Sample Repair Effect (TASD -> TASD-F)

| # | Benchmark | Idx | Sample | TASD Sp | TASD-F Sp | Delta | Repair (TASD) | Repair (TASD-F) | Repair Delta |
|---|-----------|-----|--------|---------|-----------|-------|---------------|-----------------|---------------|
| 1 | Argparse | 22 | argparse\_real\_023 | 0.24x | 0.92x | +0.67x | 14 | 1 | +13 |
| 2 | Argparse | 73 | argparse\_real\_074 | 0.29x | 1.08x | +0.79x | 15 | 1 | +14 |
| 3 | DictConfig | 59 | dict\_config\_real\_060 | 0.33x | 0.76x | +0.43x | 0 | 0 | +0 |
| 4 | DictConfig | 2 | dict\_config\_real\_003 | 0.33x | 0.73x | +0.40x | 0 | 0 | +0 |
| 5 | DictConfig | 1 | dict\_config\_real\_002 | 0.48x | 0.64x | +0.16x | 0 | 0 | +0 |
| 6 | OpenMMLab | 70 | openmmlab\_config\_real\_071 | 0.48x | 1.12x | +0.64x | 7 | 1 | +6 |
| 7 | OpenMMLab | 64 | openmmlab\_config\_real\_065 | 0.55x | 0.93x | +0.38x | 6 | 1 | +5 |
| 8 | Argparse | 33 | argparse\_real\_034 | 0.60x | 0.36x | -0.24x | 5 | 0 | +5 |
| 9 | OpenMMLab | 48 | openmmlab\_config\_real\_049 | 0.70x | 1.25x | +0.55x | 4 | 1 | +3 |
| 10 | Argparse | 30 | argparse\_real\_031 | 0.73x | 0.22x | -0.51x | 4 | 7 | -3 |
| 11 | Argparse | 61 | argparse\_real\_062 | 0.64x | 0.52x | -0.12x | 4 | 5 | -1 |
| 12 | OpenMMLab | 0 | openmmlab\_config\_real\_001 | 0.82x | 0.63x | -0.19x | 3 | 2 | +1 |
| 13 | OpenMMLab | 2 | openmmlab\_config\_real\_003 | 0.83x | 0.63x | -0.20x | 3 | 2 | +1 |
| 14 | DictConfig | 13 | dict\_config\_real\_014 | 0.88x | 1.16x | +0.28x | 3 | 2 | +1 |
| 15 | Argparse | 69 | argparse\_real\_070 | 0.90x | 0.70x | -0.19x | 2 | 2 | +0 |
| 16 | DictConfig | 18 | dict\_config\_real\_019 | 0.98x | 1.27x | +0.28x | 4 | 2 | +2 |
| 17 | Argparse | 38 | argparse\_real\_039 | 1.07x | 0.29x | -0.78x | 2 | 0 | +2 |
| 18 | DictConfig | 57 | dict\_config\_real\_058 | 1.33x | 1.36x | +0.03x | 3 | 1 | +2 |
| 19 | DictConfig | 40 | dict\_config\_real\_041 | 1.22x | 0.78x | -0.44x | 1 | 1 | +0 |
| 20 | DictConfig | 52 | dict\_config\_real\_053 | 1.34x | 1.34x | -0.00x | 1 | 1 | +0 |
| 21 | DictConfig | 77 | dict\_config\_real\_078 | 1.26x | 1.22x | -0.03x | 3 | 2 | +1 |
| 22 | OpenMMLab | 29 | openmmlab\_config\_real\_030 | 1.29x | 1.34x | +0.05x | 4 | 2 | +2 |
| 23 | DictConfig | 78 | dict\_config\_real\_079 | 1.28x | 1.25x | -0.03x | 3 | 2 | +1 |
| 24 | DictConfig | 7 | dict\_config\_real\_008 | 1.40x | 1.39x | -0.01x | 1 | 1 | +0 |

## Key Findings

- **Below-1.0x reduction**: TASD has 16/24 below-1.0x cases, TASD-F reduces this to 13/24 (3 improved)
- **Repair count**: TASD mean=3.83, TASD-F mean=1.54 (60% reduction)
- **SQ preservation**: TASD mean=0.7521, TASD-F mean=0.7636 (no degradation)
- **Off-structure**: TASD mean=0.0219, TASD-F mean=0.2706 (notable increase — 2-token unguarded fallback produces more off-structure tokens)

### Interpretation

**TASD-F is effective for the worst failures**: For cases with TASD speedup < 0.5x (severe failures), TASD-F consistently improves speedup (e.g., 0.24→0.92x, 0.29→1.08x, 0.48→1.12x) and dramatically reduces repair count (14→1, 15→1, 7→1).

**TASD-F can regress moderate hard cases**: For cases where TASD already achieves 0.6-0.8x, TASD-F sometimes degrades performance (e.g., 0.73→0.22x, 0.82→0.63x, 1.07→0.29x). This suggests the 2-token fallback should be targeted at severe failures, not applied universally.

**Tradeoff: speed vs structure**: TASD-F's higher off-structure rate (0.27 vs 0.02) indicates the unguarded fallback produces structurally invalid tokens. This is acceptable for worst-case rescue scenarios where speed is the primary concern, but not for general use.

### Conclusion

TASD-F is an **optional hard-case extension**, not a replacement for TASD. It is most effective when:
- TASD speedup < 0.5x (severe failure)
- Repair count > 5 (excessive guard intervention)
- Accept rate < 0.2 (draft model severely off-target)

For moderate hard cases (0.6-0.9x) and cases already achieving >1.0x, standard TASD should be preferred to avoid regression and structural quality degradation.
