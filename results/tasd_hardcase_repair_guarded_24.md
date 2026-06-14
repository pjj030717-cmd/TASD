# TASD-F-G Hard-Case Repair Experiment

**Hard cases**: 24 performance failures from 480-sample main experiment
**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct

### Variants

| Variant | fallback_guarded | accept_threshold | repair_threshold |
|---------|-----------------|-------------------|------------------|
| TASD-F (unguarded) | False | 0.5 | 2 |
| TASD-F-G (guarded) | **True** | 0.5 | 2 |
| TASD-F-G-Sel (selective guarded) | **True** | **0.2** | **3** |

## Table 1: Hard Cases — All Methods Summary

| Method | Mean TPS | Median TPS | Mean Speedup | Median Speedup | Mean SQ | Mean OffStr | Mean Repair | Mean Accept | Below 1.0x |
| -------|----------|------------|--------------|----------------|---------|-------------|-------------|-------------|------------ |
| AR | 31.1 | 31.2 | - | - | 0.8525 | 0.0274 | 0.00 | 1.0000 | 0/24 |
| FLY | 34.0 | 34.7 | 1.09x | 1.10x | 0.7861 | 0.0054 | 0.00 | 0.5476 | 6/24 |
| TASD | 26.0 | 25.6 | 0.83x | 0.83x | 0.7521 | 0.0219 | 3.83 | 0.4190 | 16/24 |
| TASD-F | 28.6 | 29.7 | 0.91x | 0.93x | 0.7636 | 0.2706 | 1.54 | 0.4899 | 13/24 |
| TASD-F-G | 32.9 | 35.5 | 1.06x | 1.14x | 0.7892 | 0.0438 | 1.50 | 0.5977 | 12/24 |
| TASD-F-G-Sel | 35.3 | 37.4 | 1.14x | 1.16x | 0.7877 | 0.0487 | 1.71 | 0.5811 | 9/24 |

## Table 2: Below-1.0x Count Comparison

| Variant | Below-1.0x | Rate | vs TASD |
|---------|-----------|------|---------|
| TASD-F | 13/24 | 54.2% | +3 |
| TASD-F-G | 12/24 | 50.0% | +4 |
| TASD-F-G-Sel | 9/24 | 37.5% | +7 |

## Table 3: Per-Sample Speedup Deltas (vs TASD baseline)

| # | Benchmark | Idx | Name | TASD | TASD-F | TASD-F-G | TASD-F-G-Sel |
|---|-----------|-----|------|------|--------|----------|--------------|
| 1 | Argparse | 22 | argparse\_real\_023 | 0.24x | 0.92x | 0.81x | 1.16x |
| 2 | Argparse | 73 | argparse\_real\_074 | 0.29x | 1.08x | 1.28x | 1.34x |
| 3 | DictConfig | 59 | dict\_config\_real\_060 | 0.33x | 0.76x | 0.74x | 0.79x |
| 4 | DictConfig | 2 | dict\_config\_real\_003 | 0.33x | 0.73x | 0.75x | 0.83x |
| 5 | DictConfig | 1 | dict\_config\_real\_002 | 0.48x | 0.64x | 1.14x | 0.84x |
| 6 | OpenMMLab | 70 | openmmlab\_config\_real\_ | 0.48x | 1.12x | 0.90x | 0.76x |
| 7 | OpenMMLab | 64 | openmmlab\_config\_real\_ | 0.55x | 0.93x | 0.78x | 1.01x |
| 8 | Argparse | 33 | argparse\_real\_034 | 0.60x | 0.36x | 1.76x | 1.75x |
| 9 | OpenMMLab | 48 | openmmlab\_config\_real\_ | 0.70x | 1.25x | 1.34x | 0.95x |
| 10 | Argparse | 30 | argparse\_real\_031 | 0.73x | 0.22x | 1.74x | 1.73x |
| 11 | Argparse | 61 | argparse\_real\_062 | 0.64x | 0.52x | 0.59x | 0.58x |
| 12 | OpenMMLab | 0 | openmmlab\_config\_real\_ | 0.82x | 0.63x | 0.71x | 0.87x |
| 13 | OpenMMLab | 2 | openmmlab\_config\_real\_ | 0.83x | 0.63x | 0.64x | 0.79x |
| 14 | DictConfig | 13 | dict\_config\_real\_014 | 0.88x | 1.16x | 0.65x | 0.88x |
| 15 | Argparse | 69 | argparse\_real\_070 | 0.90x | 0.70x | 1.45x | 1.47x |
| 16 | DictConfig | 18 | dict\_config\_real\_019 | 0.98x | 1.27x | 1.17x | 1.04x |
| 17 | Argparse | 38 | argparse\_real\_039 | 1.07x | 0.29x | 0.55x | 1.11x |
| 18 | DictConfig | 57 | dict\_config\_real\_058 | 1.33x | 1.36x | 0.89x | 1.34x |
| 19 | DictConfig | 40 | dict\_config\_real\_041 | 1.22x | 0.78x | 0.87x | 1.30x |
| 20 | DictConfig | 52 | dict\_config\_real\_053 | 1.34x | 1.34x | 1.42x | 1.44x |
| 21 | DictConfig | 77 | dict\_config\_real\_078 | 1.26x | 1.22x | 1.22x | 1.26x |
| 22 | OpenMMLab | 29 | openmmlab\_config\_real\_ | 1.29x | 1.34x | 1.36x | 1.29x |
| 23 | DictConfig | 78 | dict\_config\_real\_079 | 1.28x | 1.25x | 1.21x | 1.30x |
| 24 | DictConfig | 7 | dict\_config\_real\_008 | 1.40x | 1.39x | 1.47x | 1.42x |

## Criteria Check

| Variant | Below ↓ | Repair ↓ | OffStr < 0.05 | Speedup ≥ TASD | All Pass |
|---------|---------|----------|---------------|----------------|----------|
| TASD-F-G | YES | YES | YES | YES | **PASS** |
| TASD-F-G-Sel | YES | YES | YES | YES | **PASS** |

## Key Findings

### TASD-F-G

- Below-1.0x: 12/24 (TASD: 16/24)
- Mean Repair: 1.50 (TASD: 3.83)
- Off-Structure: 0.0438 (TASD: 0.0219)
- Mean Speedup: 1.06x (TASD: 0.83x)

**Verdict**: TASD-F-G passes all criteria. Can serve as a positive TASD-F extension.

### TASD-F-G-Sel

- Below-1.0x: 9/24 (TASD: 16/24)
- Mean Repair: 1.71 (TASD: 3.83)
- Off-Structure: 0.0487 (TASD: 0.0219)
- Mean Speedup: 1.14x (TASD: 0.83x)

**Verdict**: TASD-F-G-Sel passes all criteria. Can serve as a positive TASD-F extension.

