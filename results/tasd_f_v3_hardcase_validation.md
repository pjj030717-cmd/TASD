> **Abandoned variant.** TASD-F-v3 is not used in the final method because it causes severe slowdowns on short Argparse structures. All 7 Argparse hard cases dropped to ~1.9 TPS. Kept here as a negative result for reference.

# TASD-F v3 (Progressive Fallback) Hard-Case Validation

**Dataset**: 24 performance hard cases from 480-sample analysis

## Summary Table

| Method | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair | FB Triggered |
|--------|--------------|----------------|---------|---------|----|--------|--------|--------------|
| TASD | 0.83x | 0.81x | 15 | 17 | 0.0000 | 0.0000 | 3.8 | 0 |
| TASD_F | 0.92x | 0.92x | 14 | 16 | 0.0000 | 0.0000 | 1.4 | 0 |
| TASD_F_V3 | 0.79x | 1.00x | 12 | 17 | 0.0000 | 0.0000 | 1.0 | 26 |

## Per-Sample Comparison

| Benchmark | Sample | AR TPS | TASD | TASD-F | TASD-F-v3 | TASD Speedup | TASD-F-v3 Speedup | FB Level |
|-----------|--------|--------|------|--------|-----------|--------------|-------------------|----------|
| OpenMMLab-Config | 0 | 32.9 | 26.8 | 19.5 | 33.0 | 0.81x | 1.00x | L3 |
| OpenMMLab-Config | 2 | 32.9 | 26.6 | 20.1 | 33.2 | 0.81x | 1.01x | L3 |
| OpenMMLab-Config | 29 | 32.9 | 41.4 | 43.5 | 51.2 | 1.26x | 1.56x | L3 |
| OpenMMLab-Config | 48 | 32.9 | 23.0 | 41.1 | 45.8 | 0.70x | 1.39x | L3 |
| OpenMMLab-Config | 64 | 32.9 | 17.5 | 29.3 | 22.8 | 0.53x | 0.69x | L3 |
| OpenMMLab-Config | 70 | 32.9 | 16.2 | 36.7 | 33.1 | 0.49x | 1.01x | L3 |
| Real-Python-Argparse | 22 | 33.2 | 8.2 | 30.6 | 1.8 | 0.25x | 0.06x | L3 |
| Real-Python-Argparse | 30 | 33.2 | 23.3 | 6.8 | 1.9 | 0.70x | 0.06x | L3 |
| Real-Python-Argparse | 33 | 33.2 | 19.8 | 10.9 | 1.9 | 0.60x | 0.06x | L3 |
| Real-Python-Argparse | 38 | 33.2 | 35.8 | 9.9 | 1.9 | 1.08x | 0.06x | L3 |
| Real-Python-Argparse | 61 | 33.2 | 23.0 | 27.6 | 1.8 | 0.69x | 0.05x | L3 |
| Real-Python-Argparse | 69 | 33.2 | 28.2 | 22.1 | 1.9 | 0.85x | 0.06x | L3 |
| Real-Python-Argparse | 73 | 33.2 | 8.8 | 32.1 | 1.9 | 0.27x | 0.06x | L3 |
| Real-Python-DictConfig | 1 | 32.7 | 14.8 | 19.8 | 28.8 | 0.45x | 0.88x | L3 |
| Real-Python-DictConfig | 13 | 32.7 | 27.9 | 37.9 | 35.1 | 0.85x | 1.07x | L3 |
| Real-Python-DictConfig | 18 | 32.7 | 33.2 | 40.8 | 41.7 | 1.02x | 1.28x | L3 |
| Real-Python-DictConfig | 2 | 32.7 | 11.2 | 23.8 | 18.6 | 0.34x | 0.57x | L3 |
| Real-Python-DictConfig | 40 | 32.7 | 41.7 | 28.1 | 31.1 | 1.28x | 0.95x | L3 |
| Real-Python-DictConfig | 52 | 32.7 | 44.0 | 43.0 | 43.5 | 1.35x | 1.33x | No |
| Real-Python-DictConfig | 57 | 32.7 | 43.2 | 44.4 | 51.3 | 1.32x | 1.57x | L3 |
| Real-Python-DictConfig | 59 | 32.7 | 10.8 | 24.8 | 19.5 | 0.33x | 0.60x | L3 |
| Real-Python-DictConfig | 7 | 32.7 | 45.7 | 44.8 | 34.0 | 1.40x | 1.04x | L3 |
| Real-Python-DictConfig | 77 | 32.7 | 42.3 | 42.1 | 44.1 | 1.29x | 1.35x | L3 |
| Real-Python-DictConfig | 78 | 32.7 | 41.7 | 42.2 | 44.8 | 1.28x | 1.37x | L3 |

## Fallback Level Distribution

| Level | Count |
|-------|-------|
| Level 1 (mild/severe/persistent) | 6 |
| Level 2 (mild/severe/persistent) | 20 |
| Level 3 (mild/severe/persistent) | 0 |