> **Abandoned variant.** TASD-F-v3 is not used in the final method because it causes severe slowdowns on short Argparse structures. Kept here as a negative result for reference.

# TASD-F v3 (Progressive Fallback) Full Validation

**Dataset**: 6 benchmarks x 20 samples = 120 samples

## Summary Table

| Method | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair | FB Triggered | FB Rate |
|--------|--------------|----------------|---------|---------|----|--------|--------|--------------|---------|
| TASD | 1.90x | 2.00x | 6 | 6 | 0.0000 | 0.0000 | 0.2 | 0 | 0.00% |
| TASD_F | 1.91x | 1.99x | 4 | 5 | 0.0000 | 0.0000 | 0.1 | 0 | 0.00% |
| TASD_F_V3 | 1.91x | 2.01x | 4 | 6 | 0.0000 | 0.0000 | 0.1 | 13 | 10.83% |

### TASD

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|---------|----|--------|--------|
| Real-Python-Argparse | 20 | 1.95x | 2.00x | 0 | 0 | 0.0000 | 0.0000 | 0.1 |
| Real-Python-DictConfig | 20 | 1.65x | 1.99x | 4 | 4 | 0.0000 | 0.0000 | 0.6 |
| OpenMMLab-Config | 20 | 1.87x | 2.02x | 2 | 2 | 0.0000 | 0.0000 | 0.3 |
| Rich-CLI-Option-Groups | 20 | 2.00x | 2.00x | 0 | 0 | 0.0000 | 0.0000 | 0.0 |
| Complex-Nested-Config | 20 | 1.99x | 2.01x | 0 | 0 | 0.0000 | 0.0000 | 0.0 |
| Pipeline-Stage-Config | 20 | 1.97x | 2.00x | 0 | 0 | 0.0000 | 0.0000 | 0.1 |

### TASD_F

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|---------|----|--------|--------|
| Real-Python-Argparse | 20 | 1.95x | 2.01x | 0 | 0 | 0.0000 | 0.0000 | 0.1 |
| Real-Python-DictConfig | 20 | 1.69x | 1.94x | 2 | 3 | 0.0000 | 0.0000 | 0.4 |
| OpenMMLab-Config | 20 | 1.86x | 2.03x | 2 | 2 | 0.0000 | 0.0000 | 0.2 |
| Rich-CLI-Option-Groups | 20 | 1.99x | 2.01x | 0 | 0 | 0.0000 | 0.0000 | 0.0 |
| Complex-Nested-Config | 20 | 1.97x | 1.97x | 0 | 0 | 0.0000 | 0.0000 | 0.0 |
| Pipeline-Stage-Config | 20 | 1.98x | 2.01x | 0 | 0 | 0.0000 | 0.0000 | 0.1 |

### TASD_F_V3

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|---------|----|--------|--------|
| Real-Python-Argparse | 20 | 1.95x | 1.99x | 0 | 0 | 0.0000 | 0.0000 | 0.1 |
| Real-Python-DictConfig | 20 | 1.69x | 1.92x | 2 | 4 | 0.0000 | 0.0000 | 0.4 |
| OpenMMLab-Config | 20 | 1.91x | 2.04x | 2 | 2 | 0.0000 | 0.0000 | 0.1 |
| Rich-CLI-Option-Groups | 20 | 1.97x | 2.02x | 0 | 0 | 0.0000 | 0.0000 | 0.0 |
| Complex-Nested-Config | 20 | 1.98x | 1.99x | 0 | 0 | 0.0000 | 0.0000 | 0.0 |
| Pipeline-Stage-Config | 20 | 1.98x | 2.01x | 0 | 0 | 0.0000 | 0.0000 | 0.1 |

## Quality Impact

- **SQ Delta**: +0.0000
- **OffStr Delta**: +0.0000