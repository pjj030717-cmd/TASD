> **Experimental safety analysis / future work.** Not part of the final TASD or TASD-F method. Kept here for reference.

# TASD-P (ProfitGuard) Full Validation

**Dataset**: 6 benchmarks x 20 samples = 120 samples

## Summary Table

| Method | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair | PG Triggered |
|--------|--------------|----------------|---------|---------|----|--------|--------|--------------|
| TASD | 1.88x | 1.97x | 6 | 6 | 0.8985 | 0.0190 | 0.2 | 0 (0.0%) |
| TASD-P | 1.89x | 1.97x | 5 | 8 | 0.9006 | 0.0176 | 0.1 | 10 (8.3%) |

## Per-Benchmark Breakdown

### TASD

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|----|--------|--------|
| Real-Python-Argparse | 20 | 1.93x | 1.93x | 0 | 0.9200 | 0.0000 | 0.1 |
| Real-Python-DictConfig | 20 | 1.62x | 1.84x | 4 | 0.8443 | 0.0000 | 0.6 |
| OpenMMLab-Config | 20 | 1.86x | 1.98x | 2 | 0.8974 | 0.0126 | 0.3 |
| Rich-CLI-Option-Groups | 20 | 1.97x | 1.98x | 0 | 0.9413 | 0.0500 | 0.0 |
| Complex-Nested-Config | 20 | 1.98x | 1.98x | 0 | 0.8299 | 0.0208 | 0.0 |
| Pipeline-Stage-Config | 20 | 1.94x | 1.98x | 0 | 0.9581 | 0.0303 | 0.1 |

### TASD-P

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|----|--------|--------|
| Real-Python-Argparse | 20 | 1.94x | 1.95x | 0 | 0.9200 | 0.0000 | 0.1 |
| Real-Python-DictConfig | 20 | 1.62x | 1.82x | 4 | 0.8433 | 0.0018 | 0.3 |
| OpenMMLab-Config | 20 | 1.87x | 1.97x | 1 | 0.9008 | 0.0193 | 0.1 |
| Rich-CLI-Option-Groups | 20 | 1.96x | 1.98x | 0 | 0.9413 | 0.0500 | 0.0 |
| Complex-Nested-Config | 20 | 1.97x | 2.00x | 0 | 0.8299 | 0.0208 | 0.0 |
| Pipeline-Stage-Config | 20 | 1.95x | 1.99x | 0 | 0.9683 | 0.0136 | 0.1 |

## ProfitGuard Trigger Analysis

- **Triggered**: 10/120 samples

### Trigger Reasons

| Reason | Count |
|--------|-------|
| severe_low_accept_rate | 10 |

## Quality Impact

- **SQ Delta**: +0.0021
- **OffStr Delta**: -0.0014
- **Truncation Delta**: -0.0045
