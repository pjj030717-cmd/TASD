> **Experimental safety analysis / future work.** Not part of the final TASD or TASD-F method. Kept here for reference.

# TASD-P v2 (ProfitGuard) Hard-Case Validation

**Dataset**: 24 performance hard cases from 480-sample analysis

## Summary Table

| Method | Mean Speedup | Median Speedup | # <1.0x | # <1.2x | SQ | OffStr | Repair | PG Triggered |
|--------|--------------|----------------|---------|---------|----|--------|--------|--------------|
| TASD | 0.82x | 0.80x | 17 | 18 | 0.7990 | 0.0222 | 3.8 | 0 (0.0%) |
| TASD-F | 1.08x | 1.12x | 11 | 13 | 0.8540 | 0.0442 | 1.5 | 0 (0.0%) |
| TASD-P | 0.91x | 0.95x | 15 | 20 | 0.8037 | 0.0423 | 2.6 | 13 (54.2%) |

## Per-Benchmark Breakdown

### TASD

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|----|--------|--------|
| Real-Python-Argparse | 7 | 0.61x | 0.69x | 7 | 0.8101 | 0.0163 | 6.7 |
| Real-Python-DictConfig | 11 | 0.97x | 1.19x | 5 | 0.8016 | 0.0000 | 1.7 |
| OpenMMLab-Config | 6 | 0.79x | 0.74x | 5 | 0.7814 | 0.0696 | 4.3 |

### TASD-F

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|----|--------|--------|
| Real-Python-Argparse | 7 | 1.19x | 1.27x | 2 | 0.8885 | 0.0000 | 1.3 |
| Real-Python-DictConfig | 11 | 1.07x | 1.08x | 5 | 0.8725 | 0.0139 | 1.0 |
| OpenMMLab-Config | 6 | 0.97x | 0.92x | 4 | 0.7799 | 0.1514 | 2.8 |

### TASD-P

| Benchmark | N | Mean Speedup | Median Speedup | # <1.0x | SQ | OffStr | Repair |
|-----------|---|--------------|----------------|---------|----|--------|--------|
| Real-Python-Argparse | 7 | 0.66x | 0.71x | 6 | 0.8074 | 0.0163 | 5.6 |
| Real-Python-DictConfig | 11 | 1.01x | 1.04x | 5 | 0.8180 | 0.0196 | 1.4 |
| OpenMMLab-Config | 6 | 1.02x | 0.97x | 4 | 0.7731 | 0.1144 | 1.5 |

## Per-Sample Comparison

| Benchmark | Sample | AR TPS | TASD | TASD-F | TASD-P | TASD Speedup | TASD-P Speedup | PG Triggered? | PG Reason |
|-----------|--------|--------|------|--------|--------|--------------|----------------|---------------|-----------|
| Real-Python-Argparse | 22 | 34.0 | 7.6 | 39.51 | 8.73 | 0.22x | 0.2571428571428571 | No | None |
| Real-Python-Argparse | 73 | 33.5 | 8.9 | 42.93 | 12.56 | 0.27x | 0.37458991947509696 | Yes | severe_low_accept_rate |
| Real-Python-Argparse | 33 | 32.6 | 20.1 | 32.12 | 19.15 | 0.62x | 0.5877839165131983 | No | None |
| Real-Python-Argparse | 30 | 32.7 | 22.4 | 43.34 | 23.42 | 0.69x | 0.7153329260843005 | No | None |
| Real-Python-Argparse | 61 | 32.1 | 22.3 | 42.54 | 22.89 | 0.69x | 0.7124183006535947 | No | None |
| Real-Python-Argparse | 69 | 32.1 | 28.9 | 40.6 | 28.81 | 0.90x | 0.8983473651387589 | No | None |
| Real-Python-Argparse | 38 | 34.5 | 29.5 | 33.14 | 36.58 | 0.85x | 1.0609048723897911 | No | None |
| Real-Python-DictConfig | 59 | 34.2 | 11.2 | 27.01 | 21.01 | 0.33x | 0.6137890739117734 | Yes | severe_low_accept_rate |
| Real-Python-DictConfig | 2 | 32.1 | 11.7 | 27.25 | 21.06 | 0.36x | 0.6558704453441295 | Yes | severe_low_accept_rate |
| Real-Python-DictConfig | 1 | 31.7 | 15.7 | 17.67 | 28.8 | 0.50x | 0.9093779602147142 | Yes | severe_low_accept_rate |
| Real-Python-DictConfig | 13 | 34.4 | 27.3 | 33.46 | 27.52 | 0.79x | 0.7997675094449289 | Yes | severe_low_accept_rate |
| Real-Python-DictConfig | 18 | 34.2 | 32.3 | 36.95 | 32.11 | 0.94x | 0.9388888888888888 | Yes | severe_low_accept_rate |
| Real-Python-DictConfig | 57 | 34.3 | 40.9 | 47.18 | 40.51 | 1.19x | 1.1820834549168366 | No | None |
| Real-Python-DictConfig | 40 | 32.2 | 40.1 | 27.26 | 33.38 | 1.25x | 1.0363241229431854 | Yes | severe_low_accept_rate |
| Real-Python-DictConfig | 52 | 34.8 | 45.3 | 43.96 | 45.68 | 1.30x | 1.3145323741007193 | No | None |
| Real-Python-DictConfig | 77 | 32.8 | 42.0 | 45.63 | 42.62 | 1.28x | 1.2982028632348461 | No | None |
| Real-Python-DictConfig | 78 | 32.2 | 42.1 | 45.0 | 42.11 | 1.31x | 1.3077639751552794 | No | None |
| Real-Python-DictConfig | 7 | 32.4 | 45.1 | 41.96 | 34.34 | 1.40x | 1.0611866501854141 | Yes | severe_low_accept_rate |
| OpenMMLab-Config | 70 | 33.5 | 15.4 | 24.34 | 33.08 | 0.46x | 0.9880525686977301 | Yes | severe_low_accept_rate |
| OpenMMLab-Config | 64 | 31.5 | 18.7 | 41.38 | 27.12 | 0.59x | 0.8601332064700286 | Yes | severe_low_accept_rate |
| OpenMMLab-Config | 48 | 33.8 | 23.4 | 17.91 | 32.23 | 0.69x | 0.9538324948209529 | Yes | severe_low_accept_rate |
| OpenMMLab-Config | 0 | 34.2 | 27.9 | 31.55 | 32.7 | 0.81x | 0.9564200058496638 | Yes | severe_low_accept_rate |
| OpenMMLab-Config | 2 | 33.4 | 26.6 | 30.38 | 34.17 | 0.80x | 1.02213580616213 | Yes | severe_low_accept_rate |
| OpenMMLab-Config | 29 | 31.6 | 43.2 | 44.25 | 42.0 | 1.37x | 1.3278533038254823 | No | None |

## ProfitGuard Trigger Analysis

- **Triggered**: 13/24 samples

### Trigger Reasons

| Reason | Count |
|--------|-------|
| severe_low_accept_rate | 13 |

### Triggered Samples

| Benchmark | Sample | Generated Before Fallback | Remaining | Accept Rate at Trigger | Repair Count at Trigger | TASD-P TPS |
|-----------|--------|--------------------------|-----------|----------------------|----------------------|------------|
| Real-Python-Argparse | 73 | 67 | 69 | 0.17 | 1 | 12.6 |
| Real-Python-DictConfig | 59 | 49 | 83 | 0.14 | 0 | 21.0 |
| Real-Python-DictConfig | 2 | 49 | 83 | 0.14 | 0 | 21.1 |
| Real-Python-DictConfig | 1 | 49 | 84 | 0.18 | 0 | 28.8 |
| OpenMMLab-Config | 70 | 67 | 62 | 0.35 | 1 | 33.1 |
| OpenMMLab-Config | 64 | 49 | 93 | 0.16 | 1 | 27.1 |
| OpenMMLab-Config | 48 | 52 | 77 | 0.53 | 1 | 32.2 |
| OpenMMLab-Config | 0 | 68 | 66 | 0.36 | 1 | 32.7 |
| OpenMMLab-Config | 2 | 68 | 66 | 0.36 | 1 | 34.2 |
| Real-Python-DictConfig | 13 | 58 | 71 | 0.33 | 1 | 27.5 |
| Real-Python-DictConfig | 18 | 48 | 81 | 0.49 | 1 | 32.1 |
| Real-Python-DictConfig | 40 | 62 | 67 | 0.38 | 1 | 33.4 |
| Real-Python-DictConfig | 7 | 68 | 61 | 0.36 | 1 | 34.3 |

## Quality Impact

### TASD

- Mean SQ: 0.7990
- Mean Off-Structure: 0.0222
- Mean Truncation: 0.0971
- Mean Repair: 3.8

### TASD-P

- Mean SQ: 0.8037
- Mean Off-Structure: 0.0423
- Mean Truncation: 0.0889
- Mean Repair: 2.6

## Conclusion

- **Speedup < 1.0x**: TASD=17, TASD-P=15
- **Speedup < 1.2x**: TASD=18, TASD-P=20
- **Mean Speedup**: TASD=0.82x, TASD-P=0.91x
- **SQ Delta**: +0.0047
- **OffStr Delta**: +0.0201
- **Repair Delta**: -1.2
