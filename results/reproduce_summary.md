# TASD 7B+3B Reproduce Results Summary

## Speed Comparison

| Benchmark | AR TPS | TASD TPS | Speedup |
|---|---:|---:|---:|
| argparse | 40.61 | 24.83 | 0.61x |
| dict_config | 40.37 | 23.40 | 0.58x |
| openmmlab | 34.23 | 26.31 | 0.77x |

## Structural Quality (TASD)

| Benchmark | Score | Severe Rate | Off-Structure | Repetition | Truncation |
|---|---:|---:|---:|---:|---:|
| argparse | 0.9024 | 0.0 | 0.0859 | 0.0 | 0.0389 |
| dict_config | 0.909 | 0.0463 | 0.0218 | 0.0124 | 0.1124 |
| openmmlab | 0.9774 | 0.0303 | 0.0 | 0.0 | 0.0248 |

## TASD Internal Diagnostics

| Benchmark | Accept Rate | Draft Time % | Target FW | Draft FW | Guard | Trim | Repair |
|---|---:|---:|---:|---:|---:|---:|---:|
| argparse | 1.0 | 0.8936 | 21.4 | 19.0 | 3.3 | 3.3 | 2.4 |
| dict_config | 0.9844 | 0.8985 | 19.4 | 18.5 | 4.2 | 4.2 | 0.9 |
| openmmlab | 1.0 | 0.9021 | 16.7 | 16.4 | 0.4 | 0.4 | 0.3 |