# Final Total Experiment Table (1.5B Draft, d16_b2_k3)

**Model**: Qwen2.5-14B-Instruct-AWQ (target) + Qwen2.5-1.5B-Instruct (draft)
**Settings**: temperature=0.0, max_new_tokens=128, KV cache enabled
**Sample count**: n=80
**TASD config**: draft_len=16, draft_blocks=2, top_k_accept=3

1.5B draft is a strong candidate default and shows promising speed gains.

## Main Table

| Benchmark | AR TPS | TASD TPS | Speedup | Accept Mean | Accept Med | Accept P10 | Accept P90 | High/80 | Low/80 | SQ | OffStr | Trunc |
|-----------|--------|----------|---------|-------------|------------|------------|------------|---------|--------|----|--------|-------|
| Real-Python-Argparse | 32.98 | 61.89 | 1.88x | 0.93 | 1.00 | 0.78 | 1.00 | 71 | 7 | 0.9010 | 0.0014 | 0.0269 |
| Real-Python-DictConfig | 32.67 | 60.00 | 1.84x | 0.91 | 1.00 | 0.68 | 1.00 | 61 | 10 | 0.8316 | 0.0000 | 0.0590 |
| OpenMMLab-Config | 32.91 | 63.96 | 1.94x | 0.95 | 1.00 | 0.99 | 1.00 | 73 | 6 | 0.8876 | 0.0059 | 0.1134 |
| Rich-CLI-Option-Groups | 33.14 | 66.06 | 1.99x | 1.00 | 1.00 | 1.00 | 1.00 | 79 | 0 | 0.9119 | 0.1141 | 0.0395 |
| Complex-Nested-Config | 32.71 | 66.35 | 2.03x | 1.00 | 1.00 | 1.00 | 1.00 | 80 | 0 | 0.8227 | 0.0143 | 0.0727 |
| Pipeline-Stage-Config | 32.24 | 66.66 | 2.07x | 1.00 | 1.00 | 1.00 | 1.00 | 79 | 0 | 0.9405 | 0.0245 | 0.1489 |
