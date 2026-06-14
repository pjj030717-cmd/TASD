# TASD Ablation Study

**Benchmarks**: Real-Python-DictConfig, OpenMMLab-Config, Pipeline-Stage-Config (40 samples each)
**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct (default), Qwen2.5-3B-Instruct (3B variant)
**Settings**: max_new_tokens=128, temperature=0.0

## dict_config

| Variant | TPS | Speedup | SQ | OffStr | Repair | Accept |
|---------|-----|---------|----|--------|--------|--------|
| AR (baseline) | 32.1 | 1.00x | - | - | - | - |
| TASD (full) | 57.2 | 1.84x | 0.6475 | 0.0000 | 0.33 | 0.8923 |
| TASD w/o relaxed | 51.0 | 1.64x | 0.6444 | 0.0500 | 0.40 | 0.8047 |
| TASD w/o guard | 65.3 | 2.10x | 0.6515 | 0.1000 | 0.00 | 0.9998 |
| TASD single-block | 52.9 | 1.70x | 0.6468 | 0.0000 | 0.45 | 0.8925 |
| TASD draft_len=8 | 52.0 | 1.67x | 0.6468 | 0.0000 | 0.45 | 0.8925 |
| TASD (3B draft) | 48.7 | 1.57x | 0.6204 | 0.0000 | 0.35 | 0.9399 |

## openmmlab_config

| Variant | TPS | Speedup | SQ | OffStr | Repair | Accept |
|---------|-----|---------|----|--------|--------|--------|
| AR (baseline) | 32.5 | 1.00x | - | - | - | - |
| TASD (full) | 61.7 | 1.94x | 0.9161 | 0.0750 | 0.28 | 0.9567 |
| TASD w/o relaxed | 52.7 | 1.66x | 0.9185 | 0.0500 | 0.42 | 0.8192 |
| TASD w/o guard | 65.4 | 2.05x | 0.9161 | 0.1000 | 0.00 | 0.9996 |
| TASD single-block | 57.1 | 1.80x | 0.9161 | 0.0250 | 0.40 | 0.9576 |
| TASD draft_len=8 | 57.6 | 1.81x | 0.9161 | 0.0250 | 0.40 | 0.9576 |
| TASD (3B draft) | 50.8 | 1.60x | 0.9981 | 0.0500 | 0.10 | 0.9858 |

## pipeline_stage_config

| Variant | TPS | Speedup | SQ | OffStr | Repair | Accept |
|---------|-----|---------|----|--------|--------|--------|
| AR (baseline) | 31.7 | 1.00x | - | - | - | - |
| TASD (full) | 63.9 | 2.05x | 0.8975 | 0.0750 | 0.03 | 0.9935 |
| TASD w/o relaxed | 55.8 | 1.79x | 0.8960 | 0.0500 | 0.07 | 0.8808 |
| TASD w/o guard | 63.8 | 2.05x | 0.8975 | 0.0750 | 0.03 | 0.9935 |
| TASD single-block | 58.5 | 1.88x | 0.8975 | 0.0750 | 0.03 | 0.9954 |
| TASD draft_len=8 | 59.8 | 1.92x | 0.8975 | 0.0750 | 0.03 | 0.9954 |
| TASD (3B draft) | 52.8 | 1.69x | 0.8879 | 0.0000 | 0.00 | 1.0000 |

## Cross-Benchmark Summary (Speedup)

| Variant | DictConfig | OpenMMLab | Pipeline | Mean |
|---------|-----------|-----------|----------|------|
| tasd_full | 1.84x | 1.94x | 2.05x | 1.94x |
| tasd_no_relaxed | 1.64x | 1.66x | 1.79x | 1.70x |
| tasd_no_guard | 2.10x | 2.05x | 2.05x | 2.07x |
| tasd_single_block | 1.70x | 1.80x | 1.88x | 1.79x |
| tasd_draft_len_8 | 1.67x | 1.81x | 1.92x | 1.80x |
| tasd_3b | 1.57x | 1.60x | 1.69x | 1.62x |
