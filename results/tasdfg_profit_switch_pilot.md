# TASD-FG-P Profit-Aware AR Switch Pilot Report

**Samples**: 21 (3 below-AR + 18 normal)

**Window**: 48 tokens | **Triggers**: est_speedup<1.05, fb>=2, guard_trim>=3, rolling_accept<0.4, zero_accept>=2

## Below-AR Samples (Target: >=1.0x)

| # | Name | AR TPS | FG sp | FGP sp | Δ | Switched | Reason | At token | FG SQ-R/S | FGP SQ-R/S | FG Off-Str | FGP Off-Str |
|---|------|:------:|:-----:|:------:|:--:|:--------:|:------:|:--------:|:---------:|:----------:|:----------:|:----------:|
| 1 | argparse_real_062 | 34.0 | 0.595x | 0.752x | **+0.157** | YES | est_speedup=0.104_below_1.05 | 4 | 0.647/0.705 | 0.688/0.591 | 0.1000 | 0.3529 |
| 2 | dict_config_real_014 | 31.6 | 0.965x | 0.961x | -0.004 | YES | est_speedup=0.745_below_1.05 | 25 | 0.576/0.750 | 0.580/0.711 | 0.0000 | 0.0870 |
| 3 | dict_config_real_057 | 31.8 | 0.924x | 0.929x | **+0.005** | no | None | 0 | 0.578/0.750 | 0.578/0.750 | 0.0000 | 0.0000 |

**Below avg: FG=0.828x → FGP=0.881x | Switches: 2/3 | Still below-1.0: 3/3**

## Normal Samples (Target: minimal false triggers)

| # | Bench | Name | AR TPS | FG sp | FGP sp | Δ% | Switched | Reason |
|---|-------|------|:------:|:-----:|:------:|:--:|:--------:|:------:|
| 1 | argparse | argparse_real_001 | 28.6 | 2.391x | 2.374x | -0.7% | no | None |
| 2 | argparse | argparse_real_002 | 32.7 | 1.991x | 2.026x | +1.8% | no | None |
| 3 | argparse | argparse_real_003 | 33.6 | 2.034x | 2.075x | +2.0% | no | None |
| 4 | dict_config | dict_config_real_001 | 33.8 | 2.054x | 2.085x | +1.5% | no | None |
| 5 | dict_config | dict_config_real_002 | 32.7 | 2.138x | 2.105x | -1.5% | no | None |
| 6 | dict_config | dict_config_real_003 | 33.7 | 2.038x | 1.948x | -4.4% | no | None |
| 7 | openmmlab_config | openmmlab_config_real_001 | 31.8 | 2.039x | 2.116x | +3.8% | no | None |
| 8 | openmmlab_config | openmmlab_config_real_002 | 32.1 | 2.200x | 2.147x | -2.4% | no | None |
| 9 | openmmlab_config | openmmlab_config_real_003 | 32.2 | 1.985x | 1.962x | -1.2% | no | None |
| 10 | pipeline_stage_config | pipeline_stage_config_001 | 31.4 | 2.247x | 2.202x | -2.0% | no | None |
| 11 | pipeline_stage_config | pipeline_stage_config_002 | 33.9 | 1.934x | 1.946x | +0.6% | no | None |
| 12 | pipeline_stage_config | pipeline_stage_config_003 | 34.1 | 1.972x | 2.055x | +4.2% | no | None |
| 13 | complex_nested_config | complex_nested_config_001 | 32.9 | 2.135x | 2.118x | -0.8% | no | None |
| 14 | complex_nested_config | complex_nested_config_002 | 33.4 | 2.066x | 2.087x | +1.0% | no | None |
| 15 | complex_nested_config | complex_nested_config_003 | 34.2 | 2.019x | 2.003x | -0.8% | no | None |
| 16 | rich_cli_option_groups | rich_cli_option_groups_001 | 32.8 | 1.985x | 1.982x | -0.2% | no | None |
| 17 | rich_cli_option_groups | rich_cli_option_groups_002 | 31.6 | 2.212x | 2.183x | -1.3% | no | None |
| 18 | rich_cli_option_groups | rich_cli_option_groups_003 | 32.3 | 2.162x | 1.998x | -7.6% | no | None |

**False triggers on normal samples: 0/18**

## Summary

| Group | FG Mean sp | FGP Mean sp | Δ | Switches | Below-1.0 |
|-------|:----------:|:----------:|:--:|:--------:|:---------:|
| Below (3) | 0.828x | 0.881x | +0.053 | 2/3 | 3/3 |
| Normal (18) | 2.089x | 2.078x | -0.5% | 0/18 | - |
| **Total (21)** | **1.909x** | **1.907x** | **-0.002** | **2/21** | 3/21 |

### Per-benchmark normal mean speedup

| Benchmark | FG | FGP | Δ |
|-----------|:--:|:--:|:--:|
| argparse | 2.139x | 2.158x | +0.020 |
| dict_config | 2.077x | 2.046x | -0.031 |
| openmmlab_config | 2.075x | 2.075x | +0.000 |
| pipeline_stage_config | 2.051x | 2.068x | +0.017 |
| complex_nested_config | 2.073x | 2.069x | -0.004 |
| rich_cli_option_groups | 2.120x | 2.054x | -0.065 |

## Switch Detail Log

### argparse/argparse_real_062 (below)

- **Reason**: est_speedup=0.104_below_1.05
- **At token**: 4
- **Trigger values**: {'rolling_accept': 0.0156, 'fallback_count': 1, 'guard_trim': 3, 'consecutive_zero': 1}
- **FG speedup**: 0.595x → **FGP speedup**: 0.752x

### dict_config/dict_config_real_014 (below)

- **Reason**: est_speedup=0.745_below_1.05
- **At token**: 25
- **Trigger values**: {'rolling_accept': 0.375, 'fallback_count': 0, 'guard_trim': 3, 'consecutive_zero': 1}
- **FG speedup**: 0.965x → **FGP speedup**: 0.961x

## Judgment

- FAIL | Below samples >= 1.0x: ['0.752x', '0.961x', '0.929x']
- PASS | False trigger rate: 0/18 (0%)
- PASS | Normal mean speedup drop: -0.5%
- PASS | SQ-S: FG=0.7639 → FGP=0.7566
- FAIL | Off-Str: FG=0.0429 → FGP=0.0590
