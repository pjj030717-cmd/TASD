# TASD-F Hardcase Pilot Report

**Target**: Qwen2.5-14B-Instruct-AWQ  |  **Draft**: Qwen2.5-1.5B-Instruct
**Config**: max_new_tokens=128, draft_len=16, draft_blocks=2, top_k_accept=3

**TASD-F params**: enable_failure_aware_fallback=True, fallback_tokens=2, fallback_guarded=False

## Sample Composition

- Below-1.0x hardcases: 9
- Normal samples: 18
- Total: 27

## Below-1.0x Samples (Hardcase)

| Metric | TASD | TASD-F | Delta |
|--------|:----:|:------:|:-----:|
| mean_sp | 0.369 | 0.616 | +0.247 |
| median_sp | 0.306 | 0.589 | +0.283 |
| min_sp | 0.103 | 0.213 | +0.110 |
| worst10_sp | 0.103 | 0.213 | +0.110 |
| below_count | 9 | 7 | - |
| accept_avg | 0.1758 | 0.3191 | +0.143 |
| repair_avg | 16.0 | 2.0 | -14.000 |
| guard_avg | 32.3 | 13.4 | -18.900 |
| trim_avg | 32.2 | 13.2 | -19.000 |
| sq_avg | 0.5733 | 0.6105 | +0.037 |
| off_structure_avg | 0.0726 | 0.633 | +0.560 |
| rep_rate_avg | 0.1902 | 0.3855 | +0.195 |
| trunc_avg | 1.0 | 0.8889 | -0.111 |
| fallback_count | - | 122 | - |
| fallback_tokens | - | 244 | - |

## Normal Samples (Sanity)

| Metric | TASD | TASD-F | Delta |
|--------|:----:|:------:|:-----:|
| mean_sp | 2.038 | 2.03 | -0.008 |
| median_sp | 2.026 | 2.026 | +0.000 |
| min_sp | 1.987 | 1.878 | -0.109 |
| below_count | 0 | 0 | - |
| accept_avg | 0.9996 | 0.9996 | +0.000 |
| repair_avg | 0.0 | 0.0 | +0.000 |
| guard_avg | 0.6 | 0.6 | +0.000 |
| trim_avg | 0.0 | 0.0 | +0.000 |
| sq_avg | 0.5493 | 0.5493 | +0.000 |
| off_structure_avg | 0.028 | 0.028 | +0.000 |
| rep_rate_avg | 0.1027 | 0.1027 | +0.000 |
| trunc_avg | 0.8333 | 0.8333 | +0.000 |
| fallback_count | - | 0 | - |
| fallback_tokens | - | 0 | - |

## All Samples

| Metric | TASD | TASD-F | Delta |
|--------|:----:|:------:|:-----:|
| mean_sp | 1.482 | 1.558 | +0.076 |
| median_sp | 2.015 | 1.994 | -0.021 |
| min_sp | 0.103 | 0.213 | +0.110 |
| worst10_sp | 0.154 | 0.27 | +0.116 |
| below_count | 9 | 7 | - |
| accept_avg | 0.725 | 0.7727 | +0.048 |
| repair_avg | 5.3 | 0.7 | -4.600 |
| guard_avg | 11.1 | 4.9 | -6.200 |
| trim_avg | 10.7 | 4.4 | -6.300 |
| sq_avg | 0.5573 | 0.5697 | +0.012 |
| off_structure_avg | 0.0429 | 0.2297 | +0.187 |
| rep_rate_avg | 0.1319 | 0.197 | +0.065 |
| trunc_avg | 0.8889 | 0.8519 | -0.037 |
| fallback_count | - | 122 | - |
| fallback_tokens | - | 244 | - |

## Per-Sample Details

| Sample | AR TPS | TASD sp | TASD-F sp | TASD acc | TASD-F acc | TASD-F fb | TASD SQ | TASD-F SQ |
|--------|:------:|:-------:|:---------:|:--------:|:----------:|:---------:|:-------:|:---------:|
| argparse_real_023 **BELOW** | 29.8 | 0.373x | 1.084x | 0.1621 | 0.5571 | 5 | 0.5164 | 0.5919 |
| argparse_real_030 **BELOW** | 33.9 | 0.205x | 0.589x | 0.0891 | 0.3159 | 10 | 0.7000 | 0.8014 |
| argparse_real_031 **BELOW** | 33.2 | 0.103x | 0.213x | 0.0355 | 0.0691 | 30 | 0.7000 | 0.6218 |
| argparse_real_034 **BELOW** | 32.4 | 0.266x | 0.326x | 0.1258 | 0.1614 | 19 | 0.3947 | 0.4600 |
| argparse_real_039 **BELOW** | 29.8 | 0.206x | 0.339x | 0.0770 | 0.1321 | 22 | 0.8000 | 0.7000 |
| argparse_real_062 **BELOW** | 33.2 | 0.306x | 0.342x | 0.1376 | 0.1568 | 19 | 0.7538 | 0.6040 |
| argparse_real_070 **BELOW** | 33.2 | 0.354x | 0.717x | 0.1660 | 0.4179 | 7 | 0.1333 | 0.5688 |
| argparse_real_015 | 34.0 | 1.989x | 1.959x | 1.0000 | 1.0000 | 0 | 0.6797 | 0.6797 |
| argparse_real_004 | 31.8 | 2.019x | 2.131x | 1.0000 | 1.0000 | 0 | 0.5498 | 0.5498 |
| argparse_real_041 | 33.7 | 2.023x | 1.963x | 1.0000 | 1.0000 | 0 | 0.5000 | 0.5000 |
| dict_config_real_019 **BELOW** | 33.8 | 0.923x | 1.336x | 0.5041 | 0.7184 | 1 | 0.5811 | 0.5479 |
| dict_config_real_057 **BELOW** | 32.5 | 0.584x | 0.594x | 0.2847 | 0.3428 | 9 | 0.5800 | 0.5987 |
| dict_config_real_035 | 31.6 | 2.015x | 1.993x | 1.0000 | 1.0000 | 0 | 0.8679 | 0.8679 |
| dict_config_real_032 | 31.8 | 2.074x | 2.068x | 1.0000 | 1.0000 | 0 | 0.8800 | 0.8800 |
| dict_config_real_021 | 32.2 | 2.106x | 2.130x | 1.0000 | 1.0000 | 0 | 0.5664 | 0.5664 |
| openmmlab_config_real_014 | 33.8 | 2.060x | 1.878x | 1.0000 | 1.0000 | 0 | 0.6351 | 0.6351 |
| openmmlab_config_real_071 | 32.8 | 2.102x | 2.114x | 1.0000 | 1.0000 | 0 | 0.4952 | 0.4952 |
| openmmlab_config_real_012 | 33.9 | 2.003x | 1.999x | 0.9922 | 0.9922 | 0 | 0.5768 | 0.5768 |
| pipeline_stage_config_076 | 34.4 | 2.027x | 2.026x | 1.0000 | 1.0000 | 0 | 0.4800 | 0.4800 |
| pipeline_stage_config_055 | 34.2 | 2.049x | 1.994x | 1.0000 | 1.0000 | 0 | 0.5800 | 0.5800 |
| pipeline_stage_config_005 | 32.9 | 2.107x | 2.103x | 1.0000 | 1.0000 | 0 | 0.5800 | 0.5800 |
| complex_nested_config_004 | 33.9 | 2.026x | 2.006x | 1.0000 | 1.0000 | 0 | 0.1682 | 0.1682 |
| complex_nested_config_012 | 33.0 | 2.022x | 2.017x | 1.0000 | 1.0000 | 0 | 0.5261 | 0.5261 |
| complex_nested_config_028 | 33.5 | 2.019x | 2.074x | 1.0000 | 1.0000 | 0 | 0.4800 | 0.4800 |
| rich_cli_option_groups_03 | 33.6 | 2.040x | 2.035x | 1.0000 | 1.0000 | 0 | 0.4743 | 0.4743 |
| rich_cli_option_groups_06 | 33.7 | 2.013x | 2.036x | 1.0000 | 1.0000 | 0 | 0.3885 | 0.3885 |
| rich_cli_option_groups_07 | 34.1 | 1.987x | 2.006x | 1.0000 | 1.0000 | 0 | 0.4594 | 0.4594 |

## Pass/Fail Criteria

| Criterion | Pass | Note |
|-----------|:----:|------|
| below-1.0x reduced | PASS | 9 -> 7 |
| mean sp not degraded >5% | PASS | 1.482 -> 1.558 |
| SQ not degraded >0.02 | PASS | 0.5573 -> 0.5697 |

## Data

- `results/qwen_tasd_f_hardcase_pilot.json`
