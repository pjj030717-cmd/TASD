# TASD-F-G Focused Pilot

**TASD-F**: unguarded fallback  |  **TASD-F-G**: guarded fallback

## Per-Sample

| Sample | AR TPS | TASD sp | TASD-F sp | TASD-F-G sp | TASD-F fb | TASD-F-G fb | TASD-F SQ | TASD-F-G SQ |
|--------|:------:|:-------:|:---------:|:-----------:|:---------:|:-----------:|:---------:|:-----------:|
| argparse_real_023 [below] | 29.3 | **0.312x** | 1.044x | 1.752x | 5 | 1 | 0.5919 | 0.6874 |
| argparse_real_030 [below] | 33.9 | **0.209x** | **0.584x** | 1.512x | 10 | 1 | 0.8014 | 0.6500 |
| argparse_real_031 [below] | 33.3 | **0.104x** | **0.215x** | 1.236x | 30 | 1 | 0.6218 | 0.4268 |
| argparse_real_034 [below] | 34.0 | **0.269x** | **0.326x** | 1.474x | 19 | 1 | 0.4600 | 0.5143 |
| argparse_real_039 [below] | 32.4 | **0.177x** | **0.304x** | 1.560x | 22 | 1 | 0.7000 | 0.6704 |
| argparse_real_062 [below] | 32.9 | **0.296x** | **0.347x** | 1.537x | 19 | 1 | 0.6040 | 0.9954 |
| argparse_real_070 [below] | 32.6 | **0.343x** | **0.710x** | 1.573x | 7 | 1 | 0.5688 | 0.6036 |
| dict_config_real_019 [below] | 33.1 | **0.975x** | 1.371x | 1.391x | 1 | 1 | 0.5479 | 0.5479 |
| dict_config_real_057 [below] | 32.6 | **0.600x** | **0.643x** | **0.899x** | 9 | 5 | 0.5987 | 0.5784 |
| argparse_real_074 [false_trigger] | 33.7 | 1.335x | **0.961x** | 1.149x | 5 | 3 | 0.9329 | 0.7000 |
| argparse_real_015 [normal] | 33.9 | 1.957x | 2.039x | 2.041x | 0 | 0 | 0.6797 | 0.6797 |
| argparse_real_041 [normal] | 34.2 | 2.022x | 2.001x | 1.988x | 0 | 0 | 0.5000 | 0.5000 |
| dict_config_real_035 [normal] | 32.1 | 2.140x | 2.131x | 2.127x | 0 | 0 | 0.8679 | 0.8679 |

## Below-1.0x (9) (n=9)

| Metric | TASD | TASD-F | TASD-F-G |
|--------|:----:|:------:|:--------:|
| mean sp | 0.365 | 0.616 | 1.437 |
| below-1.0x | 9 | 7 | 1 |
| total fb_count | 122 | 122 | 13 |
| mean guard_trig | 13.4444 | 13.4444 | 4.6667 |
| mean trim | 13.2222 | 13.2222 | 3.0000 |
| mean repair | 2.0000 | 2.0000 | 0.3333 |
| mean SQ | 0.6105 | 0.6105 | 0.6305 |
| mean off_structure | 0.6330 | 0.6330 | 0.0000 |
| mean rep_rate | 0.3855 | 0.3855 | 0.0417 |
| mean truncation | 0.8889 | 0.8889 | 0.7778 |

## False Trigger (argparse_074) (n=1)

| Metric | TASD | TASD-F | TASD-F-G |
|--------|:----:|:------:|:--------:|
| mean sp | 1.335 | 0.961 | 1.149 |
| below-1.0x | 0 | 1 | 0 |
| total fb_count | 5 | 5 | 3 |
| mean guard_trig | 3.0000 | 3.0000 | 4.0000 |
| mean trim | 3.0000 | 3.0000 | 3.0000 |
| mean repair | 1.0000 | 1.0000 | 1.0000 |
| mean SQ | 0.9329 | 0.9329 | 0.7000 |
| mean off_structure | 0.0833 | 0.0833 | 0.0000 |
| mean rep_rate | 0.0000 | 0.0000 | 0.0000 |
| mean truncation | 1.0000 | 1.0000 | 1.0000 |

## Normal (3) (n=3)

| Metric | TASD | TASD-F | TASD-F-G |
|--------|:----:|:------:|:--------:|
| mean sp | 2.040 | 2.057 | 2.052 |
| below-1.0x | 0 | 0 | 0 |
| total fb_count | 0 | 0 | 0 |
| mean guard_trig | 0.0000 | 0.0000 | 0.0000 |
| mean trim | 0.0000 | 0.0000 | 0.0000 |
| mean repair | 0.0000 | 0.0000 | 0.0000 |
| mean SQ | 0.6825 | 0.6825 | 0.6825 |
| mean off_structure | 0.0000 | 0.0000 | 0.0000 |
| mean rep_rate | 0.0000 | 0.0000 | 0.0000 |
| mean truncation | 1.0000 | 1.0000 | 1.0000 |

## All (13) (n=13)

| Metric | TASD | TASD-F | TASD-F-G |
|--------|:----:|:------:|:--------:|
| mean sp | 0.826 | 0.975 | 1.557 |
| below-1.0x | 9 | 8 | 1 |
| total fb_count | 127 | 127 | 16 |
| mean guard_trig | 9.5385 | 9.5385 | 3.5385 |
| mean trim | 9.3846 | 9.3846 | 2.3077 |
| mean repair | 1.4615 | 1.4615 | 0.3077 |
| mean SQ | 0.6519 | 0.6519 | 0.6478 |
| mean off_structure | 0.4446 | 0.4446 | 0.0000 |
| mean rep_rate | 0.2669 | 0.2669 | 0.0288 |
| mean truncation | 0.9231 | 0.9231 | 0.8462 |

## Pass/Fail

| Criterion | Result | Note |
|-----------|:------:|------|
| below-1.0x reduced in hardcases | PASS | 9 → 1 |
| dict_019 stays >1.0x | PASS | TASD-F-G=1.391x |
| argparse_074 NOT <1.0x | PASS | TASD-F-G=1.149x |
| off_structure <= TASD-F×1.05 | PASS | 0.4446 → 0.0000 |
| mean sp >= TASD-F×0.97 | PASS | 0.975 → 1.557 |

**Overall**: ALL PASS

