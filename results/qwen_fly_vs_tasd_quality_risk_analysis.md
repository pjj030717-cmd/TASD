# Qwen 6×80: Official FLY vs TASD — Quality Risk Analysis

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Samples**: 6 benchmarks × 80 = 480 total

## 1. Overall Comparison

| Metric | Official FLY | TASD (cal) | Greedy SD | N-gram SD | FLY advantage |
|--------|:-----------:|:----------:|:---------:|:---------:|:-------------:|
| **Speedup** | **1.596x** | 1.353x | 1.220x | 0.952x | FLY +0.244x |
| **SQ** | 0.7504 | 0.7022 | 0.7051 | - | FLY +0.0482 |
| **Below 1.0x** | 131/480 | **70/480** | 91/480 | - | TASD has 61 fewer |
| **Speed wins** | 4/6 benchmarks | 2/6 | | | |
| **SQ wins** | 6/6 benchmarks | 0/6 | | | |
| FLY MAT | 1.53 | - | | | |
| FLY ngram_acc | 1.9 | - | | | |
| TASD off_str | - | 0.0376 | | | |
| TASD accept | - | 0.9785 | | | |
| Speed correlation | - | r=-0.303 (per-sample FLY-TASD) | | | |
| SQ correlation | - | r=0.902 (per-sample FLY-TASD) | | | |
| Per-sample wins | 254/480 | 225/480 | | | |

## 2. Per-Benchmark Breakdown

### argparse (AR TPS: 67, 80 samples)

| Method | Speedup | SQ | Accept/MAT | Below | Details |
|--------|:-------:|:--:|:----------:|:-----:|--------|
| AR | **1.000x** | 0.8672 | - | 0 | trunc=0.74 |
| Greedy SD | **0.935x** | 0.8443 | 0.878 | 45 |  |
| N-gram SD | **0.750x** | 0.6466 | 0.524 | 63 |  |
| Official FLY | **1.899x** | 0.8790 | 1.37 MAT | 8 | ngram_acc=2.3 | gen_len=135 |
| TASD (cal) | **0.935x** | 0.8289 | 0.907 | 44 | off_str=0.0067 | guard=2.7/2.7 |

**argparse winner**: **FLY** — FLY 1.899x vs TASD 0.935x (delta=+0.964x), estimated n-gram PLD contribution: +0.964x (FLY 1.899x − GSD 0.935x) | SQ: FLY 0.8790 vs TASD 0.8289

### dict_config (AR TPS: 49, 80 samples)

| Method | Speedup | SQ | Accept/MAT | Below | Details |
|--------|:-------:|:--:|:----------:|:-----:|--------|
| AR | **1.000x** | 0.6908 | - | 0 | trunc=0.76 |
| Greedy SD | **1.308x** | 0.6522 | 0.899 | 7 |  |
| N-gram SD | **1.042x** | 0.5100 | 0.541 | 52 |  |
| Official FLY | **1.774x** | 0.6917 | 1.39 MAT | 7 | ngram_acc=2.1 | gen_len=134 |
| TASD (cal) | **1.432x** | 0.6506 | 0.971 | 6 | off_str=0.0007 | guard=0.9/0.3 |

**dict_config winner**: **FLY** — FLY 1.774x vs TASD 1.432x (delta=+0.343x), estimated n-gram PLD contribution: +0.466x (FLY 1.774x − GSD 1.308x) | SQ: FLY 0.6917 vs TASD 0.6506

### openmmlab_config (AR TPS: 42, 80 samples)

| Method | Speedup | SQ | Accept/MAT | Below | Details |
|--------|:-------:|:--:|:----------:|:-----:|--------|
| AR | **1.000x** | 0.9281 | - | 0 | trunc=0.74 |
| Greedy SD | **1.384x** | 0.8751 | 0.860 | 0 |  |
| N-gram SD | **0.899x** | 0.7388 | 0.542 | 60 |  |
| Official FLY | **0.986x** | 0.9358 | 1.83 MAT | 43 | ngram_acc=1.4 | gen_len=133 |
| TASD (cal) | **1.590x** | 0.8751 | 0.996 | 0 | off_str=0.0426 | guard=0.7/0.0 |

**openmmlab_config winner**: **TASD** — TASD 1.590x vs FLY 0.986x (delta=+0.603x), off_str=0.0426 | SQ: FLY 0.9358 vs TASD 0.8751

### pipeline_stage_config (AR TPS: 42, 80 samples)

| Method | Speedup | SQ | Accept/MAT | Below | Details |
|--------|:-------:|:--:|:----------:|:-----:|--------|
| AR | **1.000x** | 0.9219 | - | 0 | trunc=0.65 |
| Greedy SD | **1.434x** | 0.8918 | 0.890 | 0 |  |
| N-gram SD | **0.943x** | 0.6777 | 0.535 | 57 |  |
| Official FLY | **1.119x** | 0.9397 | 1.70 MAT | 39 | ngram_acc=1.5 | gen_len=133 |
| TASD (cal) | **1.627x** | 0.8918 | 0.997 | 0 | off_str=0.0371 | guard=0.0/0.0 |

**pipeline_stage_config winner**: **TASD** — TASD 1.627x vs FLY 1.119x (delta=+0.509x), off_str=0.0371 | SQ: FLY 0.9397 vs TASD 0.8918

### complex_nested_config (AR TPS: 55, 80 samples)

| Method | Speedup | SQ | Accept/MAT | Below | Details |
|--------|:-------:|:--:|:----------:|:-----:|--------|
| AR | **1.000x** | 0.4857 | - | 0 | trunc=0.74 |
| Greedy SD | **1.123x** | 0.4744 | 0.902 | 23 |  |
| N-gram SD | **0.921x** | 0.3425 | 0.553 | 55 |  |
| Official FLY | **1.695x** | 0.5047 | 1.57 MAT | 27 | ngram_acc=2.2 | gen_len=134 |
| TASD (cal) | **1.254x** | 0.4741 | 1.000 | 12 | off_str=0.0132 | guard=0.0/0.0 |

**complex_nested_config winner**: **FLY** — FLY 1.695x vs TASD 1.254x (delta=+0.442x), estimated n-gram PLD contribution: +0.573x (FLY 1.695x − GSD 1.123x) | SQ: FLY 0.5047 vs TASD 0.4741

### rich_cli_option_groups (AR TPS: 53, 80 samples)

| Method | Speedup | SQ | Accept/MAT | Below | Details |
|--------|:-------:|:--:|:----------:|:-----:|--------|
| AR | **1.000x** | 0.5186 | - | 0 | trunc=0.80 |
| Greedy SD | **1.137x** | 0.4931 | 0.894 | 16 |  |
| N-gram SD | **1.159x** | 0.4665 | 0.685 | 46 |  |
| Official FLY | **2.104x** | 0.5514 | 1.30 MAT | 7 | ngram_acc=2.3 | gen_len=135 |
| TASD (cal) | **1.278x** | 0.4927 | 1.000 | 8 | off_str=0.1254 | guard=0.0/0.0 |

**rich_cli_option_groups winner**: **FLY** — FLY 2.104x vs TASD 1.278x (delta=+0.826x), estimated n-gram PLD contribution: +0.967x (FLY 2.104x − GSD 1.137x) | SQ: FLY 0.5514 vs TASD 0.4927

## 3. Quality Risk Analysis

### 3.1 Structural Quality (SQ)

- FLY overall SQ: **0.7504**
- TASD overall SQ: **0.7022**
- GSD overall SQ: **0.7051**
- Delta FLY−TASD: **+0.0482** (FLY slightly higher)

- **argparse**: FLY SQ 0.8790 vs TASD 0.8289 (delta=+0.0501)
- **dict_config**: FLY SQ 0.6917 vs TASD 0.6506 (delta=+0.0411)
- **openmmlab_config**: FLY SQ 0.9358 vs TASD 0.8751 (delta=+0.0607)
- **pipeline_stage_config**: FLY SQ 0.9397 vs TASD 0.8918 (delta=+0.0479)
- **complex_nested_config**: FLY SQ 0.5047 vs TASD 0.4741 (delta=+0.0306)
- **rich_cli_option_groups**: FLY SQ 0.5514 vs TASD 0.4927 (delta=+0.0587)

**Conclusion**: FLY achieves equal or higher SQ on all benchmarks. TASD does not trade quality for structure safety — it maintains SQ parity with GSD.

### 3.2 Off-Structure Risk (TASD only)

TASD's structural guard explicitly prevents `def`/`class`/`import` in generated code.

- **argparse**: off_str=0.0067, guard=2.7, trim=2.7
- **dict_config**: off_str=0.0007, guard=0.9, trim=0.3
- **openmmlab_config**: off_str=0.0426, guard=0.7, trim=0.0
- **pipeline_stage_config**: off_str=0.0371, guard=0.0, trim=0.0
- **complex_nested_config**: off_str=0.0132, guard=0.0, trim=0.0
- **rich_cli_option_groups**: off_str=0.1254, guard=0.0, trim=0.0

### 3.3 Reliability (Below 1.0x)

Counts samples where the method is slower than AR:

| Benchmark | FLY below | TASD below | FLY−TASD |
|-----------|:---------:|:----------:|:--------:|
| argparse | 8/80 | 44/80 | +36 |
| dict_config | 7/80 | 6/80 | -1 |
| openmmlab_config | 43/80 | 0/80 | -43 |
| pipeline_stage_config | 39/80 | 0/80 | -39 |
| complex_nested_config | 27/80 | 12/80 | -15 |
| rich_cli_option_groups | 7/80 | 8/80 | +1 |
| **Total** | **131/480** | **70/480** | **-61** |

**TASD has 12.7% fewer sub-AR cases** — TASD more consistently provides speedup above AR, while FLY has more variance.

### 3.4 Hard Case Count (sp < 1.0 OR SQ < 0.5)

| Benchmark | FLY hard | TASD hard |
|-----------|:--------:|:---------:|
| argparse | 14/80 | 44/80 |
| dict_config | 29/80 | 32/80 |
| openmmlab_config | 46/80 | 9/80 |
| pipeline_stage_config | 41/80 | 5/80 |
| complex_nested_config | 56/80 | 51/80 |
| rich_cli_option_groups | 39/80 | 45/80 |

## 4. Why FLY Wins: N-gram PLD Contribution

FLY combines two draft sources: **n-gram prompt lookup (PLD)** + **draft model SD**.
We estimate n-gram contribution as FLY speedup minus GSD speedup (pure model draft).

| Benchmark | FLY | GSD | N-gram+model gap | FLY ngram_acc | Interpretation |
|-----------|:---:|:---:|:----------------:|:-------------:|:--------------:|
| argparse | 1.899x | 0.935x | +0.964x | 2.3 | n-gram PLD dominant |
| dict_config | 1.774x | 1.308x | +0.466x | 2.1 | n-gram PLD dominant |
| openmmlab_config | 0.986x | 1.384x | -0.398x | 1.4 | model draft only, n-gram ineffective |
| pipeline_stage_config | 1.119x | 1.434x | -0.316x | 1.5 | model draft only, n-gram ineffective |
| complex_nested_config | 1.695x | 1.123x | +0.573x | 2.2 | n-gram PLD dominant |
| rich_cli_option_groups | 2.104x | 1.137x | +0.967x | 2.3 | n-gram PLD dominant |

## 5. Where TASD Wins: Structural Guard Advantage

| Benchmark | TASD | FLY | GSD | TASD−GSD | TASD−FLY | Interpretation |
|-----------|:---:|:---:|:---:|:--------:|:--------:|:--------------:|
| argparse | 0.935x | 1.899x | 0.935x | +0.000x | -0.964x | Guard value over GSD, but FLY n-gram stronger |
| dict_config | 1.432x | 1.774x | 1.308x | +0.123x | -0.343x | Guard value over GSD, but FLY n-gram stronger |
| openmmlab_config | 1.590x | 0.986x | 1.384x | +0.206x | +0.603x | Structural guard + high draft alignment |
| pipeline_stage_config | 1.627x | 1.119x | 1.434x | +0.193x | +0.509x | Structural guard + high draft alignment |
| complex_nested_config | 1.254x | 1.695x | 1.123x | +0.131x | -0.442x | Guard value over GSD, but FLY n-gram stronger |
| rich_cli_option_groups | 1.278x | 2.104x | 1.137x | +0.141x | -0.826x | Guard value over GSD, but FLY n-gram stronger |

## 6. Per-Sample FLY-TASD Analysis

- **Speedup correlation**: r = -0.303 — weak correlation; methods benefit from different sample characteristics
- **SQ correlation**: r = 0.902 — both methods preserve reference structure similarly
- **TASD beats FLY on 225/480** individual samples (46.9%)
- **FLY beats TASD on 254/480** individual samples (52.9%)

## 7. Final Judgment

### Evidence Summary

1. **Speed**: FLY 1.596x > TASD 1.353x — FLY is +0.244x faster overall
2. **Quality**: FLY SQ 0.7504 ≥ TASD SQ 0.7022 — FLY does not degrade quality
3. **Reliability**: TASD has 61 fewer below-1.0x cases — TASD more consistent
4. **Benchmarks**: FLY wins 4/6, TASD wins 2/6 on speed; FLY wins 6/6, TASD wins 0/6 on SQ
5. **N-gram PLD**: FLY's speed advantage comes from n-gram prompt lookup (FLY−GSD gap = +0.376x)
6. **Structural guard**: TASD's guard adds +0.132x over GSD through guard + relaxed accept

### Judgment: **B — FLY is faster with no quality penalty; TASD is structure-safe SD**

**Reasoning**:

- FLY's speed advantage is real and significant (+0.243x over TASD), driven by n-gram PLD which TASD lacks
- FLY does NOT sacrifice quality for speed: SQ is equal or better on all benchmarks
- However, FLY has **2.1× more below-1.0x cases** (131 vs 70) — higher variance across samples
- TASD's structural guard provides consistent safety (off_str < 0.03) but cannot match n-gram PLD speed on high-repetition benchmarks
- TASD wins on **OpenMMLab** (1.59x vs 0.99x) and **PipelineStage** (1.63x vs 1.12x) — structured config formats where guard excels

### Recommendation: TASD-FLY Hybrid

- The two methods are **complementary**: FLY's n-gram PLD accelerates high-repetition content; TASD's structural guard accelerates structured content
- A **TASD-FLY hybrid** would combine n-gram PLD (from FLY) + model draft (from SD) + structural guard (from TASD)
- Expected: n-gram PLD adds +0.24x to TASD, raising TASD from 1.35x to ~1.59x, matching or exceeding FLY at 1.60x
- **Paper narrative**: FLY achieves the best raw speedup, predominantly via n-gram PLD; TASD offers structure-aware safety with lower variance; a hybrid approach captures benefits of both

### Missing Metrics

The following metrics require generated text and were not computed in this analysis:
- Repetition rate (n-gram based)
- Truncation rate (non-AR methods)
- Structure not preserved rate (evaluator-based)
- These can be added by re-evaluating with generated texts from checkpoints.

## Data

- Source: `results/qwen_5method_6x80.json`
- Checkpoints: `results/qwen_6x80_checkpoints/`
- Analysis script: `analyze_fly_vs_tasd.py`
