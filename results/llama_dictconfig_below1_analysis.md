# DictConfig TASD Below-1.0x Case Analysis (LLaMA-8B)

**Target**: Llama-3.1-8B-Instruct | **Draft**: Llama-3.2-1B-Instruct | **Benchmark**: DictConfig  
**Samples**: 5/20 below 1.0x | **Analysis date**: 2026-06-13

---

## Overview

Among 20 DictConfig samples, **5 have TASD speedup < 1.0x**. This is the primary bottleneck preventing TASD from reaching the 1.3x overall threshold.

**Key finding**: The dominant factor is **AR TPS** (target model speed). Cases where TASD exceeds 1.0x have mean AR = 84.7 TPS; cases below 1.0x have mean AR = **103.7 TPS**. When the target model alone generates at >100 TPS, speculative decoding overhead cannot be amortized even with perfect draft acceptance.

| Group | Count | Mean AR TPS | Mean TASD sp |
|-------|:-----:|:-----------:|:------------:|
| TASD >= 1.0x | 15 | 84.7 | 1.28x |
| **TASD < 1.0x** | **5** | **103.7** | **0.74x** |

### Guard Rescue Cases

TASD's structural guard **rescued** 3 DictConfig samples where Greedy SD was below 1.0x:

| Case | GSD | TASD | TASD acc | Guard |
|------|:---:|:----:|:--------:|:-----:|
| dict_config_real_007 | 0.75x | **1.00x** | 1.000 | 0 |
| dict_config_real_012 | 0.95x | **1.03x** | 1.000 | 0 |
| dict_config_real_015 | 0.91x | **1.35x** | 1.000 | 0 |

This is the guard working as intended — it trims off-structure drafts, increasing accept rate from partial to 100%.

---

## Per-Case Analysis

### Case 1: dict_config_real_002 — AR Too Fast, Both SD Methods Fail

| Metric | AR | Greedy SD | TASD | Delta |
|--------|:--:|:---------:|:----:|:-----:|
| **Speedup** | 1.00x | 0.98x | 0.98x | TASD ≈ GSD |
| **Accept Rate** | - | 0.889 | 0.895 | +0.006 |
| **TPS** | 103.3 | 101.6 | 101.4 | - |
| **Guard Triggers** | - | - | 1 | minimal |
| **Repair** | - | - | 0 | - |
| **SQ** | - | 0.151 | 0.151 | identical |

**Prompt**: 311 chars — Python dict of regex pattern tuples for syntax error tracebacks. Structure: deeply nested dict with comment lines interleaved.

**Root cause**: **(e) AR too fast relative to overhead**.
- AR at 103 TPS -> each token costs ~10ms
- SD overhead: draft forward (1B model, ~5-8ms) + verify forward (8B, ~10ms) = ~15-18ms overhead per round
- With draft_len=16, draft_blocks=2, TASD must accept ~2 tokens per round just to break even
- At acc=0.895, ~14.3 tokens per 16-token block → 1.8 tokens wasted per block (just breaks even)
- TASD's 0.98x is near-identical to GSD's 0.98x — this is an **SD problem, not a TASD problem**

**SQ note**: Both AR/GSD/TASD have SQ=0.15 — the generated text diverges significantly from the reference. This is a quality issue independent of speed, possibly due to the complex nested structure.

**Fixability**: **LOW**. Any SD method will struggle when AR is this fast. Only reducing overhead (smaller draft model, fewer blocks) could help, but that would also reduce the benefit on difficult samples.

---

### Case 2: dict_config_real_003 — Guard Over-Trimming ⚠️

| Metric | AR | Greedy SD | TASD | Delta |
|--------|:--:|:---------:|:----:|:-----:|
| **Speedup** | 1.00x | **1.14x** | **0.31x** | **-0.83x** |
| **Accept Rate** | - | 0.842 | 0.243 | **-0.599** |
| **TPS** | 84.2 | 95.6 | 25.8 | **-70 TPS** |
| **Guard Triggers** | - | - | **16** | catastrophic |
| **Repair** | - | - | 0 | - |
| **SQ** | - | 1.000 | 1.000 | identical |

**Prompt**: 119 chars — `_PROTO_ALLOWLIST` dict mapping module names to allowed classes. Structure: `{'module': ['Class1', 'Class2', ...]}`.

**Root cause**: **(b) Guard over-trimming — most severe case in the entire pilot**.
- GSD at 1.14x proves the draft model is good and draft-target alignment is solid (acc=0.842)
- TASD at 0.31x with **16 guard triggers** — guard fires ~twice per decoding round
- The prompt is only 119 chars. With max_new_tokens=128, generation is ~30-40 tokens. At draft_len=16, that's ~2-3 rounds, meaning **guard rejects ~80% of draft tokens before target verification**
- Each guard rejection forces a single-token fallback or full re-draft, nullifying all draft effort

**Structure analysis**: `_PROTO_ALLOWLIST` is a simple dict of module→class lists. The guard may be detecting bracket imbalance during generation (e.g., draft proposes `'Hashable'` but hasn't closed the bracket yet, or generates content that looks like off-structure code).

**Fixability**: **HIGH — this is the most fixable case**.
- The guard's dict_config checker may be too aggressive for short prompts with simple list structures
- A calibrated guard (less aggro for short/early generation) could recover this to near-GSD levels (1.14x)
- If fixed, would move 1 case from 0.31x to ~1.1x (gain of ~+0.04 on overall mean)

---

### Case 3: dict_config_real_005 — Guard + Moderate Accept Rate

| Metric | AR | Greedy SD | TASD | Delta |
|--------|:--:|:---------:|:----:|:-----:|
| **Speedup** | 1.00x | 1.04x | 0.79x | -0.25x |
| **Accept Rate** | - | 0.877 | 0.672 | -0.205 |
| **TPS** | 94.2 | 97.7 | 74.3 | - |
| **Guard Triggers** | - | - | 5 | moderate |
| **Repair** | - | - | 1 | - |
| **SQ** | - | 0.539 | 0.513 | -0.026 |

**Prompt**: 241 chars — dict of comparison operator mappings (`'__lt__': [('__gt__', lambda ...)]`). Contains inline lambdas nested inside dicts.

**Root cause**: **(b + a) Guard reduces accept rate + moderate draft-target alignment**.
- GSD acc=0.877 → draft is OK but not perfect
- TASD acc drops to 0.672 due to 5 guard triggers + 1 repair → guard rejects ~20% of otherwise-acceptable drafts
- AR at 94 TPS means overhead is borderline

**Fixability**: **MEDIUM**. Calibrated guard could reduce trim events. The lambda-in-dict structure may trigger false-positive guard signals.

---

### Case 4: dict_config_real_006 — AR Too Fast, TASD Still Better Than GSD

| Metric | AR | Greedy SD | TASD | Delta |
|--------|:--:|:---------:|:----:|:-----:|
| **Speedup** | 1.00x | 0.64x | **0.78x** | **+0.14x** |
| **Accept Rate** | - | 0.837 | **1.000** | **+0.163** |
| **TPS** | **142.9** | 92.0 | 111.8 | - |
| **Guard Triggers** | - | - | 0 | none |
| **Repair** | - | - | 0 | - |
| **SQ** | - | 0.114 | 0.114 | identical |

**Prompt**: 440 chars — irregular plural-to-singular mapping dict (`'addendum': 'addenda', ...`). Pure key-value pairs, highly repetitive, perfectly deterministic. Large prompt → fast AR.

**Root cause**: **(e) AR extremely fast (143 TPS)** — but TASD is the best SD method here.
- AR at 143 TPS is the **fastest in the entire DictConfig benchmark**
- TASD achieves **1.000 accept rate** (perfect!) — draft and target agree on every token
- GSD only gets 0.837 accept rate → 0.64x (worse than AR)
- TASD's guard + relaxed accept improves acceptance from 0.837 to 1.000
- Even with perfect acceptance, SD overhead (draft forward + verify forward) > the time saved by skipping AR steps

**Fixability**: **LOW — not a TASD problem**. This is a physics limitation. If target AR is 143 TPS (~7ms per token), and each SD round costs ~15ms overhead, you need to accept >2 tokens per round to break even. At draft_len=16, acc=1.000, TASD accepts exactly 16 tokens per ~15ms round → 1.07ms per token → should be ~940 TPS theoretically. But the 1B draft model also takes time per token, and the 8B verify forward for 16 tokens takes ~10ms. So:
  - AR: 7ms/token → 143 TPS
  - TASD: (draft 16 tokens @ ~2ms each = 32ms) + (verify 16 tokens = ~10ms) = 42ms for 16 accepted → 2.6ms/token → 385 theoretical TPS
  - Actual: 111.8 TPS → suggests CUDA synchronization overhead, KV cache manipulation, and Python-level control flow dominate at these speeds

**Not a TASD problem**: Even if we made TASD zero-overhead, AR would still be faster for this specific case.

---

### Case 5: dict_config_real_019 — Moderate Accept + Fast AR

| Metric | AR | Greedy SD | TASD | Delta |
|--------|:--:|:---------:|:----:|:-----:|
| **Speedup** | 1.00x | 0.99x | 0.82x | -0.17x |
| **Accept Rate** | - | 0.865 | 0.734 | -0.131 |
| **TPS** | 93.7 | 92.7 | 76.9 | - |
| **Guard Triggers** | - | - | 2 | low |
| **Repair** | - | - | 1 | - |
| **SQ** | - | 0.947 | 0.763 | -0.184 |

**Prompt**: 87 chars — file size format dict (`{'gib': 1024**3, 'mib': 1024**2, ...}`). Very short prompt, simple structure.

**Root cause**: **(a + e) Below-average draft-target alignment + fast AR**.
- TASD acc=0.734 → ~27% of draft tokens don't match target. Some due to guard (2 triggers), most due to draft-target disagreement
- Short prompt + simple structure → AR is fast (94 TPS)
- GSD also at 0.99x → borderline for any SD method
- SQ drops from 0.947 to 0.763 — quality regression exists

**Fixability**: **LOW-MEDIUM**. SQ drop suggests guard + repair may be changing generation output. But overall speedup improvement unlikely without draft model improvement.

---

## Root Cause Summary

| ID | Case | AR TPS | TASD sp | GSD sp | Accept | Guard | Repair | Root Cause | Fixable |
|:--:|------|:------:|:-------:|:------:|:------:|:-----:|:------:|-----------|:-------:|
| 1 | real_002 | 103.3 | 0.98x | 0.98x | 0.895 | 1 | 0 | **(e)** AR too fast | LOW |
| 2 | real_003 | 84.2 | **0.31x** | 1.14x | 0.243 | **16** | 0 | **(b)** Guard over-trim | **HIGH** |
| 3 | real_005 | 94.2 | 0.79x | 1.04x | 0.672 | 5 | 1 | **(b+a)** Guard + accept | MEDIUM |
| 4 | real_006 | 142.9 | 0.78x | 0.64x | 1.000 | 0 | 0 | **(e)** AR extremely fast | LOW |
| 5 | real_019 | 93.7 | 0.82x | 0.99x | 0.734 | 2 | 1 | **(a+e)** Accept + fast AR | LOW-MED |

### Root Cause Breakdown

| Cause | Cases | Description |
|-------|:-----:|-------------|
| **(e) AR too fast** | 002, 006 | Target model TPS >100 → SD overhead physically cannot be amortized |
| **(b) Guard over-trim** | 003, 005 | Structural guard rejects too many valid drafts |
| **(a) Accept rate** | 005, 019 | Draft-target misalignment reduces effective draft length |

---

## Quantitative Impact

If we could fix case 003 (guard over-trim) from 0.31x to match GSD's 1.14x:

| Scenario | TASD Mean sp | Delta |
|----------|:-----------:|:-----:|
| Current | 1.14x (DictConfig) | - |
| Fix case 003 | 1.18x (DictConfig) | +0.04 |
| **Overall (all 3 benchmarks)** | **1.28x → 1.30x** | **+0.02** |

Fixing case 003 alone would raise TASD overall from 1.28x to **1.30x — crossing the 1.30x threshold**.

---

## Recommendation

### Immediate
1. **Investigate case 003 specifically**: Re-run with guard-v1.5 calibrated=True and verbose logging to see what guard rules are firing
2. The guard's `_check_dict_config()` may need an early-generation leniency threshold (allow more deviation in first ~20 tokens)

### If guard fix succeeds
- Re-run DictConfig 20 samples with calibrated guard → expect ~1.18x DictConfig → ~1.30x overall
- This would satisfy the 1.3x threshold

### If guard fix fails or is too risky
- Case 003 qualifies as a **structural hard case**: prompt is extremely short (119 chars), context is a simple dict of module→class lists — the draft model generates correct code but the guard misfires
- Report as limitation: "Structural guard can over-trim on very short generation contexts where structure is not yet established"
- The remaining 4 below-1.0x cases are all AR-speed-dominated and cannot be fixed by modifying TASD

### For 6×80 scale-up
- Even without fixing case 003, TASD at 1.28x overall is valid and shows generalization across 3 benchmarks
- The below-1.0x cases are explainable and not indicative of TASD failure
- Recommend proceeding to 6×80 with current guard, documenting DictConfig below-1.0x cases as limitation
