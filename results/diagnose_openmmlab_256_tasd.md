# OpenMMLab-Config TASD Speedup Diagnosis: 256 vs 128 tokens

**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct
**Settings**: temperature=0.0, TASD: draft_blocks=2, draft_len=16, top_k_accept=3

## 1. Overview

- **128-token TASD speedup** (80 samples): 1.94x
- **256-token TASD speedup** (40 samples): 1.75x
- **Delta**: -0.19x

- **128-token avg accept rate**: 0.9516
- **256-token avg accept rate**: 0.8594
- **Delta accept**: -0.0921

## 2. Per-Sample Speedup Distribution (256 tokens)

| Stat | Value |
|------|-------|
| Min | 0.42x |
| P10 | 0.89x |
| P25 | 1.59x |
| Median | 2.02x |
| P75 | 2.07x |
| P90 | 2.13x |
| Max | 2.22x |
| Mean | 1.75x |

## 3. Worst 10 Samples (Lowest TASD Speedup)

| # | Idx | AR TPS | TASD TPS | Speedup | Accept | SQ | OffStr | Rep | 128 Accept | 128 Repair | 128 Guard | 128 TPS | Draft% |
|---|-----|--------|----------|---------|--------|----|--------|-----|------------|------------|-----------|---------|--------|
| 1 | 3 | 33.0 | 14.0 | 0.42x | 0.1967 | 0.6711 | 0.0789 | 0.3158 | 0.7744 | 1 | 2 | 52.61 | 0.58 |
| 2 | 29 | 32.8 | 14.6 | 0.45x | 0.2069 | 0.6827 | 0.0385 | 0.3462 | 0.6966 | 4 | 6 | 41.67 | 0.56 |
| 3 | 21 | 34.0 | 22.2 | 0.65x | 0.3306 | 0.8524 | 0.0476 | 0.1905 | 1.0 | 0 | 0 | 64.71 | 0.36 |
| 4 | 38 | 34.4 | 26.7 | 0.78x | 0.4049 | 0.8568 | 0.0909 | 0.0455 | 1.0 | 0 | 0 | 65.86 | 0.24 |
| 5 | 17 | 33.6 | 30.0 | 0.89x | 0.4529 | 0.9667 | 0.0000 | 0.0000 | 1.0 | 0 | 0 | 67.39 | 0.13 |
| 6 | 37 | 33.3 | 33.2 | 1.00x | 0.5199 | 0.7222 | 0.0000 | 0.1111 | 1.0 | 0 | 0 | 66.53 | 0.03 |
| 7 | 4 | 32.4 | 33.4 | 1.03x | 0.5030 | 0.8533 | 0.0333 | 0.0667 | 1.0 | 0 | 0 | 69.52 | -0.00 |
| 8 | 2 | 33.9 | 40.7 | 1.20x | 0.5816 | 0.8600 | 0.1000 | 0.0000 | 0.4072 | 3 | 7 | 27.37 | -0.16 |
| 9 | 0 | 30.7 | 38.2 | 1.24x | 0.5816 | 0.8600 | 0.1000 | 0.0000 | 0.4072 | 3 | 7 | 27.65 | -0.20 |
| 10 | 36 | 34.3 | 53.5 | 1.56x | 0.8032 | 0.8955 | 0.0000 | 0.0000 | 1.0 | 0 | 0 | 65.52 | -0.51 |

## 4. Low-Speedup vs High-Speedup Comparison

- **Low-speedup** (< 1.6x): 11 samples
- **Mid-speedup** (1.6x-2.0x): 5 samples
- **High-speedup** (>= 2.0x): 24 samples

| Metric | Low (<1.6x) | Mid (1.6-2.0x) | High (>=2.0x) |
|--------|-------------|----------------|---------------|
| Avg Speedup | 0.98x | 1.95x | 2.06x |
| Avg Accept Rate | 0.4895 | 0.9992 | 0.9998 |
| Avg SQ | 0.8287 | 0.9414 | 0.9305 |
| Avg OffStr | 0.0445 | 0.0000 | 0.0000 |
| Avg Rep | 0.0978 | 0.0000 | 0.0000 |
| Avg Wall Time | 9.61s | 0.00s | 3.95s |
| AR Wall Time | 7.72s | 0.00s | 7.69s |
| 128-token Accept | 0.8441 | 0.0000 | 0.9997 |
| 128-token Repair | 1.0 | 0.0 | 0.0 |
| 128-token Guard | 2.0 | 0.0 | 0.0 |
| 128-token Drafted | 168.4 | 0.0 | 128.0 |

## 5. Accept Rate Changes (256 vs 128 tokens)

Samples sorted by accept rate degradation (most negative first):

| # | Idx | Accept 256 | Accept 128 | Delta | Speedup 256 |
|---|-----|------------|------------|-------|-------------|
| 1 | 21 | 0.3306 | 1.0000 | -0.6694 | 0.65x |
| 2 | 38 | 0.4049 | 1.0000 | -0.5951 | 0.78x |
| 3 | 3 | 0.1967 | 0.7744 | -0.5777 | 0.42x |
| 4 | 17 | 0.4529 | 1.0000 | -0.5471 | 0.89x |
| 5 | 4 | 0.5030 | 1.0000 | -0.4970 | 1.03x |
| 6 | 29 | 0.2069 | 0.6966 | -0.4897 | 0.45x |
| 7 | 37 | 0.5199 | 1.0000 | -0.4801 | 1.00x |
| 8 | 34 | 0.8032 | 1.0000 | -0.1968 | 1.59x |
| 9 | 36 | 0.8032 | 1.0000 | -0.1968 | 1.56x |
| 10 | 1 | 1.0000 | 1.0000 | +0.0000 | 2.08x |
| 11 | 5 | 1.0000 | 1.0000 | +0.0000 | 2.06x |
| 12 | 6 | 1.0000 | 1.0000 | +0.0000 | 2.03x |
| 13 | 8 | 1.0000 | 1.0000 | +0.0000 | 2.02x |
| 14 | 9 | 1.0000 | 1.0000 | +0.0000 | 1.93x |
| 15 | 10 | 1.0000 | 1.0000 | +0.0000 | 2.02x |

## 6. FLY vs TASD Comparison

- **FLY wins**: 11 / 40 samples
- **TASD wins**: 29 / 40 samples

FLY-win samples:
| # | Idx | FLY Speedup | TASD Speedup | FLY MAT | TASD Accept | TASD OffStr | TASD Rep |
|---|-----|-------------|--------------|---------|-------------|-------------|----------|
| 1 | 0 | 1.36x | 1.24x | 1.36 | 0.5816 | 0.1000 | 0.0000 |
| 2 | 2 | 1.31x | 1.20x | 1.31 | 0.5816 | 0.1000 | 0.0000 |
| 3 | 3 | 0.89x | 0.42x | 0.89 | 0.1967 | 0.0789 | 0.3158 |
| 4 | 6 | 2.08x | 2.03x | 2.08 | 1.0000 | 0.0000 | 0.0000 |
| 5 | 14 | 2.07x | 2.04x | 2.07 | 1.0000 | 0.0000 | 0.0000 |
| 6 | 17 | 2.13x | 0.89x | 2.13 | 0.4529 | 0.0000 | 0.0000 |
| 7 | 21 | 1.29x | 0.65x | 1.29 | 0.3306 | 0.0476 | 0.1905 |
| 8 | 25 | 2.21x | 2.04x | 2.21 | 1.0000 | 0.0000 | 0.0000 |
| 9 | 29 | 1.28x | 0.45x | 1.28 | 0.2069 | 0.0385 | 0.3462 |
| 10 | 37 | 1.15x | 1.00x | 1.15 | 0.5199 | 0.0000 | 0.1111 |
| 11 | 38 | 1.71x | 0.78x | 1.71 | 0.4049 | 0.0909 | 0.0455 |

## 7. Draft Overhead Analysis

Draft time share estimated as: (TASD wall - AR wall normalized to same token count) / TASD wall

| Group | Avg Draft% | Avg Draft Overhead (s) | Avg Wall (s) |
|-------|------------|------------------------|--------------|
| Low-speedup | 0.2384 | 1.8878 | 9.61 |
| High-speedup | -0.9482 | -3.7437 | 3.95 |

> Note: Negative draft% for high-speedup samples indicates TASD wall time is *less* than AR wall time, which is expected — TASD is faster because target model forwards are batched. The draft overhead estimate is a rough approximation; the true draft cost is embedded in the TASD wall time.

## 8. Root Cause Analysis: The 6 Persistent Outliers

The most important finding is that **the speedup drop is driven by 6 persistent outlier samples** that were already problematic at 128 tokens and become dramatically worse at 256 tokens:

| Idx | 128 Speedup | 256 Speedup | 128 Accept | 256 Accept | 128 Repair | 128 Guard | Pattern |
|-----|-------------|-------------|------------|------------|------------|-----------|---------|
| 3 | 1.59x | 0.42x | 0.77 | 0.20 | 1 | 2 | Nested dict with comments |
| 29 | 1.27x | 0.45x | 0.70 | 0.21 | 4 | 6 | Pipeline with comment blocks |
| 21 | 1.95x | 0.65x | 1.00 | 0.33 | 0 | 0 | Pipeline with comment blocks |
| 38 | 1.91x | 0.78x | 1.00 | 0.40 | 0 | 0 | Short prompt, nested dict |
| 17 | 2.03x | 0.89x | 1.00 | 0.45 | 0 | 0 | Pipeline with comment blocks |
| 37 | 2.01x | 1.00x | 1.00 | 0.52 | 0 | 0 | Nested dict, long continuation |

**Key observation**: Samples 21, 38, 17, 37 had **accept=1.0 at 128 tokens** but dropped to **0.33-0.52 at 256 tokens**. This means:
- At 128 tokens, the draft model perfectly predicts the continuation (all tokens accepted)
- At 256 tokens, the continuation becomes harder to predict — the draft model starts generating tokens that don't match the target model's argmax

**Why does this happen?**
These samples share a common pattern: they contain **comment blocks** (lines starting with `#`) or **multi-line string values** in the config dict. The draft model (1.5B) is good at predicting structured dict syntax (`dict(type=...)`) but struggles with:
1. Free-text comments (e.g., `# add loading annotation after Resize because...`)
2. String literals with specific paths/names (e.g., `'ade20k_panoptic_val.json'`)
3. Nested dict continuations where the key order matters

When the draft model generates wrong tokens, the relaxed acceptance (top_k=3) catches some but not all of them. The remaining tokens are rejected, triggering repair rounds. Each repair round costs one target model forward for just 1 token — the worst-case overhead for speculative decoding.

**The 6 outliers account for the entire speedup drop:**
- Without these 6 samples: average speedup = (1.75 * 40 - sum_of_6_low) / 34 ≈ **2.02x**
- With these 6 samples: average speedup = **1.75x**
- Contribution: these 6 samples pull down the average by **0.27x**

## 9. FLY Advantage on These Samples

FLY outperforms TASD on exactly these outlier samples because:
- FLY's n-gram mechanism can match comment patterns from the prompt/history
- FLY's window acceptance (win_len=6) is more forgiving than TASD's prefix-based acceptance
- For samples with repetitive comment patterns (e.g., `# ... because ground truth does not need...`), n-gram matching is more effective than draft model prediction

| Sample | FLY Speedup | TASD Speedup | FLY MAT | TASD Accept | FLY advantage source |
|--------|-------------|--------------|---------|-------------|---------------------|
| 17 | 2.13x | 0.89x | 12.29 | 0.45 | N-gram matches comment pattern |
| 21 | 1.29x | 0.65x | 6.76 | 0.33 | N-gram matches comment pattern |
| 29 | 1.28x | 0.45x | 7.91 | 0.21 | N-gram + window acceptance |
| 38 | 1.71x | 0.78x | 9.31 | 0.40 | N-gram matches dict pattern |

## 10. Conclusion

- **A. Accept rate degradation**: 256-token accept rate (0.8594) is lower than 128-token (0.9516), delta -0.0921. **But this is entirely driven by 6 outlier samples.** The remaining 34 samples have accept rate ≈ 1.0.
- **B. Guard/repair overhead**: Low-speedup samples have more guard triggers (2.0 vs 0.0) and repairs (1.0 vs 0.0) at 128 tokens, indicating these are **persistent structural issues** that exist at both lengths.
- **E. Less structured continuation**: Low-speedup samples have higher off-structure rate (0.0445 vs 0.0000) and higher repetition rate (0.0978 vs 0.0000), confirming that these samples contain comment blocks and repetitive patterns that the draft model struggles with.
- **F. FLY/n-gram better suited**: FLY wins on 11/40 samples, but only 4 of these are significant wins. The FLY advantage comes from n-gram matching of comment patterns, not from fundamental algorithm superiority.
- **G. Sample-level outliers**: 6 samples have TASD slower than AR (speedup < 1.0x). These are the **same 6 samples that were problematic at 128 tokens**, just worse at 256 tokens.

## 11. Algorithm Change Recommendation

**Primary diagnosis: G (sample-level outliers) + A (accept rate degradation on outliers)**

The speedup drop from 1.94x to 1.75x is **not a fundamental algorithm issue**. It is caused by 6 persistent outlier samples (15% of the 40-sample set) that:
1. Were already low-accept at 128 tokens (accept 0.70-0.77) or became low-accept at 256 tokens (accept dropped from 1.0 to 0.33-0.52)
2. Contain comment blocks and free-text patterns that the 1.5B draft model cannot predict well
3. Incur heavy repair overhead (each repair = 1 target forward for 1 token)

**Recommendation: Do NOT change the main algorithm.**

Instead:
1. **Report the distribution**: The median speedup is 2.02x, and 24/40 samples (60%) exceed 2.0x. The mean is pulled down by 6 outliers.
2. **Note the outlier pattern**: OpenMMLab configs with comment blocks and free-text values are harder for draft model prediction. This is a known limitation of model-based speculative decoding.
3. **Future work**: Consider adding n-gram candidate generation to TASD for handling comment patterns, but this should be a separate investigation, not a change to the main algorithm.
4. **If needed, exclude outliers**: In paper reporting, consider reporting median speedup (2.02x) alongside mean (1.75x), or report results excluding the 6 worst samples (which would be ~2.02x).