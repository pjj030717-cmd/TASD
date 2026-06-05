# TASD Algorithm Simplification Audit

**Purpose**: Distinguish core algorithm components from auxiliary/legacy/future-work components, based on experimental evidence from ablation, speed search, and full benchmarks.

**Evidence base**: Ablation (n=10, 3 benchmarks), speed parameter search (n=20, 2 benchmarks), 6-benchmark n=80 runs (3B and 1.5B drafts).

---

## 1. Final TASD Core Algorithm

```
TASD(prompt, structure_type) → generated_text

1. PRE-FILL: Run prompt through target + draft KV caches.
2. for each round:
   a. MULTI-BLOCK DRAFT: Draft model generates draft_blocks × draft_len tokens
      incrementally, using its KV cache. Early-stop on EOS or off-structure tokens.
   b. RELAXED VERIFY: Target model forward on all draft tokens in one batch.
      Accept tokens matching target argmax (strict), or in top_k_accept (relaxed),
      or above min_token_prob (prob threshold).
   c. PREFIX/WINDOW: Extend accepted prefix via window-based acceptance and
      prefix-budget fallback for remaining draft tokens.
   d. GUARD CHECK: StructuralGuard checks accepted text for off-structure,
      duplicate options, repetition, bad delimiters. Trims if unsafe.
   e. TRIM/APPLY: Append accepted tokens. Trim both KV caches to match.
   f. REPAIR: If no tokens accepted, fall back to single token from target argmax.
```

---

## 2. Components Retained

### 2.1 Multi-Block Draft (CORE)

| Evidence | Value |
|----------|-------|
| draft_blocks=1 ablation | -14.0% speedup (1.50x → 1.29x) |
| DictConfig with draft_blocks=1 | -20.1% speedup (1.49x → 1.19x) |
| Per-block computation | 2 blocks × 16 tokens = 32 draft tokens per target verification |

**Conclusion**: Multi-block draft is the single largest speed contributor. A single draft block cannot amortize the verification cost. Retained as a primary contribution.

### 2.2 Relaxed Acceptance (CORE)

| Evidence | Value |
|----------|-------|
| strict-only ablation | -10.7% speedup (1.50x → 1.34x) |
| 3B draft accept rate strict | 0.87 vs relaxed 0.98 |
| 1.5B draft accept rate | 0.93-1.00 across 6 benchmarks |

**Conclusion**: If only exact argmax matches are accepted, the draft model discards 13% of tokens that are valid structural continuations. Relaxed acceptance (`top_k_accept=3`, `min_token_prob=1e-4`) recovers these tokens. This is the defining difference between TASD and Greedy SD.

### 2.3 KV-Cache Incremental Execution (CORE)

**Conclusion**: Without KV caching, both draft and target would require full-sequence forward passes each round, eliminating the speed advantage. The cache is the implementation backbone of speculative decoding. Retained as a necessary component (not a novel contribution per se, but required for correctness).

### 2.4 Trim / Repair Fallback (CORE)

| Evidence | Value |
|----------|-------|
| Repair count (3B d16, avg across 6 benches) | 0.31/80 samples |
| Repair count (1.5B d16, avg across 6 benches) | 0.21/80 samples |
| Consecutive repair limit | 5 → degrades to AR fallback |

**Conclusion**: Repair handles the edge case where all draft tokens are rejected (structure guard trim, or target disagrees with all). The AR fallback ensures graceful degradation. Retained as decoding-loop closure.

### 2.5 Structural Quality Evaluator (EXPERIMENT SUPPORT)

| Evidence | Value |
|----------|-------|
| SQ (penalty-based) | 0.80-0.94 across 6 benchmarks |
| Detected evaluator bug | Inline evaluator was too optimistic; fixed to use `src/evaluator.py` |

**Conclusion**: Not part of the decoding algorithm, but required for experimental validation. The penalty-based scoring (off_structure, severe, repetition, truncation, duplicate_option, unbalanced_delimiter, bad_tail, structure_not_preserved) provides the quality evidence that makes TASD credible. Retained in the experiment tooling.

---

## 3. Components Downgraded

### 3.1 Structural Guard (AUXILIARY SAFETY LAYER)

| Evidence | 3B Draft | 1.5B Draft |
|----------|----------|------------|
| Guard triggers/80 samples (avg 6 benches) | 0.69 | 0.62 |
| Guard triggers on extended benches | 0.00 | 0.00 |
| no-guard ablation speedup | +3.9% (1.50x → 1.56x) | — |
| no-guard SQ change | None (SQ unchanged) | — |

**Conclusion**: The guard triggers rarely on the current benchmarks (0-3 times per 80 samples). Removing it yields a small speed gain (+4%) with no measurable quality impact on the tested benchmarks. The guard is a **conservative safety layer** — it exists to handle edge cases but is not the mechanism delivering the primary speedup or quality stability. In the paper, it should be described as an optional safety mechanism, not as a primary contribution.

### 3.2 Prefix Budget (FALLBACK EXTENSION)

**Conclusion**: The current acceptance pipeline is dominated by strict (argmax match) and top-k/prob relaxed acceptance. The prefix_budget mechanism triggers infrequently because most tokens are already accepted through the higher-priority rules. It should be described as a **prefix-budget fallback** that extends acceptance when strict + top-k/prob rules are insufficient, not as a primary acceptance mechanism.

### 3.3 Window Acceptance (FALLBACK EXTENSION)

**Conclusion**: Similar to prefix budget — window-level rules extend the accepted prefix beyond the strict boundary but are secondary to the main acceptance pipeline. The `window_len=2` and 50% threshold rule is a lightweight extension that is described as part of the relaxed verification framework, not as an independent mechanism.

---

## 4. Components Removed from Main Narrative

### 4.1 Adaptive Policy (FUTURE WORK)

| Evidence | v1 | v2 |
|----------|----|----|
| DictConfig TPS | +6.3% | +3.9% |
| OpenMMLab TPS | -0.5% | -1.9% |
| Pipeline-Stage TPS | -0.4% | -4.1% |
| Overall | FAIL | FAIL |
| k=5 triggered | 0 times | 0 times |

**Conclusion**: Both v1 and v2 adaptive policies fail to produce consistent throughput gains. The `draft_len` increase (20/24) on already-high-accept benchmarks introduces pure compute overhead. The `top_k_accept=5` condition (`top5 - top3 >= 0.08`) never fires in practice because the gap is consistently below the threshold. Adaptive scheduling is documented as **future work / exploratory optimization** with known failure modes. Do not include in the main method description.

### 4.2 Auto Threshold Solving (NOT IMPLEMENTED)

**Conclusion**: No automatic threshold solver was implemented. The current configuration was selected through grid-style parameter search (7 configs × 2 benchmarks × 20 samples). The paper should describe this as **fixed optimized policy selected by parameter search**, not as an auto-tuned or learned policy.

### 4.3 Dynamic Draft Model Selection (NOT IMPLEMENTED)

**Conclusion**: The project uses a single draft model (3B or 1.5B) but does not switch between them at runtime. Dual-draft selection would require loading both models simultaneously (+10GB VRAM) and introduce switching logic. Listed as future work.

---

## 5. Recommended Method Description for Paper

TASD (Training-free Structure-Aware Speculative Decoding) consists of the following components:

**Core decoding pipeline:**
1. **Multi-block draft proposal** (`draft_blocks=2`, `draft_len=16`): The draft model generates 32 tokens in two sequential blocks using incremental KV-cache execution. Multi-block drafting is the single largest speed contributor — reducing to a single block costs ~14% throughput (Table X).

2. **Relaxed target verification** (`top_k_accept=3`, `min_token_prob=1e-4`): The target model verifies all draft tokens in one batch forward pass. Tokens are accepted if they match the target argmax (strict), fall within the top-3 candidates (relaxed), or exceed a minimum probability threshold. Relaxed acceptance is the defining mechanism of TASD — disabling it (strict-only) reduces speedup by ~11% (Table Y).

3. **Structural safety guard** (optional, enabled by default): A rule-based checker inspects the accepted text for off-structure tokens, duplicate options, repetition, and unbalanced delimiters. When triggered, it trims the accepted prefix to the last safe position. The guard triggers rarely on current benchmarks (~0.7 times per 80 samples) and acts as a conservative safety layer — removing it yields a marginal speed gain (+4%) with no measurable quality impact on tested benchmarks. It is retained for safety but is not the primary speed or quality mechanism.

4. **Trim / repair fallback**: When all draft tokens are rejected (by target verification or guard trim), the system falls back to a single target argmax token. After 5 consecutive repairs, the system degrades to autoregressive generation to avoid unbounded slowdown. Repair events are rare (~0.3 per 80 samples) but required for decoding-loop closure.

**Secondary mechanisms (included for completeness, not dominant):**
- **Prefix-budget fallback**: Extends accepted prefix using log-prob risk budgeting when strict + top-k rules are insufficient. Triggers infrequently in current experiments.
- **Window-based acceptance**: Extends the accepted boundary beyond the strict prefix using a 2-token sliding window with 50% threshold. Described as part of the relaxed verification framework.

**What TASD is not:**
- TASD does not require training, fine-tuning, or architecture modification.
- TASD does not depend on an adaptive scheduling policy (explored in Appendix X but found unstable).
- TASD does not require dynamic draft model switching (single draft model per run).
- TASD does not use reference outputs during decoding.

**Configuration**: The final parameters (`d16_b2_k3`) were selected through parameter search over 7 configurations on 2 benchmarks (n=20). The search considered `draft_len ∈ {8, 12, 16, 24}`, `draft_blocks ∈ {2, 3}`, and `top_k_accept ∈ {3, 5}`. The optimized configuration achieves 1.84x-2.07x speedup (1.5B draft) or 1.44x-1.65x speedup (3B draft) across 6 structured code generation benchmarks, with structural quality within ±0.02 of the corresponding autoregressive baseline.

---

## 6. Parameter Summary

### Final Fixed Policy

| Parameter | Value | Selected By |
|-----------|-------|-------------|
| `draft_len` | 16 | Speed parameter search |
| `draft_blocks` | 2 | Ablation (blocks=1 → -14% speed) |
| `top_k_accept` | 3 | Speed parameter search |
| `min_token_prob` | 1e-4 | Fixed; catch-all tolerance |
| `prefix_budget` | 0.2 | Conservative default |
| `window_len` | 2 | Conservative default |
| `enable_guard` | True | Conservative safety layer |
| `enable_relaxed_accept` | True | Core mechanism (strict → -11%) |
| `temperature` | 0.0 | Greedy; reproducibility |

### Draft Model Recommendation

| Draft Model | Speedup Range | Quality | Recommendation |
|-------------|--------------|---------|----------------|
| Qwen2.5-1.5B-Instruct | 1.84x-2.07x | SQ within ±0.02 of baseline | **Optimized speed default** |
| Qwen2.5-3B-Instruct | 1.44x-1.65x | SQ within ±0.02 of baseline | Conservative / stable baseline |

---

## 7. Representative FLY Pilot Comparison

As a representative comparison, we evaluate TASD against FLY (Li et al., 2025; https://github.com/AMD-AIG-AIMA/FLy), a training-free relaxed speculative decoding method with window-based acceptance. This is a limited pilot study (3 benchmarks × 20 samples), not a full-scale benchmark.

**FLY parameters**:

| Parameter | Value |
|-----------|-------|
| Target model | Qwen2.5-14B-Instruct-AWQ |
| Draft model | Qwen2.5-1.5B-Instruct |
| `k` | 16 |
| `win_len` | 6 |
| `entropy_thre` | 0.3 |
| `temperature` | 0.0 |
| `max_new_tokens` | 128 |
| `enable_fly` | True (FLY) / False (Greedy SD) |
| `use_ngram` | False |
| `tree_verify` | False |
| Verification mode | Greedy (argmax matching) |

**Results**:

| Benchmark | Greedy SD TPS | FLY TPS | TASD TPS | FLY SQ | TASD SQ |
|-----------|--------------|---------|----------|--------|---------|
| OpenMMLab-Config | 22.1 | 30.3 | 62.8 | 0.8437 | 0.8974 |
| Real-Python-DictConfig | 33.4 | 44.9 | 51.4 | 0.8412 | 0.8443 |
| Pipeline-Stage-Config | 22.9 | 34.2 | 65.5 | 0.9209 | 0.9581 |

**Key findings**:

- FLY's window acceptance provides a ~10 TPS boost over standard SD, confirming the value of relaxed verification.
- In the deterministic structured code completion setting studied in this paper, TASD is more effective than FLY.
- FLY improves over Greedy SD but remains below TASD on these representative benchmarks.
- TASD's advantage mainly comes from multi-block draft proposal and structure-aware relaxed verification, rather than generic relaxed verification alone.

**Qualification**: FLY was tested in greedy verification mode (argmax matching) for fair comparison with TASD's verification baseline at temperature 0.0. FLY's original paper also supports modified rejection sampling and n-gram draft, which may yield different results. This pilot does not cover all FLY variants and should be considered a preliminary baseline.
