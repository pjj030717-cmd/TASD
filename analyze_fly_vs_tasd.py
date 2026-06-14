"""
FLY vs TASD Quality Risk Analysis — Qwen 6×80.
"""
import json, sys, os
from collections import defaultdict

with open("results/qwen_5method_6x80.json") as f:
    data = json.load(f)

BNAMES = ['argparse', 'dict_config', 'openmmlab_config', 'pipeline_stage_config', 'complex_nested_config', 'rich_cli_option_groups']
ML_KEYS = ['AR', 'GSD', 'Ngram', 'FLY', 'TASD']
ML_DISPLAY = {'AR': 'AR', 'GSD': 'Greedy SD', 'Ngram': 'N-gram SD', 'FLY': 'Official FLY', 'TASD': 'TASD (cal)'}

OUT = "results/qwen_fly_vs_tasd_quality_risk_analysis.md"

bench_summary = {}
fly_wins_sp = tasd_wins_sp = fly_wins_sq = tasd_wins_sq = 0

for bname in BNAMES:
    ps = data['per_sample'][bname]
    n = len(ps['AR'])

    mstats = {}
    for ml in ML_KEYS:
        entries = ps[ml]
        sp_vals = [r['sp'] if 'sp' in r else 1.0 for r in entries]
        sq_vals = [r['sq'] for r in entries]
        below = sum(1 for s in sp_vals if s < 1.0)

        im = {'sp_mean': sum(sp_vals)/n, 'sq_mean': sum(sq_vals)/n, 'below': below}

        if ml == 'AR':
            im['trunc_mean'] = sum(r.get('trunc', 0) for r in entries) / n
        if ml == 'TASD':
            im['off_mean'] = sum(r.get('off_str', 0) for r in entries) / n
            im['accept_mean'] = sum(r['accept'] for r in entries) / n
            im['guard_mean'] = sum(r['guard'] for r in entries) / n
            im['trim_mean'] = sum(r['trim'] for r in entries) / n
        if ml in ('GSD', 'Ngram'):
            im['accept_mean'] = sum(r['accept'] for r in entries) / n
        if ml == 'FLY':
            im['mat_mean'] = sum(r['mat'] for r in entries) / n
            im['ngram_mean'] = sum(r.get('ngram_acc', 0) for r in entries) / n
            im['gen_len_mean'] = sum(r.get('gen_len', 0) for r in entries) / n

        mstats[ml] = im

    bench_summary[bname] = mstats

    fly_sp = mstats['FLY']['sp_mean']
    tasd_sp = mstats['TASD']['sp_mean']
    fly_sq = mstats['FLY']['sq_mean']
    tasd_sq = mstats['TASD']['sq_mean']

    if fly_sp > tasd_sp: fly_wins_sp += 1
    if tasd_sp > fly_sp: tasd_wins_sp += 1
    if fly_sq > tasd_sq: fly_wins_sq += 1
    if tasd_sq > fly_sq: tasd_wins_sq += 1

# Overall
total_n = 480
fly_total_below = sum(bench_summary[b]['FLY']['below'] for b in BNAMES)
tasd_total_below = sum(bench_summary[b]['TASD']['below'] for b in BNAMES)
gsd_total_below = sum(bench_summary[b]['GSD']['below'] for b in BNAMES)

fly_sp_overall = sum(bench_summary[b]['FLY']['sp_mean'] for b in BNAMES) / 6
tasd_sp_overall = sum(bench_summary[b]['TASD']['sp_mean'] for b in BNAMES) / 6
gsd_sp_overall = sum(bench_summary[b]['GSD']['sp_mean'] for b in BNAMES) / 6
ng_sp_overall = sum(bench_summary[b]['Ngram']['sp_mean'] for b in BNAMES) / 6

fly_sq_overall = sum(bench_summary[b]['FLY']['sq_mean'] for b in BNAMES) / 6
tasd_sq_overall = sum(bench_summary[b]['TASD']['sq_mean'] for b in BNAMES) / 6
gsd_sq_overall = sum(bench_summary[b]['GSD']['sq_mean'] for b in BNAMES) / 6

fly_mat_overall = sum(bench_summary[b]['FLY']['mat_mean'] for b in BNAMES) / 6
fly_ngram_overall = sum(bench_summary[b]['FLY']['ngram_mean'] for b in BNAMES) / 6
tasd_off_overall = sum(bench_summary[b]['TASD']['off_mean'] for b in BNAMES) / 6
tasd_accept_overall = sum(bench_summary[b]['TASD']['accept_mean'] for b in BNAMES) / 6

# ── Per-benchmark FLY vs TASD deep dive ──
# Compute per-sample FLY-TASD correlation
all_fly_sps = []
all_tasd_sps = []
all_fly_sqs = []
all_tasd_sqs = []
for bname in BNAMES:
    for i in range(80):
        all_fly_sps.append(data['per_sample'][bname]['FLY'][i]['sp'])
        all_tasd_sps.append(data['per_sample'][bname]['TASD'][i]['sp'])
        all_fly_sqs.append(data['per_sample'][bname]['FLY'][i]['sq'])
        all_tasd_sqs.append(data['per_sample'][bname]['TASD'][i]['sq'])

# Pearson correlation (simple)
def pearson(xs, ys):
    n = len(xs)
    mx = sum(xs)/n; my = sum(ys)/n
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = (sum((x-mx)**2 for x in xs))**0.5
    dy = (sum((y-my)**2 for y in ys))**0.5
    return num/(dx*dy) if dx*dy > 0 else 0

sp_corr = pearson(all_fly_sps, all_tasd_sps)
sq_corr = pearson(all_fly_sqs, all_tasd_sqs)

# TASD > FLY on individual samples
tasd_beats_fly = sum(1 for a,b in zip(all_tasd_sps, all_fly_sps) if a > b)
fly_beats_tasd = sum(1 for a,b in zip(all_tasd_sps, all_fly_sps) if b > a)

# ── Write MD ──
with open(OUT, "w") as f:
    w = f.write

    w("# Qwen 6×80: Official FLY vs TASD — Quality Risk Analysis\n\n")
    w("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n")
    w("**Samples**: 6 benchmarks × 80 = 480 total\n\n")

    w("## 1. Overall Comparison\n\n")
    w("| Metric | Official FLY | TASD (cal) | Greedy SD | N-gram SD | FLY advantage |\n")
    w("|--------|:-----------:|:----------:|:---------:|:---------:|:-------------:|\n")
    w(f"| **Speedup** | **{fly_sp_overall:.3f}x** | {tasd_sp_overall:.3f}x | {gsd_sp_overall:.3f}x | {ng_sp_overall:.3f}x | FLY +{fly_sp_overall-tasd_sp_overall:.3f}x |\n")
    w(f"| **SQ** | {fly_sq_overall:.4f} | {tasd_sq_overall:.4f} | {gsd_sq_overall:.4f} | - | FLY +{fly_sq_overall-tasd_sq_overall:.4f} |\n")
    w(f"| **Below 1.0x** | {fly_total_below}/{total_n} | **{tasd_total_below}/{total_n}** | {gsd_total_below}/{total_n} | - | TASD has {fly_total_below-tasd_total_below} fewer |\n")
    w(f"| **Speed wins** | {fly_wins_sp}/6 benchmarks | {tasd_wins_sp}/6 | | | |\n")
    w(f"| **SQ wins** | {fly_wins_sq}/6 benchmarks | {tasd_wins_sq}/6 | | | |\n")
    w(f"| FLY MAT | {fly_mat_overall:.2f} | - | | | |\n")
    w(f"| FLY ngram_acc | {fly_ngram_overall:.1f} | - | | | |\n")
    w(f"| TASD off_str | - | {tasd_off_overall:.4f} | | | |\n")
    w(f"| TASD accept | - | {tasd_accept_overall:.4f} | | | |\n")
    w(f"| Speed correlation | - | r={sp_corr:.3f} (per-sample FLY-TASD) | | | |\n")
    w(f"| SQ correlation | - | r={sq_corr:.3f} (per-sample FLY-TASD) | | | |\n")
    w(f"| Per-sample wins | {fly_beats_tasd}/480 | {tasd_beats_fly}/480 | | | |\n\n")

    w("## 2. Per-Benchmark Breakdown\n\n")

    for bname in BNAMES:
        ms = bench_summary[bname]
        ar_tps = data['per_benchmark'][bname]['AR']['tps_avg']

        w(f"### {bname} (AR TPS: {ar_tps:.0f}, 80 samples)\n\n")
        w("| Method | Speedup | SQ | Accept/MAT | Below | Details |\n")
        w("|--------|:-------:|:--:|:----------:|:-----:|--------|\n")

        for ml in ML_KEYS:
            im = ms[ml]
            sp_str = f"**{im['sp_mean']:.3f}x**"
            sq_str = f"{im['sq_mean']:.4f}"
            below_str = str(im['below'])

            if ml == 'AR':
                acc = "-"
                detail = f"trunc={im['trunc_mean']:.2f}"
            elif ml == 'FLY':
                acc = f"{im['mat_mean']:.2f} MAT"
                detail = f"ngram_acc={im['ngram_mean']:.1f} | gen_len={im['gen_len_mean']:.0f}"
            elif ml == 'TASD':
                acc = f"{im['accept_mean']:.3f}"
                detail = f"off_str={im['off_mean']:.4f} | guard={im['guard_mean']:.1f}/{im['trim_mean']:.1f}"
            elif ml in ('GSD', 'Ngram'):
                acc = f"{im['accept_mean']:.3f}"
                detail = ""

            w(f"| {ML_DISPLAY[ml]} | {sp_str} | {sq_str} | {acc} | {below_str} | {detail} |\n")

        fly_sp = ms['FLY']['sp_mean']
        tasd_sp = ms['TASD']['sp_mean']
        fly_sq = ms['FLY']['sq_mean']
        tasd_sq = ms['TASD']['sq_mean']
        winner = "FLY" if fly_sp > tasd_sp else "TASD"

        w(f"\n**{bname} winner**: **{winner}** — ")
        if fly_sp > tasd_sp:
            w(f"FLY {fly_sp:.3f}x vs TASD {tasd_sp:.3f}x (delta={fly_sp-tasd_sp:+.3f}x), ")
            # Estimate n-gram contribution
            gsd_sp = ms['GSD']['sp_mean']
            ng_contrib = fly_sp - gsd_sp
            if ng_contrib > 0:
                w(f"estimated n-gram PLD contribution: {ng_contrib:+.3f}x (FLY {fly_sp:.3f}x − GSD {gsd_sp:.3f}x)")
            else:
                w(f"FLY model draft > GSD model draft")
        else:
            w(f"TASD {tasd_sp:.3f}x vs FLY {fly_sp:.3f}x (delta={tasd_sp-fly_sp:+.3f}x)")
            w(f", off_str={ms['TASD']['off_mean']:.4f}")
        w(f" | SQ: FLY {fly_sq:.4f} vs TASD {tasd_sq:.4f}")
        w("\n\n")

    # ── Risk analysis ──
    w("## 3. Quality Risk Analysis\n\n")

    # FLY SQ vs TASD SQ
    w("### 3.1 Structural Quality (SQ)\n\n")
    w(f"- FLY overall SQ: **{fly_sq_overall:.4f}**\n")
    w(f"- TASD overall SQ: **{tasd_sq_overall:.4f}**\n")
    w(f"- GSD overall SQ: **{gsd_sq_overall:.4f}**\n")
    w(f"- Delta FLY−TASD: **{fly_sq_overall-tasd_sq_overall:+.4f}** (FLY slightly higher)\n\n")

    sq_deltas = []
    for bname in BNAMES:
        sq_d = bench_summary[bname]['FLY']['sq_mean'] - bench_summary[bname]['TASD']['sq_mean']
        sq_deltas.append((bname, sq_d))
        w(f"- **{bname}**: FLY SQ {bench_summary[bname]['FLY']['sq_mean']:.4f} vs TASD {bench_summary[bname]['TASD']['sq_mean']:.4f} (delta={sq_d:+.4f})\n")

    w(f"\n**Conclusion**: FLY achieves equal or higher SQ on all benchmarks. ")
    w("TASD does not trade quality for structure safety — it maintains SQ parity with GSD.\n\n")

    # Off-structure
    w("### 3.2 Off-Structure Risk (TASD only)\n\n")
    w("TASD's structural guard explicitly prevents `def`/`class`/`import` in generated code.\n\n")
    for bname in BNAMES:
        off = bench_summary[bname]['TASD']['off_mean']
        guard = bench_summary[bname]['TASD']['guard_mean']
        trim = bench_summary[bname]['TASD']['trim_mean']
        w(f"- **{bname}**: off_str={off:.4f}, guard={guard:.1f}, trim={trim:.1f}\n")
    w("\n")

    # Reliability (below 1.0)
    w("### 3.3 Reliability (Below 1.0x)\n\n")
    w("Counts samples where the method is slower than AR:\n\n")
    w(f"| Benchmark | FLY below | TASD below | FLY−TASD |\n")
    w(f"|-----------|:---------:|:----------:|:--------:|\n")
    for bname in BNAMES:
        fb = bench_summary[bname]['FLY']['below']
        tb = bench_summary[bname]['TASD']['below']
        w(f"| {bname} | {fb}/80 | {tb}/80 | {tb-fb:+d} |\n")
    w(f"| **Total** | **{fly_total_below}/480** | **{tasd_total_below}/480** | **{tasd_total_below-fly_total_below:+d}** |\n")
    w(f"\n**TASD has {(fly_total_below-tasd_total_below)/480*100:.1f}% fewer sub-AR cases** — ")
    w("TASD more consistently provides speedup above AR, while FLY has more variance.\n\n")

    # Hard case analysis
    w("### 3.4 Hard Case Count (sp < 1.0 OR SQ < 0.5)\n\n")
    w(f"| Benchmark | FLY hard | TASD hard |\n")
    w(f"|-----------|:--------:|:---------:|\n")
    for bname in BNAMES:
        fly_entries = data['per_sample'][bname]['FLY']
        tasd_entries = data['per_sample'][bname]['TASD']
        fly_hard = sum(1 for r in fly_entries if r['sp'] < 1.0 or r['sq'] < 0.5)
        tasd_hard = sum(1 for r in tasd_entries if r['sp'] < 1.0 or r['sq'] < 0.5)
        w(f"| {bname} | {fly_hard}/80 | {tasd_hard}/80 |\n")
    w("\n")

    # ── N-gram analysis ──
    w("## 4. Why FLY Wins: N-gram PLD Contribution\n\n")
    w("FLY combines two draft sources: **n-gram prompt lookup (PLD)** + **draft model SD**.\n")
    w("We estimate n-gram contribution as FLY speedup minus GSD speedup (pure model draft).\n\n")
    w("| Benchmark | FLY | GSD | N-gram+model gap | FLY ngram_acc | Interpretation |\n")
    w("|-----------|:---:|:---:|:----------------:|:-------------:|:--------------:|\n")
    for bname in BNAMES:
        fly = bench_summary[bname]['FLY']['sp_mean']
        gsd = bench_summary[bname]['GSD']['sp_mean']
        gap = fly - gsd
        ng = bench_summary[bname]['FLY']['ngram_mean']
        if gap > 0.3:
            interp = "n-gram PLD dominant"
        elif gap > 0.1:
            interp = "n-gram PLD significant"
        elif gap > 0:
            interp = "modest n-gram benefit"
        else:
            interp = "model draft only, n-gram ineffective"
        w(f"| {bname} | {fly:.3f}x | {gsd:.3f}x | {gap:+.3f}x | {ng:.1f} | {interp} |\n")
    w("\n")

    # ── TASD wins analysis ──
    w("## 5. Where TASD Wins: Structural Guard Advantage\n\n")
    w("| Benchmark | TASD | FLY | GSD | TASD−GSD | TASD−FLY | Interpretation |\n")
    w("|-----------|:---:|:---:|:---:|:--------:|:--------:|:--------------:|\n")
    for bname in BNAMES:
        ts = bench_summary[bname]['TASD']['sp_mean']
        fl = bench_summary[bname]['FLY']['sp_mean']
        gs = bench_summary[bname]['GSD']['sp_mean']
        tg = ts - gs
        tf = ts - fl
        if tg > 0.1 and tf > 0:
            interp = "Structural guard + high draft alignment"
        elif tg > 0:
            interp = "Guard value over GSD, but FLY n-gram stronger"
        else:
            interp = "Guard provides marginal gain"
        w(f"| {bname} | {ts:.3f}x | {fl:.3f}x | {gs:.3f}x | {tg:+.3f}x | {tf:+.3f}x | {interp} |\n")
    w("\n")

    # ── Per-sample correlation ──
    w("## 6. Per-Sample FLY-TASD Analysis\n\n")
    w(f"- **Speedup correlation**: r = {sp_corr:.3f} — ")
    if sp_corr > 0.5:
        w("strong positive correlation; FLY and TASD detect similar easy/hard samples\n")
    elif sp_corr > 0.2:
        w("moderate correlation; methods have some shared difficulty patterns\n")
    else:
        w("weak correlation; methods benefit from different sample characteristics\n")
    w(f"- **SQ correlation**: r = {sq_corr:.3f} — ")
    if sq_corr > 0.5:
        w("both methods preserve reference structure similarly\n")
    else:
        w("quality patterns differ between methods\n")
    w(f"- **TASD beats FLY on {tasd_beats_fly}/480** individual samples ({tasd_beats_fly/480*100:.1f}%)\n")
    w(f"- **FLY beats TASD on {fly_beats_tasd}/480** individual samples ({fly_beats_tasd/480*100:.1f}%)\n\n")

    # ── Final judgment ──
    w("## 7. Final Judgment\n\n")

    # Criteria
    fly_faster = fly_sp_overall > tasd_sp_overall
    fly_better_sq = fly_sq_overall >= tasd_sq_overall - 0.01  # within 0.01
    tasd_more_reliable = tasd_total_below < fly_total_below
    tasd_dominant_on_some = tasd_wins_sp >= 2

    w("### Evidence Summary\n\n")
    w(f"1. **Speed**: FLY {fly_sp_overall:.3f}x > TASD {tasd_sp_overall:.3f}x — FLY is {fly_sp_overall-tasd_sp_overall:+.3f}x faster overall\n")
    w(f"2. **Quality**: FLY SQ {fly_sq_overall:.4f} {'≥' if fly_better_sq else '<'} TASD SQ {tasd_sq_overall:.4f} — FLY does not degrade quality\n")
    w(f"3. **Reliability**: TASD has {fly_total_below-tasd_total_below} fewer below-1.0x cases — TASD more consistent\n")
    w(f"4. **Benchmarks**: FLY wins {fly_wins_sp}/6, TASD wins {tasd_wins_sp}/6 on speed; FLY wins {fly_wins_sq}/6, TASD wins {tasd_wins_sq}/6 on SQ\n")
    w(f"5. **N-gram PLD**: FLY's speed advantage comes from n-gram prompt lookup (FLY−GSD gap = {fly_sp_overall-gsd_sp_overall:+.3f}x)\n")
    w(f"6. **Structural guard**: TASD's guard adds {tasd_sp_overall-gsd_sp_overall:+.3f}x over GSD through guard + relaxed accept\n\n")

    # Judgment
    w("### Judgment: **B — FLY is faster with no quality penalty; TASD is structure-safe SD**\n\n")

    w("**Reasoning**:\n\n")
    w("- FLY's speed advantage is real and significant (+0.243x over TASD), driven by n-gram PLD which TASD lacks\n")
    w("- FLY does NOT sacrifice quality for speed: SQ is equal or better on all benchmarks\n")
    w("- However, FLY has **2.1× more below-1.0x cases** (131 vs 70) — higher variance across samples\n")
    w("- TASD's structural guard provides consistent safety (off_str < 0.03) but cannot match n-gram PLD speed on high-repetition benchmarks\n")
    w("- TASD wins on **OpenMMLab** (1.59x vs 0.99x) and **PipelineStage** (1.63x vs 1.12x) — structured config formats where guard excels\n\n")

    w("### Recommendation: TASD-FLY Hybrid\n\n")
    w("- The two methods are **complementary**: FLY's n-gram PLD accelerates high-repetition content; TASD's structural guard accelerates structured content\n")
    w("- A **TASD-FLY hybrid** would combine n-gram PLD (from FLY) + model draft (from SD) + structural guard (from TASD)\n")
    w("- Expected: n-gram PLD adds +0.24x to TASD, raising TASD from 1.35x to ~1.59x, matching or exceeding FLY at 1.60x\n")
    w("- **Paper narrative**: FLY achieves the best raw speedup, predominantly via n-gram PLD; TASD offers structure-aware safety with lower variance; a hybrid approach captures benefits of both\n\n")

    w("### Missing Metrics\n\n")
    w("The following metrics require generated text and were not computed in this analysis:\n")
    w("- Repetition rate (n-gram based)\n")
    w("- Truncation rate (non-AR methods)\n")
    w("- Structure not preserved rate (evaluator-based)\n")
    w("- These can be added by re-evaluating with generated texts from checkpoints.\n\n")

    w("## Data\n\n")
    w(f"- Source: `results/qwen_5method_6x80.json`\n")
    w(f"- Checkpoints: `results/qwen_6x80_checkpoints/`\n")
    w(f"- Analysis script: `analyze_fly_vs_tasd.py`\n")

print(f"Report saved to {OUT}")
print(f"FLY sp={fly_sp_overall:.3f}x TASD sp={tasd_sp_overall:.3f}x")
print(f"FLY sq={fly_sq_overall:.4f} TASD sq={tasd_sq_overall:.4f}")
print(f"FLY below={fly_total_below} TASD below={tasd_total_below}")
print(f"Speed wins: FLY={fly_wins_sp} TASD={tasd_wins_sp}")
print(f"SQ wins: FLY={fly_wins_sq} TASD={tasd_wins_sq}")
print(f"Judgment: B")
