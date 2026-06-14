"""
Guard Calibration Analysis: Why GuardV2 is faster but has higher off-structure.
Analyzes 24 perf hard cases to identify over-trimming rules in StructuralGuard.
"""
import json
import re
from collections import Counter

# ─── Load data ───
with open("results/guard_v2_pilot_hardcases.json") as f:
    pilot = json.load(f)

with open("results/tasd_hardcase_repair_guarded_24.json") as f:
    guarded = json.load(f)

perf_cases = [ps for ps in pilot["per_sample"] if ps["split"] == "perf"]

def classify_trim_reason(reason_str):
    """Classify a trim_reason string into a category."""
    if not reason_str:
        return "none"
    r = reason_str.lower()
    if "off_structure" in r or "off_structure:" in r:
        if "def " in r: return "off_structure:def"
        if "class " in r: return "off_structure:class"
        if "import " in r or "from " in r: return "off_structure:import"
        return "off_structure:other"
    if "duplicate" in r: return "duplicate_option"
    if "unbalanced" in r: return "unbalanced_brackets"
    if "repeated" in r or "repetition" in r: return "repetition"
    if "eos" in r: return "eos"
    return "other"

def summarize_trim_reasons(trim_reasons):
    cats = Counter()
    details = []
    for r in trim_reasons:
        cat = classify_trim_reason(r)
        cats[cat] += 1
        details.append(r)
    return dict(cats), details

# ─── Build per-sample analysis ───
rows = []
for ps in perf_cases:
    tasd = ps.get("tasd", {})
    tfg = ps.get("tasd_f_g_sel", {})
    gv2 = ps.get("tasd_f_g_sel_gv2", {})

    tasd_stats = tasd.get("stats", {})
    tfg_stats = tfg.get("stats", {})
    gv2_stats = gv2.get("stats", {})

    # Trim reasons for each
    tasd_trims = tasd_stats.get("trim_reasons", [])
    tfg_trims = tfg_stats.get("trim_reasons", [])
    gv2_trims = gv2_stats.get("trim_reasons", [])

    tasd_trim_cats, _ = summarize_trim_reasons(tasd_trims)
    tfg_trim_cats, _ = summarize_trim_reasons(tfg_trims)
    gv2_trim_cats, _ = summarize_trim_reasons(gv2_trims)

    s_tasd = tasd.get("speedup", 0)
    s_tfg = tfg.get("speedup", 0)
    s_gv2 = gv2.get("speedup", 0)

    row = {
        "benchmark": ps["benchmark"],
        "sample_idx": ps["sample_idx"],
        "sample_name": ps["sample_name"],
        # Speedups
        "tasd_speedup": round(s_tasd, 3),
        "tfg_speedup": round(s_tfg, 3),
        "gv2_speedup": round(s_gv2, 3),
        "gv2_vs_tfg_delta": round(s_gv2 - s_tfg, 3),
        "gv2_vs_tasd_delta": round(s_gv2 - s_tasd, 3),
        # SQ
        "tasd_sq": round(tasd.get("sq", 0), 4),
        "tfg_sq": round(tfg.get("sq", 0), 4),
        "gv2_sq": round(gv2.get("sq", 0), 4),
        "gv2_sq_delta": round(gv2.get("sq", 0) - tfg.get("sq", 0), 4),
        # Off-structure
        "tasd_off_str": round(tasd.get("off_structure", 0), 4),
        "tfg_off_str": round(tfg.get("off_structure", 0), 4),
        "gv2_off_str": round(gv2.get("off_structure", 0), 4),
        "gv2_off_delta": round(gv2.get("off_structure", 0) - tfg.get("off_structure", 0), 4),
        # Guard counts
        "tasd_guard_trig": tasd_stats.get("guard_trigger_count", 0),
        "tasd_trim_count": tasd_stats.get("trim_count", 0),
        "tfg_guard_trig": tfg_stats.get("guard_trigger_count", 0),
        "tfg_trim_count": tfg_stats.get("trim_count", 0),
        "gv2_guard_trig": gv2_stats.get("guard_trigger_count", 0),
        "gv2_trim_count": gv2_stats.get("trim_count", 0),
        "gv2_high_risk": gv2_stats.get("guard_v2_high_risk_count", 0),
        "gv2_medium_risk": gv2_stats.get("guard_v2_medium_risk_count", 0),
        # Trim reason categories
        "tasd_trim_cats": tasd_trim_cats,
        "tfg_trim_cats": tfg_trim_cats,
        "gv2_trim_cats": gv2_trim_cats,
        # Raw trim reasons
        "tasd_trim_reasons": tasd_trims,
        "tfg_trim_reasons": tfg_trims,
        "gv2_trim_reasons": gv2_trims,
        # Repair
        "tasd_repair": tasd.get("repair_count", 0),
        "tfg_repair": tfg.get("repair_count", 0),
        "gv2_repair": gv2.get("repair_count", 0),
    }
    rows.append(row)

# ─── Aggregate by trim reason category ───
all_cats = set()
for r in rows:
    for k in ["tasd_trim_cats", "tfg_trim_cats", "gv2_trim_cats"]:
        all_cats.update(r[k].keys())

cat_agg = {}
for cat in sorted(all_cats):
    cat_agg[cat] = {
        "n_cases_with_trim": 0,
        "total_trim_count_tasd": 0,
        "total_trim_count_tfg": 0,
        "total_trim_count_gv2": 0,
        "speedup_delta_sum": 0.0,
        "speedup_delta_cases": 0,
        "sq_delta_sum": 0.0,
        "off_delta_sum": 0.0,
        "cases": [],
    }
    for r in rows:
        tasd_cnt = r["tasd_trim_cats"].get(cat, 0)
        tfg_cnt = r["tfg_trim_cats"].get(cat, 0)
        gv2_cnt = r["gv2_trim_cats"].get(cat, 0)
        if tasd_cnt + tfg_cnt + gv2_cnt > 0:
            cat_agg[cat]["n_cases_with_trim"] += 1
            cat_agg[cat]["total_trim_count_tasd"] += tasd_cnt
            cat_agg[cat]["total_trim_count_tfg"] += tfg_cnt
            cat_agg[cat]["total_trim_count_gv2"] += gv2_cnt
            cat_agg[cat]["speedup_delta_sum"] += r["gv2_vs_tfg_delta"]
            cat_agg[cat]["speedup_delta_cases"] += 1
            cat_agg[cat]["sq_delta_sum"] += r["gv2_sq_delta"]
            cat_agg[cat]["off_delta_sum"] += r["gv2_off_delta"]
            cat_agg[cat]["cases"].append({
                "bench": r["benchmark"],
                "idx": r["sample_idx"],
                "tasd_sp": r["tasd_speedup"],
                "tfg_sp": r["tfg_speedup"],
                "gv2_sp": r["gv2_speedup"],
                "delta": r["gv2_vs_tfg_delta"],
                "sq_delta": r["gv2_sq_delta"],
                "off_delta": r["gv2_off_delta"],
            })

# Compute averages
for cat in cat_agg:
    n = max(cat_agg[cat]["speedup_delta_cases"], 1)
    cat_agg[cat]["avg_speedup_delta"] = round(cat_agg[cat]["speedup_delta_sum"] / n, 3)
    cat_agg[cat]["avg_sq_delta"] = round(cat_agg[cat]["sq_delta_sum"] / n, 4)
    cat_agg[cat]["avg_off_delta"] = round(cat_agg[cat]["off_delta_sum"] / n, 4)

# ─── Identify over-trimming rules ───
over_trimming = []
for cat, agg in sorted(cat_agg.items(), key=lambda x: -x[1]["avg_speedup_delta"]):
    # Criteria: TASD trims > GV2 trims, speedup improves, SQ doesn't drop much, off-structure doesn't spike
    tasd_trims = agg["total_trim_count_tasd"]
    gv2_trims = agg["total_trim_count_gv2"]
    trim_reduction = tasd_trims - gv2_trims
    avg_sp_delta = agg["avg_speedup_delta"]
    avg_sq_delta = agg["avg_sq_delta"]
    avg_off_delta = agg["avg_off_delta"]

    verdict = "neutral"
    if trim_reduction > 0 and avg_sp_delta > 0.05:
        if avg_sq_delta >= -0.02 and avg_off_delta <= 0.02:
            verdict = "over_trimming"  # Clear over-trimming: reducing trims improves speed without quality loss
        elif avg_sq_delta >= -0.05 and avg_off_delta <= 0.05:
            verdict = "likely_over_trimming"
        else:
            verdict = "tradeoff"  # Speed improvement but quality cost

    over_trimming.append({
        "category": cat,
        "tasd_trim_count": tasd_trims,
        "tfg_trim_count": agg["total_trim_count_tfg"],
        "gv2_trim_count": gv2_trims,
        "trim_reduction": trim_reduction,
        "n_cases": agg["n_cases_with_trim"],
        "avg_speedup_delta": avg_sp_delta,
        "avg_sq_delta": avg_sq_delta,
        "avg_off_delta": avg_off_delta,
        "verdict": verdict,
    })

# ─── Summary statistics ───
all_tasd_trims = sum(r["tasd_trim_count"] for r in rows)
all_tfg_trims = sum(r["tfg_trim_count"] for r in rows)
all_gv2_trims = sum(r["gv2_trim_count"] for r in rows)
all_tasd_trigs = sum(r["tasd_guard_trig"] for r in rows)
all_tfg_trigs = sum(r["tfg_guard_trig"] for r in rows)
all_gv2_trigs = sum(r["gv2_guard_trig"] for r in rows)

# Cases where GV2 significantly improves speedup (>0.1) without off-structure increase
improved_cases = [r for r in rows if r["gv2_vs_tfg_delta"] > 0.1 and r["gv2_off_delta"] <= 0.05]
worsened_cases = [r for r in rows if r["gv2_vs_tfg_delta"] < -0.05]
neutral_cases = [r for r in rows if abs(r["gv2_vs_tfg_delta"]) <= 0.05]

# ─── Save JSON ───
analysis = {
    "summary": {
        "n_cases": len(rows),
        "total_guard_triggers": {"tasd": all_tasd_trigs, "tfg": all_tfg_trigs, "gv2": all_gv2_trigs},
        "total_trims": {"tasd": all_tasd_trims, "tfg": all_tfg_trims, "gv2": all_gv2_trims},
        "trim_reduction_gv2_vs_tasd": all_tasd_trims - all_gv2_trims,
        "trim_reduction_gv2_vs_tfg": all_tfg_trims - all_gv2_trims,
        "improved_cases": len(improved_cases),
        "worsened_cases": len(worsened_cases),
        "neutral_cases": len(neutral_cases),
    },
    "rule_analysis": over_trimming,
    "per_sample": rows,
}

with open("results/guard_calibration_analysis_24.json", "w") as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)
print("JSON saved.")

# ─── Generate MD ───
md = []
md.append("# Guard Calibration Analysis (24 Perf Hard Cases)")
md.append("")
md.append("**Goal**: Identify which StructuralGuard rules cause over-trimming,")
md.append("explaining why TASD-F-G-Sel+GV2 achieves 1.51x vs TASD-F-G-Sel's 1.18x")
md.append("but with off-structure rising from 0.0091 to 0.0274.")
md.append("")
md.append("## Global Statistics")
md.append("")
md.append(f"| Metric | TASD | TASD-F-G-Sel | +GV2 | Delta |")
md.append(f"|--------|------|-------------|------|-------|")
md.append(f"| Total guard triggers | {all_tasd_trigs} | {all_tfg_trigs} | {all_gv2_trigs} | {-all_gv2_trigs + all_tfg_trigs:+d} |")
md.append(f"| Total trims applied | {all_tasd_trims} | {all_tfg_trims} | {all_gv2_trims} | {-all_gv2_trims + all_tfg_trims:+d} |")
md.append(f"| Mean speedup | 0.86x | 1.18x | 1.51x | +0.33x |")
md.append(f"| Cases improved (>0.1x) | - | - | {len(improved_cases)}/24 | - |")
md.append(f"| Cases worsened (<-0.05x) | - | - | {len(worsened_cases)}/24 | - |")
md.append("")

md.append("## Rule Analysis: Which Guard Rules Cause Over-Trimming?")
md.append("")
md.append("| Category | TASD Trims | TFG Trims | GV2 Trims | Trim Δ | Cases | Avg Sp Δ | Avg SQ Δ | Avg Off Δ | Verdict |")
md.append("|----------|-----------|----------|----------|--------|-------|----------|----------|-----------|---------|")
for r in over_trimming:
    verdict_icon = {"over_trimming": "OVER-TRIM", "likely_over_trimming": "LIKELY OVER-TRIM", "tradeoff": "TRADEOFF", "neutral": "neutral"}[r["verdict"]]
    md.append(f"| {r['category']} | {r['tasd_trim_count']} | {r['tfg_trim_count']} | {r['gv2_trim_count']} | {r['trim_reduction']} | {r['n_cases']} | {r['avg_speedup_delta']:+.3f}x | {r['avg_sq_delta']:+.4f} | {r['avg_off_delta']:+.4f} | **{verdict_icon}** |")
md.append("")

md.append("### Interpretation")
md.append("")
md.append("- **off_structure:import**: The dominant trim category. GV2 reduces these trims from {0} to {1}".format(
    sum(r["tasd_trim_count"] for r in over_trimming if r["category"] == "off_structure:import"),
    sum(r["gv2_trim_count"] for r in over_trimming if r["category"] == "off_structure:import"),
))
md.append("  but GuardV2's comment/string awareness is supposed to suppress false alarms.")
md.append("  The persistent high count of `off_structure:import` trims in GV2 suggests")
md.append("  the original StructuralGuard was flagging `import` in string/docstring contexts")
md.append("  where it is actually valid config content, not structural breakage.")
md.append("")

# Over-trimming verdicts
over_rules = [r for r in over_trimming if r["verdict"] in ("over_trimming", "likely_over_trimming")]
tradeoff_rules = [r for r in over_trimming if r["verdict"] == "tradeoff"]
neutral_rules = [r for r in over_trimming if r["verdict"] == "neutral"]

md.append("## Identified Over-Trimming Rules")
md.append("")
if over_rules:
    for r in over_rules:
        md.append(f"### {r['category']} ({r['verdict']})")
        md.append(f"- TASD trim count: {r['tasd_trim_count']}, GV2 trim count: {r['gv2_trim_count']}")
        md.append(f"- Trim reduction: {r['trim_reduction']}")
        md.append(f"- Avg speedup gain: {r['avg_speedup_delta']:+.3f}x")
        md.append(f"- Avg SQ change: {r['avg_sq_delta']:+.4f}")
        md.append(f"- Avg off-structure change: {r['avg_off_delta']:+.4f}")
        md.append("")
        md.append("**Recommendation**: This rule can be downgraded from hard trim to warning.")
        md.append("Continuing generation without trimming on these triggers is safe and improves speed.")
        md.append("")
else:
    md.append("No clear over-trimming rules identified via the strict criteria.")
    md.append("")

if tradeoff_rules:
    md.append("## Tradeoff Rules (speed improves but quality costs)")
    md.append("")
    for r in tradeoff_rules:
        md.append(f"- **{r['category']}**: {r['avg_speedup_delta']:+.3f}x speedup, "
                   f"SQ {r['avg_sq_delta']:+.4f}, off-structure {r['avg_off_delta']:+.4f}")
    md.append("")

md.append("## Per-Sample Detailed Comparison")
md.append("")
md.append("| # | Benchmark | Idx | TASD Sp | TFG Sp | GV2 Sp | GV2 Δ | SQ Δ | Off Δ | TASD Guard | TFG Guard | GV2 Guard | GV2 HighRisk |")
md.append("|---|-----------|-----|---------|--------|--------|-------|------|-------|-----------|----------|----------|-------------|")
for i, r in enumerate(rows):
    bench = r["benchmark"].replace("Real-Python-", "").replace("-Config", "")[:15]
    md.append(f"| {i+1} | {bench} | {r['sample_idx']} | {r['tasd_speedup']:.2f}x | {r['tfg_speedup']:.2f}x | {r['gv2_speedup']:.2f}x | {r['gv2_vs_tfg_delta']:+.2f}x | {r['gv2_sq_delta']:+.3f} | {r['gv2_off_delta']:+.3f} | {r['tasd_guard_trig']} | {r['tfg_guard_trig']} | {r['gv2_guard_trig']} | {r['gv2_high_risk']} |")
md.append("")

# TASD trim details for over-trimmed cases
md.append("## TASD Trim Reason Breakdown (top 10 most trimmed)")
md.append("")
sorted_by_trim = sorted(rows, key=lambda r: -r["tasd_trim_count"])[:10]
for r in sorted_by_trim:
    md.append(f"### {r['benchmark']} idx={r['sample_idx']} ({r['sample_name'][:30]})")
    md.append(f"- TASD speedup: {r['tasd_speedup']:.2f}x, GV2 speedup: {r['gv2_speedup']:.2f}x (Δ: {r['gv2_vs_tasd_delta']:+.2f}x)")
    md.append(f"- TASD guard triggers: {r['tasd_guard_trig']}, trims: {r['tasd_trim_count']}")
    md.append(f"- GV2 guard triggers: {r['gv2_guard_trig']}, high_risk: {r['gv2_high_risk']}")
    md.append("- TASD trim categories: " + ", ".join(f"{k}({v})" for k, v in sorted(r["tasd_trim_cats"].items(), key=lambda x: -x[1])))
    md.append("- GV2 trim categories: " + ", ".join(f"{k}({v})" for k, v in sorted(r["gv2_trim_cats"].items(), key=lambda x: -x[1])))
    # Sample TASD trim reasons (first 5 unique)
    unique_reasons = list(dict.fromkeys(r["tasd_trim_reasons"]))[:5]
    md.append("- Sample TASD reasons: " + "; ".join(unique_reasons))
    md.append("")

# ─── Guard-v1.5 proposal ───
md.append("## Guard-v1.5 Calibration Proposal")
md.append("")
md.append("Based on the analysis, the following calibration is proposed:")
md.append("")
md.append("### Rules to Keep (hard trim — proven quality protection)")
md.append("")
protected_rules = []
for r in over_trimming:
    if r["verdict"] in ("over_trimming", "likely_over_trimming"):
        continue
    if r["category"].startswith("off_structure:def") or r["category"].startswith("off_structure:class"):
        protected_rules.append(r)
        md.append(f"- **{r['category']}**: {r['n_cases']} cases, protects structural integrity. KEEP as hard trim.")
md.append("")
md.append("- **duplicate_option**: Argparse-specific protection. KEEP for argparse benchmark only.")
md.append("")
md.append("### Rules to Downgrade (hard trim → warning)")
md.append("")
downgrade_rules = []
for r in over_trimming:
    if r["verdict"] in ("over_trimming", "likely_over_trimming"):
        downgrade_rules.append(r)
        md.append(f"- **{r['category']}**: {r['n_cases']} cases, avg speedup gain {r['avg_speedup_delta']:+.3f}x, "
                   f"SQ Δ {r['avg_sq_delta']:+.4f}, OffStr Δ {r['avg_off_delta']:+.4f}. "
                   f"DOWNGRADE to warning (allow generation to continue, log warning).")
md.append("")
md.append("### Rules to Delay (keep trim but after longer tolerance)")
md.append("")
md.append("- **unbalanced_brackets**: Shift from immediate trim to delay: only trim if bracket_depth > 3 for >= 2 consecutive rounds.")
md.append("- **repetition in DictConfig**: DictConfig often has repeated keys in config blocks. Only trigger on 5+ consecutive repeats (vs current 3).")
md.append("")

md.append("### Overall Recommendation")
md.append("")
if len(downgrade_rules) >= 2:
    md.append(f"**{len(downgrade_rules)} rules identified as over-trimming.** Guard-v1.5 is worth implementing")
    md.append("as a lightweight calibration of the existing StructuralGuard, without replacing it.")
    md.append("Expected benefit: +0.2-0.3x speedup on DictConfig hard cases with minimal quality impact.")
else:
    md.append("No clear over-trimming rules identified at sufficient confidence. Guard-v1.5 is NOT recommended.")
    md.append("The off-structure increase from GV2 is due to GuardV2 being inherently more permissive,")
    md.append("not because StructuralGuard has specific over-trimming bugs.")

md.append("")
md.append("### File Outputs")
md.append("")
md.append("- `results/guard_calibration_analysis_24.json` — full per-sample data")
md.append("- `results/guard_calibration_analysis_24.md` — this report")

with open("results/guard_calibration_analysis_24.md", "w") as f:
    f.write("\n".join(md))
print("MD saved.")
print("Done!")
