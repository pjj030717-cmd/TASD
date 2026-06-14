#!/usr/bin/env python3
"""Regenerate final master report with SQ-R / SQ-S / Off-Str.

Note: N-gram SD, Greedy SD, and FLY baseline data was collected before
quality sub-metric recording was added. Their SQ values are legacy
structural_char_recall only. AR, TASD, TASD-FG have full sub-metrics.

For the ablation study (TASD variants 4-7), quality sub-metrics are available
from checkpoint files."""
import json, sys, os

sys.path.insert(0, os.path.dirname(__file__))

SHORT = ["argparse", "dict_config", "openmmlab", "pipeline", "complex_nested", "rich_cli"]
BENCHMARKS = ["argparse", "dict_config", "openmmlab_config",
              "pipeline_stage_config", "complex_nested_config", "rich_cli_option_groups"]

# Per-benchmark structure type for quality metrics
STYPE_MAP = dict(zip(SHORT, BENCHMARKS))

COVERAGE = {
    "argparse": {"repos": 12, "files": 41, "type": "CLI argument parser (argparse/click)"},
    "dict_config": {"repos": 14, "files": 55, "type": "Dict/list lookup table (device/setting maps)"},
    "openmmlab": {"repos": 5, "files": 30, "type": "ML pipeline test config (mmengine-based)"},
    "pipeline": {"repos": 5, "files": 77, "type": "ML pipeline stage config (mmengine-based)"},
    "complex_nested": {"repos": 20, "files": 68, "type": "Deeply nested config dicts (nesting depth 3-21)"},
    "rich_cli": {"repos": 29, "files": 65, "type": "CLI args with option groups (accelerate/transformers)"},
}
PS_LENS = {
    "argparse": "78-1431", "dict_config": "56-1178", "openmmlab": "50-218",
    "pipeline": "61-268", "complex_nested": "95-739", "rich_cli": "118-1134",
}
RS_LENS = {
    "argparse": "73-1493", "dict_config": "80-71474", "openmmlab": "107-1416",
    "pipeline": "171-2305", "complex_nested": "220-59263", "rich_cli": "256-50015",
}

def _agg(entries):
    n = len(entries)
    sps = [e["sp"] for e in entries]
    sqrs = [e.get("sq_r", None) for e in entries]
    sqss = [e.get("sq_s", None) for e in entries]
    offs = [e.get("off_structure_rate", None) for e in entries]

    below = sum(1 for s in sps if s < 1.0)
    sps_sorted = sorted(sps)
    w10n = max(1, n // 10)
    fb_total = sum(e.get("fb_count", 0) for e in entries)

    def _safe_mean(vals):
        v = [x for x in vals if x is not None]
        return round(sum(v)/len(v), 4) if v else None

    return {
        "n": n, "sp_mean": round(sum(sps)/n, 3), "sp_median": round(sps_sorted[n//2], 3),
        "below": below, "worst10": round(sum(sps_sorted[:w10n])/w10n, 3),
        "sq_r": _safe_mean(sqrs), "sq_s": _safe_mean(sqss),
        "off_str": _safe_mean(offs), "fb": fb_total,
    }


def main():
    with open("results/qwen_5method_6x80_quality.json") as f:
        q = json.load(f)["per_sample"]
    with open("results/qwen_tasd_fg_6x80.json") as f:
        tfg_raw = json.load(f)["per_sample"]

    # Build per-sample collections
    per_bench = {}
    for bn, sf in zip(BENCHMARKS, SHORT):
        per_bench[sf] = {}
        st = STYPE_MAP[sf]

        # AR (sp=1.0, sub-metrics from quality JSON)
        ar_entries = []
        for s in q[bn]["AR"]:
            ar_entries.append({"name": s["name"], "sp": 1.0})
            # Copy all sub-metric fields
            for k in ["structural_char_f1", "bracket_balance_score", "structure_type_preservation",
                      "no_repetition_score", "off_structure_rate", "repetition_rate",
                      "is_truncated", "duplicate_option_rate"]:
                if k in s:
                    ar_entries[-1][k] = s[k]
        # Compute SQ-R/SQ-S on AR (1.0 sp)
        for e in ar_entries:
            e["sq_r"] = round(_sq_r(e), 4)
            e["sq_s"] = round(_sq_s(e, sf), 4)
        per_bench[sf]["AR"] = ar_entries

        # Ngram/GSD/FLY — no sub-metrics available
        for jk in ["Ngram", "GSD", "FLY"]:
            if jk not in q[bn]:
                continue
            entries = []
            for s in q[bn][jk]:
                entries.append({"name": s["name"], "sp": s.get("sp", 0)})
            per_bench[sf][jk] = entries

        # TASD (sub-metrics from quality JSON)
        tasd_entries = []
        for s in q[bn]["TASD"]:
            e = {"name": s["name"], "sp": s.get("sp", 0)}
            for k in ["structural_char_f1", "bracket_balance_score", "structure_type_preservation",
                      "no_repetition_score", "off_structure_rate", "repetition_rate",
                      "is_truncated", "duplicate_option_rate", "fb_count"]:
                if k in s:
                    e[k] = s[k]
            e["fb_count"] = 0
            tasd_entries.append(e)
        for e in tasd_entries:
            e["sq_r"] = round(_sq_r(e), 4)
            e["sq_s"] = round(_sq_s(e, sf), 4)
        per_bench[sf]["TASD"] = tasd_entries

        # TASD-FG (sub-metrics from tfg JSON)
        if bn in tfg_raw and "TASD-F-G" in tfg_raw[bn]:
            tfg_entries = []
            for s in tfg_raw[bn]["TASD-F-G"]:
                e = {"name": s["name"], "sp": s.get("sp", 0)}
                for k in ["structural_char_f1", "bracket_balance_score", "structure_type_preservation",
                          "no_repetition_score", "off_structure_rate", "repetition_rate",
                          "is_truncated", "duplicate_option_rate", "fb_count"]:
                    if k in s:
                        e[k] = s[k]
                tfg_entries.append(e)
            for e in tfg_entries:
                e["sq_r"] = round(_sq_r(e), 4)
                e["sq_s"] = round(_sq_s(e, sf), 4)
            per_bench[sf]["TASD-FG"] = tfg_entries

    # Aggregate
    methods = ["AR", "Ngram", "GSD", "FLY", "TASD", "TASD-FG"]
    vlabels = {"AR": "AR", "Ngram": "N-gram SD", "GSD": "Greedy SD",
               "FLY": "Official FLY", "TASD": "TASD", "TASD-FG": "TASD-FG"}

    overall_agg = {}
    for m in methods:
        all_entries = []
        for sf in SHORT:
            if m in per_bench[sf]:
                all_entries.extend(per_bench[sf][m])
        if all_entries:
            overall_agg[m] = _agg(all_entries)

    bench_agg = {}
    for sf in SHORT:
        bench_agg[sf] = {}
        for m in methods:
            if m in per_bench[sf]:
                bench_agg[sf][m] = _agg(per_bench[sf][m])

    def _fmt(val, digits=4):
        if val is None: return "—"
        return f"{val:.{digits}f}"

    # ── Write report ──
    lines = []
    lines.append("# TASD-FG Final Master Report\n\n")
    lines.append("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct\n")
    lines.append("**Config**: max_new_tokens=128, temperature=0.0, samples=80 per benchmark\n\n")

    # Table 0
    lines.append("## Table 0: Structure Coverage / Benchmark Construction\n\n")
    lines.append("| Benchmark | Type | Source Repos | Unique Files | Prompt (chars) | Ref (chars) |\n")
    lines.append("|-----------|------|:----------:|:----------:|:---:|:---:|\n")
    for sf in SHORT:
        cv = COVERAGE[sf]
        lines.append(f"| {sf} | {cv['type']} | {cv['repos']} | {cv['files']} | {PS_LENS[sf]} | {RS_LENS[sf]} |\n")
    total_repos = sum(cv["repos"] for cv in COVERAGE.values())
    total_files = sum(cv["files"] for cv in COVERAGE.values())
    lines.append(f"\nAll six benchmark types are constructed from real-world open-source code repositories (CodeSearchNet-Python and OpenMMLab ecosystem), covering **{total_repos}** unique repos and **{total_files}** unique source files.\n\n")

    # Quality metrics
    lines.append("## Quality Metrics\n\n")
    lines.append("- **SQ-R** (Reference-aware Structural Quality): 0.4×structural_char_F1 + 0.3×bracket_balance + 0.2×type_preservation + 0.1×no_repetition\n")
    lines.append("- **SQ-S** (Structure Safety Score): 1.0 − 0.45×off_structure − 0.25×truncation − 0.20×repetition − 0.10×duplicate_option\n")
    lines.append("- **Off-Str**: lines starting with `def`/`class`/`import`/`from` ÷ total lines\n")
    lines.append("- SQ-R/SQ-S available for: AR, TASD, TASD-FG (have per-sample quality sub-metrics). Baselines (N-gram SD, Greedy SD, Official FLY) were collected before sub-metric recording was added.\n\n")

    # Table 1
    lines.append("## Table 1: Main Results (480 samples)\n\n")
    lines.append("| Method | Speedup | Below | Worst-10 | SQ-R | SQ-S | Off-Str | FB |\n")
    lines.append("|--------|:-------:|:-----:|:--------:|:----:|:----:|:-------:|:--:|\n")
    for m in methods:
        if m not in overall_agg: continue
        o = overall_agg[m]
        bold = "**" if m in ("TASD", "TASD-FG") else ""
        fb_str = str(o["fb"]) if m == "TASD-FG" else "-"
        lines.append(f"| {bold}{vlabels[m]}{bold} | {bold}{o['sp_mean']:.3f}x{bold} | {o['below']}/480 | {o['worst10']:.3f}x | {_fmt(o['sq_r'])} | {_fmt(o['sq_s'])} | {_fmt(o['off_str'])} | {fb_str} |\n")
    lines.append("\n")

    # Table 2
    lines.append("## Table 2: Per-Benchmark Results\n\n")
    for sf in SHORT:
        lines.append(f"### {sf} (80)\n\n")
        lines.append("| Method | Speedup | Below | SQ-R | SQ-S | Off-Str | FB |\n")
        lines.append("|--------|:-------:|:-----:|:----:|:----:|:-------:|:--:|\n")
        for m in methods:
            if m not in bench_agg[sf]: continue
            a = bench_agg[sf][m]
            bold = "**" if m in ("TASD", "TASD-FG") else ""
            fb_str = str(a["fb"]) if m == "TASD-FG" else "-"
            lines.append(f"| {bold}{vlabels[m]}{bold} | {bold}{a['sp_mean']:.3f}x{bold} | {a['below']} | {_fmt(a['sq_r'])} | {_fmt(a['sq_s'])} | {_fmt(a['off_str'])} | {fb_str} |\n")
        lines.append("\n")

    # Methods
    lines.append("## Methods\n\n")
    lines.append("| # | Method | Description |\n")
    lines.append("|---|--------|-------------|\n")
    lines.append("| 1 | AR | Target autoregressive (greedy) |\n")
    lines.append("| 2 | N-gram SD | Prompt-lookup SD (n=3-8, no draft model) |\n")
    lines.append("| 3 | Greedy SD | Strict greedy SD (draft_len=16, blocks=2) |\n")
    lines.append("| 4 | Official FLY | AMD FLy SPDGenerate (k=15, win_len=6, entropy_thre=0.3) |\n")
    lines.append("| 5 | TASD | Structure-aware SD + Guard v1.5 calibrated |\n")
    lines.append("| 6 | **TASD-FG** | **TASD + Guarded Failure-Aware Fallback (proposed)** |\n\n")

    # Conclusion
    tfg_o = overall_agg["TASD-FG"]
    fly_o = overall_agg["FLY"]
    lines.append("## Conclusion\n\n")
    lines.append(f"TASD-FG achieves **{tfg_o['sp_mean']:.3f}x** average speedup with only **{tfg_o['below']}/480 below-AR** cases ({tfg_o['below']/480*100:.1f}%), outperforming Official FLY ({fly_o['sp_mean']:.3f}x, {fly_o['below']}/480 below) in both average speedup and robustness.\n\n")

    tsd_o = overall_agg["TASD"]
    lines.append(f"Compared to TASD ({tsd_o['sp_mean']:.3f}x, {tsd_o['below']}/480 below, worst-10={tsd_o['worst10']:.3f}x):\n")
    lines.append(f"- Speedup: +{tfg_o['sp_mean']-tsd_o['sp_mean']:.3f}x\n")
    lines.append(f"- Below-AR: {tsd_o['below']} → {tfg_o['below']} (−{tsd_o['below']-tfg_o['below']})\n")
    lines.append(f"- Worst-10: {tsd_o['worst10']:.3f}x → {tfg_o['worst10']:.3f}x (+{tfg_o['worst10']-tsd_o['worst10']:.3f})\n")
    lines.append(f"- SQ-S: {tsd_o['sq_s']:.4f} → {tfg_o['sq_s']:.4f}\n")
    lines.append(f"- Off-Str: {tsd_o['off_str']:.4f} → {tfg_o['off_str']:.4f}\n\n")

    lines.append("TASD-FG achieves the best speed-robustness-safety trade-off. It does not maximize reference similarity (SQ-R), but maintains competitive structure safety (SQ-S) while achieving the highest speedup and the fewest below-AR failures among all methods.\n\n")

    # Supplementary tables
    lines.append("## Supplementary Tables\n\n")
    lines.append("| Table | Description | File |\n")
    lines.append("|-------|-------------|------|\n")
    lines.append("| Table 3 | Ablation Study (7 TASD variants) | `results/qwen_ablation_7variant.md` |\n")
    lines.append("| Table 4 | 256-token Scaling Pilot (3 benchmarks × 20) | `results/qwen_256token_pilot_3x20.md` |\n")
    lines.append("| Table 5 | Hard Case / Negative Results | `results/tasdfg_below_vs_noguard_analysis.md` |\n")
    lines.append("| - | Profit-aware switch pilot | `results/tasdfg_profit_switch_pilot.md` |\n")
    lines.append("| - | TASD-NG (n-gram draft channel, negative) | `results/tasd_ng_pilot_3x20.md` |\n")

    with open("results/final_master_report.md", "w") as f:
        f.write("".join(lines))

    out_json = {"overall": overall_agg, "per_benchmark": bench_agg}
    with open("results/final_master_report.json", "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    print("Report generated.")
    for m in methods:
        if m in overall_agg:
            o = overall_agg[m]
            print(f"  {vlabels[m]:20s}: sp={o['sp_mean']:.3f}x  below={o['below']}/480  sq_r={_fmt(o['sq_r'])}  sq_s={_fmt(o['sq_s'])}  off_str={_fmt(o['off_str'])}")


def _sq_r(s):
    return 0.4 * s.get("structural_char_f1", 0) + \
           0.3 * s.get("bracket_balance_score", 0) + \
           0.2 * s.get("structure_type_preservation", 0) + \
           0.1 * s.get("no_repetition_score", 0)

def _sq_s(s, stype):
    off = s.get("off_structure_rate", 0)
    trunc = s.get("is_truncated", 0)
    rep = s.get("repetition_rate", 0)
    dup_opt = s.get("duplicate_option_rate", rep)
    sq = 1.0 - 0.45 * off - 0.25 * trunc - 0.20 * rep - 0.10 * dup_opt
    return max(0.0, min(1.0, sq))

if __name__ == "__main__":
    main()
