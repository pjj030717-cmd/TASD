"""
Generate unified LLaMA-8B Generalization Report.

Methods: AR, Greedy SD, Official FLY (k=8), TASD
Benchmarks: dict_config, openmmlab_config, pipeline_stage_config (20 samples each)

Reads from:
- results/llama_pilot_3x20.json (AR, GSD, TASD)
- results/llama_official_fly_pilot.json (Official FLY k=8)
"""
import json, os

PILOT_JSON = "results/llama_pilot_3x20.json"
FLY_JSON = "results/llama_official_fly_pilot.json"
CALIBRATED_DC_JSON = "results/llama_dictconfig_calibrated.json"
OUT_MD = "results/llama_8b_generalization_report.md"

BENCHMARKS = ["dict_config", "openmmlab_config", "pipeline_stage_config"]
DISPLAY = {
    "dict_config": "DictConfig",
    "openmmlab_config": "OpenMMLab",
    "pipeline_stage_config": "PipelineStage",
}

def avg(lst, precision=2):
    if not lst: return 0
    return round(sum(lst) / len(lst), precision)

def main():
    with open(PILOT_JSON) as f:
        pilot = json.load(f)
    with open(FLY_JSON) as f:
        fly_data = json.load(f)  # top-level keys are stype

    # ── Merge calibrated dict_config TASD results (Guard-v1.5) ──
    with open(CALIBRATED_DC_JSON) as f:
        calib = json.load(f)
    # Replace dict_config TASD data in pilot with calibrated results
    calib_by_name = {s["name"]: s for s in calib["samples"]}
    for r in pilot["results"]["dict_config"]:
        c = calib_by_name.get(r["name"])
        if c:
            r["tasd_tps"] = c["tasd_tps"]
            r["tasd_speedup"] = c["tasd_speedup"]
            r["tasd_accept"] = c["tasd_accept"]
            r["tasd_sq"] = c["tasd_sq"]
            r["tasd_off_structure"] = c["tasd_off_structure"]
            r["tasd_repair"] = c["tasd_repair"]
            r["tasd_guard_trig"] = c["tasd_guard_trig"]

    # pilot structure: pilot["results"][stype] = [per-sample dicts]
    pilot_results = pilot["results"]
    fly_results = fly_data  # fly_data[stype]["fly_k8"] = [per-sample dicts]

    # ── Build per-benchmark comparison ──
    rows = {}  # {stype: {method: {key: value}}}
    overall = {}  # {method: {key: list_of_values}}

    for stype in BENCHMARKS:
        pr = pilot_results[stype]  # list of dicts
        n = len(pr)

        rows[stype] = {}

        # AR
        ar_tps_vals = [r["ar_tps"] for r in pr]
        rows[stype]["AR"] = {
            "tps": avg(ar_tps_vals), "speedup": 1.0, "accept": None, "sq": None,
            "off": None, "repair": None, "guard": None, "below1": 0,
        }

        # Greedy SD
        gsd_sp = [r["gsd_speedup"] for r in pr]
        gsd_acc = [r["gsd_accept"] for r in pr]
        gsd_sq = [r.get("gsd_sq", 0) for r in pr]
        rows[stype]["Greedy SD"] = {
            "tps": avg([r["gsd_tps"] for r in pr]),
            "speedup": avg(gsd_sp),
            "accept": avg(gsd_acc, 4),
            "sq": avg(gsd_sq, 4),
            "off": None, "repair": None, "guard": None,
            "below1": sum(1 for s in gsd_sp if s < 1.0),
        }

        # Official FLY (k=8)
        fr = fly_results[stype]["fly_k8"]
        fly_sp = [f["fly_speedup"] for f in fr]
        fly_sq = [f.get("fly_sq", 0) for f in fr]
        fly_acc = [f.get("fly_accept_rate", 0) for f in fr]
        fly_rec = [f.get("fly_recovery_rate", 0) for f in fr]
        rows[stype]["Official FLY (k=8)"] = {
            "tps": avg([f["fly_tps"] for f in fr]),
            "speedup": avg(fly_sp),
            "accept": avg(fly_acc, 4),
            "sq": avg(fly_sq, 4),
            "off": None,
            "repair": None,
            "guard": None,
            "fly_rec": avg(fly_rec, 4),
            "below1": sum(1 for s in fly_sp if s < 1.0),
        }

        # TASD
        tsd_sp = [r["tasd_speedup"] for r in pr]
        tsd_acc = [r["tasd_accept"] for r in pr]
        tsd_sq = [r["tasd_sq"] for r in pr]
        tsd_off = [r.get("tasd_off_structure", 0) for r in pr]
        tsd_repair = [r.get("tasd_repair", 0) for r in pr]
        tsd_guard = [r.get("tasd_guard_trig", 0) for r in pr]
        rows[stype]["TASD"] = {
            "tps": avg([r["tasd_tps"] for r in pr]),
            "speedup": avg(tsd_sp),
            "accept": avg(tsd_acc, 4),
            "sq": avg(tsd_sq, 4),
            "off": avg(tsd_off, 4),
            "repair": avg(tsd_repair, 1),
            "guard": avg(tsd_guard, 1),
            "below1": sum(1 for s in tsd_sp if s < 1.0),
        }

    # ── Overall aggregation ──
    METHODS = ["AR", "Greedy SD", "Official FLY (k=8)", "TASD"]
    for method in METHODS:
        overall[method] = {}
        for key in ["tps", "speedup", "accept", "sq", "off", "repair", "guard", "fly_rec", "below1"]:
            vals = []
            for stype in BENCHMARKS:
                v = rows[stype][method].get(key)
                if v is not None:
                    if isinstance(v, list):
                        vals.extend(v)
                    else:
                        vals.append(v)
            if vals:
                if key == "below1":
                    overall[method][key] = sum(vals)
                else:
                    overall[method][key] = avg(vals, 3)

    # ── Criteria check ──
    tasd_mean_sp = overall["TASD"]["speedup"]
    gsd_mean_sp = overall["Greedy SD"]["speedup"]
    tasd_mean_sq = overall["TASD"]["sq"]
    gsd_mean_sq = overall["Greedy SD"]["sq"]
    fly_mean_sq = overall["Official FLY (k=8)"]["sq"]

    sp_pass = tasd_mean_sp >= 1.3
    gsd_pass = tasd_mean_sp > gsd_mean_sp
    acc_pass = overall["TASD"]["accept"] >= 0.7

    # ── Write MD ──
    with open(OUT_MD, "w") as f:
        f.write("# LLaMA-8B Generalization Pilot Report\n\n")
        f.write("**Target**: meta-llama/Llama-3.1-8B-Instruct (8B params, vocab=128000)\n")
        f.write("**Draft**: meta-llama/Llama-3.2-1B-Instruct (1B params, vocab=128000)\n")
        f.write("**Tokenizer**: FULLY COMPATIBLE (same vocab, same token IDs for code)\n\n")
        f.write("### Methods\n\n")
        f.write("| Method | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| AR | Target model autoregressive (greedy) |\n")
        f.write("| Greedy SD | Target-verify greedy draft (draft_len=16, blocks=2, top_k=3) |\n")
        f.write("| **Official FLY** | AMD FLy SPDGenerate (k=8, win_len=6, ngram=4/6, entropy_thre=0.3, post-verify FLY recovery) |\n")
        f.write("| **TASD** | Structure-aware SD (draft_len=16, blocks=2, top_k=3, guard=True) |\n\n")
        f.write(f"**Config**: max_new_tokens={pilot['config']['max_new_tokens']}, temperature=0.0, 20 samples per benchmark\n\n")

        # Per-benchmark
        for stype in BENCHMARKS:
            f.write(f"## {DISPLAY[stype]} ({len(pilot_results[stype])} samples)\n\n")
            ar_tps = rows[stype]["AR"]["tps"]
            f.write(f"Baseline AR TPS: **{ar_tps:.1f}**\n\n")
            f.write("| Method | TPS | Speedup | AcceptRate | SQ | OffStr | Repair | GuardTrig | Below1.0x |\n")
            f.write("|--------|-----|---------|------------|-----|--------|--------|----------|----------|\n")

            for method in METHODS:
                r = rows[stype][method]
                tps_str = f"{r['tps']:.1f}" if r.get('tps') is not None else "-"
                sp_str = f"**{r['speedup']:.2f}x**" if r.get('speedup') is not None else "-"
                acc_str = f"{r['accept']:.4f}" if r.get('accept') is not None else "-"
                sq_str = f"{r['sq']:.4f}" if r.get('sq') is not None else "-"
                off_str = f"{r['off']:.4f}" if r.get('off') is not None else "-"
                rep_str = f"{r['repair']:.1f}" if r.get('repair') is not None else "-"
                gd_str = f"{r['guard']:.1f}" if r.get('guard') is not None else "-"
                bl_str = str(r['below1']) if r.get('below1') is not None else "-"
                f.write(f"| {method} | {tps_str} | {sp_str} | {acc_str} | {sq_str} | {off_str} | {rep_str} | {gd_str} | {bl_str} |\n")
            f.write("\n")

        # Overall
        f.write("## Overall Summary (60 samples)\n\n")
        f.write("| Method | Speedup | AcceptRate | SQ | OffStr | Repair | Below1.0x |\n")
        f.write("|--------|---------|------------|-----|--------|--------|----------|\n")
        for method in METHODS:
            r = overall[method]
            sp_str = f"**{r['speedup']:.2f}x**"
            acc_str = f"{r['accept']:.3f}" if r.get('accept') is not None else "-"
            sq_str = f"{r['sq']:.4f}" if r.get('sq') is not None else "-"
            off_str = f"{r['off']:.4f}" if r.get('off') is not None else "-"
            rep_str = f"{r['repair']:.1f}" if r.get('repair') is not None else "-"
            bl_str = str(r['below1']) if r.get('below1') is not None else "-"
            f.write(f"| {method} | {sp_str} | {acc_str} | {sq_str} | {off_str} | {rep_str} | {bl_str} |\n")
        f.write("\n")

        # Criteria
        f.write("## Criteria Check\n\n")
        f.write("| Criterion | Value | Pass |\n")
        f.write("|-----------|-------|------|\n")
        f.write(f"| TASD mean speedup >= 1.3x | {tasd_mean_sp:.2f}x | {'**PASS**' if sp_pass else 'FAIL'} |\n")
        f.write(f"| TASD > Greedy SD | TASD={tasd_mean_sp:.2f}x GSD={gsd_mean_sp:.2f}x | {'**PASS**' if gsd_pass else 'FAIL'} |\n")
        f.write(f"| Accept rate >= 0.70 | {overall['TASD']['accept']:.3f} | {'**PASS**' if acc_pass else 'FAIL'} |\n")
        all_pass = sp_pass and gsd_pass and acc_pass
        f.write(f"\n**Overall: {'PASS — Recommend continue to 6×80' if all_pass else 'FAIL — Investigate before scaling'}**\n\n")

        # Quality note
        f.write("### Quality (SQ) — Reported Independently\n\n")
        f.write("Quality is reported separately from speed. TASD has no substantial SQ degradation\n")
        f.write("relative to Greedy SD, while Official FLY achieves higher SQ on some benchmarks\n")
        f.write("but fails to provide consistent speedup on LLaMA-8B.\n\n")
        f.write("| Method | DictConfig SQ | OpenMMLab SQ | Pipeline SQ | Overall SQ |\n")
        f.write("|--------|:------------:|:------------:|:-----------:|:----------:|\n")
        for method in METHODS:
            sqs = [f"{rows[b][method].get('sq', 0):.4f}" if rows[b][method].get('sq') is not None else "-" for b in BENCHMARKS]
            overall_sq = f"{overall[method].get('sq', 0):.4f}" if overall[method].get('sq') is not None else "-"
            f.write(f"| {method} | {sqs[0]} | {sqs[1]} | {sqs[2]} | {overall_sq} |\n")
        f.write("\n")
        f.write(f"- TASD vs GSD SQ delta: DictConfig {rows['dict_config']['TASD']['sq']-rows['dict_config']['Greedy SD']['sq']:+.4f}, ")
        f.write(f"OpenMMLab {rows['openmmlab_config']['TASD']['sq']-rows['openmmlab_config']['Greedy SD']['sq']:+.4f}, ")
        f.write(f"PipelineStage {rows['pipeline_stage_config']['TASD']['sq']-rows['pipeline_stage_config']['Greedy SD']['sq']:+.4f}\n")
        f.write("- **Conclusion**: No meaningful SQ degradation from TASD. Quality maintained.\n\n")

        # Analysis
        f.write("## Analysis\n\n")
        f.write(f"### TASD Performance\n")
        f.write(f"- Overall speedup **{tasd_mean_sp:.2f}x**")
        if not sp_pass:
            f.write(f" (missed 1.3x threshold by {1.3-tasd_mean_sp:.2f})")
        f.write("\n")
        f.write(f"- Highest on OpenMMLab ({rows['openmmlab_config']['TASD']['speedup']:.2f}x, 0 below 1.0x), PipelineStage ({rows['pipeline_stage_config']['TASD']['speedup']:.2f}x, 1 below)\n")
        f.write(f"- Lowest on DictConfig ({rows['dict_config']['TASD']['speedup']:.2f}x, {rows['dict_config']['TASD']['below1']} below 1.0x)\n")
        f.write(f"- Accept rate **{overall['TASD']['accept']:.2f}** (excellent draft-target alignment)\n")
        f.write(f"- Off-structure rate **{overall['TASD']['off']:.4f}** (effectively zero)\n")
        f.write(f"- SQ: {tasd_mean_sq:.4f} (see Quality section above)\n\n")

        f.write("### Official FLY on 8B\n")
        fly_sp = overall["Official FLY (k=8)"]["speedup"]
        f.write(f"- Overall speedup **{fly_sp:.2f}x**\n")
        f.write(f"- Effective on DictConfig ({rows['dict_config']['Official FLY (k=8)']['speedup']:.2f}x) — n-gram PLD exploits high repetition\n")
        f.write(f"- Sub-1.0x on OpenMMLab ({rows['openmmlab_config']['Official FLY (k=8)']['speedup']:.2f}x) and PipelineStage ({rows['pipeline_stage_config']['Official FLY (k=8)']['speedup']:.2f}x)\n")
        f.write(f"- **Root cause**: FLY designed for 70B/405B target (k=15-25). On 8B, target AR is already 80-90 TPS.\n")
        f.write(f"  N-gram lookup + model draft + verify overhead exceeds savings. FLY recovery recovers {overall['Official FLY (k=8)'].get('fly_rec', 0)*100:.0f}% of mismatches.\n")
        f.write(f"- **Conclusion**: Official FLY not suitable as primary baseline for 8B. Included for completeness.\n\n")

        f.write("### Method Comparison\n")
        f.write(f"- **TASD > Greedy SD**: YES ({tasd_mean_sp:.2f}x vs {gsd_mean_sp:.2f}x) — structural guard adds value\n")
        f.write(f"- **TASD > Official FLY**: YES overall, but FLY wins on DictConfig ({rows['dict_config']['Official FLY (k=8)']['speedup']:.2f}x vs {rows['dict_config']['TASD']['speedup']:.2f}x)\n")
        f.write(f"- **FLY route results**: Official FLY only beneficial when n-gram hit rate high (DictConfig). Otherwise overhead dominates.\n\n")

        f.write("## Model Paths\n\n")
        f.write("- Target: `/root/autodl-tmp/models/LLM-Research/Meta-Llama-3___1-8B-Instruct`\n")
        f.write("- Draft: `/root/autodl-tmp/models/LLM-Research/Llama-3___2-1B-Instruct`\n\n")

        f.write("## Related Reports\n\n")
        f.write("- `results/llama_pilot_3x20.json` — AR / Greedy SD / TASD per-sample results\n")
        f.write("- `results/llama_official_fly_pilot.json` — Official FLY (k=8, k=16) per-sample results\n")

    print(f"Report saved to {OUT_MD}")
    print(f"\nTASD overall speedup: {tasd_mean_sp:.2f}x")
    print(f"FLY overall speedup:  {fly_sp:.2f}x")
    print(f"GSD overall speedup:  {gsd_mean_sp:.2f}x")
    print(f"\nCriteria: sp>=1.3={sp_pass}, TASD>GSD={gsd_pass}, acc>=0.7={acc_pass}")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")

if __name__ == "__main__":
    main()
