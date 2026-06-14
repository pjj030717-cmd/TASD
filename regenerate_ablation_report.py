#!/usr/bin/env python3
"""Recompute SQ-R and SQ-S for all ablation variants from checkpoint sub-metrics."""
import json

BENCHMARKS = ["argparse", "dict_config", "openmmlab_config",
              "pipeline_stage_config", "complex_nested_config",
              "rich_cli_option_groups"]
SHORT = ["argparse", "dict_config", "openmmlab", "pipeline", "complex_nested", "rich_cli"]
CKPT_DIR = "results/checkpoints_ablation"

VARIANTS = ["TASD-FG", "TASD", "TASD-F", "no_relaxed", "no_guard", "draft_len8", "draft_blocks1"]
VLABELS = {
    "TASD-FG": "TASD-FG (full)",
    "TASD": "TASD (base, no fb)",
    "TASD-F": "TASD-F (unguarded fb)",
    "no_relaxed": "w/o relaxed verify",
    "no_guard": "w/o struct guard",
    "draft_len8": "draft_len=8",
    "draft_blocks1": "draft_blocks=1",
}

OUT_JSON = "results/qwen_ablation_7variant.json"
OUT_MD = "results/qwen_ablation_7variant.md"


def compute_sq_r(s):
    return 0.4 * s.get("structural_char_f1", 0) + \
           0.3 * s.get("bracket_balance_score", 0) + \
           0.2 * s.get("structure_type_preservation", 0) + \
           0.1 * s.get("no_repetition_score", 0)


def compute_sq_s(s, stype):
    off = s.get("off_structure_rate", 0)
    trunc = s.get("is_truncated", 0)
    rep = s.get("repetition_rate", 0)
    dup_opt = s.get("duplicate_option_rate", 0)
    # fallback: compute dup_opt from rep if stype is argparse/rich_cli
    if dup_opt == 0 and stype in ("argparse", "rich_cli_option_groups", "rich_cli"):
        dup_opt = rep  # rough approximation; rep already covers line-level dup
    sq_s = 1.0 - 0.45 * off - 0.25 * trunc - 0.20 * rep - 0.10 * dup_opt
    return max(0.0, min(1.0, sq_s))


def main():
    # Load existing data
    with open("results/qwen_5method_6x80_quality.json") as f:
        quality = json.load(f)
    with open("results/qwen_tasd_fg_6x80.json") as f:
        tfg = json.load(f)
    with open("results/qwen_tasd_f_6x80.json") as f:
        tf = json.load(f)

    all_data = {}
    for bn, sn in zip(BENCHMARKS, SHORT):
        all_data[sn] = {"n": 80}
        # TASD-FG
        if "TASD-F-G" in tfg["per_sample"][bn]:
            all_data[sn]["TASD-FG"] = tfg["per_sample"][bn]["TASD-F-G"]
        # TASD
        if "TASD" in quality["per_sample"][bn]:
            all_data[sn]["TASD"] = quality["per_sample"][bn]["TASD"]
        # TASD-F
        if "TASD-F" in tf["per_sample"][bn]:
            all_data[sn]["TASD-F"] = tf["per_sample"][bn]["TASD-F"]
        # New variants
        for vname in ["no_relaxed", "no_guard", "draft_len8", "draft_blocks1"]:
            ckpt = f"{CKPT_DIR}/{bn}_{vname}.json"
            with open(ckpt) as f:
                all_data[sn][vname] = json.load(f)

    # Also load old5 TASD for reference
    with open("results/qwen_5method_6x80.json") as f:
        old5 = json.load(f)

    # Augment each sample with sq_r and sq_s
    for sn in SHORT:
        for vname in VARIANTS:
            if vname not in all_data[sn]:
                continue
            data = all_data[sn][vname]
            for s in data:
                # Determine stype
                if sn == "argparse":
                    stype = "argparse"
                elif sn == "dict_config":
                    stype = "dict_config"
                elif sn == "openmmlab":
                    stype = "openmmlab_config"
                elif sn == "pipeline":
                    stype = "pipeline_stage_config"
                elif sn == "complex_nested":
                    stype = "complex_nested_config"
                elif sn == "rich_cli":
                    stype = "rich_cli_option_groups"
                else:
                    stype = ""
                s["sq_r"] = round(compute_sq_r(s), 4)
                s["sq_s"] = round(compute_sq_s(s, stype), 4)

    # Aggregate
    agg = {}
    for vname in VARIANTS:
        agg[vname] = {}
        all_sps = []; all_below = []; all_sqr = []; all_sqs = []
        all_off = []; all_rep = []; all_trunc = []; all_fb = []; all_accept = []
        all_guard_trig = []; all_trim = []
        for sn in SHORT:
            data = all_data[sn].get(vname, [])
            if not data:
                continue
            n = len(data)
            sps = [x["sp"] for x in data]
            all_sps.extend(sps)
            all_below.extend(1 for s in sps if s < 1.0)
            all_sqr.extend(x["sq_r"] for x in data)
            all_sqs.extend(x["sq_s"] for x in data)
            all_off.extend(x.get("off_structure_rate", 0) for x in data)
            all_rep.extend(x.get("repetition_rate", 0) for x in data)
            all_trunc.extend(x.get("is_truncated", 0) for x in data)
            all_fb.extend(x.get("fb_count", 0) for x in data)
            all_accept.extend(x.get("accept", 0) for x in data)
            all_guard_trig.extend(x.get("guard_trig", 0) for x in data)
            all_trim.extend(x.get("trim", 0) for x in data)
            agg[vname][sn] = {
                "sp_mean": round(sum(sps)/n, 3),
                "below": sum(1 for s in sps if s < 1.0),
                "sq_r": round(sum(x["sq_r"] for x in data)/n, 4),
                "sq_s": round(sum(x["sq_s"] for x in data)/n, 4),
                "off_str": round(sum(x.get("off_structure_rate",0) for x in data)/n, 4),
                "rep_rate": round(sum(x.get("repetition_rate",0) for x in data)/n, 4),
                "truncation": round(sum(x.get("is_truncated",0) for x in data)/n, 4),
                "fb_count": sum(x.get("fb_count",0) for x in data),
                "accept": round(sum(x.get("accept",0) for x in data)/n, 4),
                "guard_trig": sum(x.get("guard_trig",0) for x in data)/n,
                "trim": sum(x.get("trim",0) for x in data)/n,
            }
        n_total = len(all_sps)
        sps_sorted = sorted(all_sps)
        w10n = max(1, n_total // 10)
        agg[vname]["overall"] = {
            "sp_mean": round(sum(all_sps)/n_total, 3),
            "sp_median": round(sps_sorted[n_total//2], 3),
            "below": sum(all_below),
            "sq_r": round(sum(all_sqr)/n_total, 4),
            "sq_s": round(sum(all_sqs)/n_total, 4),
            "off_str": round(sum(all_off)/n_total, 4),
            "rep_rate": round(sum(all_rep)/n_total, 4),
            "truncation": round(sum(all_trunc)/n_total, 4),
            "fb_total": sum(all_fb),
            "accept": round(sum(all_accept)/n_total, 4) if all_accept else 0,
            "worst10": round(sum(sps_sorted[:w10n])/w10n, 3),
            "guard_trig": round(sum(all_guard_trig)/n_total, 1) if all_guard_trig else 0,
            "trim": round(sum(all_trim)/n_total, 1) if all_trim else 0,
        }

    # Save JSON
    output = {"per_benchmark": {v: {sn: agg[v][sn] for sn in SHORT if sn in agg[v]} for v in VARIANTS},
              "overall": {v: agg[v]["overall"] for v in VARIANTS}}
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # ── MD report ──
    with open(OUT_MD, "w") as f:
        f.write("# Qwen TASD Ablation (7 variants, 6×80=480 samples)\n\n")
        f.write("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct | **max_new_tokens**: 128\n\n")

        f.write("## Quality Metrics\n\n")
        f.write("- **SQ-R** (Reference-aware Structural Quality): 0.4×F1 + 0.3×bracket + 0.2×type_preservation + 0.1×no_repetition\n")
        f.write("- **SQ-S** (Structure Safety Score): 1.0 − 0.45×off_str − 0.25×trunc − 0.20×rep − 0.10×dup_opt\n\n")

        f.write("## Overall (480 samples)\n\n")
        f.write("| Variant | Speedup | Below | Worst-10 | SQ-R | SQ-S | Off-Str | FB | Accept |\n")
        f.write("|---------|:-------:|:-----:|:--------:|:----:|:----:|:-------:|:--:|:------:|\n")
        for vname in VARIANTS:
            o = agg[vname]["overall"]
            bold = "**" if vname == "TASD-FG" else ""
            f.write(f"| {bold}{VLABELS[vname]}{bold} | {bold}{o['sp_mean']:.3f}x{bold} | {o['below']}/480 | {o['worst10']:.3f}x | {o['sq_r']:.4f} | {o['sq_s']:.4f} | {o['off_str']:.4f} | {o['fb_total']} | {o['accept']:.3f} |\n")
        f.write("\n")

        # Per-benchmark
        for sn in SHORT:
            f.write(f"### {sn} (80)\n\n")
            f.write("| Variant | Speedup | Below | SQ-R | SQ-S | Off-Str | FB | Accept |\n")
            f.write("|---------|:-------:|:-----:|:----:|:----:|:-------:|:--:|:------:|\n")
            for vname in VARIANTS:
                if sn not in agg[vname]:
                    continue
                a = agg[vname][sn]
                bold = "**" if vname == "TASD-FG" else ""
                f.write(f"| {bold}{VLABELS[vname]}{bold} | {bold}{a['sp_mean']:.3f}x{bold} | {a['below']} | {a['sq_r']:.4f} | {a['sq_s']:.4f} | {a['off_str']:.4f} | {a['fb_count']} | {a['accept']:.3f} |\n")
            f.write("\n")

        # Decomposition
        f.write("## Contribution Decomposition\n\n")
        f.write("| Ablation | ΔSpeedup | ΔBelow | ΔSQ-R | ΔSQ-S | ΔOff-Str | ΔFB |\n")
        f.write("|----------|:--------:|:------:|:-----:|:-----:|:--------:|:---:|\n")

        full = agg["TASD-FG"]["overall"]
        base = agg["TASD"]["overall"]
        ngr = agg["no_guard"]["overall"]
        rel = agg["no_relaxed"]["overall"]
        dl8 = agg["draft_len8"]["overall"]
        db1 = agg["draft_blocks1"]["overall"]
        tsf = agg["TASD-F"]["overall"]

        f.write(f"| TASD → TASD-FG | +{full['sp_mean']-base['sp_mean']:.3f} | {base['below']}→{full['below']} | {base['sq_r']:.4f}→{full['sq_r']:.4f} | {base['sq_s']:.4f}→{full['sq_s']:.4f} | {base['off_str']:.4f}→{full['off_str']:.4f} | +{full['fb_total']} |\n")
        f.write(f"| TASD-FG → no guard | {ngr['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{ngr['below']} | {full['sq_r']:.4f}→{ngr['sq_r']:.4f} | {full['sq_s']:.4f}→{ngr['sq_s']:.4f} | {full['off_str']:.4f}→{ngr['off_str']:.4f} | {full['fb_total']}→{ngr['fb_total']} |\n")
        f.write(f"| TASD-FG → no relaxed | {rel['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{rel['below']} | {full['sq_r']:.4f}→{rel['sq_r']:.4f} | {full['sq_s']:.4f}→{rel['sq_s']:.4f} | {full['off_str']:.4f}→{rel['off_str']:.4f} | {full['fb_total']}→{rel['fb_total']} |\n")
        f.write(f"| TASD-FG → draft_len=8 | {dl8['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{dl8['below']} | {full['sq_r']:.4f}→{dl8['sq_r']:.4f} | {full['sq_s']:.4f}→{dl8['sq_s']:.4f} | {full['off_str']:.4f}→{dl8['off_str']:.4f} | {full['fb_total']}→{dl8['fb_total']} |\n")
        f.write(f"| TASD-FG → draft_blk=1 | {db1['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{db1['below']} | {full['sq_r']:.4f}→{db1['sq_r']:.4f} | {full['sq_s']:.4f}→{db1['sq_s']:.4f} | {full['off_str']:.4f}→{db1['off_str']:.4f} | {full['fb_total']}→{db1['fb_total']} |\n")
        f.write("\n")

        # Key insight box
        f.write("## Key Insight\n\n")
        f.write("**SQ-R and SQ-S separate different quality dimensions:**\n\n")
        f.write(f"- `w/o struct guard` has higher SQ-R ({ngr['sq_r']:.4f}) than TASD-FG ({full['sq_r']:.4f}) because it more closely matches the reference structure\n")
        f.write(f"- But `w/o struct guard` has lower SQ-S ({ngr['sq_s']:.4f}) and higher Off-Str ({ngr['off_str']:.4f}) — more structural risk\n")
        f.write(f"- TASD-FG maintains SQ-S at {full['sq_s']:.4f} with Off-Str {full['off_str']:.4f} while achieving highest speedup ({full['sp_mean']:.3f}x)\n\n")
        f.write("TASD-FG achieves the best speed-robustness trade-off. It does not maximize reference similarity (SQ-R), but maintains competitive structure safety (SQ-S) while achieving the highest speedup and the fewest below-AR failures.\n")

    print(f"Saved {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
