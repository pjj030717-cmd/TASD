#!/usr/bin/env python3
"""Merge Qwen 128-token ablation data and generate 7-variant report."""
import json

BENCHMARKS = ["argparse", "dict_config", "openmmlab_config",
              "pipeline_stage_config", "complex_nested_config",
              "rich_cli_option_groups"]
SHORT = ["argparse", "dict_config", "openmmlab", "pipeline", "complex_nested", "rich_cli"]
CKPT_DIR = "results/checkpoints_ablation"

OUT_JSON = "results/qwen_ablation_7variant.json"
OUT_MD = "results/qwen_ablation_7variant.md"

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

def main():
    # Load existing data
    with open("results/qwen_5method_6x80_quality.json") as f:
        quality = json.load(f)
    with open("results/qwen_tasd_fg_6x80.json") as f:
        tfg = json.load(f)
    with open("results/qwen_tasd_f_6x80.json") as f:
        tf = json.load(f)
    with open("results/qwen_5method_6x80.json") as f:
        old5 = json.load(f)

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

    # Aggregate per-variant
    agg = {}
    for vname in VARIANTS:
        agg[vname] = {}
        all_sps = []
        all_below = []
        all_sq = []
        all_off = []
        all_rep = []
        all_trunc = []
        all_fb = []
        all_accept = []
        all_guard_trig = []
        all_trim = []
        for bn, sn in zip(BENCHMARKS, SHORT):
            data = all_data[sn].get(vname, [])
            if not data:
                continue
            sps = [x["sp"] for x in data]
            all_sps.extend(sps)
            all_below.extend(1 for s in sps if s < 1.0)
            all_sq.extend(x.get("composite_sq", 0) for x in data)
            all_off.extend(x.get("off_structure_rate", 0) for x in data)
            all_rep.extend(x.get("repetition_rate", 0) for x in data)
            all_trunc.extend(1 if x.get("is_truncated", False) else 0 for x in data)
            all_fb.extend(x.get("fb_count", 0) for x in data)
            all_accept.extend(x.get("accept", 0) for x in data)
            all_guard_trig.extend(x.get("guard_trig", 0) for x in data)
            all_trim.extend(x.get("trim", 0) for x in data)
            agg[vname][sn] = {
                "sp_mean": round(sum(sps)/len(sps), 3),
                "below": sum(1 for s in sps if s < 1.0),
                "sq": round(sum(x.get("composite_sq",0) for x in data)/len(data), 4),
                "off_str": round(sum(x.get("off_structure_rate",0) for x in data)/len(data), 4),
                "rep_rate": round(sum(x.get("repetition_rate",0) for x in data)/len(data), 4),
                "truncation": round(sum(1 for x in data if x.get("is_truncated",False))/len(data), 4),
                "fb_count": sum(x.get("fb_count",0) for x in data),
                "accept": round(sum(x.get("accept",0) for x in data)/len(data), 4) if any(x.get("accept",0) for x in data) else 0,
                "guard_trig": sum(x.get("guard_trig",0) for x in data)/len(data),
                "trim": sum(x.get("trim",0) for x in data)/len(data),
            }
        sps_sorted = sorted(all_sps)
        w10n = max(1, len(sps_sorted)//10)
        agg[vname]["overall"] = {
            "sp_mean": round(sum(all_sps)/len(all_sps), 3),
            "sp_median": round(sps_sorted[len(sps_sorted)//2], 3),
            "below": sum(all_below),
            "sq": round(sum(all_sq)/len(all_sq), 4),
            "off_str": round(sum(all_off)/len(all_off), 4),
            "rep_rate": round(sum(all_rep)/len(all_rep), 4),
            "truncation": round(sum(all_trunc)/len(all_trunc), 4),
            "fb_total": sum(all_fb),
            "accept": round(sum(all_accept)/len(all_accept), 4),
            "worst10": round(sum(sps_sorted[:w10n])/w10n, 3),
            "guard_trig": round(sum(all_guard_trig)/len(all_guard_trig), 1),
            "trim": round(sum(all_trim)/len(all_trim), 1),
        }

    # Save JSON
    output = {"per_benchmark": {v: {sn: agg[v][sn] for sn in SHORT if sn in agg[v]} for v in VARIANTS},
              "overall": {v: agg[v]["overall"] for v in VARIANTS}}
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # MD report
    with open(OUT_MD, "w") as f:
        f.write("# Qwen TASD Ablation (7 variants, 6×80=480 samples)\n\n")
        f.write("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct | **max_new_tokens**: 128\n\n")

        f.write("## Overall (480 samples)\n\n")
        f.write("| Variant | Speedup | Median | Below | Worst-10 | SQ | Off-Str | Rep | Trunc | FB | Accept | Guard | Trim |\n")
        f.write("|---------|:-------:|:------:|:-----:|:--------:|:--:|:-------:|:---:|:-----:|:--:|:------:|:-----:|:----:|\n")
        for vname in VARIANTS:
            o = agg[vname]["overall"]
            bold = "**" if vname == "TASD-FG" else ""
            f.write(f"| {bold}{VLABELS[vname]}{bold} | {bold}{o['sp_mean']:.3f}x{bold} | {o['sp_median']:.3f}x | {o['below']}/480 | {o['worst10']:.3f}x | {o['sq']:.4f} | {o['off_str']:.4f} | {o['rep_rate']:.4f} | {o['truncation']:.4f} | {o['fb_total']} | {o['accept']:.3f} | {o['guard_trig']:.0f} | {o['trim']:.0f} |\n")
        f.write("\n")

        # Per-benchmark
        for sn in SHORT:
            f.write(f"### {sn} (80)\n\n")
            f.write("| Variant | Speedup | Below | SQ | Off-Str | Rep | Trunc | FB | Accept | Guard | Trim |\n")
            f.write("|---------|:-------:|:-----:|:--:|:-------:|:---:|:-----:|:--:|:------:|:-----:|:----:|\n")
            for vname in VARIANTS:
                if sn not in agg[vname]:
                    continue
                a = agg[vname][sn]
                bold = "**" if vname == "TASD-FG" else ""
                f.write(f"| {bold}{VLABELS[vname]}{bold} | {bold}{a['sp_mean']:.3f}x{bold} | {a['below']} | {a['sq']:.4f} | {a['off_str']:.4f} | {a['rep_rate']:.4f} | {a['truncation']:.4f} | {a['fb_count']} | {a['accept']:.3f} | {a['guard_trig']:.0f} | {a['trim']:.0f} |\n")
            f.write("\n")

        # Decomposition analysis
        f.write("## Contribution Decomposition\n\n")
        f.write("| Ablation | ΔSpeedup | ΔBelow | ΔOff-Str | ΔSQ | ΔFB | Interpretation |\n")
        f.write("|----------|:--------:|:------:|:--------:|:---:|:---:|---------------|\n")

        base = agg["TASD"]["overall"]
        full = agg["TASD-FG"]["overall"]
        rel = agg["no_relaxed"]["overall"]
        ngr = agg["no_guard"]["overall"]
        dl8 = agg["draft_len8"]["overall"]
        db1 = agg["draft_blocks1"]["overall"]

        f.write(f"| TASD → TASD-FG | +{full['sp_mean']-base['sp_mean']:.3f} | {base['below']}→{full['below']} | {base['off_str']:.4f}→{full['off_str']:.4f} | {base['sq']:.4f}→{full['sq']:.4f} | {full['fb_total']} | failure-aware fb |\n")
        f.write(f"| TASD-FG → no relaxed | {rel['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{rel['below']} | {full['off_str']:.4f}→{rel['off_str']:.4f} | {full['sq']:.4f}→{rel['sq']:.4f} | {full['fb_total']}→{rel['fb_total']} | top-k=1 strict |\n")
        f.write(f"| TASD-FG → no guard | {ngr['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{ngr['below']} | {full['off_str']:.4f}→{ngr['off_str']:.4f} | {full['sq']:.4f}→{ngr['sq']:.4f} | {full['fb_total']}→{ngr['fb_total']} | structural guard |\n")
        f.write(f"| TASD-FG → draft_len=8 | {dl8['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{dl8['below']} | {full['off_str']:.4f}→{dl8['off_str']:.4f} | {full['sq']:.4f}→{dl8['sq']:.4f} | {full['fb_total']}→{dl8['fb_total']} | shorter draft |\n")
        f.write(f"| TASD-FG → draft_blk=1 | {db1['sp_mean']-full['sp_mean']:.3f} | {full['below']}→{db1['below']} | {full['off_str']:.4f}→{db1['off_str']:.4f} | {full['sq']:.4f}→{db1['sq']:.4f} | {full['fb_total']}→{db1['fb_total']} | single block |\n")

    print(f"Saved {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
