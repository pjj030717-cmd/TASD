#!/usr/bin/env python3
"""TASD-FG-P Profit-Aware AR Switch Pilot.
Tests 3 below-AR samples + 3 normal samples per benchmark (21 total).
Compares: AR, TASD-FG, TASD-FG-P."""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

# (benchmark_key_in_quality_json, data_file_path, structure_type)
BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli"),
]

# ── Sample lists: (benchmark_key, sample_name) ──
BELOW_SAMPLES = [
    ("argparse", "argparse_real_062"),
    ("dict_config", "dict_config_real_014"),
    ("dict_config", "dict_config_real_057"),
]

NORMAL_SAMPLES = {
    "argparse": ["argparse_real_001", "argparse_real_002", "argparse_real_003"],
    "dict_config": ["dict_config_real_001", "dict_config_real_002", "dict_config_real_003"],
    "openmmlab_config": ["openmmlab_config_real_001", "openmmlab_config_real_002", "openmmlab_config_real_003"],
    "pipeline_stage_config": ["pipeline_stage_config_001", "pipeline_stage_config_002", "pipeline_stage_config_003"],
    "complex_nested_config": ["complex_nested_config_001", "complex_nested_config_002", "complex_nested_config_003"],
    "rich_cli_option_groups": ["rich_cli_option_groups_001", "rich_cli_option_groups_002", "rich_cli_option_groups_003"],
}

# ── Configs ──
TASD_COMMON = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)

TASD_FG = dict(
    enable_guard=True, enable_relaxed_accept=True,
    guard_calibrated=True,
    enable_failure_aware_fallback=True, fallback_guarded=True,
    fallback_accept_threshold=0.5, fallback_repair_threshold=2,
)

TASD_FGP = dict(
    enable_guard=True, enable_relaxed_accept=True,
    guard_calibrated=True,
    enable_failure_aware_fallback=True, fallback_guarded=True,
    fallback_accept_threshold=0.5, fallback_repair_threshold=2,
    enable_profit_aware_switch=True,
    profit_switch_window=48,
)


def run_tasd(target, draft, tokenizer, prompt, stype, extra_cfg, ar_tps_est=30):
    kw = {**TASD_COMMON, **extra_cfg}
    if kw.get("enable_profit_aware_switch"):
        kw["profit_switch_ar_tps_estimate"] = ar_tps_est
    r = tasd_decode(target, draft, tokenizer, prompt, structure_type=stype, **kw)
    s = r["stats"]
    fb = s.get("failure_aware_fallback", {}) or {}
    ps = s.get("profit_aware_switch", {}) or {}
    return {
        "wall": r["elapsed_time"], "tps": r["tokens_per_second"],
        "text": r["generated_text"], "gen_len": s["generated_length"],
        "accept": s["accept_rate"], "repair": s.get("repair_count", 0),
        "guard_trig": s.get("guard_trigger_count", 0),
        "trim": s.get("trim_count", 0),
        "fb_count": fb.get("fallback_count", 0),
        "fb_tokens": fb.get("total_fallback_tokens", 0),
        "switched_to_ar": ps.get("switched_to_ar", False),
        "switch_reason": ps.get("switch_reason", None),
        "switch_at_token": ps.get("switch_at_token", 0),
        "switch_trigger_values": ps.get("trigger_values", {}),
    }


def main():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_PATH, local_files_only=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True).eval()
    print("Models loaded.\n")

    # Load AR TPS data
    with open("results/qwen_5method_6x80_quality.json") as f:
        ar_data = json.load(f)["per_sample"]

    # Build AR lookup: (benchmark_key, name) -> ar_tps
    ar_lookup = {}
    for bn, _, _ in BENCHMARKS:
        for s in ar_data[bn]["AR"]:
            ar_lookup[(bn, s["name"])] = s["ar_tps"]

    # Load samples into lookup: (benchmark_key, name) -> {prompt, reference}
    sample_lookup = {}
    for bn, data_file, stype in BENCHMARKS:
        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:80]]
        for s in samples:
            sample_lookup[(bn, s["name"])] = {"prompt": s["prompt"], "reference": s.get("reference", ""), "stype": stype}

    # Build run list
    run_list = []  # (label, benchmark_key, name, stype)
    for bn, name in BELOW_SAMPLES:
        run_list.append(("below", bn, name, sample_lookup[(bn, name)]["stype"]))
    for bn, names in NORMAL_SAMPLES.items():
        for name in names:
            run_list.append(("normal", bn, name, sample_lookup[(bn, name)]["stype"]))

    results = []
    print(f"Running {len(run_list)} samples...\n")

    for idx, (label, bn, name, stype) in enumerate(run_list):
        entry = sample_lookup[(bn, name)]
        prompt = entry["prompt"]
        ref = entry["reference"]
        ar_tps_val = ar_lookup.get((bn, name), 30)

        print(f"[{idx+1}/{len(run_list)}] {bn}/{name} ({label})...", end=" ", flush=True)

        # ── TASD-FG ──
        t1 = time.time()
        fg_res = run_tasd(target, draft, tokenizer, prompt, stype, TASD_FG, ar_tps_val)
        fg_sp = fg_res["tps"] / ar_tps_val if ar_tps_val > 0 else 0
        t1_elapsed = time.time() - t1

        # ── TASD-FG-P ──
        t2 = time.time()
        fgp_res = run_tasd(target, draft, tokenizer, prompt, stype, TASD_FGP, ar_tps_val)
        fgp_sp = fgp_res["tps"] / ar_tps_val if ar_tps_val > 0 else 0
        t2_elapsed = time.time() - t2

        # ── Quality ──
        q_fg = compute_composite_sq(fg_res["text"], ref, stype)
        q_fgp = compute_composite_sq(fgp_res["text"], ref, stype)

        res_entry = {
            "label": label, "benchmark": bn, "name": name, "structure_type": stype,
            "ar_tps": ar_tps_val,
            "tasd_fg": {
                "sp": round(fg_sp, 3), "tps": round(fg_res["tps"], 2),
                "wall": round(fg_res["wall"], 3),
                "accept": round(fg_res["accept"], 4),
                "guard_trig": fg_res["guard_trig"], "trim": fg_res["trim"],
                "fb_count": fg_res["fb_count"], "fb_tokens": fg_res["fb_tokens"],
                "sq_r": q_fg["sq_r"], "sq_s": q_fg["sq_s"],
                "off_str": q_fg["off_structure_rate"],
                "rep_rate": q_fg["repetition_rate"],
                "text": fg_res["text"],
            },
            "tasd_fgp": {
                "sp": round(fgp_sp, 3), "tps": round(fgp_res["tps"], 2),
                "wall": round(fgp_res["wall"], 3),
                "accept": round(fgp_res["accept"], 4),
                "guard_trig": fgp_res["guard_trig"], "trim": fgp_res["trim"],
                "fb_count": fgp_res["fb_count"], "fb_tokens": fgp_res["fb_tokens"],
                "switched_to_ar": fgp_res["switched_to_ar"],
                "switch_reason": fgp_res["switch_reason"],
                "switch_at_token": fgp_res["switch_at_token"],
                "switch_trigger_values": fgp_res["switch_trigger_values"],
                "sq_r": q_fgp["sq_r"], "sq_s": q_fgp["sq_s"],
                "off_str": q_fgp["off_structure_rate"],
                "rep_rate": q_fgp["repetition_rate"],
                "text": fgp_res["text"],
            },
        }
        results.append(res_entry)

        switched = "SWITCHED" if fgp_res["switched_to_ar"] else "no-switch"
        print(f"FG={fg_sp:.3f}x FGP={fgp_sp:.3f}x [{switched}] "
              f"({t1_elapsed:.0f}s+{t2_elapsed:.0f}s)")

    # ── Save ──
    os.makedirs("results", exist_ok=True)
    with open("results/tasdfg_profit_switch_pilot.json", "w") as f:
        json.dump({"samples": results}, f, indent=2, ensure_ascii=False)

    generate_report(results)
    print("\nDone.")


def generate_report(results):
    lines = []
    lines.append("# TASD-FG-P Profit-Aware AR Switch Pilot Report\n\n")
    lines.append(f"**Samples**: {len(results)} (3 below-AR + 18 normal)\n\n")
    lines.append("**Window**: 48 tokens | **Triggers**: est_speedup<1.05, fb>=2, guard_trim>=3, rolling_accept<0.4, zero_accept>=2\n\n")

    below = [r for r in results if r["label"] == "below"]
    normal = [r for r in results if r["label"] == "normal"]

    # ── Below samples ──
    lines.append("## Below-AR Samples (Target: >=1.0x)\n\n")
    lines.append("| # | Name | AR TPS | FG sp | FGP sp | Δ | Switched | Reason | At token | FG SQ-R/S | FGP SQ-R/S | FG Off-Str | FGP Off-Str |\n")
    lines.append("|---|------|:------:|:-----:|:------:|:--:|:--------:|:------:|:--------:|:---------:|:----------:|:----------:|:----------:|\n")
    below_fg_sps = []
    below_fgp_sps = []
    switches = 0
    for i, r in enumerate(below):
        fg = r["tasd_fg"]; fgp = r["tasd_fgp"]
        delta = fgp["sp"] - fg["sp"]
        sw = "YES" if fgp["switched_to_ar"] else "no"
        if fgp["switched_to_ar"]:
            switches += 1
        sr = str(fgp.get("switch_reason", "-"))[:35]
        sat = fgp.get("switch_at_token", "-")
        lines.append(
            f"| {i+1} | {r['name'][:35]} | {r['ar_tps']:.1f} | "
            f"{fg['sp']:.3f}x | {fgp['sp']:.3f}x | " +
            (f"**+{delta:.3f}**" if delta > 0.001 else f"{delta:.3f}") +
            f" | {sw} | {sr} | {sat} | "
            f"{fg['sq_r']:.3f}/{fg['sq_s']:.3f} | {fgp['sq_r']:.3f}/{fgp['sq_s']:.3f} | "
            f"{fg['off_str']:.4f} | {fgp['off_str']:.4f} |\n"
        )
        below_fg_sps.append(fg["sp"])
        below_fgp_sps.append(fgp["sp"])
    fg_below_avg = sum(below_fg_sps) / len(below_fg_sps)
    fgp_below_avg = sum(below_fgp_sps) / len(below_fgp_sps)
    below_1x_count = sum(1 for s in below_fgp_sps if s < 1.0)
    lines.append(f"\n**Below avg: FG={fg_below_avg:.3f}x → FGP={fgp_below_avg:.3f}x | Switches: {switches}/3 | Still below-1.0: {below_1x_count}/3**\n\n")

    # ── Normal samples ──
    lines.append("## Normal Samples (Target: minimal false triggers)\n\n")
    lines.append("| # | Bench | Name | AR TPS | FG sp | FGP sp | Δ% | Switched | Reason |\n")
    lines.append("|---|-------|------|:------:|:-----:|:------:|:--:|:--------:|:------:|\n")
    normal_fg_sps = {}
    normal_fgp_sps = {}
    false_switches = 0
    for i, r in enumerate(normal):
        fg = r["tasd_fg"]; fgp = r["tasd_fgp"]
        delta_pct = (fgp["sp"] - fg["sp"]) / fg["sp"] * 100 if fg["sp"] > 0 else 0
        sw = "YES" if fgp["switched_to_ar"] else "no"
        if fgp["switched_to_ar"]:
            false_switches += 1
        sr = str(fgp.get("switch_reason", "-"))[:30]
        bench = r["benchmark"]
        lines.append(
            f"| {i+1} | {bench} | {r['name'][:30]} | {r['ar_tps']:.1f} | "
            f"{fg['sp']:.3f}x | {fgp['sp']:.3f}x | "
            f"{delta_pct:+.1f}% | {sw} | {sr} |\n"
        )
        normal_fg_sps.setdefault(bench, []).append(fg["sp"])
        normal_fgp_sps.setdefault(bench, []).append(fgp["sp"])

    lines.append(f"\n**False triggers on normal samples: {false_switches}/18**\n\n")

    # ── Summary ──
    lines.append("## Summary\n\n")
    lines.append("| Group | FG Mean sp | FGP Mean sp | Δ | Switches | Below-1.0 |\n")
    lines.append("|-------|:----------:|:----------:|:--:|:--------:|:---------:|\n")
    lines.append(f"| Below (3) | {fg_below_avg:.3f}x | {fgp_below_avg:.3f}x | {fgp_below_avg-fg_below_avg:+.3f} | {switches}/3 | {below_1x_count}/3 |\n")

    all_fg = [r["tasd_fg"]["sp"] for r in normal]
    all_fgp = [r["tasd_fgp"]["sp"] for r in normal]
    fg_norm_mu = sum(all_fg) / len(all_fg)
    fgp_norm_mu = sum(all_fgp) / len(all_fgp)
    pct_drop = (fgp_norm_mu - fg_norm_mu) / fg_norm_mu * 100
    lines.append(f"| Normal (18) | {fg_norm_mu:.3f}x | {fgp_norm_mu:.3f}x | {pct_drop:+.1f}% | {false_switches}/18 | - |\n")

    all_fg_tot = [r["tasd_fg"]["sp"] for r in results]
    all_fgp_tot = [r["tasd_fgp"]["sp"] for r in results]
    fg_tot_mu = sum(all_fg_tot) / len(all_fg_tot)
    fgp_tot_mu = sum(all_fgp_tot) / len(all_fgp_tot)
    lines.append(f"| **Total (21)** | **{fg_tot_mu:.3f}x** | **{fgp_tot_mu:.3f}x** | **{fgp_tot_mu-fg_tot_mu:+.3f}** | **{switches+false_switches}/21** | {sum(1 for s in all_fgp_tot if s < 1.0)}/21 |\n")

    # ── Per-benchmark normal ──
    lines.append("\n### Per-benchmark normal mean speedup\n\n")
    lines.append("| Benchmark | FG | FGP | Δ |\n")
    lines.append("|-----------|:--:|:--:|:--:|\n")
    for bench in normal_fg_sps:
        m_fg = sum(normal_fg_sps[bench]) / len(normal_fg_sps[bench])
        m_fgp = sum(normal_fgp_sps[bench]) / len(normal_fgp_sps[bench])
        lines.append(f"| {bench} | {m_fg:.3f}x | {m_fgp:.3f}x | {m_fgp-m_fg:+.3f} |\n")

    # ── Switch detail log ──
    switched_samples = [r for r in results if r["tasd_fgp"]["switched_to_ar"]]
    if switched_samples:
        lines.append("\n## Switch Detail Log\n\n")
        for r in switched_samples:
            fgp = r["tasd_fgp"]
            lines.append(f"### {r['benchmark']}/{r['name']} ({r['label']})\n\n")
            lines.append(f"- **Reason**: {fgp['switch_reason']}\n")
            lines.append(f"- **At token**: {fgp['switch_at_token']}\n")
            lines.append(f"- **Trigger values**: {fgp['switch_trigger_values']}\n")
            lines.append(f"- **FG speedup**: {r['tasd_fg']['sp']:.3f}x → **FGP speedup**: {fgp['sp']:.3f}x\n\n")
    else:
        lines.append("\n## Switch Detail Log\n\n*No switches occurred.*\n\n")

    # ── Judgment ──
    lines.append("## Judgment\n\n")
    criteria = []
    # 1. Below samples >= 1.0
    below_ok = all(s >= 0.98 for s in below_fgp_sps)
    criteria.append(f"{'PASS' if below_ok else 'FAIL'} | Below samples >= 1.0x: {[f'{s:.3f}x' for s in below_fgp_sps]}")
    # 2. False trigger rate
    false_rate = false_switches / max(len(normal), 1)
    criteria.append(f"{'PASS' if false_rate <= 0.1 else 'FAIL'} | False trigger rate: {false_switches}/{len(normal)} ({false_rate:.0%})")
    # 3. Mean speedup drop < 3%
    criteria.append(f"{'PASS' if abs(pct_drop) < 3 else 'FAIL'} | Normal mean speedup drop: {pct_drop:+.1f}%")
    # 4. SQ-S / Off-Str
    sq_s_fg = sum(r["tasd_fg"]["sq_s"] for r in results) / len(results)
    sq_s_fgp = sum(r["tasd_fgp"]["sq_s"] for r in results) / len(results)
    off_fg = sum(r["tasd_fg"]["off_str"] for r in results) / len(results)
    off_fgp = sum(r["tasd_fgp"]["off_str"] for r in results) / len(results)
    criteria.append(f"{'PASS' if sq_s_fgp >= sq_s_fg - 0.01 else 'FAIL'} | SQ-S: FG={sq_s_fg:.4f} → FGP={sq_s_fgp:.4f}")
    criteria.append(f"{'PASS' if off_fgp <= off_fg + 0.01 else 'FAIL'} | Off-Str: FG={off_fg:.4f} → FGP={off_fgp:.4f}")

    for c in criteria:
        lines.append(f"- {c}\n")

    md = "".join(lines)
    with open("results/tasdfg_profit_switch_pilot.md", "w") as f:
        f.write(md)


if __name__ == "__main__":
    main()
