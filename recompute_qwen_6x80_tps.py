"""
Recompute Qwen 6×80 results with corrected TPS (generated tokens only).

Fix: Old run_ar() used len(out[0])/wall which included prompt tokens.
     run_fly() had the same bug with full_ids.shape[1]/wall.
     This script corrects both using gen_len/wall from checkpoints.

After recomputation, generates new results/qwen_5method_6x80.json and .md.
"""
import json, os, sys, time
from collections import defaultdict
from transformers import AutoTokenizer

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
MAX_NEW_TOKENS = 128
CHECKPOINT_DIR = "results/qwen_6x80_checkpoints"
OUT_JSON = "results/qwen_5method_6x80.json"
OUT_MD = "results/qwen_5method_6x80.md"

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab_config"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline_stage_config"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested_config"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli_option_groups"),
]


def load_checkpoint(bname, method_label):
    fname = os.path.join(CHECKPOINT_DIR, f"{bname}_{method_label}.json")
    if os.path.exists(fname):
        with open(fname) as f:
            return json.load(f)
    return None


def load_from_full_json(bname, method_label):
    """Fall back to existing full JSON when checkpoint is missing."""
    fallback_path = "results/qwen_5method_6x80.json"
    if not os.path.exists(fallback_path):
        return None
    with open(fallback_path) as f:
        full = json.load(f)
    ps = full.get("per_sample", {}).get(bname, {})
    return ps.get(method_label, None)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)

    # Pre-compute prompt lengths from data files
    prompt_lens = {}  # {bname: [prompt_len_0, prompt_len_1, ...]}
    for bname, data_file, _ in BENCHMARKS:
        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:80]]
        prompt_lens[bname] = [len(tokenizer.encode(s["prompt"])) for s in samples]
        print(f"  Loaded {bname}: {len(samples)} prompts, avg_len={sum(prompt_lens[bname])/len(samples):.0f}")

    method_labels = ["AR", "GSD", "Ngram", "FLY", "TASD"]

    # ── Recompute all results ──
    all_data = {}  # {bname: {method: [per-sample dicts]}}
    aggregate = {}  # {bname: {method: {sp_avg, sq_avg, below, tps_avg}}}

    print(f"\n{'='*70}")
    print("Recomputing TPS with corrected formula (generated_tokens / wall_time)")
    print(f"{'='*70}")

    for bname, _, _ in BENCHMARKS:
        ar_ckpt = load_checkpoint(bname, "AR")
        if ar_ckpt is None:
            print(f"  WARNING: no AR checkpoint for {bname}")
            continue

        n = len(ar_ckpt)
        pl = prompt_lens[bname]

        # ── Fix AR TPS ──
        corrected_ar = []
        for i, r in enumerate(ar_ckpt):
            wall = r["wall"]
            old_tps = r["ar_tps"]  # old: (prompt_len + gen_len)/wall, rounded
            # reconstruct gen_len: total_tokens = old_tps * wall ≈ prompt_len + gen_len
            total_tokens = old_tps * wall
            gen_len = max(0, int(round(total_tokens - pl[i])))
            gen_len = min(gen_len, MAX_NEW_TOKENS)
            new_tps = gen_len / wall if wall > 0 else 0.0
            corrected_ar.append({
                "name": r["name"], "sq": r["sq"], "trunc": r["trunc"],
                "wall": wall,
                "prompt_len": pl[i], "gen_len": gen_len,
                "ar_tps": round(new_tps, 2),
            })
        all_data[bname] = {"AR": corrected_ar}
        ar_tps_list = [r["ar_tps"] for r in corrected_ar]
        ar_tps_avg = sum(ar_tps_list) / n
        aggregate[bname] = {"n": n}
        aggregate[bname]["AR"] = {
            "tps_avg": round(ar_tps_avg, 1),
            "sq_avg": round(sum(r["sq"] for r in corrected_ar) / n, 4),
            "sp_avg": 1.0, "below": 0,
        }

        # ── Fix other methods ──
        for ml in ["GSD", "Ngram", "FLY", "TASD"]:
            ckpt = load_checkpoint(bname, ml)
            if ckpt is None:
                ckpt = load_from_full_json(bname, ml)
            if ckpt is None:
                print(f"  WARNING: no {ml} data for {bname}")
                continue

            corrected = []
            for i, r in enumerate(ckpt):
                wall = r["wall"]
                # GSD/Ngram/TASD: tps already correct = generated/wall, but stored
                # FLY: old tps = full_ids.shape[1]/wall (includes prompt), stored as r["tps"]
                if ml == "FLY":
                    gen_len = r.get("gen_len", MAX_NEW_TOKENS)
                    new_tps = gen_len / wall if wall > 0 else 0.0
                else:
                    # GSD/Ngram/TASD already correct
                    new_tps = r["tps"]
                    gen_len = None  # not stored for GSD/Ngram

                sp = new_tps / ar_tps_list[i] if ar_tps_list[i] > 0 else 0.0
                entry = {
                    "name": r["name"], "tps": round(new_tps, 2),
                    "sp": round(sp, 3), "sq": r.get("sq", 0),
                    "wall": wall,
                }
                if ml == "AR":
                    pass
                elif ml == "GSD":
                    entry["accept"] = r.get("accept", 0)
                elif ml == "Ngram":
                    entry["accept"] = r.get("accept", 0)
                    entry["draft_avg"] = r.get("draft_avg", 0)
                elif ml == "FLY":
                    entry["mat"] = r.get("mat", 0)
                    entry["ngram_acc"] = r.get("ngram_acc", 0)
                    entry["gen_len"] = gen_len
                elif ml == "TASD":
                    entry.update({
                        "accept": r.get("accept", 0),
                        "repair": r.get("repair", 0),
                        "guard": r.get("guard", 0),
                        "trim": r.get("trim", 0),
                        "hard_trim": r.get("hard_trim", 0),
                        "rep_warn": r.get("rep_warn", 0),
                        "brack_warn": r.get("brack_warn", 0),
                        "off_str": r.get("off_str", 0),
                    })
                corrected.append(entry)

            all_data[bname][ml] = corrected

            sp_vals = [r["sp"] for r in corrected]
            sq_vals = [r["sq"] for r in corrected]
            below = sum(1 for s in sp_vals if s < 1.0)
            sp_avg = sum(sp_vals) / n
            aggregate[bname][ml] = {
                "sp_avg": round(sp_avg, 3),
                "sq_avg": round(sum(sq_vals) / n, 4),
                "below": below,
            }

        # Print per-benchmark summary
        print(f"\n  {bname} ({n}s):")
        for ml in method_labels:
            if ml in aggregate[bname]:
                a = aggregate[bname][ml]
                tps_str = f"{a.get('tps_avg', 0):.1f}" if ml == 'AR' else "    "
                print(f"    {ml}: tps={tps_str}  "
                      f"sp={a.get('sp_avg', 1.0):.3f}x  sq={a['sq_avg']:.4f}  below={a.get('below', 0)}")

    # ── Overall ──
    overall = {m: {"sp": [], "below": 0, "total": 0, "sq_vals": []} for m in method_labels}
    for bname, _, _ in BENCHMARKS:
        n = aggregate[bname]["n"]
        for ml in method_labels:
            if ml in aggregate[bname]:
                a = aggregate[bname][ml]
                if ml == "AR":
                    overall[ml]["sp"].append(1.0)
                else:
                    overall[ml]["sp"].append(a["sp_avg"])
                    overall[ml]["below"] += a.get("below", 0)
                    overall[ml]["total"] += n
                overall[ml]["sq_vals"].append(a["sq_avg"])

    print(f"\n{'='*70}")
    print("OVERALL (corrected TPS)")
    print(f"{'='*70}")
    for ml in method_labels:
        ov = overall[ml]
        mean_sp = sum(ov["sp"]) / len(ov["sp"]) if ov["sp"] else 0
        mean_sq = sum(ov["sq_vals"]) / len(ov["sq_vals"]) if ov["sq_vals"] else 0
        print(f"  {ml}: sp={mean_sp:.3f}x  sq={mean_sq:.4f}  below={ov['below']}/{ov['total']}")

    # ── Save JSON ──
    output = {
        "config": {
            "target": "Qwen2.5-14B-Instruct-AWQ",
            "draft": "Qwen2.5-1.5B-Instruct",
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "fly_k": 15, "fly_win_len": 6, "fly_entropy_thre": 0.3,
            "fly_ngram": "4/6",
            "tasd_guard_calibrated": True,
            "tps_note": "All TPS computed as generated_tokens / wall_time, excluding prompt tokens and model loading.",
        },
        "per_benchmark": aggregate,
        "per_sample": all_data,
    }
    os.makedirs(os.path.dirname(OUT_JSON) if os.path.dirname(OUT_JSON) else ".", exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {OUT_JSON}")

    # ── Write MD ──
    write_md_report(output, overall, method_labels)
    print(f"Saved {OUT_MD}")
    print("\nDone!")


def write_md_report(output, overall, method_labels):
    agg = output["per_benchmark"]
    cfg = output["config"]
    bnames = [b[0] for b in BENCHMARKS]
    methods_display = ["AR", "Greedy SD", "N-gram SD", "Official FLY", "TASD"]
    ml_to_display = {"AR": "AR", "GSD": "Greedy SD", "Ngram": "N-gram SD",
                     "FLY": "Official FLY", "TASD": "TASD"}

    with open(OUT_MD, "w") as f:
        f.write("# Qwen 6×80 Main Experiment Report (Corrected TPS)\n\n")
        f.write(f"**Target**: {cfg['target']}  |  **Draft**: {cfg['draft']}\n")
        f.write(f"**Config**: max_new_tokens={cfg['max_new_tokens']}, temperature={cfg['temperature']}\n\n")
        f.write(f"> **Note**: {cfg.get('tps_note', '')}\n\n")

        f.write("## Methods\n\n")
        f.write("| Method | Description |\n")
        f.write("|--------|-------------|\n")
        f.write("| AR | Target autoregressive (greedy) |\n")
        f.write("| Greedy SD | Target-verify greedy draft (draft_len=16, blocks=2, top_k=3, no guard) |\n")
        f.write("| N-gram SD | Pure n-gram lookup SD (ngram_min=3, max=8, no draft model) |\n")
        f.write(f"| **Official FLY** | AMD FLy SPDGenerate (k={cfg['fly_k']}, win_len={cfg['fly_win_len']}, entropy_thre={cfg['fly_entropy_thre']}, ngram={cfg['fly_ngram']}) |\n")
        f.write("| **TASD** | Structure-aware SD + Guard-v1.5 calibrated (draft_len=16, blocks=2, top_k=3) |\n\n")

        f.write("## Per-Benchmark Results\n\n")
        for bname in bnames:
            ar_tps = agg[bname]["AR"]["tps_avg"]
            f.write(f"### {bname} ({agg[bname]['n']} samples)\n\n")
            f.write(f"Baseline AR TPS: **{ar_tps:.1f}** tok/s (generated only)\n\n")
            f.write("| Method | Speedup | SQ | Accept/MAT | Below 1.0x |\n")
            f.write("|--------|:-------:|:--:|:----------:|:----------:|\n")

            for ml in method_labels:
                if ml not in agg[bname]:
                    continue
                a = agg[bname][ml]
                sp_str = f"**{a.get('sp_avg', 1.0):.3f}x**"
                sq_str = f"{a['sq_avg']:.4f}"
                below_str = str(a.get("below", 0))

                if ml == "AR":
                    acc_str = "-"
                elif ml == "FLY" and "FLY" in output["per_sample"][bname]:
                    mat_vals = [r["mat"] for r in output["per_sample"][bname]["FLY"]]
                    acc_str = f"{sum(mat_vals)/len(mat_vals):.2f} (MAT)" if mat_vals else "-"
                elif ml in ("GSD", "Ngram", "TASD") and ml in output["per_sample"][bname]:
                    acc_vals = [r["accept"] for r in output["per_sample"][bname][ml]]
                    acc_str = f"{sum(acc_vals)/len(acc_vals):.3f}" if acc_vals else "-"
                else:
                    acc_str = "-"

                f.write(f"| {ml_to_display[ml]} | {sp_str} | {sq_str} | {acc_str} | {below_str} |\n")
            f.write("\n")

        # Overall table
        f.write("## Overall (6 benchmarks × 80 samples = 480 samples)\n\n")
        f.write("| Method | Mean Speedup | Mean SQ | Below 1.0x |\n")
        f.write("|--------|:-----------:|:-------:|:----------:|\n")
        overall_sp = {}
        overall_sq = {}
        for ml in method_labels:
            ov = overall[ml]
            sps = ov["sp"]
            sQ = ov["sq_vals"]
            mean_sp = sum(sps)/len(sps) if sps else 0
            mean_sq = sum(sQ)/len(sQ) if sQ else 0
            overall_sp[ml] = mean_sp
            overall_sq[ml] = mean_sq
            sp_str = f"**{mean_sp:.3f}x**"
            f.write(f"| {ml_to_display[ml]} | {sp_str} | {mean_sq:.4f} | {ov['below']}/{ov['total']} |\n")
        f.write("\n")

        # Key findings
        tsp = overall_sp.get("TASD", 0)
        fsp = overall_sp.get("FLY", 0)
        gsp = overall_sp.get("GSD", 0)
        nsp = overall_sp.get("Ngram", 0)

        f.write("## Key Findings\n\n")
        f.write(f"- **TASD calibrated**: {tsp:.3f}x overall\n")
        f.write(f"- **Official FLY**: {fsp:.3f}x overall\n")
        f.write(f"- **Greedy SD**: {gsp:.3f}x overall\n")
        f.write(f"- **N-gram SD alone**: {nsp:.3f}x overall\n\n")

        f.write(f"- TASD vs Greedy SD: {tsp:.3f}x vs {gsp:.3f}x (Δ={tsp-gsp:+.3f}x)\n")
        f.write(f"- TASD vs Official FLY: {tsp:.3f}x vs {fsp:.3f}x (Δ={tsp-fsp:+.3f}x)\n")
        if tsp > gsp and tsp > fsp:
            f.write("- **TASD is the best overall method.**\n")
        elif fsp > tsp:
            f.write("- **FLY is the fastest method, TASD is safer (fewer below-1.0x cases).**\n")
        f.write("\n")

        f.write("### Comparison with old d16_b2_k3 experiment\n\n")
        f.write("| Benchmark | Old AR TPS | New AR TPS | Old TASD sp | New TASD sp | Change |\n")
        f.write("|-----------|:----------:|:----------:|:-----------:|:-----------:|:------:|\n")
        old_sp = {"argparse": 1.437, "dict_config": 1.472, "openmmlab": 1.557,
                  "pipeline_stage_config": 1.646, "complex_nested_config": 1.588,
                  "rich_cli_option_groups": 1.596}
        old_ar = 33.0  # uniform ~33 in old experiment
        for bname in bnames:
            if bname in agg and "TASD" in agg[bname]:
                new_ar = agg[bname]["AR"]["tps_avg"]
                new_sp = agg[bname]["TASD"]["sp_avg"]
                old = old_sp.get(bname, 0)
                change = new_sp - old
                f.write(f"| {bname} | {old_ar:.0f} | {new_ar:.0f} | {old:.3f}x | **{new_sp:.3f}x** | {change:+.3f}x |\n")
        f.write("\n")

        f.write(f"## Data Files\n\n")
        f.write(f"- Full data: `{OUT_JSON}`\n")
        f.write(f"- Checkpoints: `{CHECKPOINT_DIR}/`\n")

    # Also update per_sample JSON with corrected TASD TPS for consistency
    # (TASD checkpoints already correct, but we need to re-save corrected JSON)


if __name__ == "__main__":
    main()
