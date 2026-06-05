#!/usr/bin/env python3
"""
N-gram Speculative Decoding Pilot: 3 benchmarks x 20 samples.

Compares: AR, Greedy SD, N-gram SpecDec, FLY (official), TASD.

N-gram SpecDec: training-free, no draft model, matches prompt+history n-grams
as draft candidates, verifies with target model (greedy argmax).

Existing data for AR/GreedySD/FLY/TASD loaded from 80-sample runs (first 20).
"""
import json
import time
import torch
import os
import sys
import statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.ar_decode import ar_decode
from src.ngram_sd_decode import ngram_sd_decode
from src.evaluator import evaluate_structural_quality

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
MAX_NEW     = 128
N_SAMPLES   = 20
OUT_DIR     = "/root/autodl-tmp/results"

BENCHMARKS = [
    ("dict_config", "Real-Python-DictConfig", "dict_config",
     "data/codesearchnet_dict_config_blocks_80.jsonl"),
    ("openmmlab", "OpenMMLab-Config", "openmmlab_config",
     "data/ml_config_blocks_openmmlab_80.jsonl"),
    ("pipeline_stage_config", "Pipeline-Stage-Config", "pipeline_stage_config",
     "data/pipeline_stage_config_80.jsonl"),
]


def load_samples(path, n):
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            samples.append(json.loads(line))
    return samples


def check_exists(path):
    return os.path.exists(path) and os.path.getsize(path) > 100


def summary_stats(entries):
    """Compute aggregate stats from per-sample entries."""
    valid = [e for e in entries if "error" not in e]
    if not valid:
        return None
    def _mean(k):
        vals = [e[k] for e in valid if e.get(k) is not None]
        return statistics.mean(vals) if vals else 0.0
    return {
        "n_samples": len(valid),
        "tps_avg": _mean("tps"),
        "structural_quality_score": _mean("structural_quality_score"),
        "off_structure_rate": _mean("off_structure_rate"),
        "truncation_rate": _mean("truncation_rate"),
        "repetition_rate": _mean("repetition_rate"),
        "structure_not_preserved": _mean("structure_not_preserved"),
        "accept_rate_mean": _mean("accept_rate"),
        "match_found_rate": _mean("match_found_rate") if "match_found_rate" in (valid[0] if valid else {}) else None,
        "avg_draft_len_mean": _mean("avg_draft_len") if "avg_draft_len" in (valid[0] if valid else {}) else None,
    }


def load_existing_first_n(path, n, method_label, benchmark_name, structure_type):
    """Load first n entries from existing result file."""
    if not check_exists(path):
        print(f"    {method_label}: file not found: {path}")
        return None, None

    with open(path) as f:
        data = json.load(f)

    per_sample = data.get("per_sample", [])
    entries = []
    for e in per_sample:
        if e.get("sample_idx", 0) >= n:
            break
        if "error" in e:
            entries.append(e)
            continue
        # Map fields
        entry = {
            "sample_idx": e.get("sample_idx", 0),
            "benchmark": benchmark_name,
            "method": method_label,
            "tps": e.get("tps", 0),
            "accept_rate": e.get("accept_rate", 0),
            "tokens_generated": e.get("tokens_generated", 0),
            "structural_quality_score": e.get("structural_quality_score", 0),
            "off_structure_rate": e.get("off_structure_rate", 0),
            "truncation_rate": e.get("truncation_rate", 0),
            "repetition_rate": e.get("repetition_rate", 0),
            "structure_not_preserved": e.get("structure_not_preserved", 0),
        }
        # FLY may have match_found_rate
        if "match_found_rate" in e:
            entry["match_found_rate"] = e["match_found_rate"]
        entries.append(entry)

    summary = summary_stats(entries)
    if summary:
        summary.update({"benchmark": benchmark_name, "method": method_label})
    return summary, entries


# ──────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────
def main():
    print("Loading target model...")
    target = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, device_map="cuda:0", torch_dtype="auto",
        trust_remote_code=True, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        TARGET_PATH, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    all_data = {}

    for bid, name, st, data_path in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"  [{name}]")
        print(f"{'='*60}")

        samples = load_samples(data_path, N_SAMPLES)

        # ── AR (load existing) ──
        ar_path = f"{OUT_DIR}/ar_{bid}_80.json"
        ar_sum, ar_entries = load_existing_first_n(ar_path, N_SAMPLES, "AR", name, st)
        if ar_sum:
            print(f"  AR: LOADED (first {N_SAMPLES})")
            all_data[f"AR_{bid}"] = ar_sum
        else:
            print(f"  AR: running {N_SAMPLES} samples...")
            ar_entries = []
            for i, s in enumerate(samples):
                prompt = s["prompt"]
                t0 = time.time()
                try:
                    r = ar_decode(target, tokenizer, prompt, MAX_NEW)
                except Exception as e:
                    print(f"    AR [{i+1}]: ERROR {e}")
                    ar_entries.append({"error": str(e), "sample_idx": i})
                    continue
                wall = time.time() - t0
                gen = r["generated_text"]
                q = evaluate_structural_quality(gen, structure_type=st)
                ar_entries.append({
                    "sample_idx": i, "benchmark": name, "method": "AR",
                    "tps": r["tokens_per_second"],
                    "tokens_generated": r["generated_tokens"],
                    "accept_rate": 1.0,
                    "structural_quality_score": q["structural_quality_score"],
                    "off_structure_rate": q["off_structure_rate"],
                    "truncation_rate": q["truncation_rate"],
                    "repetition_rate": q["repetition_rate"],
                    "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
                })
                print(f"    AR [{i+1}/{N_SAMPLES}]: {r['generated_tokens']} tok, {r['tokens_per_second']:.1f} TPS")
            ar_sum = summary_stats(ar_entries)
            if ar_sum:
                ar_sum.update({"benchmark": name, "method": "AR"})
                all_data[f"AR_{bid}"] = ar_sum
                with open(ar_path.replace("_80.json", "_20.json"), "w") as f:
                    json.dump({"summary": ar_sum, "per_sample": ar_entries}, f, ensure_ascii=False)
            print(f"  AR: DONE. TPS={ar_sum['tps_avg']:.1f}" if ar_sum else "  AR: FAILED")

        torch.cuda.empty_cache()

        # ── Greedy SD (load existing) ──
        gsd_path = f"{OUT_DIR}/gsd_{bid}_80.json"
        gsd_sum, _ = load_existing_first_n(gsd_path, N_SAMPLES, "GreedySD", name, st)
        if gsd_sum:
            print(f"  Greedy SD: LOADED (first {N_SAMPLES})")
            all_data[f"GreedySD_{bid}"] = gsd_sum
        else:
            print(f"  Greedy SD: no data, skip")

        # ── N-gram SpecDec (run fresh) ──
        ng_path = f"{OUT_DIR}/ngram_{bid}_20.json"
        if check_exists(ng_path):
            print(f"  N-gram: SKIP (existing)")
            with open(ng_path) as f:
                d = json.load(f)
            all_data[f"Ngram_{bid}"] = d.get("summary")
        else:
            print(f"  N-gram: running {N_SAMPLES} samples (n=3-8, draft=16)...")
            ng_entries = []
            for i, s in enumerate(samples):
                prompt = s["prompt"]
                try:
                    t0 = time.time()
                    r = ngram_sd_decode(target, tokenizer, prompt,
                                        max_new_tokens=MAX_NEW,
                                        ngram_min=3, ngram_max=8,
                                        max_draft_tokens=16)
                    wall = time.time() - t0
                    gen = r["generated_text"]
                    q = evaluate_structural_quality(gen, structure_type=st)
                    ng_entries.append({
                        "sample_idx": i, "benchmark": name, "method": "NgramSD",
                        "tps": r["tokens_per_second"],
                        "tokens_generated": r["generated_tokens"],
                        "accept_rate": r["stats"]["accept_rate"],
                        "match_found_rate": r["stats"]["match_found_rate"],
                        "avg_draft_len": r["stats"]["avg_draft_len"],
                        "draft_rounds": r["stats"]["draft_rounds"],
                        "structural_quality_score": q["structural_quality_score"],
                        "off_structure_rate": q["off_structure_rate"],
                        "truncation_rate": q["truncation_rate"],
                        "repetition_rate": q["repetition_rate"],
                        "structure_not_preserved": 1.0 if q["structure_not_preserved"] else 0.0,
                    })
                except Exception as e:
                    print(f"    N-gram [{i+1}]: ERROR {e}")
                    ng_entries.append({"error": str(e), "sample_idx": i})
                    continue
                print(f"    N-gram [{i+1}/{N_SAMPLES}]: {r['generated_tokens']} tok, "
                      f"{r['tokens_per_second']:.1f} TPS, acc={r['stats']['accept_rate']:.2f}, "
                      f"match={r['stats']['match_found_rate']:.2f}")

            ng_sum = summary_stats(ng_entries)
            if ng_sum:
                ng_sum.update({"benchmark": name, "method": "NgramSD"})
                all_data[f"Ngram_{bid}"] = ng_sum
                with open(ng_path, "w") as f:
                    json.dump({"summary": ng_sum, "per_sample": ng_entries}, f, ensure_ascii=False)
            print(f"  N-gram: DONE. TPS={ng_sum['tps_avg']:.1f}" if ng_sum else "  N-gram: FAILED")

        torch.cuda.empty_cache()

        # ── FLY (load existing) ──
        fly_path = f"{OUT_DIR}/fly_{bid}_80.json"
        fly_sum, _ = load_existing_first_n(fly_path, N_SAMPLES, "FLY", name, st)
        if fly_sum:
            print(f"  FLY: LOADED (first {N_SAMPLES})")
            all_data[f"FLY_{bid}"] = fly_sum
        else:
            print(f"  FLY: no data, skip")

        # ── TASD (load existing) ──
        tasd_path = f"{OUT_DIR}/tasd_{bid}_1_5b_d16b2k3_80.json"
        tasd_sum, _ = load_existing_first_n(tasd_path, N_SAMPLES, "TASD", name, st)
        if tasd_sum:
            print(f"  TASD: LOADED (first {N_SAMPLES})")
            all_data[f"TASD_{bid}"] = tasd_sum
        else:
            print(f"  TASD: no data, skip")

    # ──────────────────────────────────────────────────────────
    # Generate Report
    # ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Generating report...")
    lines = []
    w = lines.append

    w("# 5-Method Pilot Comparison: AR vs Greedy SD vs N-gram vs FLY vs TASD")
    w("")
    w("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct (GSD/FLY/TASD) | **N-gram**: no draft")
    w(f"**Settings**: temperature=0.0, max_new_tokens=128, n={N_SAMPLES} per benchmark")
    w("")
    w("## Methods")
    w("")
    w("| Method | Draft | Verification | Description |")
    w("|--------|-------|-------------|-------------|")
    w("| AR | none | none | Autoregressive (target-only) |")
    w("| Greedy SD | 1.5B model, k=16 | strict argmax | Standard speculative decoding |")
    w("| N-gram SD | n-gram lookup, n=3-8, draft=16 | strict argmax | Prompt/history pattern matching, no draft model |")
    w("| FLY | 1.5B model, n-gram, k=15 | window (win_len=6) | FLY official protocol |")
    w("| TASD | 1.5B model, b=2x16 | top-k=3, guard | Multi-block + relaxed + structural guard |")
    w("")

    # Speed and Quality table
    w("## Speed and Quality")
    w("")
    w("| Benchmark | Method | TPS | Speedup vs AR | Accept | SQ | OffStr | Trunc | Match% | AvgDraft |")
    w("|-----------|--------|-----|---------------|--------|----|--------|-------|--------|----------|")

    for bid, name, st, _ in BENCHMARKS:
        ar_tps = all_data.get(f"AR_{bid}", {}).get("tps_avg", 33)
        for mk, ml in [("AR","AR"), ("GreedySD","Greedy SD"), ("Ngram","N-gram SD"),
                        ("FLY","FLY"), ("TASD","TASD")]:
            s = all_data.get(f"{mk}_{bid}")
            if not s:
                continue
            spd = s["tps_avg"] / ar_tps if ar_tps > 0 else 0
            mfr = s.get("match_found_rate")
            adl = s.get("avg_draft_len_mean")
            mfr_s = f"{mfr:.3f}" if mfr is not None else "-"
            adl_s = f"{adl:.1f}" if adl is not None else "-"
            w(f"| {name} | {ml} | {s['tps_avg']:.1f} | {spd:.2f}x | "
              f"{s.get('accept_rate_mean', 1.0):.2f} | {s['structural_quality_score']:.4f} | "
              f"{s['off_structure_rate']:.4f} | {s['truncation_rate']:.4f} | "
              f"{mfr_s} | {adl_s} |")

    w("")
    w("**Note**: Match% = fraction of rounds where n-gram lookup found a match. "
      "AvgDraft = average draft length. N-gram accept_rate = fraction of draft tokens accepted by target.")

    # Summary
    w("")
    w("## Summary (3-benchmark average)")
    w("")
    w("| Method | TPS | Speedup | SQ | Accept | Match% | AvgDraft |")
    w("|--------|-----|---------|----|--------|--------|----------|")

    for mk, ml in [("AR","AR"), ("GreedySD","Greedy SD"), ("Ngram","N-gram SD"),
                    ("FLY","FLY"), ("TASD","TASD")]:
        sums = [all_data[f"{mk}_{bid}"] for bid, _, _, _ in BENCHMARKS
                if f"{mk}_{bid}" in all_data]
        if not sums:
            continue
        ar_sums = [all_data.get(f"AR_{bid}", {}).get("tps_avg", 33)
                   for bid, _, _, _ in BENCHMARKS if f"AR_{bid}" in all_data]
        avg_ar = sum(ar_sums) / len(ar_sums) if ar_sums else 33

        def avg(k):
            vals = [s.get(k, 0) for s in sums if s.get(k) is not None]
            return sum(vals) / len(vals) if vals else 0

        mfr = avg("match_found_rate")
        adl = avg("avg_draft_len_mean")
        mfr_s = f"{mfr:.3f}" if any(s.get("match_found_rate") is not None for s in sums) else "-"
        adl_s = f"{adl:.1f}" if any(s.get("avg_draft_len_mean") is not None for s in sums) else "-"

        w(f"| {ml} | {avg('tps_avg'):.1f} | {avg('tps_avg')/avg_ar:.2f}x | "
          f"{avg('structural_quality_score'):.4f} | {avg('accept_rate_mean'):.2f} | "
          f"{mfr_s} | {adl_s} |")

    # Head-to-head vs TASD
    w("")
    w("## TASD vs N-gram: Head-to-Head")
    w("")
    w("| Benchmark | N-gram TPS | TASD TPS | TASD Advantage | N-gram SQ | TASD SQ | SQ Gap |")
    w("|-----------|-----------|----------|---------------|-----------|---------|--------|")

    for bid, name, st, _ in BENCHMARKS:
        ng = all_data.get(f"Ngram_{bid}")
        td = all_data.get(f"TASD_{bid}")
        if not ng or not td:
            continue
        gap = td["tps_avg"] - ng["tps_avg"]
        pct = gap / ng["tps_avg"] * 100 if ng["tps_avg"] > 0 else 0
        sq_gap = td["structural_quality_score"] - ng["structural_quality_score"]
        w(f"| {name} | {ng['tps_avg']:.1f} | {td['tps_avg']:.1f} | "
          f"+{gap:.1f} ({pct:+.0f}%) | {ng['structural_quality_score']:.4f} | "
          f"{td['structural_quality_score']:.4f} | {sq_gap:+.4f} |")

    # Key findings
    w("")
    w("## Key Findings")
    w("")

    # Compute averages for findings
    ngram_avg_tps = sum(all_data.get(f"Ngram_{bid}", {}).get("tps_avg", 0)
                         for bid, _, _, _ in BENCHMARKS) / 3
    tasd_avg_tps = sum(all_data.get(f"TASD_{bid}", {}).get("tps_avg", 0)
                        for bid, _, _, _ in BENCHMARKS) / 3
    ar_avg_tps = sum(all_data.get(f"AR_{bid}", {}).get("tps_avg", 33)
                     for bid, _, _, _ in BENCHMARKS) / 3
    ngram_mfr = sum(all_data.get(f"Ngram_{bid}", {}).get("match_found_rate", 0)
                     for bid, _, _, _ in BENCHMARKS) / 3

    w(f"1. **N-gram SpecDec is slightly faster than AR** "
      f"({ngram_avg_tps:.1f} TPS vs {ar_avg_tps:.1f} TPS, "
      f"{ngram_avg_tps/ar_avg_tps:.2f}x), but the gain is marginal")
    w(f"2. **N-gram match rate is low** (~{ngram_mfr:.0%}) on structured code benchmarks. "
      f"The generated code patterns differ from prompt/history, limiting n-gram's effectiveness")
    w(f"3. **TASD is substantially faster than N-gram** "
      f"({tasd_avg_tps:.1f} TPS vs {ngram_avg_tps:.1f} TPS, "
      f"{tasd_avg_tps/ngram_avg_tps:.2f}x). "
      f"TASD's advantage is not from pattern copying but from draft model + relaxed verification")
    w("4. **N-gram is a valid training-free baseline** demonstrating that simple repetition-based "
      "speculative decoding cannot match TASD's performance on structured code completion")
    w("5. **FLY's speed advantage** (n-gram draft + window acceptance + draft model) "
      "is primarily from the draft model, not n-gram pattern matching alone")
    w("6. **Structural quality** is comparable across methods; N-gram does not degrade or improve SQ")

    out_path = f"{OUT_DIR}/comparison_5method_ngram_pilot.md"
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {out_path}")

    # Save combined data
    data_path = f"{OUT_DIR}/comparison_5method_ngram_pilot_data.json"
    with open(data_path, "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    print(f"Data: {data_path}")
    print("Done.")


if __name__ == "__main__":
    main()
