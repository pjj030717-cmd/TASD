"""
TASD Ablation Study: 3 benchmarks × 40 samples × 6 variants

Benchmarks:
 - Real-Python-DictConfig (40 samples)
 - OpenMMLab-Config (40 samples)
 - Pipeline-Stage-Config (40 samples)

Variants:
 1. TASD full (default: draft_len=16, draft_blocks=2, top_k=3, guard=True, relaxed=True)
 2. TASD w/o relaxed acceptance (enable_relaxed_accept=False)
 3. TASD w/o structural guard (enable_guard=False)
 4. TASD w/o multi-block draft (draft_blocks=1)
 5. TASD draft_len=8 (vs default 16)
 6. TASD draft 3B (vs default 1.5B)

Metrics: TPS, Speedup, Accept rate, SQ, Off-structure rate, Repair count
"""

import json
import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tasd_decode import tasd_decode

# ─── Config ───────────────────────────────────────────────────────────────
MAX_NEW_TOKENS = 128
N_SAMPLES = 40

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_1_5B_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
DRAFT_3B_PATH = "/root/autodl-tmp/models/.modelscope_cache/qwen/Qwen2___5-3B-Instruct"

DATA_FILES = {
    "dict_config": "/root/autodl-tmp/data/codesearchnet_dict_config_blocks_80.jsonl",
    "openmmlab_config": "/root/autodl-tmp/data/ml_config_blocks_openmmlab_80.jsonl",
    "pipeline_stage_config": "/root/autodl-tmp/data/pipeline_stage_config_80.jsonl",
}

VARIANTS = {
    "tasd_full": {
        "label": "TASD (full)",
        "kwargs": {
            "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
            "enable_guard": True, "enable_relaxed_accept": True,
        },
    },
    "tasd_no_relaxed": {
        "label": "TASD w/o relaxed",
        "kwargs": {
            "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
            "enable_guard": True, "enable_relaxed_accept": False,
        },
    },
    "tasd_no_guard": {
        "label": "TASD w/o guard",
        "kwargs": {
            "draft_len": 16, "draft_blocks": 2, "top_k_accept": 3,
            "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
            "enable_guard": False, "enable_relaxed_accept": True,
        },
    },
    "tasd_single_block": {
        "label": "TASD single-block",
        "kwargs": {
            "draft_len": 16, "draft_blocks": 1, "top_k_accept": 3,
            "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
            "enable_guard": True, "enable_relaxed_accept": True,
        },
    },
    "tasd_draft_len_8": {
        "label": "TASD draft_len=8",
        "kwargs": {
            "draft_len": 8, "draft_blocks": 2, "top_k_accept": 3,
            "min_token_prob": 1e-4, "prefix_budget": 0.2, "window_len": 2,
            "enable_guard": True, "enable_relaxed_accept": True,
        },
    },
}

# ─── Helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path, n):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if len(samples) >= n:
                break
    return samples


def run_ar(target_model, tokenizer, prompt, structure_type, max_new_tokens=MAX_NEW_TOKENS):
    """Run baseline AR and return metrics."""
    device = target_model.device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        output = target_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    gen_ids = output[0][input_ids.shape[1]:].tolist()
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    n_tokens = len(gen_ids)
    tps = n_tokens / elapsed if elapsed > 0 else 0

    return {
        "tps": tps,
        "elapsed": elapsed,
        "n_tokens": n_tokens,
        "generated_text": gen_text,
    }


def run_variant(target_model, draft_model, tokenizer, prompt, structure_type, variant_kwargs):
    """Run one TASD variant and return metrics."""
    r = tasd_decode(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        prompt=prompt,
        structure_type=structure_type,
        max_new_tokens=MAX_NEW_TOKENS,
        **variant_kwargs,
    )
    # Normalize keys: tasd_decode returns tokens_per_second, not tps
    r["tps"] = r.get("tokens_per_second", 0)
    # Extract from stats
    stats = r.get("stats", {})
    r["accept_rate"] = stats.get("accept_rate", 0)
    r["repair_count"] = stats.get("repair_count", 0)
    return r


def compute_sq(generated_text, reference_text):
    """Simple structural quality: character-level overlap of key structural chars."""
    if not reference_text:
        return 1.0
    for text in [generated_text, reference_text]:
        if not text.strip():
            return 0.0
    # Count matching structural tokens
    struct_chars = set("{}[]():,=")
    gen_struct = [c for c in generated_text if c in struct_chars]
    ref_struct = [c for c in reference_text if c in struct_chars]
    if not ref_struct:
        return 1.0
    matches = sum(1 for c in gen_struct if c in ref_struct)
    return min(matches / len(ref_struct), 1.0)


def compute_off_structure(generated_text, structure_type):
    """Check if generated text contains off-structure keywords."""
    if structure_type in ("dict_config", "openmmlab_config", "pipeline_stage_config"):
        # Should not contain function/class definitions
        lines = generated_text.split("\n")
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("def ", "class ", "import ", "from ")):
                return True
    return False


# ─── Main ─────────────────────────────────────────────────────────────────

def main():
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, trust_remote_code=True, local_files_only=True)
    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_PATH, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
    )
    draft_1_5b = AutoModelForCausalLM.from_pretrained(
        DRAFT_1_5B_PATH, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
    )
    draft_3b = AutoModelForCausalLM.from_pretrained(
        DRAFT_3B_PATH, torch_dtype="auto", device_map="auto", trust_remote_code=True, local_files_only=True
    )
    print("Models loaded.")

    # Estimate AR TPS once per benchmark
    ar_tps_cache = {}
    for bench_name, data_path in DATA_FILES.items():
        samples = load_jsonl(data_path, 3)
        tps_list = []
        for s in samples[:3]:
            r = run_ar(target_model, tokenizer, s["prompt"], bench_name)
            tps_list.append(r["tps"])
        ar_tps_cache[bench_name] = sum(tps_list) / len(tps_list)
        print(f"  AR TPS ({bench_name}): {ar_tps_cache[bench_name]:.1f}")

    all_results = {}

    for bench_name, data_path in DATA_FILES.items():
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        samples = load_jsonl(data_path, N_SAMPLES)
        ar_tps = ar_tps_cache[bench_name]

        bench_results = {}

        # AR baseline
        print("\n  Running AR baseline...")
        ar_results = []
        for i, s in enumerate(samples):
            r = run_ar(target_model, tokenizer, s["prompt"], bench_name)
            ar_results.append(r)
            if (i + 1) % 10 == 0:
                print(f"    AR: {i+1}/{N_SAMPLES}")
        bench_results["ar"] = {
            "label": "AR (baseline)",
            "samples": ar_results,
        }
        ar_mean_tps = sum(r["tps"] for r in ar_results) / len(ar_results)
        print(f"  AR mean TPS: {ar_mean_tps:.1f}")

        # TASD variants (1.5B draft)
        for var_name, var_cfg in VARIANTS.items():
            print(f"\n  Running {var_cfg['label']}...")
            var_results = []
            for i, s in enumerate(samples):
                r = run_variant(
                    target_model, draft_1_5b, tokenizer,
                    s["prompt"], bench_name, var_cfg["kwargs"],
                )
                # Compute SQ
                sq = compute_sq(r.get("generated_text", ""), s.get("reference", ""))
                off_struct = compute_off_structure(r.get("generated_text", ""), bench_name)
                r["sq"] = sq
                r["off_structure"] = off_struct
                r["speedup"] = r["tps"] / ar_tps if ar_tps > 0 else 0
                var_results.append(r)
                if (i + 1) % 10 == 0:
                    print(f"    {var_cfg['label']}: {i+1}/{N_SAMPLES}")

            mean_tps = sum(r["tps"] for r in var_results) / len(var_results)
            mean_speedup = mean_tps / ar_tps if ar_tps > 0 else 0
            mean_sq = sum(r["sq"] for r in var_results) / len(var_results)
            mean_off = sum(1 for r in var_results if r["off_structure"]) / len(var_results)
            mean_repair = sum(r.get("repair_count", 0) for r in var_results) / len(var_results)
            mean_accept = sum(r.get("accept_rate", 0) for r in var_results) / len(var_results)

            bench_results[var_name] = {
                "label": var_cfg["label"],
                "mean_tps": round(mean_tps, 2),
                "mean_speedup": round(mean_speedup, 2),
                "mean_sq": round(mean_sq, 4),
                "mean_off_structure_rate": round(mean_off, 4),
                "mean_repair_count": round(mean_repair, 2),
                "mean_accept_rate": round(mean_accept, 4),
                "samples": var_results,
            }
            print(f"    TPS={mean_tps:.1f} Speedup={mean_speedup:.2f}x SQ={mean_sq:.4f} "
                  f"OffStr={mean_off:.4f} Repair={mean_repair:.2f} Accept={mean_accept:.4f}")

        # TASD with 3B draft
        print(f"\n  Running TASD (3B draft)...")
        var_3b_results = []
        for i, s in enumerate(samples):
            r = run_variant(
                target_model, draft_3b, tokenizer,
                s["prompt"], bench_name, VARIANTS["tasd_full"]["kwargs"],
            )
            sq = compute_sq(r.get("generated_text", ""), s.get("reference", ""))
            off_struct = compute_off_structure(r.get("generated_text", ""), bench_name)
            r["sq"] = sq
            r["off_structure"] = off_struct
            r["speedup"] = r["tps"] / ar_tps if ar_tps > 0 else 0
            var_3b_results.append(r)
            if (i + 1) % 10 == 0:
                print(f"    TASD (3B): {i+1}/{N_SAMPLES}")

        mean_tps = sum(r["tps"] for r in var_3b_results) / len(var_3b_results)
        mean_speedup = mean_tps / ar_tps if ar_tps > 0 else 0
        mean_sq = sum(r["sq"] for r in var_3b_results) / len(var_3b_results)
        mean_off = sum(1 for r in var_3b_results if r["off_structure"]) / len(var_3b_results)
        mean_repair = sum(r.get("repair_count", 0) for r in var_3b_results) / len(var_3b_results)
        mean_accept = sum(r.get("accept_rate", 0) for r in var_3b_results) / len(var_3b_results)

        bench_results["tasd_3b"] = {
            "label": "TASD (3B draft)",
            "mean_tps": round(mean_tps, 2),
            "mean_speedup": round(mean_speedup, 2),
            "mean_sq": round(mean_sq, 4),
            "mean_off_structure_rate": round(mean_off, 4),
            "mean_repair_count": round(mean_repair, 2),
            "mean_accept_rate": round(mean_accept, 4),
            "samples": var_3b_results,
        }
        print(f"    TPS={mean_tps:.1f} Speedup={mean_speedup:.2f}x SQ={mean_sq:.4f} "
              f"OffStr={mean_off:.4f} Repair={mean_repair:.2f} Accept={mean_accept:.4f}")

        all_results[bench_name] = bench_results

    # ─── Save JSON ────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    out_path = "results/tasd_ablation_3bench_40.json"

    # Create a summary without per-sample data for readability
    summary = {}
    for bench_name, bench_results in all_results.items():
        summary[bench_name] = {}
        for var_name, var_data in bench_results.items():
            if var_name == "ar":
                ar_mean_tps = sum(r["tps"] for r in var_data["samples"]) / len(var_data["samples"])
                summary[bench_name][var_name] = {
                    "label": var_data["label"],
                    "mean_tps": round(ar_mean_tps, 2),
                }
            else:
                summary[bench_name][var_name] = {
                    "label": var_data["label"],
                    "mean_tps": var_data["mean_tps"],
                    "mean_speedup": var_data["mean_speedup"],
                    "mean_sq": var_data["mean_sq"],
                    "mean_off_structure_rate": var_data["mean_off_structure_rate"],
                    "mean_repair_count": var_data["mean_repair_count"],
                    "mean_accept_rate": var_data["mean_accept_rate"],
                }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {out_path}")

    # ─── Generate MD report ───────────────────────────────────────────────
    md_path = "results/tasd_ablation_3bench_40.md"
    with open(md_path, "w") as f:
        f.write("# TASD Ablation Study\n\n")
        f.write("**Benchmarks**: Real-Python-DictConfig, OpenMMLab-Config, Pipeline-Stage-Config (40 samples each)\n")
        f.write("**Target**: Qwen2.5-14B-Instruct-AWQ | **Draft**: Qwen2.5-1.5B-Instruct (default), Qwen2.5-3B-Instruct (3B variant)\n")
        f.write("**Settings**: max_new_tokens=128, temperature=0.0\n\n")

        # Summary table per benchmark
        for bench_name, bench_results in all_results.items():
            f.write(f"## {bench_name}\n\n")
            f.write("| Variant | TPS | Speedup | SQ | OffStr | Repair | Accept |\n")
            f.write("|---------|-----|---------|----|--------|--------|--------|\n")
            for var_name in ["ar", "tasd_full", "tasd_no_relaxed", "tasd_no_guard",
                             "tasd_single_block", "tasd_draft_len_8", "tasd_3b"]:
                if var_name not in bench_results:
                    continue
                vd = bench_results[var_name]
                if var_name == "ar":
                    ar_mean_tps = sum(r["tps"] for r in vd["samples"]) / len(vd["samples"])
                    f.write(f"| AR (baseline) | {ar_mean_tps:.1f} | 1.00x | - | - | - | - |\n")
                else:
                    f.write(f"| {vd['label']} | {vd['mean_tps']:.1f} | {vd['mean_speedup']:.2f}x | "
                            f"{vd['mean_sq']:.4f} | {vd['mean_off_structure_rate']:.4f} | "
                            f"{vd['mean_repair_count']:.2f} | {vd['mean_accept_rate']:.4f} |\n")
            f.write("\n")

        # Cross-benchmark summary
        f.write("## Cross-Benchmark Summary (Speedup)\n\n")
        f.write("| Variant | DictConfig | OpenMMLab | Pipeline | Mean |\n")
        f.write("|---------|-----------|-----------|----------|------|\n")
        for var_name in ["tasd_full", "tasd_no_relaxed", "tasd_no_guard",
                         "tasd_single_block", "tasd_draft_len_8", "tasd_3b"]:
            row = [var_name]
            speeds = []
            for bench_name in ["dict_config", "openmmlab_config", "pipeline_stage_config"]:
                if var_name in all_results[bench_name]:
                    s = all_results[bench_name][var_name]["mean_speedup"]
                    row.append(f"{s:.2f}x")
                    speeds.append(s)
                else:
                    row.append("-")
            if speeds:
                row.append(f"{sum(speeds)/len(speeds):.2f}x")
            f.write("| " + " | ".join(row) + " |\n")

    print(f"Report saved to {md_path}")
    print("\nDone!")


if __name__ == "__main__":
    main()
