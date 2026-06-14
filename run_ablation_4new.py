#!/usr/bin/env python3
"""Qwen 128-token Ablation: 7 TASD variants on 6 benchmarks x 80 samples.
Only runs new variants (4-7), reuses existing results for 1-3."""
import json, os, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(__file__))
from src.tasd_decode import tasd_decode
from src.quality_metrics import compute_composite_sq

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

BENCHMARKS = [
    ("argparse", "data/codesearchnet_argparse_blocks_80.jsonl", "argparse"),
    ("dict_config", "data/codesearchnet_dict_config_blocks_80.jsonl", "dict_config"),
    ("openmmlab_config", "data/ml_config_blocks_openmmlab_80.jsonl", "openmmlab"),
    ("pipeline_stage_config", "data/pipeline_stage_config_80.jsonl", "pipeline"),
    ("complex_nested_config", "data/complex_nested_config_80.jsonl", "complex_nested"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl", "rich_cli"),
]
CHECKPOINT_DIR = "results/checkpoints_ablation"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ── New variants ──
NEW_VARIANTS = {
    "no_relaxed": dict(
        enable_guard=True, enable_relaxed_accept=False,
        guard_calibrated=True,
        enable_failure_aware_fallback=True, fallback_guarded=True,
        fallback_accept_threshold=0.5, fallback_repair_threshold=2,
    ),
    "no_guard": dict(
        enable_guard=False, enable_relaxed_accept=True,
        guard_calibrated=False,
        enable_failure_aware_fallback=True, fallback_guarded=True,
        fallback_accept_threshold=0.5, fallback_repair_threshold=2,
    ),
    "draft_len8": dict(
        enable_guard=True, enable_relaxed_accept=True,
        guard_calibrated=True,
        enable_failure_aware_fallback=True, fallback_guarded=True,
        fallback_accept_threshold=0.5, fallback_repair_threshold=2,
        draft_len=8,
    ),
    "draft_blocks1": dict(
        enable_guard=True, enable_relaxed_accept=True,
        guard_calibrated=True,
        enable_failure_aware_fallback=True, fallback_guarded=True,
        fallback_accept_threshold=0.5, fallback_repair_threshold=2,
        draft_blocks=1,
    ),
}

TASD_COMMON_BASE = dict(
    max_new_tokens=MAX_NEW_TOKENS, draft_len=16, draft_blocks=2,
    top_k_accept=3, min_token_prob=1e-4, prefix_budget=0.2, window_len=2,
)


def run_one(target, draft, tokenizer, prompt, stype, variant_cfg):
    extra = {k: v for k, v in variant_cfg.items()
             if k not in ("draft_len", "draft_blocks")}
    overrides = {}
    if "draft_len" in variant_cfg:
        overrides["draft_len"] = variant_cfg["draft_len"]
    if "draft_blocks" in variant_cfg:
        overrides["draft_blocks"] = variant_cfg["draft_blocks"]
    kw = {**TASD_COMMON_BASE, **overrides, **extra}
    r = tasd_decode(target, draft, tokenizer, prompt, structure_type=stype, **kw)
    s = r["stats"]
    fb_s = s.get("failure_aware_fallback", {}) or {}
    return {"wall": r["elapsed_time"], "tps": r["tokens_per_second"],
            "gen_len": s["generated_length"], "text": r["generated_text"],
            "accept": s["accept_rate"], "repair": s.get("repair_count", 0),
            "guard_trig": s.get("guard_trigger_count", 0),
            "trim": s.get("trim_count", 0),
            "fb_count": fb_s.get("fallback_count", 0),
            "fb_tokens": fb_s.get("total_fallback_tokens", 0)}


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

    with open("results/qwen_5method_6x80_quality.json") as f:
        ar_data = json.load(f)["per_sample"]

    for sn, data_file, stype in BENCHMARKS:
        with open(data_file) as f:
            samples = [json.loads(l.strip()) for l in f.readlines()[:80]]
        n = len(samples)
        prompts = [s["prompt"] for s in samples]
        refs = [s.get("reference", "") for s in samples]
        names = [s["name"] for s in samples]
        ar_tps_list = [r["ar_tps"] for r in ar_data[sn]["AR"]]

        for vname, vcfg in NEW_VARIANTS.items():
            ckpt = os.path.join(CHECKPOINT_DIR, f"{sn}_{vname}.json")
            if os.path.exists(ckpt):
                print(f"  [{sn}] {vname}: already done, skip")
                continue
            print(f"  [{sn}] {vname}...", end=" ", flush=True)
            res = []
            for i in range(n):
                r = run_one(target, draft, tokenizer, prompts[i], stype, vcfg)
                sp = r["tps"] / ar_tps_list[i] if ar_tps_list[i] > 0 else 0
                q = compute_composite_sq(r["text"], refs[i], stype)
                res.append({
                    "name": names[i], "sp": round(sp, 3), "tps": round(r["tps"], 2),
                    "wall": round(r["wall"], 3), "accept": round(r["accept"], 4),
                    "repair": r["repair"], "guard_trig": r["guard_trig"],
                    "trim": r["trim"], "fb_count": r["fb_count"],
                    "fb_tokens": r["fb_tokens"], "gen_len": r["gen_len"],
                    **q,
                })
                if (i+1) % 20 == 0:
                    sp_mu = sum(x["sp"] for x in res)/(i+1)
                    sq_mu = sum(x["composite_sq"] for x in res)/(i+1)
                    fb_t = sum(x["fb_count"] for x in res)
                    print(f"{i+1}(sp={sp_mu:.2f}x,sq={sq_mu:.3f},fb={fb_t})...", end=" ", flush=True)
            with open(ckpt, "w") as fout:
                json.dump(res, fout, indent=2, ensure_ascii=False)
            print("done")

    print("\nAll done.")


if __name__ == "__main__":
    main()
