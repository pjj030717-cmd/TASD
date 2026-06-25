#!/usr/bin/env python3
"""
Generate FLY texts for the 60 prompts selected for human blind review.

FLY checkpoint files (results/qwen_6x80_checkpoints/*_FLY.json) only store
aggregate stats (tps, sp, sq, mat, ngram_acc) without the generated text.
This script re-runs Official FLY (k=15) on exactly those 60 prompts
and saves the texts to a standalone file.

REQUIRES: GPU with Qwen2.5-14B-Instruct-AWQ + Qwen2.5-1.5B-Instruct loaded.
Runtime: ~10-15 minutes for 60 prompts.

Output: results/human_blind_review/fly_texts_for_blind_review.json
"""

import json
import os
import time
import logging
import importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Paths ────────────────────────────────────────────────────────────────
TARGET_PATH = "Qwen/Qwen2.5-14B-Instruct-AWQ"
DRAFT_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

FLY_K15 = {
    "k": 15, "total_gen_tok": MAX_NEW_TOKENS,
    "enable_fly": True, "win_len": 6, "entropy_thre": 0.3,
    "use_ngram": True, "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
    "verbose": False, "abla_no_window": False, "enable_statistics": True,
}

# ─── Import FLY ──────────────────────────────────────────────────────────
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

# ─── Load missing sample list ────────────────────────────────────────────
missing_file = "results/human_blind_review_missing_samples.json"
with open(missing_file) as f:
    missing = json.load(f)

fly_missing = missing["fly_missing_samples"]
print(f"FLY texts to generate: {len(fly_missing)}")

# ─── Load models ─────────────────────────────────────────────────────────
print("Loading models...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True,
                                           trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

target = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, local_files_only=True, device_map="auto",
    torch_dtype=torch.float16, trust_remote_code=True).eval()

draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, local_files_only=True, device_map="auto",
    torch_dtype=torch.float16, trust_remote_code=True).eval()
print("Models loaded.")

# ─── FLY logger ──────────────────────────────────────────────────────────
fly_logger = logging.getLogger("fly")
fly_logger.setLevel(logging.WARNING)
if not fly_logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    fly_logger.addHandler(h)

# ─── Generate ────────────────────────────────────────────────────────────
fly_texts = {}  # {blind_id: text}

for i, item in enumerate(fly_missing):
    bid = item["blind_id"]
    prompt_text = item["prompt_preview"].rstrip("...")

    # We stored only a 200-char preview; need full prompt from original data
    # The preview may be truncated, so reconstruct from the sample name
    bname = item["benchmark"]
    sample_name = item["original_sample_name"]

    # Load full prompt from benchmark file
    # Map benchmark to data file
    bench_files = {
        "argparse": "data/codesearchnet_argparse_blocks_80.jsonl",
        "dict_config": "data/codesearchnet_dict_config_blocks_80.jsonl",
        "openmmlab_config": "data/ml_config_blocks_openmmlab_80.jsonl",
        "pipeline_stage_config": "data/pipeline_stage_config_80.jsonl",
        "complex_nested_config": "data/complex_nested_config_80.jsonl",
        "rich_cli_option_groups": "data/rich_cli_option_groups_80.jsonl",
    }

    data_file = bench_files.get(bname)
    if not data_file or not os.path.exists(data_file):
        print(f"  [{i+1}/{len(fly_missing)}] {bid} SKIP: data file not found for {bname}")
        continue

    with open(data_file) as f:
        samples = [json.loads(line.strip()) for line in f.readlines()]

    full_prompt = None
    for s in samples:
        if s["name"] == sample_name:
            full_prompt = s["prompt"]
            break

    if full_prompt is None:
        print(f"  [{i+1}/{len(fly_missing)}] {bid} SKIP: sample {sample_name} not found")
        continue

    # Run FLY
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
    prompt_len = input_ids.shape[1]
    spd_gen = SPDGenerate(draft_model=draft, target_model=target,
                          tokenizer=tokenizer, cuslog=fly_logger, spd_args=FLY_K15)

    torch.cuda.synchronize()
    t0 = time.time()
    full_ids = spd_gen.generate_chunks(input_ids, temperature=0.0)
    torch.cuda.synchronize()
    wall = time.time() - t0

    gen_ids = full_ids[0][prompt_len:].tolist()
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    gen_len = len(gen_ids)
    tps = gen_len / wall if wall > 0 else 0.0
    n_acc = spd_gen.num_accepted_tokens.item() if spd_gen._counter_inited else 0
    n_emitted = spd_gen.num_emitted_tokens.item() if spd_gen._counter_inited else gen_len
    mat = n_acc / n_emitted if n_emitted > 0 else 0

    fly_texts[bid] = {
        "blind_id": bid,
        "benchmark": bname,
        "sample_name": sample_name,
        "text": text,
        "tps": round(tps, 2),
        "wall": round(wall, 3),
        "gen_len": gen_len,
        "mat": round(mat, 2),
    }

    print(f"  [{i+1}/{len(fly_missing)}] {bid} ({bname}/{sample_name}): "
          f"tps={tps:.1f}, len={gen_len}")

# ─── Save ────────────────────────────────────────────────────────────────
out_path = "results/human_blind_review/fly_texts_for_blind_review.json"
with open(out_path, "w") as f:
    json.dump(fly_texts, f, indent=2, ensure_ascii=False)

print(f"\nGenerated {len(fly_texts)}/{len(fly_missing)} FLY texts.")
print(f"Saved to: {out_path}")

# Instructions for next step
if len(fly_texts) < len(fly_missing):
    print(f"\nWARNING: {len(fly_missing) - len(fly_texts)} texts could not be generated.")
    print("Check the missing sample names and data files.")

print("\nNext: Run python prepare_blind_review.py again with FLY texts available")
print("  to regenerate HTML annotators with complete data.")
