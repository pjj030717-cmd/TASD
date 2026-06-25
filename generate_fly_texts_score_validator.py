#!/usr/bin/env python3
"""
Generate FLY texts for all 480 prompts (6 benchmarks x 80) needed by score validator.

REQUIRES: GPU with Qwen2.5-14B-Instruct-AWQ + Qwen2.5-1.5B-Instruct loaded.
Runtime: ~30-60 minutes for 480 prompts.

Output: results/score_validator_review/private/fly_texts_all_480.json
"""

import json, os, random, time, logging, sys, importlib.util
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TARGET_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2___5-14B-Instruct-AWQ"
DRAFT_PATH = "/root/autodl-tmp/models/.modelscope_cache/Qwen/Qwen2.5-1.5B-Instruct"
MAX_NEW_TOKENS = 128

FLY_K15 = {
    "k": 15, "total_gen_tok": MAX_NEW_TOKENS,
    "enable_fly": True, "win_len": 6, "entropy_thre": 0.3,
    "use_ngram": True, "max_ngram_size": 4, "num_ngram_pred_tokens": 6,
    "verbose": False, "abla_no_window": False, "enable_statistics": True,
}

BENCHMARKS = [
    ("argparse",               "data/codesearchnet_argparse_blocks_80.jsonl"),
    ("dict_config",            "data/codesearchnet_dict_config_blocks_80.jsonl"),
    ("openmmlab_config",       "data/ml_config_blocks_openmmlab_80.jsonl"),
    ("pipeline_stage_config",  "data/pipeline_stage_config_80.jsonl"),
    ("complex_nested_config",  "data/complex_nested_config_80.jsonl"),
    ("rich_cli_option_groups", "data/rich_cli_option_groups_80.jsonl"),
]

OUT_PATH = "results/score_validator_review/private/fly_texts_all_480.json"

# Import FLY
fly_path = os.path.join(os.path.dirname(__file__), "FLy", "fly", "models", "FLy.py")
spec = importlib.util.spec_from_file_location("FLy", fly_path)
FLy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(FLy)
SPDGenerate = FLy.SPDGenerate

# Collect all prompts
all_prompts = []
for bname, data_file in BENCHMARKS:
    with open(data_file) as f:
        samples = [json.loads(line.strip()) for line in f.readlines()]
    for s in samples:
        all_prompts.append((bname, s["name"], s["prompt"]))
    print(f"  {bname}: {len(samples)} prompts")

print(f"\nTotal: {len(all_prompts)} prompts")

# Load models
print("\nLoading models...")
tokenizer = AutoTokenizer.from_pretrained(TARGET_PATH, local_files_only=True, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

target = AutoModelForCausalLM.from_pretrained(
    TARGET_PATH, local_files_only=True, device_map="auto",
    torch_dtype=torch.float16, trust_remote_code=True).eval()

draft = AutoModelForCausalLM.from_pretrained(
    DRAFT_PATH, local_files_only=True, device_map="auto",
    torch_dtype=torch.float16, trust_remote_code=True).eval()
print("Models loaded.")

# Logger
import logging
fly_logger = logging.getLogger("fly")
fly_logger.setLevel(logging.WARNING)
if not fly_logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    fly_logger.addHandler(h)

# Generate
print(f"\nGenerating FLY texts for {len(all_prompts)} prompts...")
fly_texts = {}

for i, (bname, sample_name, full_prompt) in enumerate(all_prompts):
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

    key = f"{bname}/{sample_name}"
    fly_texts[key] = {
        "benchmark": bname,
        "sample_name": sample_name,
        "text": text,
        "tps": round(tps, 2),
        "wall": round(wall, 3),
        "gen_len": gen_len,
        "mat": round(mat, 2),
    }

    if (i + 1) % 50 == 0 or (i + 1) == len(all_prompts):
        print(f"  [{i+1}/{len(all_prompts)}] {key}: tps={tps:.1f}, len={gen_len}")

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(fly_texts, f, indent=2, ensure_ascii=False)

print(f"\nDone! Generated {len(fly_texts)} FLY texts -> {OUT_PATH}")
